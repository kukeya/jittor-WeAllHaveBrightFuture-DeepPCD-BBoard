#!/usr/bin/env python3
"""Run one frozen PGD1 over a PGD2 noisy cache in resumable GPU chunks."""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
import queue
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from pcdenoise.inference_config import load_inference_config


EXPECTED_SAMPLE_COUNT = 35_534
EXPECTED_POINT_COUNT = 50_000
DEFAULT_CHUNK_SIZE = 512
FROZEN_E200_CHECKPOINT_SHA256 = (
    "d679e174564442f1f6c031b50fbd45f94d6035c7c089e555be5aa54f2834f2c0"
)
FROZEN_E200_CONFIG_SHA256 = (
    "fba86f44277df8d81ff301e6201a4d1157e0a3052658f32f61e8bda572e22d99"
)
FROZEN_E200_CHECKPOINT_STEP = 712_800
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYNSET_RE = re.compile(r"^[0-9]{8}$")
_MODEL_RE = re.compile(r"^[0-9a-f]{28,32}$")
_MANIFEST_FIELDS = {
    "format",
    "format_version",
    "status",
    "sample_count",
    "sample_ids",
    "patch_size",
    "seed_k",
    "patch_batch_size",
    "niters",
    "normalization_mode",
    "robust_quantile",
    "fusion_mode",
    "iteration_damping",
    "model_reference",
    "elapsed_seconds",
    "samples",
}


@dataclass(frozen=True)
class ModelBinding:
    checkpoint_sha256: str
    config_sha256: str
    checkpoint_step: int


@dataclass(frozen=True)
class InferenceOptions:
    patch_size: int
    seed_k: float
    patch_batch_size: int
    niters: int
    normalization_mode: str
    robust_quantile: float | None
    fusion_mode: str
    iteration_damping: float

    def manifest_fields(self) -> dict[str, object]:
        return {
            "patch_size": self.patch_size,
            "seed_k": self.seed_k,
            "patch_batch_size": self.patch_batch_size,
            "niters": self.niters,
            "normalization_mode": self.normalization_mode,
            "robust_quantile": self.robust_quantile,
            "fusion_mode": self.fusion_mode,
            "iteration_damping": self.iteration_damping,
        }


@dataclass(frozen=True)
class Chunk:
    index: int
    sample_ids: tuple[str, ...]
    root: Path

    @property
    def ids_path(self) -> Path:
        return self.root / "sample_ids.txt"

    @property
    def prediction_dir(self) -> Path:
        return self.root / "prediction"

    @property
    def log_path(self) -> Path:
        return self.root / "denoise.log"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=EXPECTED_SAMPLE_COUNT,
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--patch-batch-size", type=int, default=20)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument(
        "--normalization-mode",
        choices=("noisy_max", "identity", "robust_quantile"),
        default="noisy_max",
    )
    parser.add_argument("--robust-quantile", type=float)
    parser.add_argument(
        "--fusion-mode",
        choices=("hard_best",),
        default="hard_best",
    )
    parser.add_argument("--iteration-damping", type=float, default=1.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path("/usr/local/cuda-12.4"),
    )
    parser.add_argument(
        "--cc-path",
        type=Path,
        default=Path(os.environ.get("cc_path", "/usr/bin/g++")),
    )
    parser.add_argument("--jittor-home-root", type=Path)
    return parser


def _strict_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            payload = stream.read(chunk_size)
            if not payload:
                break
            digest.update(payload)
    return digest.hexdigest()


def _preflight_checkpoint(path: Path, *, expected_sha256: str) -> None:
    expected = _strict_sha256(
        expected_sha256,
        name="expected checkpoint SHA256",
    )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"checkpoint must be a regular file: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError(f"checkpoint SHA256 sidecar is missing: {sidecar}")
    claimed = sidecar.read_text(encoding="ascii").strip()
    _strict_sha256(claimed, name="checkpoint SHA256 sidecar")
    actual = _sha256_file(path)
    if claimed != actual:
        raise ValueError("checkpoint SHA256 does not match its sidecar")
    if actual != expected:
        raise ValueError("checkpoint SHA256 does not match frozen expectation")


def _parse_physical_gpus(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--gpus must be a comma-separated GPU ID list") from error
    if not parsed or len(parsed) != len(set(parsed)) or any(gpu < 0 for gpu in parsed):
        raise ValueError("--gpus must contain distinct nonnegative GPU IDs")
    return parsed


def _sample_id(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError as error:
        raise ValueError(f"noisy sample is outside input cache: {path}") from error
    if len(parts) != 4 or parts[0] != "shapenet" or parts[-1] != "noisy.npy":
        raise ValueError(f"invalid noisy sample layout: {path}")
    synset, model_id = parts[1], parts[2]
    if not _SYNSET_RE.fullmatch(synset) or not _MODEL_RE.fullmatch(model_id):
        raise ValueError(f"invalid noisy sample ID: {synset}/{model_id}")
    return f"{synset}/{model_id}"


def scan_input_ids(root: Path) -> tuple[str, ...]:
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"input cache must be a directory: {root}")
    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("noisy.npy")):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"noisy input must be a regular file: {path}")
        sample_id = _sample_id(path, root)
        if sample_id in indexed:
            raise ValueError(f"duplicate noisy sample ID: {sample_id}")
        indexed[sample_id] = path
    if not indexed:
        raise ValueError("input cache contains no noisy.npy samples")
    return tuple(sorted(indexed))


def verify_input_cache_manifest(
    root: Path,
    *,
    scanned_ids: Sequence[str],
) -> dict[str, object]:
    manifest = _load_manifest(root / "manifest.json")
    expected_ids = tuple(sorted(scanned_ids))
    if (
        manifest.get("format") != "pcdenoise_pgd2_noisy_cache_v1"
        or manifest.get("format_version") != 1
        or manifest.get("status") != "completed"
    ):
        raise ValueError("input cache manifest is not completed PGD2 noisy v1")
    if manifest.get("shape_count") != len(expected_ids):
        raise ValueError("input cache manifest shape_count mismatch")
    if manifest.get("shape_ids") != list(expected_ids):
        raise ValueError("input cache manifest shape_ids mismatch")
    noisy_input_sha256_by_id(manifest, sample_ids=expected_ids)
    return manifest


def noisy_input_sha256_by_id(
    manifest: Mapping[str, object],
    *,
    sample_ids: Sequence[str],
) -> dict[str, str]:
    """Return the noisy bytes bound to every selected sample in the manifest."""

    expected_ids = tuple(sorted(sample_ids))
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise ValueError("input cache manifest sample records are incomplete")
    result: dict[str, str] = {}
    for expected_id, record in zip(expected_ids, records, strict=True):
        if not isinstance(record, dict) or record.get("shape_id") != expected_id:
            raise ValueError("input cache manifest sample record order mismatch")
        result[expected_id] = _strict_sha256(
            record.get("noisy_sha256"),
            name=f"{expected_id} noisy SHA256",
        )
    return result


def _expected_input_sha256(
    sample_ids: Sequence[str],
    input_sha256_by_id: Mapping[str, str],
) -> dict[str, str]:
    expected_ids = tuple(sorted(sample_ids))
    if not isinstance(input_sha256_by_id, Mapping):
        raise TypeError("input_sha256_by_id must be a mapping")
    if set(input_sha256_by_id) != set(expected_ids):
        raise ValueError("input SHA256 mapping does not exactly cover sample IDs")
    return {
        sample_id: _strict_sha256(
            input_sha256_by_id[sample_id],
            name=f"{sample_id} expected input SHA256",
        )
        for sample_id in expected_ids
    }


def partition_chunks(
    sample_ids: Sequence[str],
    *,
    chunk_size: int,
    state_dir: Path,
) -> tuple[Chunk, ...]:
    size = _positive_integer(chunk_size, name="chunk_size")
    ordered = tuple(sorted(sample_ids))
    if len(set(ordered)) != len(ordered):
        raise ValueError("sample IDs contain duplicates")
    return tuple(
        Chunk(
            index=index,
            sample_ids=ordered[start : start + size],
            root=state_dir / "chunks" / f"chunk_{index:05d}",
        )
        for index, start in enumerate(range(0, len(ordered), size))
    )


def _write_or_verify_ids(chunk: Chunk) -> None:
    chunk.root.mkdir(parents=True, exist_ok=True)
    expected = "".join(f"{sample_id}\n" for sample_id in chunk.sample_ids)
    if chunk.ids_path.exists():
        if chunk.ids_path.is_symlink() or not chunk.ids_path.is_file():
            raise ValueError(f"chunk sample list is not a regular file: {chunk.ids_path}")
        if chunk.ids_path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"chunk sample list changed: {chunk.ids_path}")
        return
    temporary = chunk.ids_path.with_name(
        f".{chunk.ids_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, chunk.ids_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"inference manifest must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid inference manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"inference manifest root must be a mapping: {path}")
    return value


def _regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")


def _verify_prediction_array(
    path: Path,
    *,
    sample_id: str,
    expected_sha256: str,
    expected_point_count: int,
) -> None:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"prediction output SHA256 mismatch: {sample_id}")
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (EOFError, OSError, ValueError) as error:
        raise ValueError(
            f"prediction output must be a non-pickle NPY: {sample_id}"
        ) from error
    if not isinstance(values, np.ndarray):
        raise ValueError(f"prediction output must be an NPY array: {sample_id}")
    if values.dtype != np.float32:
        raise ValueError(f"prediction output must use float32: {sample_id}")
    if values.shape != (expected_point_count, 3):
        raise ValueError(
            "prediction output shape mismatch: "
            f"{sample_id} expected=({expected_point_count},3) actual={values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"prediction output must be finite: {sample_id}")


def verify_prediction_dir(
    prediction_dir: Path,
    *,
    sample_ids: Sequence[str],
    binding: ModelBinding,
    options: InferenceOptions,
    input_sha256_by_id: Mapping[str, str],
) -> dict[str, object]:
    if not prediction_dir.is_dir() or prediction_dir.is_symlink():
        raise ValueError(f"prediction directory is invalid: {prediction_dir}")
    expected_ids = tuple(sorted(sample_ids))
    expected_input_sha256 = _expected_input_sha256(
        expected_ids,
        input_sha256_by_id,
    )
    manifest = _load_manifest(prediction_dir / "inference_manifest.json")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("inference manifest fields are not standard")
    if (
        manifest.get("format") != "pcdenoise_prediction_v1"
        or manifest.get("format_version") != 1
        or manifest.get("status") != "completed"
    ):
        raise ValueError("inference manifest is not completed prediction v1")
    if manifest.get("sample_count") != len(expected_ids):
        raise ValueError("inference manifest sample_count mismatch")
    if manifest.get("sample_ids") != list(expected_ids):
        raise ValueError("inference manifest sample IDs mismatch")
    for name, expected in options.manifest_fields().items():
        if manifest.get(name) != expected:
            raise ValueError(f"inference manifest {name} mismatch")
    reference = manifest.get("model_reference")
    if not isinstance(reference, dict):
        raise ValueError("inference manifest model_reference is invalid")
    if reference.get("checkpoint_sha256") != binding.checkpoint_sha256:
        raise ValueError("inference manifest checkpoint SHA256 mismatch")
    if reference.get("config_sha256") != binding.config_sha256:
        raise ValueError("inference manifest config SHA256 mismatch")
    if reference.get("checkpoint_step") != binding.checkpoint_step:
        raise ValueError("inference manifest checkpoint step mismatch")
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise ValueError("inference manifest sample records are incomplete")
    record_ids: list[str] = []
    expected_files = {"inference_manifest.json"}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("inference manifest sample record is invalid")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("inference manifest sample record ID is invalid")
        relative = f"shapenet/{sample_id}/denoised.npy"
        if record.get("relative_path") != relative:
            raise ValueError(f"inference relative path mismatch: {sample_id}")
        input_sha256 = _strict_sha256(
            record.get("input_sha256"),
            name=f"{sample_id} input SHA256",
        )
        if input_sha256 != expected_input_sha256.get(sample_id):
            raise ValueError(f"inference input SHA256 mismatch: {sample_id}")
        _strict_sha256(
            record.get("output_sha256"),
            name=f"{sample_id} output SHA256",
        )
        point_count = record.get("point_count")
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count != EXPECTED_POINT_COUNT
        ):
            raise ValueError(f"inference point_count mismatch: {sample_id}")
        output_path = prediction_dir / relative
        _regular_file(output_path, label="prediction output")
        _verify_prediction_array(
            output_path,
            sample_id=sample_id,
            expected_sha256=record["output_sha256"],
            expected_point_count=point_count,
        )
        expected_files.add(relative)
        record_ids.append(sample_id)
    if record_ids != list(expected_ids):
        raise ValueError("inference manifest sample record order mismatch")
    actual_files = {
        path.relative_to(prediction_dir).as_posix()
        for path in prediction_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("prediction directory files are not exact")
    return manifest


def _chunk_command(
    chunk: Chunk,
    *,
    args: argparse.Namespace,
) -> list[str]:
    script = Path(__file__).resolve().with_name("denoise.py")
    command = [
        str(args.python),
        str(script),
        "--inference-config",
        str(args.inference_config),
        "--checkpoint",
        str(args.checkpoint),
        "--input-dir",
        str(args.input_cache),
        "--output-dir",
        str(chunk.prediction_dir),
        "--sample-ids",
        str(chunk.ids_path),
        "--patch-size",
        str(args.patch_size),
        "--seed-k",
        str(args.seed_k),
        "--patch-batch-size",
        str(args.patch_batch_size),
        "--niters",
        str(args.niters),
        "--normalization-mode",
        str(args.normalization_mode),
        "--fusion-mode",
        str(args.fusion_mode),
        "--iteration-damping",
        str(args.iteration_damping),
    ]
    if args.robust_quantile is not None:
        command.extend(("--robust-quantile", str(args.robust_quantile)))
    return command


def _chunk_environment(
    gpu: int,
    *,
    args: argparse.Namespace,
) -> dict[str, str]:
    if gpu < 0:
        raise ValueError(f"GPU ID must be nonnegative: {gpu}")
    environment = os.environ.copy()
    # Importing Jittor mutates the parent environment with cache selectors.
    # Never let an unrelated earlier import override this launcher's pinned,
    # shared cache and architecture contract.
    environment.pop("cache_name", None)
    environment.pop("cuda_arch", None)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CUDA_HOME"] = str(args.cuda_home)
    environment["cc_path"] = str(args.cc_path)
    environment["nvcc_path"] = str(args.cuda_home / "bin" / "nvcc")
    environment["conv_opt"] = "1"
    environment["cuda_archs"] = "89"
    # Jittor 1.3.10 rewrites ~/.cache/jittor/config.json non-atomically when
    # two processes request different homes.  Both cards are sm_89 4090s, so
    # one prewarmed home is valid for both and avoids that upstream race.
    environment["JITTOR_HOME"] = str(args.jittor_home_root)
    return environment


def _run_chunk_process(
    chunk: Chunk,
    *,
    gpu: int,
    args: argparse.Namespace,
) -> None:
    environment = _chunk_environment(gpu, args=args)
    command = _chunk_command(chunk, args=args)
    chunk.log_path.parent.mkdir(parents=True, exist_ok=True)
    with chunk.log_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "event": "chunk_start",
                    "chunk_index": chunk.index,
                    "gpu": gpu,
                    "sample_count": len(chunk.sample_ids),
                    "command": command,
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        log.write(
            json.dumps(
                {
                    "event": "chunk_exit",
                    "chunk_index": chunk.index,
                    "gpu": gpu,
                    "returncode": completed.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
    if completed.returncode != 0:
        raise RuntimeError(
            f"chunk {chunk.index:05d} failed on GPU {gpu}; "
            f"see {chunk.log_path}"
        )


def run_missing_chunks(
    chunks: Sequence[Chunk],
    *,
    args: argparse.Namespace,
    binding: ModelBinding,
    options: InferenceOptions,
    input_sha256_by_id: Mapping[str, str],
) -> None:
    all_sample_ids = tuple(
        sample_id for chunk in chunks for sample_id in chunk.sample_ids
    )
    expected_input_sha256 = _expected_input_sha256(
        all_sample_ids,
        input_sha256_by_id,
    )
    pending: queue.Queue[Chunk] = queue.Queue()
    for chunk in chunks:
        _write_or_verify_ids(chunk)
        if chunk.prediction_dir.exists():
            verify_prediction_dir(
                chunk.prediction_dir,
                sample_ids=chunk.sample_ids,
                binding=binding,
                options=options,
                input_sha256_by_id={
                    sample_id: expected_input_sha256[sample_id]
                    for sample_id in chunk.sample_ids
                },
            )
        else:
            pending.put(chunk)
    if pending.empty():
        return

    stop = threading.Event()
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while not stop.is_set():
            try:
                chunk = pending.get_nowait()
            except queue.Empty:
                return
            try:
                _run_chunk_process(chunk, gpu=gpu, args=args)
                verify_prediction_dir(
                    chunk.prediction_dir,
                    sample_ids=chunk.sample_ids,
                    binding=binding,
                    options=options,
                    input_sha256_by_id={
                        sample_id: expected_input_sha256[sample_id]
                        for sample_id in chunk.sample_ids
                    },
                )
            except BaseException as error:
                with failure_lock:
                    failures.append(error)
                stop.set()
            finally:
                pending.task_done()

    workers = [
        threading.Thread(
            target=worker,
            args=(gpu,),
            name=f"pgd1-gpu{gpu}",
        )
        for gpu in args.physical_gpus
    ]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    if failures:
        raise RuntimeError(
            f"PGD1 chunk preparation stopped after {len(failures)} failure(s)"
        ) from failures[0]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination, follow_symlinks=False)


def merge_chunks(
    chunks: Sequence[Chunk],
    *,
    output_root: Path,
    all_sample_ids: Sequence[str],
    binding: ModelBinding,
    options: InferenceOptions,
    input_sha256_by_id: Mapping[str, str],
) -> dict[str, object]:
    ordered_ids = tuple(sorted(all_sample_ids))
    expected_input_sha256 = _expected_input_sha256(
        ordered_ids,
        input_sha256_by_id,
    )
    if output_root.exists():
        return verify_prediction_dir(
            output_root,
            sample_ids=ordered_ids,
            binding=binding,
            options=options,
            input_sha256_by_id=expected_input_sha256,
        )

    manifests = [
        verify_prediction_dir(
            chunk.prediction_dir,
            sample_ids=chunk.sample_ids,
            binding=binding,
            options=options,
            input_sha256_by_id={
                sample_id: expected_input_sha256[sample_id]
                for sample_id in chunk.sample_ids
            },
        )
        for chunk in chunks
    ]
    references = [manifest["model_reference"] for manifest in manifests]
    reference_payloads = {
        json.dumps(
            reference,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        for reference in references
    }
    if len(reference_payloads) != 1:
        raise ValueError("chunk model_reference values differ")
    records = {
        str(record["sample_id"]): (chunk, record)
        for chunk, manifest in zip(chunks, manifests, strict=True)
        for record in manifest["samples"]
    }
    if tuple(sorted(records)) != ordered_ids:
        raise ValueError("chunk sample records do not exactly cover input cache")

    stage = _create_owned_stage(output_root)
    try:
        merged_records = []
        for sample_id in ordered_ids:
            chunk, record = records[sample_id]
            relative = Path(str(record["relative_path"]))
            _link_or_copy(
                chunk.prediction_dir / relative,
                stage.path / relative,
            )
            merged_records.append(record)
        merged = {
            "format": "pcdenoise_prediction_v1",
            "format_version": 1,
            "status": "completed",
            "sample_count": len(ordered_ids),
            "sample_ids": list(ordered_ids),
            **options.manifest_fields(),
            "model_reference": references[0],
            "elapsed_seconds": sum(
                float(manifest["elapsed_seconds"]) for manifest in manifests
            ),
            "samples": merged_records,
        }
        manifest_path = stage.path / "inference_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(
                merged,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        published = _publish_owned_stage(stage)
    except BaseException:
        _cleanup_owned_stage(stage)
        raise
    return verify_prediction_dir(
        published,
        sample_ids=ordered_ids,
        binding=binding,
        options=options,
        input_sha256_by_id=expected_input_sha256,
    )


def _state_directory(output_root: Path, requested: Path | None) -> Path:
    state = (
        requested
        if requested is not None
        else output_root.with_name(f".{output_root.name}.pgd1-chunks")
    )
    output = output_root.resolve(strict=False)
    resolved_state = state.resolve(strict=False)
    if resolved_state == output or output in resolved_state.parents:
        raise ValueError("state directory must be outside final output root")
    if resolved_state.exists() and (
        not resolved_state.is_dir() or resolved_state.is_symlink()
    ):
        raise ValueError(f"state directory is invalid: {resolved_state}")
    resolved_state.mkdir(parents=True, exist_ok=True)
    return resolved_state


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    args.physical_gpus = _parse_physical_gpus(args.gpus)
    expected_count = _positive_integer(
        args.expected_sample_count,
        name="expected_sample_count",
    )
    if expected_count != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"expected_sample_count must remain frozen at {EXPECTED_SAMPLE_COUNT}"
        )
    _positive_integer(args.chunk_size, name="chunk_size")
    args.input_cache = args.input_cache.resolve()
    args.output_root = args.output_root.resolve(strict=False)
    args.inference_config = args.inference_config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.cuda_home = args.cuda_home.resolve()
    args.cc_path = args.cc_path.resolve()
    if not args.cc_path.is_file():
        raise FileNotFoundError(f"C++ compiler is missing: {args.cc_path}")
    state_dir = _state_directory(args.output_root, args.state_dir)
    args.jittor_home_root = (
        args.jittor_home_root.resolve(strict=False)
        if args.jittor_home_root is not None
        else state_dir / "jittor_home"
    )
    args.jittor_home_root.mkdir(parents=True, exist_ok=True)

    inference_config = load_inference_config(args.inference_config)
    binding = ModelBinding(
        checkpoint_sha256=str(inference_config["checkpoint_sha256"]),
        config_sha256=str(inference_config["canonical_config_sha256"]),
        checkpoint_step=int(inference_config["checkpoint_step"]),
    )
    if binding.checkpoint_sha256 != FROZEN_E200_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint SHA256 does not match frozen e200 expectation"
        )
    if binding.config_sha256 != FROZEN_E200_CONFIG_SHA256:
        raise ValueError("config SHA256 does not match frozen e200 expectation")
    if binding.checkpoint_step != FROZEN_E200_CHECKPOINT_STEP:
        raise ValueError("checkpoint step does not match frozen e200 expectation")
    options = InferenceOptions(
        patch_size=_positive_integer(args.patch_size, name="patch_size"),
        seed_k=float(args.seed_k),
        patch_batch_size=_positive_integer(
            args.patch_batch_size,
            name="patch_batch_size",
        ),
        niters=_positive_integer(args.niters, name="niters"),
        normalization_mode=str(args.normalization_mode),
        robust_quantile=args.robust_quantile,
        fusion_mode=str(args.fusion_mode),
        iteration_damping=float(args.iteration_damping),
    )
    _preflight_checkpoint(
        args.checkpoint,
        expected_sha256=binding.checkpoint_sha256,
    )
    sample_ids = scan_input_ids(args.input_cache)
    if len(sample_ids) != expected_count:
        raise ValueError(
            f"input cache sample count {len(sample_ids)} does not match "
            f"required {expected_count}"
        )
    input_manifest = verify_input_cache_manifest(
        args.input_cache,
        scanned_ids=sample_ids,
    )
    input_sha256_by_id = noisy_input_sha256_by_id(
        input_manifest,
        sample_ids=sample_ids,
    )
    chunks = partition_chunks(
        sample_ids,
        chunk_size=args.chunk_size,
        state_dir=state_dir,
    )
    run_missing_chunks(
        chunks,
        args=args,
        binding=binding,
        options=options,
        input_sha256_by_id=input_sha256_by_id,
    )
    manifest = merge_chunks(
        chunks,
        output_root=args.output_root,
        all_sample_ids=sample_ids,
        binding=binding,
        options=options,
        input_sha256_by_id=input_sha256_by_id,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "chunk_count": len(chunks),
                "chunk_size": args.chunk_size,
                "output_root": str(args.output_root),
                "state_dir": str(state_dir),
                "elapsed_seconds": time.perf_counter() - started,
                "model_reference": manifest["model_reference"],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
