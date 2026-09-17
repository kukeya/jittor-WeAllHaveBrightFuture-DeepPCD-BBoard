#!/usr/bin/env python3
"""Fast PGD1 -> FPS4 PGD2 -> FPS4 PGD2-beta inference pipeline.

One resident worker is created per physical GPU.  The four semantic FPS
strategies are distributed round-robin across those workers, so two GPUs run
two strategies each.  Every worker loads the frozen PGD2 model once and keeps
it resident across both PGD2 passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.inference_config import load_inference_config


STRATEGIES = ("index0", "centroid_far", "x_min", "x_max")


def _select_start_index(points: np.ndarray, strategy: str) -> int:
    """Select one README semantic FPS start from the original point order."""

    cloud = np.asarray(points)
    if (
        cloud.dtype != np.float32
        or cloud.ndim != 2
        or cloud.shape[0] == 0
        or cloud.shape[1] != 3
        or not np.isfinite(cloud).all()
    ):
        raise ValueError("FPS selection points must be finite float32 (N, 3)")
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported FPS strategy: {strategy}")
    if strategy == "index0":
        return 0
    coordinates = cloud.astype(np.float64)
    if strategy == "centroid_far":
        center = coordinates.mean(axis=0, dtype=np.float64)
        delta = coordinates - center
        return int(np.argmax(np.einsum("ij,ij->i", delta, delta)))
    if strategy == "x_min":
        return int(np.argmin(coordinates[:, 0]))
    return int(np.argmax(coordinates[:, 0]))


def _assign_strategies(
    devices: list[int],
    strategy_counts: list[int] | None = None,
) -> dict[int, tuple[str, ...]]:
    """Distribute strategies evenly or by explicit per-device counts."""

    if not devices or len(devices) > len(STRATEGIES):
        raise ValueError("devices must contain between one and four GPU IDs")
    if len(devices) != len(set(devices)) or any(device < 0 for device in devices):
        raise ValueError("devices must contain distinct nonnegative GPU IDs")
    assigned: dict[int, list[str]] = {device: [] for device in devices}
    if strategy_counts is None:
        for index, strategy in enumerate(STRATEGIES):
            assigned[devices[index % len(devices)]].append(strategy)
    else:
        if (
            len(strategy_counts) != len(devices)
            or any(count <= 0 for count in strategy_counts)
            or sum(strategy_counts) != len(STRATEGIES)
        ):
            raise ValueError(
                "strategy-counts must contain one positive count per device "
                f"and sum to {len(STRATEGIES)}"
            )
        offset = 0
        for device, count in zip(devices, strategy_counts):
            assigned[device].extend(STRATEGIES[offset : offset + count])
            offset += count
    return {device: tuple(values) for device, values in assigned.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    """Fast atomic-tree writer; the parent stage is published atomically."""

    result = np.ascontiguousarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("prediction must be finite float32 with shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, result, allow_pickle=False)


def _scan_ids(root: Path, filename: str) -> list[str]:
    indexed = []
    for path in root.glob(f"shapenet/*/*/{filename}"):
        relative = path.relative_to(root).parts
        indexed.append(f"{relative[1]}/{relative[2]}")
    result = sorted(indexed)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"invalid or empty {filename} input tree: {root}")
    return result


def _relative(sample_id: str, filename: str) -> Path:
    return Path("shapenet").joinpath(*sample_id.split("/"), filename)


def _worker(
    *,
    strategies: tuple[str, ...],
    device: int,
    project_root: str,
    jittor_home: str | None,
    checkpoint: str,
    inference_config_path: str,
    pgd1_root: str,
    x2_root: str,
    first_root: str,
    second_root: str,
    skip_first_pass: bool,
    sample_ids: list[str],
    seed_k: float,
    patch_batch_size: int,
    random_seed: int,
    import_lock: object,
    x2_ready: object,
    abort: object,
    messages: object,
) -> None:
    try:
        config = load_inference_config(Path(inference_config_path))
        config_sha256 = str(config["canonical_config_sha256"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        # Match the submitted PGD inference implementation without requiring
        # a separate cuDNN installation in the CUDA 12.4 environment.
        os.environ.setdefault("conv_opt", "1")
        if jittor_home is not None:
            Path(jittor_home).mkdir(parents=True, exist_ok=True)
            os.environ["JITTOR_HOME"] = jittor_home
        sys.path.insert(0, project_root)

        # Jittor updates a process-global cache config during import.  Serialize
        # the two imports and use one shared cascade cache to avoid corrupting
        # that config while still keeping both models resident concurrently.
        with import_lock:
            import jittor as jt
            from pcdenoise.data.mesh_dataset import fit_unit_sphere
            from pcdenoise.inference import denoise_normalized_cloud
            from pcdenoise.models.factory import build_pgd_model
            from pcdenoise.training.checkpoint import load_checkpoint

        jt.flags.use_cuda = int(bool(config["use_cuda"]))
        jt.set_global_seed(random_seed)
        model = build_pgd_model(config["model"], config["data"])
        loaded = load_checkpoint(
            checkpoint,
            model=model,
            expected_config_sha256=config_sha256,
        )
        if (
            loaded["checkpoint_sha256"] != config["checkpoint_sha256"]
            or loaded["step"] != config["checkpoint_step"]
        ):
            raise ValueError(
                "PGD2 checkpoint digest or step differs from inference config"
            )
        model.eval()
        model_reference = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": loaded["checkpoint_sha256"],
            "checkpoint_step": int(loaded["step"]),
            "config_sha256": config_sha256,
        }

        def run_pass(
            input_root: Path,
            output_root: Path,
            pass_index: int,
        ) -> None:
            output_root.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(
                prefix=f".{output_root.name}.build-", dir=output_root.parent
            ))
            started = time.perf_counter()
            try:
                for index, sample_id in enumerate(sample_ids, start=1):
                    if abort.is_set():
                        raise RuntimeError("cascade aborted by another worker")
                    path = input_root / _relative(sample_id, "denoised.npy")
                    points = np.load(path, allow_pickle=False)
                    if (
                        points.dtype != np.float32
                        or points.ndim != 2
                        or points.shape[1] != 3
                        or not np.isfinite(points).all()
                    ):
                        raise ValueError(f"invalid stage input: {sample_id}")
                    transform = fit_unit_sphere(points)
                    normalized = transform.apply(points)
                    for strategy in strategies:
                        start_index = _select_start_index(points, strategy)
                        prediction, _ = denoise_normalized_cloud(
                            model,
                            normalized,
                            patch_size=int(config["model"]["patch_size"]),
                            seed_k=seed_k,
                            patch_batch_size=patch_batch_size,
                            fusion_mode="hard_best",
                            fps_start_index=start_index,
                        )
                        relative = _relative(sample_id, "normalized.npy")
                        _save_npy(
                            stage / strategy / relative,
                            np.asarray(prediction, dtype=np.float32),
                        )
                    print(
                        f"pass={pass_index} gpu={device} "
                        f"strategies={','.join(strategies)} "
                        f"sample={index}/{len(sample_ids)} {sample_id}",
                        flush=True,
                    )
                manifest = {
                    "format": "pcdenoise_fast_fps4_gpu_plans_v1",
                    "status": "completed",
                    "pass_index": pass_index,
                    "strategies": list(strategies),
                    "physical_device": device,
                    "sample_count": len(sample_ids),
                    "sample_ids": sample_ids,
                    "seed_k": seed_k,
                    "patch_size": int(config["model"]["patch_size"]),
                    "patch_batch_size": patch_batch_size,
                    "random_seed": random_seed,
                    "model_reference": model_reference,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                _write_json(stage / "partial_manifest.json", manifest)
                os.replace(stage, output_root)
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise

        if not skip_first_pass:
            run_pass(
                Path(pgd1_root),
                Path(first_root) / f"gpu{device}",
                1,
            )
        messages.put({"status": "pass1_done", "device": device})
        while not abort.is_set() and not x2_ready.wait(timeout=1.0):
            pass
        if abort.is_set():
            return
        run_pass(
            Path(x2_root),
            Path(second_root) / f"gpu{device}",
            2,
        )
        messages.put({"status": "pass2_done", "device": device})
    except BaseException as error:
        abort.set()
        x2_ready.set()
        messages.put({
            "status": "error",
            "strategies": list(strategies),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })


def _wait_stage(messages: object, processes: list[mp.Process], expected: str) -> None:
    completed = set()
    while len(completed) < len(processes):
        try:
            message = messages.get(timeout=2.0)
        except queue.Empty:
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"PGD2 worker exited unexpectedly: {failed[0].exitcode}")
            continue
        if message.get("status") == "error":
            raise RuntimeError(
                f"PGD2 worker {message.get('strategies')} failed: {message['error']}\n"
                f"{message['traceback']}"
            )
        if message.get("status") != expected:
            raise RuntimeError(f"unexpected worker message: {message}")
        completed.add(message["device"])


def _merge(
    *,
    input_root: Path,
    plan_root: Path,
    output_root: Path,
    assignment: dict[int, tuple[str, ...]],
    sample_ids: list[str],
    residual_scale: float,
    stage_name: str,
    output_reference: dict[str, object],
) -> float:
    from pcdenoise.data.mesh_dataset import fit_unit_sphere

    if output_root.exists():
        raise FileExistsError(output_root)
    manifests = []
    for device, strategies in assignment.items():
        path = plan_root / f"gpu{device}" / "partial_manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document.get("status") != "completed"
            or document.get("physical_device") != device
            or document.get("strategies") != list(strategies)
            or document.get("sample_ids") != sample_ids
        ):
            raise ValueError(f"invalid or misaligned GPU plan set: {path}")
        manifests.append(document)
    references = {
        json.dumps(document.get("model_reference"), sort_keys=True)
        for document in manifests
    }
    pass_indices = {document.get("pass_index") for document in manifests}
    if len(references) != 1 or len(pass_indices) != 1:
        raise ValueError("FPS4 plans disagree on model reference or pass index")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.build-", dir=output_root.parent
    ))
    records = []
    strategy_devices = {
        strategy: device
        for device, strategies in assignment.items()
        for strategy in strategies
    }
    started = time.perf_counter()
    try:
        for sample_id in sample_ids:
            input_path = input_root / _relative(sample_id, "denoised.npy")
            points = np.load(input_path, allow_pickle=False)
            plans = [
                np.load(
                    plan_root
                    / f"gpu{strategy_devices[strategy]}"
                    / strategy
                    / _relative(sample_id, "normalized.npy"),
                    allow_pickle=False,
                )
                for strategy in STRATEGIES
            ]
            stacked = np.stack(plans, axis=0)
            if stacked.dtype != np.float32:
                raise ValueError(f"FPS4 components must be float32: {sample_id}")
            mean_normalized = np.mean(stacked, axis=0, dtype=np.float32)
            refined = fit_unit_sphere(points).restore(mean_normalized)
            output = (
                points
                + np.float32(residual_scale)
                * (refined.astype(np.float32) - points.astype(np.float32))
            ).astype(np.float32)
            relative = _relative(sample_id, "denoised.npy")
            _save_npy(stage / relative, output)
            records.append({
                "sample_id": sample_id,
                "relative_path": relative.as_posix(),
                "point_count": len(points),
            })
        elapsed = time.perf_counter() - started
        _write_json(stage / "inference_manifest.json", {
            "format": "pcdenoise_cascade_prediction_v1",
            "format_version": 1,
            "status": "completed",
            "stage": stage_name,
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "fps_strategies": list(STRATEGIES),
            "cross_plan_fusion": "arithmetic_mean",
            "residual_scale": residual_scale,
            "elapsed_seconds": elapsed,
            "model_reference": output_reference,
            "samples": records,
        })
        os.replace(stage, output_root)
        return elapsed
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="checkout containing the compatible pcdenoise package and scripts",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--pgd1-root",
        type=Path,
        help="reuse a completed PGD1 output tree instead of rerunning PGD1",
    )
    parser.add_argument("--pgd1-inference-config", type=Path)
    parser.add_argument("--pgd1-checkpoint", type=Path)
    parser.add_argument("--pgd2-checkpoint", type=Path, required=True)
    parser.add_argument("--pgd2-inference-config", type=Path, required=True)
    parser.add_argument(
        "--reuse-x2-root",
        type=Path,
        help=(
            "reuse a completed first-pass FPS4 plus alpha output and run only "
            "the second PGD2 pass"
        ),
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        required=True,
        help="one to four distinct physical GPU IDs; two GPUs get two plans each",
    )
    parser.add_argument(
        "--strategy-counts",
        type=int,
        nargs="+",
        help=(
            "optional number of FPS strategies assigned to each --devices entry; "
            "counts must be positive and sum to four"
        ),
    )
    parser.add_argument("--pgd1-device", type=int)
    parser.add_argument(
        "--jittor-home-root",
        type=Path,
        help="optional shared persistent Jittor cache for the cascade workers",
    )
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=1.12)
    parser.add_argument("--beta", type=float, default=0.59)
    parser.add_argument("--pgd1-patch-batch-size", type=int, default=40)
    parser.add_argument("--pgd2-patch-batch-size", type=int, default=160)
    parser.add_argument("--random-seed", type=int, default=20260819)
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N samples for bounded diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_root.exists() or args.work_root.exists():
        raise FileExistsError("output-root and work-root must not already exist")
    assignment = _assign_strategies(args.devices, args.strategy_counts)
    if args.pgd1_device is None:
        args.pgd1_device = args.devices[0]
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.seed_k <= 0 or args.pgd1_patch_batch_size <= 0 or args.pgd2_patch_batch_size <= 0:
        raise ValueError("seed-k and patch batch sizes must be positive")
    if not all(np.isfinite(value) and value >= 0 for value in (args.alpha, args.beta)):
        raise ValueError("alpha and beta must be finite and nonnegative")
    if not args.project_root.is_dir():
        raise FileNotFoundError(args.project_root)
    sys.path.insert(0, str(args.project_root.resolve()))
    for path in (args.pgd2_checkpoint, args.pgd2_inference_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    pgd2_config = load_inference_config(args.pgd2_inference_config)
    pgd2_config_sha256 = str(pgd2_config["canonical_config_sha256"])
    if _sha256(args.pgd2_checkpoint) != pgd2_config["checkpoint_sha256"]:
        raise ValueError("PGD2 checkpoint digest differs from inference config")
    if int(pgd2_config["model"]["patch_size"]) != 1000:
        raise ValueError("README FPS4 requires PGD2 patch_size=1000")
    if args.pgd1_root is None:
        if (
            args.pgd1_inference_config is None
            or args.pgd1_checkpoint is None
        ):
            raise ValueError(
                "PGD1 execution requires --pgd1-inference-config and "
                "--pgd1-checkpoint"
            )
        for path in (args.pgd1_inference_config, args.pgd1_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        pgd1_config = load_inference_config(args.pgd1_inference_config)
        if _sha256(args.pgd1_checkpoint) != pgd1_config["checkpoint_sha256"]:
            raise ValueError("PGD1 checkpoint digest differs from inference config")
    else:
        pgd1_config = None
    if args.pgd1_root is not None and not args.pgd1_root.is_dir():
        raise FileNotFoundError(args.pgd1_root)
    if args.reuse_x2_root is not None and not args.reuse_x2_root.is_dir():
        raise FileNotFoundError(args.reuse_x2_root)

    sample_ids = _scan_ids(args.input_root, "noisy.npy")
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    args.work_root.mkdir(parents=True)
    status_path = args.work_root / "cascade_status.json"
    status = {
        "format": "pcdenoise_pgd1_pgd2_pgd2_pipeline_v1",
        "status": "pgd1",
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "devices": args.devices,
        "device_strategy_assignment": {
            str(device): list(strategies)
            for device, strategies in assignment.items()
        },
        "pgd1_device": args.pgd1_device,
        "seed_k": args.seed_k,
        "alpha": args.alpha,
        "beta": args.beta,
        "pgd2_patch_batch_size": args.pgd2_patch_batch_size,
        "cross_plan_accumulator_dtype": "float32",
        "random_seed": args.random_seed,
        "timings": {},
    }
    _write_json(status_path, status)
    ids_path = args.work_root / "sample_ids.txt"
    ids_path.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
    started_total = time.perf_counter()
    if args.pgd1_root is not None:
        pgd1_root = args.pgd1_root.resolve()
        pgd1_ids = _scan_ids(pgd1_root, "denoised.npy")
        available_pgd1_ids = set(pgd1_ids)
        if any(sample_id not in available_pgd1_ids for sample_id in sample_ids):
            raise ValueError("reused PGD1 sample IDs differ from input-root")
        status["pgd1_reused"] = True
        status["pgd1_root"] = str(pgd1_root)
        status["timings"]["pgd1_wall_seconds"] = 0.0
    else:
        pgd1_root = args.work_root / "pgd1"
        pgd1_started = time.perf_counter()
        denoise_script = args.project_root / "scripts" / "denoise.py"
        if not denoise_script.is_file():
            raise FileNotFoundError(denoise_script)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.pgd1_device)
        if args.jittor_home_root is not None:
            shared_jittor_home = (args.jittor_home_root / "shared").resolve()
            shared_jittor_home.mkdir(parents=True, exist_ok=True)
            environment["JITTOR_HOME"] = str(shared_jittor_home)
        command = [
            sys.executable,
            str(denoise_script),
            "--inference-config",
            str(args.pgd1_inference_config),
            "--checkpoint",
            str(args.pgd1_checkpoint),
            "--input-dir",
            str(args.input_root),
            "--output-dir",
            str(pgd1_root),
            "--patch-batch-size",
            str(args.pgd1_patch_batch_size),
            "--sample-ids",
            str(ids_path),
        ]
        subprocess.run(
            command,
            check=True,
            cwd=args.project_root,
            env=environment,
        )
        pgd1_manifest_path = pgd1_root / "inference_manifest.json"
        pgd1_manifest = json.loads(
            pgd1_manifest_path.read_text(encoding="utf-8")
        )
        pgd1_reference = pgd1_manifest.get("model_reference")
        if (
            pgd1_manifest.get("status") != "completed"
            or not isinstance(pgd1_reference, dict)
            or pgd1_reference.get("config_sha256")
            != pgd1_config["canonical_config_sha256"]
            or pgd1_reference.get("checkpoint_sha256")
            != pgd1_config["checkpoint_sha256"]
            or pgd1_reference.get("checkpoint_step")
            != pgd1_config["checkpoint_step"]
        ):
            raise ValueError("generated PGD1 manifest differs from expectation")
        status["pgd1_reused"] = False
        status["pgd1_root"] = str(pgd1_root.resolve())
        status["timings"]["pgd1_wall_seconds"] = (
            time.perf_counter() - pgd1_started
        )

    context = mp.get_context("spawn")
    x2_ready, abort = context.Event(), context.Event()
    import_lock = context.Lock()
    messages = context.Queue()
    first_root = args.work_root / "pgd2_first_plans"
    second_root = args.work_root / "pgd2_second_plans"
    x2_root = (
        args.reuse_x2_root.resolve()
        if args.reuse_x2_root is not None
        else args.work_root / "x2"
    )
    processes = []
    persistent_started = time.perf_counter()
    for device, strategies in assignment.items():
        jittor_home = None
        if args.jittor_home_root is not None:
            jittor_home = str(
                (args.jittor_home_root / "shared").resolve()
            )
        process = context.Process(target=_worker, kwargs={
            "strategies": strategies,
            "device": device,
            "project_root": str(args.project_root.resolve()),
            "jittor_home": jittor_home,
            "checkpoint": str(args.pgd2_checkpoint),
            "inference_config_path": str(args.pgd2_inference_config),
            "pgd1_root": str(pgd1_root),
            "x2_root": str(x2_root),
            "first_root": str(first_root),
            "second_root": str(second_root),
            "skip_first_pass": args.reuse_x2_root is not None,
            "sample_ids": sample_ids,
            "seed_k": args.seed_k,
            "patch_batch_size": args.pgd2_patch_batch_size,
            "random_seed": args.random_seed,
            "import_lock": import_lock,
            "x2_ready": x2_ready,
            "abort": abort,
            "messages": messages,
        })
        process.start()
        processes.append(process)
    try:
        status["status"] = "pgd2_first"
        _write_json(status_path, status)
        first_started = time.perf_counter()
        _wait_stage(messages, processes, "pass1_done")
        status["timings"]["pgd2_first_wall_seconds"] = time.perf_counter() - first_started
        reference = {
            "checkpoint": str(args.pgd2_checkpoint.resolve()),
            "checkpoint_sha256": pgd2_config["checkpoint_sha256"],
            "config_sha256": pgd2_config_sha256,
        }
        if args.reuse_x2_root is None:
            status["timings"]["x2_merge_seconds"] = _merge(
                input_root=pgd1_root,
                plan_root=first_root,
                output_root=x2_root,
                assignment=assignment,
                sample_ids=sample_ids,
                residual_scale=args.alpha,
                stage_name="pgd2_alpha",
                output_reference=reference,
            )
            status["x2_reused"] = False
        else:
            x2_manifest = json.loads(
                (x2_root / "inference_manifest.json").read_text(encoding="utf-8")
            )
            if (
                _scan_ids(x2_root, "denoised.npy") != sample_ids
                or x2_manifest.get("status") != "completed"
                or x2_manifest.get("sample_ids") != sample_ids
                or x2_manifest.get("residual_scale") != args.alpha
                or x2_manifest.get("model_reference") != reference
            ):
                raise ValueError("reused x2 root differs from the requested contract")
            status["timings"]["x2_merge_seconds"] = 0.0
            status["x2_reused"] = True
            status["x2_root"] = str(x2_root)
        status["status"] = "pgd2_second"
        _write_json(status_path, status)
        second_started = time.perf_counter()
        x2_ready.set()
        _wait_stage(messages, processes, "pass2_done")
        status["timings"]["pgd2_second_wall_seconds"] = time.perf_counter() - second_started
        status["timings"]["final_merge_seconds"] = _merge(
            input_root=x2_root,
            plan_root=second_root,
            output_root=args.output_root,
            assignment=assignment,
            sample_ids=sample_ids,
            residual_scale=args.beta,
            stage_name="pgd2_beta",
            output_reference=reference,
        )
    except BaseException:
        abort.set(); x2_ready.set()
        raise
    finally:
        for process in processes:
            process.join(timeout=60.0)
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join()
    status["timings"]["persistent_pgd2_total_wall_seconds"] = time.perf_counter() - persistent_started
    status["timings"]["total_wall_seconds"] = time.perf_counter() - started_total
    status["status"] = "completed"
    status["output_root"] = str(args.output_root.resolve())
    _write_json(status_path, status)
    final_manifest = json.loads((args.output_root / "inference_manifest.json").read_text(encoding="utf-8"))
    final_manifest["pipeline"] = status
    _write_json(args.output_root / "inference_manifest.json", final_manifest)
    console_status = dict(status)
    console_status.pop("sample_ids", None)
    print(json.dumps(console_status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
