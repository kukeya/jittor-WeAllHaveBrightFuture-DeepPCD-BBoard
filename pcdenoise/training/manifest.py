"""Atomic, auditable experiment-run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_snapshot(value: object, *, name: str) -> object:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be finite JSON-compatible data") from error
    return json.loads(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(
    source_paths: Mapping[str, os.PathLike[str] | str],
) -> dict[str, str]:
    if not isinstance(source_paths, Mapping) or not source_paths:
        raise ValueError("source_paths must be a non-empty mapping")
    hashes = {}
    for logical_name, raw_path in source_paths.items():
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or "\x00" in logical_name
        ):
            raise ValueError("source hash names must be non-empty strings")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[logical_name] = _sha256_file(path)
    return dict(sorted(hashes.items()))


def capture_environment(
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Capture stable runtime facts without importing the Jittor package."""

    packages = {}
    for distribution in ("jittor", "numpy", "tensorboard", "tensorboardX"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    environment: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "cc_path",
        "nvcc_path",
    ):
        if variable in os.environ:
            environment[variable] = os.environ[variable]
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise TypeError("environment must be a mapping")
        captured_extra = _json_snapshot(extra, name="environment")
        environment.update(captured_extra)
    return environment


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".manifest.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _rollback_initial_run_directory(
    run_path: Path,
    *,
    device: int,
    inode: int,
) -> None:
    """Remove only the just-created run directory after initial-write failure."""

    try:
        metadata = os.lstat(run_path)
    except FileNotFoundError:
        return
    if (
        metadata.st_dev != device
        or metadata.st_ino != inode
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
    ):
        raise RuntimeError(
            f"refusing to clean a replaced run directory: {run_path}"
        )
    for child in run_path.iterdir():
        child_metadata = os.lstat(child)
        is_owned_manifest = (
            child.name == "manifest.json"
            and stat.S_ISREG(child_metadata.st_mode)
            and not stat.S_ISLNK(child_metadata.st_mode)
        )
        is_owned_temporary = (
            child.name.startswith(".manifest.")
            and child.name.endswith(".tmp")
            and stat.S_ISREG(child_metadata.st_mode)
            and not stat.S_ISLNK(child_metadata.st_mode)
        )
        if not (is_owned_manifest or is_owned_temporary):
            raise RuntimeError(
                f"refusing to remove unexpected run artifact: {child}"
            )
        child.unlink()
    run_path.rmdir()


@dataclass
class RunManifest:
    """Mutable status handle backed by atomically replaced JSON."""

    run_dir: Path
    path: Path
    _data: dict[str, object]

    @property
    def status(self) -> str:
        return str(self._data["status"])

    @property
    def data(self) -> dict[str, object]:
        return _json_snapshot(self._data, name="manifest")

    def _finish(
        self,
        status: str,
        *,
        result: Mapping[str, object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self.status != "running":
            raise RuntimeError(
                f"run is already terminal with status {self.status!r}"
            )
        candidate = _json_snapshot(self._data, name="manifest")
        if result is not None:
            if not isinstance(result, Mapping):
                raise TypeError("result must be a mapping")
            candidate["result"] = _json_snapshot(result, name="result")
        if error is not None:
            candidate["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        now = _utc_now()
        candidate["status"] = status
        candidate["updated_at"] = now
        candidate["ended_at"] = now
        try:
            _atomic_write_json(self.path, candidate)
        except OSError:
            # os.replace may already have committed candidate before a
            # directory-fsync error. Re-read the authoritative file so this
            # handle never reports "running" while disk reports a terminal
            # state. A pre-replace failure leaves both states unchanged.
            try:
                disk_data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                disk_data = None
            if disk_data == candidate:
                self._data = candidate
            elif disk_data != self._data:
                raise RuntimeError(
                    "manifest write failed with ambiguous on-disk state"
                )
            raise
        self._data = candidate

    def complete(
        self, result: Mapping[str, object] | None = None
    ) -> None:
        self._finish("completed", result=result)

    def fail(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        self._finish("failed", error=error)

    def __enter__(self) -> "RunManifest":
        if self.status != "running":
            raise RuntimeError("only a running manifest can enter a context")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_value is None:
            if self.status == "running":
                self.complete()
        elif self.status == "running":
            self.fail(exc_value)
        return False


def create_run(
    run_dir: os.PathLike[str] | str,
    *,
    config: Mapping[str, object],
    seed: int,
    split_sha256: str,
    source_paths: Mapping[str, os.PathLike[str] | str],
    environment: Mapping[str, object] | None = None,
) -> RunManifest:
    """Create a new run directory and its initial ``running`` manifest."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    config_snapshot = _json_snapshot(config, name="config")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(split_sha256, str) or not SHA256_RE.fullmatch(
        split_sha256
    ):
        raise ValueError("split_sha256 must be 64 lowercase hexadecimal chars")
    hashes = _source_hashes(source_paths)
    captured_environment = capture_environment(environment)

    run_path = Path(run_dir)
    if run_path.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {run_path}")
    run_path.mkdir(parents=True, exist_ok=False)
    run_metadata = os.lstat(run_path)
    manifest_path = run_path / "manifest.json"
    now = _utc_now()
    data: dict[str, object] = {
        "format_version": 1,
        "run_id": run_path.name,
        "status": "running",
        "config": config_snapshot,
        "seed": seed,
        "split_sha256": split_sha256,
        "environment": captured_environment,
        "source_sha256": hashes,
        "started_at": now,
        "updated_at": now,
        "ended_at": None,
    }
    try:
        _atomic_write_json(manifest_path, data)
    except BaseException:
        _rollback_initial_run_directory(
            run_path,
            device=run_metadata.st_dev,
            inode=run_metadata.st_ino,
        )
        raise
    return RunManifest(
        run_dir=run_path.resolve(),
        path=manifest_path.resolve(),
        _data=data,
    )
