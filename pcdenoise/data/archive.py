"""Safe archive inspection and extraction for the official competition data.

The validators deliberately understand only the two official layouts.  They
first scan every member and reject unsafe or unexpected metadata; extraction
then copies regular-file payloads one by one without using ``extractall``.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Dict, Iterable, Iterator, List, Sequence, Tuple


SYNSET_RE = re.compile(r"^[0-9]{8}$")
MODEL_RE = re.compile(r"^[0-9a-f]{28,32}$")
TRAIN_ROOT = ("dataset_train", "shapenet")
TEST_ARCHIVE_PREFIX = "dataset_test_noisy"
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class ArchiveValidationError(ValueError):
    """Raised when an archive is unsafe or does not match the official layout."""


@dataclass(frozen=True)
class _Member:
    source_name: str
    target_name: str
    is_dir: bool
    size: int


@dataclass(frozen=True)
class _OwnedStage:
    path: Path
    parent: Path
    prefix: str
    destination: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _OwnedArchiveSource:
    path: Path
    descriptor: int
    device: int
    inode: int
    size: int

    def open(self) -> BinaryIO:
        metadata = os.fstat(self.descriptor)
        if (
            metadata.st_dev != self.device
            or metadata.st_ino != self.inode
            or metadata.st_size != self.size
            or not stat.S_ISREG(metadata.st_mode)
        ):
            raise ArchiveValidationError(
                f"archive source identity changed: {self.path}"
            )
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        return os.fdopen(os.dup(self.descriptor), "rb")


@contextmanager
def _open_owned_archive_source(
    archive_path: os.PathLike[str] | str,
) -> Iterator[_OwnedArchiveSource]:
    requested = Path(archive_path)
    if not requested.is_file():
        raise FileNotFoundError(requested)
    absolute = requested.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArchiveValidationError(
                f"archive source must be a regular file: {absolute}"
            )
        yield _OwnedArchiveSource(
            path=absolute,
            descriptor=descriptor,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
        )
    finally:
        os.close(descriptor)


def _sha256_owned_source(
    source: _OwnedArchiveSource,
    chunk_size: int = 8 << 20,
) -> str:
    digest = hashlib.sha256()
    with source.open() as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 << 20) -> str:
    """Return the lowercase SHA256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_parts(raw_name: str, is_dir: bool) -> Tuple[str, ...]:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ArchiveValidationError(f"invalid archive path: {raw_name!r}")
    if raw_name.startswith("/") or re.match(r"^[A-Za-z]:", raw_name):
        raise ArchiveValidationError(f"absolute archive path: {raw_name!r}")
    has_trailing_slash = raw_name.endswith("/")
    if has_trailing_slash and not is_dir:
        raise ArchiveValidationError(
            f"file path has a directory suffix: {raw_name!r}"
        )
    name = raw_name[:-1] if has_trailing_slash else raw_name
    if not name:
        raise ArchiveValidationError(f"empty archive path: {raw_name!r}")
    parts = tuple(name.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ArchiveValidationError(f"unsafe archive path: {raw_name!r}")
    return parts


def _valid_ids(synset: str, model: str) -> bool:
    return bool(SYNSET_RE.fullmatch(synset) and MODEL_RE.fullmatch(model))


def _validate_train_path(
    raw_name: str, is_dir: bool
) -> Tuple[str, str | None, bool]:
    parts = _safe_parts(raw_name, is_dir)
    if parts == (TRAIN_ROOT[0],) and is_dir:
        return parts[0], None, False
    if parts[:2] != TRAIN_ROOT:
        raise ArchiveValidationError(f"unexpected train path: {raw_name!r}")

    if is_dir:
        if len(parts) == 2:
            return "/".join(parts), None, False
        if len(parts) == 3 and SYNSET_RE.fullmatch(parts[2]):
            return "/".join(parts), None, False
        if (
            len(parts) in (4, 5)
            and _valid_ids(parts[2], parts[3])
            and (len(parts) == 4 or parts[4] == "models")
        ):
            shape_id = f"{parts[2]}/{parts[3]}"
            return "/".join(parts), shape_id, True
        raise ArchiveValidationError(
            f"unexpected or invalid train directory: {raw_name!r}"
        )

    if (
        len(parts) == 6
        and _valid_ids(parts[2], parts[3])
        and parts[4:] == ("models", "model_normalized.obj")
    ):
        return "/".join(parts), f"{parts[2]}/{parts[3]}", False
    raise ArchiveValidationError(
        f"unexpected or invalid train file: {raw_name!r}"
    )


def _normalize_test_path(
    raw_name: str, is_dir: bool
) -> Tuple[str, str | None, bool]:
    parts = _safe_parts(raw_name, is_dir)
    if parts and parts[0] == TEST_ARCHIVE_PREFIX:
        parts = parts[1:]
    if not parts and is_dir:
        return "", None, False
    if not parts or parts[0] != "shapenet":
        raise ArchiveValidationError(f"unexpected test path: {raw_name!r}")

    if is_dir:
        if len(parts) == 1:
            return "/".join(parts), None, False
        if len(parts) == 2 and SYNSET_RE.fullmatch(parts[1]):
            return "/".join(parts), None, False
        if len(parts) == 3 and _valid_ids(parts[1], parts[2]):
            shape_id = f"{parts[1]}/{parts[2]}"
            return "/".join(parts), shape_id, True
        raise ArchiveValidationError(
            f"unexpected or invalid test directory: {raw_name!r}"
        )

    if (
        len(parts) == 4
        and _valid_ids(parts[1], parts[2])
        and parts[3] == "noisy.npy"
    ):
        return "/".join(parts), f"{parts[1]}/{parts[2]}", False
    raise ArchiveValidationError(
        f"unexpected or invalid test file: {raw_name!r}"
    )


def _base_inventory(
    source: _OwnedArchiveSource,
    kind: str,
    members: Sequence[_Member],
    shape_ids: Iterable[str],
    declared_ids: Iterable[str],
) -> Dict[str, object]:
    ids = sorted(shape_ids)
    declared = set(declared_ids)
    id_set = set(ids)
    counts: Dict[str, int] = {}
    for shape_id in ids:
        synset = shape_id.split("/", 1)[0]
        counts[synset] = counts.get(synset, 0) + 1
    return {
        "format_version": 1,
        "archive_kind": kind,
        "archive_path": str(source.path),
        "archive_sha256": _sha256_owned_source(source),
        "compressed_bytes": source.size,
        "uncompressed_bytes": sum(
            member.size for member in members if not member.is_dir
        ),
        "member_count": len(members),
        "directory_count": sum(member.is_dir for member in members),
        "shape_ids": ids,
        "synset_counts": dict(sorted(counts.items())),
        "duplicate_ids": [],
        "missing_ids": sorted(declared - id_set),
        "invalid_ids": [],
    }


def _validated_train_member(
    info: tarfile.TarInfo,
) -> Tuple[_Member, str | None, bool]:
    if info.issparse():
        raise ArchiveValidationError(
            f"sparse tar member is forbidden: {info.name!r}"
        )
    if info.type == tarfile.DIRTYPE:
        is_dir = True
    elif info.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
        is_dir = False
    else:
        raise ArchiveValidationError(
            "tar member type is forbidden: "
            f"{info.name!r} type={info.type!r}"
        )
    target, shape_id, declares_shape = _validate_train_path(
        info.name, is_dir
    )
    return (
        _Member(
            source_name=info.name,
            target_name=target,
            is_dir=is_dir,
            size=info.size if not is_dir else 0,
        ),
        shape_id,
        declares_shape,
    )


def _scan_train_source(
    source: _OwnedArchiveSource,
) -> Tuple[Dict[str, object], List[_Member]]:
    members: List[_Member] = []
    targets = set()
    shape_ids = set()
    declared_ids = set()
    try:
        with source.open() as archive_stream:
            with tarfile.open(fileobj=archive_stream, mode="r:*") as stream:
                for info in stream:
                    member, shape_id, declares_shape = _validated_train_member(
                        info
                    )
                    if member.target_name in targets:
                        raise ArchiveValidationError(
                            "duplicate extraction target: "
                            f"{member.target_name!r}"
                        )
                    targets.add(member.target_name)
                    if shape_id is not None:
                        if not member.is_dir:
                            if shape_id in shape_ids:
                                raise ArchiveValidationError(
                                    f"duplicate shape ID: {shape_id!r}"
                                )
                            shape_ids.add(shape_id)
                        if declares_shape:
                            declared_ids.add(shape_id)
                    members.append(member)
    except (tarfile.TarError, EOFError) as error:
        raise ArchiveValidationError(f"invalid tar archive: {error}") from error

    inventory = _base_inventory(
        source, "train_tar", members, shape_ids, declared_ids
    )
    inventory["obj_count"] = len(shape_ids)
    inventory["npy_count"] = 0
    inventory["normalized_files"] = sorted(
        member.target_name for member in members if not member.is_dir
    )
    return inventory, members


def _scan_train_tar(
    archive_path: os.PathLike[str] | str,
) -> Tuple[Dict[str, object], List[_Member]]:
    with _open_owned_archive_source(archive_path) as source:
        return _scan_train_source(source)


def _zip_member_kind(info: zipfile.ZipInfo) -> bool:
    """Validate a ZIP member's Unix type and return whether it is a directory."""

    raw_name = info.orig_filename
    if "\x00" in raw_name:
        raise ArchiveValidationError(
            f"NUL is forbidden in ZIP filename: {raw_name!r}"
        )
    if raw_name != info.filename:
        raise ArchiveValidationError(
            "ZIP filename changed during parsing: "
            f"raw={raw_name!r} parsed={info.filename!r}"
        )

    name_marks_directory = raw_name.endswith("/")
    unix_mode = info.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        raise ArchiveValidationError(
            f"ZIP symlink is forbidden: {raw_name!r}"
        )
    if file_type == stat.S_IFDIR:
        if not name_marks_directory:
            raise ArchiveValidationError(
                f"ZIP directory lacks trailing slash: {raw_name!r}"
            )
        return True
    if file_type == stat.S_IFREG:
        if name_marks_directory:
            raise ArchiveValidationError(
                f"ZIP regular file has trailing slash: {raw_name!r}"
            )
        return False
    if file_type != 0:
        raise ArchiveValidationError(
            f"ZIP special member is forbidden: {raw_name!r}"
        )
    return name_marks_directory


def _validated_test_member(
    info: zipfile.ZipInfo,
) -> Tuple[_Member, str | None, bool]:
    is_dir = _zip_member_kind(info)
    target, shape_id, declares_shape = _normalize_test_path(
        info.orig_filename, is_dir
    )
    return (
        _Member(
            source_name=info.filename,
            target_name=target,
            is_dir=is_dir,
            size=info.file_size if not is_dir else 0,
        ),
        shape_id,
        declares_shape,
    )


def _scan_test_source(
    source: _OwnedArchiveSource,
) -> Tuple[Dict[str, object], List[_Member]]:
    members: List[_Member] = []
    targets = set()
    shape_ids = set()
    declared_ids = set()
    try:
        with source.open() as archive_stream:
            with zipfile.ZipFile(archive_stream, mode="r") as stream:
                for info in stream.infolist():
                    member, shape_id, declares_shape = _validated_test_member(
                        info
                    )
                    if member.target_name in targets:
                        raise ArchiveValidationError(
                            "duplicate extraction target: "
                            f"{member.target_name!r}"
                        )
                    targets.add(member.target_name)
                    if shape_id is not None:
                        if not member.is_dir:
                            if shape_id in shape_ids:
                                raise ArchiveValidationError(
                                    f"duplicate shape ID: {shape_id!r}"
                                )
                            shape_ids.add(shape_id)
                        if declares_shape:
                            declared_ids.add(shape_id)
                    members.append(member)
    except (zipfile.BadZipFile, EOFError) as error:
        raise ArchiveValidationError(f"invalid ZIP archive: {error}") from error

    inventory = _base_inventory(
        source, "test_zip", members, shape_ids, declared_ids
    )
    inventory["obj_count"] = 0
    inventory["npy_count"] = len(shape_ids)
    inventory["normalized_files"] = sorted(
        member.target_name for member in members if not member.is_dir
    )
    return inventory, members


def _scan_test_zip(
    archive_path: os.PathLike[str] | str,
) -> Tuple[Dict[str, object], List[_Member]]:
    with _open_owned_archive_source(archive_path) as source:
        return _scan_test_source(source)


def inspect_train_tar(
    archive_path: os.PathLike[str] | str,
) -> Dict[str, object]:
    """Validate and inventory a training tar archive without extracting it."""

    inventory, _ = _scan_train_tar(archive_path)
    return inventory


def inspect_test_zip(
    archive_path: os.PathLike[str] | str,
) -> Dict[str, object]:
    """Validate and inventory a test ZIP archive without extracting it."""

    inventory, _ = _scan_test_zip(archive_path)
    return inventory


def _path_lexists(path: os.PathLike[str] | str) -> bool:
    return os.path.lexists(os.fspath(path))


def _create_owned_stage(
    output_dir: os.PathLike[str] | str,
) -> _OwnedStage:
    requested = Path(output_dir)
    if not requested.name:
        raise ValueError(f"output must name a new directory: {requested}")
    if _path_lexists(requested):
        raise FileExistsError(f"refusing to overwrite output: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve()
    destination = parent / requested.name
    if _path_lexists(destination):
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    prefix = f".{requested.name}.pcdenoise-stage-"
    stage_path = Path(
        tempfile.mkdtemp(prefix=prefix, dir=os.fspath(parent))
    ).resolve()
    metadata = os.lstat(stage_path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"created staging path is not a directory: {stage_path}")
    return _OwnedStage(
        path=stage_path,
        parent=parent,
        prefix=prefix,
        destination=destination,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _verified_owned_stage_metadata(stage: _OwnedStage) -> os.stat_result:
    if (
        stage.path.parent != stage.parent
        or not stage.path.name.startswith(stage.prefix)
    ):
        raise RuntimeError(f"refusing unsafe staging cleanup: {stage.path}")
    metadata = os.lstat(stage.path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"staging path is no longer a directory: {stage.path}")
    if (
        metadata.st_dev != stage.device
        or metadata.st_ino != stage.inode
    ):
        raise RuntimeError(f"staging directory identity changed: {stage.path}")
    return metadata


def _cleanup_owned_stage(stage: _OwnedStage) -> None:
    if not _path_lexists(stage.path):
        return
    _verified_owned_stage_metadata(stage)
    shutil.rmtree(stage.path)


def _publish_owned_stage(stage: _OwnedStage) -> Path:
    _verified_owned_stage_metadata(stage)
    if _path_lexists(stage.destination):
        raise FileExistsError(
            f"refusing to overwrite output: {stage.destination}"
        )
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
            os.fspath(stage.destination),
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    parent_descriptor = os.open(
        stage.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        result = renameat2(
            parent_descriptor,
            os.fsencode(stage.path.name),
            parent_descriptor,
            os.fsencode(stage.destination.name),
            RENAME_NOREPLACE,
        )
        if result == 0:
            published = os.lstat(stage.destination)
            if (
                published.st_dev != stage.device
                or published.st_ino != stage.inode
            ):
                rollback_result = renameat2(
                    parent_descriptor,
                    os.fsencode(stage.destination.name),
                    parent_descriptor,
                    os.fsencode(stage.path.name),
                    RENAME_NOREPLACE,
                )
                if rollback_result != 0:
                    rollback_errno = ctypes.get_errno()
                    raise RuntimeError(
                        "staging identity changed during publication and "
                        "the unexpected directory could not be restored: "
                        f"{os.strerror(rollback_errno)}"
                    )
                raise RuntimeError(
                    f"staging directory identity changed: {stage.path}"
                )
            return stage.destination
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error_number,
                "refusing to overwrite output",
                os.fspath(stage.destination),
            )
        if error_number in (errno.ENOSYS, errno.EINVAL):
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory publication is unavailable",
                os.fspath(stage.destination),
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(stage.destination),
        )
    finally:
        os.close(parent_descriptor)


def _member_mismatch(
    kind: str,
    index: int,
    expected: _Member,
    actual: _Member,
) -> ArchiveValidationError:
    return ArchiveValidationError(
        f"{kind} member changed at index {index}: "
        f"expected={expected!r}, actual={actual!r}"
    )


def _verify_archive_hash(
    archive_path: os.PathLike[str] | str | _OwnedArchiveSource,
    inventory: Dict[str, object],
) -> None:
    if isinstance(archive_path, _OwnedArchiveSource):
        actual = _sha256_owned_source(archive_path)
    else:
        actual = sha256_file(archive_path)
    expected = inventory["archive_sha256"]
    if actual != expected:
        raise ArchiveValidationError(
            f"archive changed during extraction: expected {expected}, got {actual}"
        )


def _extract_train_into(
    archive_path: _OwnedArchiveSource,
    output: Path,
    members: Sequence[_Member],
) -> None:
    seen = 0
    with archive_path.open() as archive_stream:
        with tarfile.open(fileobj=archive_stream, mode="r:*") as stream:
            for info in stream:
                actual, _, _ = _validated_train_member(info)
                if seen >= len(members):
                    raise ArchiveValidationError(
                        f"train archive gained member at index {seen}: {actual!r}"
                    )
                expected = members[seen]
                if actual != expected:
                    raise _member_mismatch("train", seen, expected, actual)
                destination = output.joinpath(
                    *PurePosixPath(actual.target_name).parts
                )
                if actual.is_dir:
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = stream.extractfile(info)
                    if source is None:
                        raise ArchiveValidationError(
                            f"could not read regular member: {info.name!r}"
                        )
                    with source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target, length=8 << 20)
                seen += 1
    if seen != len(members):
        raise ArchiveValidationError(
            f"train archive lost members: expected {len(members)}, got {seen}"
        )


def _extract_test_into(
    archive_path: _OwnedArchiveSource,
    output: Path,
    members: Sequence[_Member],
) -> None:
    with archive_path.open() as archive_stream:
        with zipfile.ZipFile(archive_stream, mode="r") as stream:
            infos = stream.infolist()
            if len(infos) != len(members):
                raise ArchiveValidationError(
                    f"test archive member count changed: "
                    f"expected {len(members)}, got {len(infos)}"
                )
            for index, (info, expected) in enumerate(zip(infos, members)):
                actual, _, _ = _validated_test_member(info)
                if actual != expected:
                    raise _member_mismatch("test", index, expected, actual)
                destination = output.joinpath(
                    *PurePosixPath(actual.target_name).parts
                )
                if actual.is_dir:
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(info, mode="r") as source:
                        with destination.open("xb") as target:
                            shutil.copyfileobj(source, target, length=8 << 20)


def extract_train_tar(
    archive_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> Dict[str, object]:
    """Fully validate, then safely extract the official train archive."""

    with _open_owned_archive_source(archive_path) as source:
        inventory, members = _scan_train_source(source)
        stage = _create_owned_stage(output_dir)
        try:
            _extract_train_into(source, stage.path, members)
            _verify_archive_hash(source, inventory)
            _publish_owned_stage(stage)
        except BaseException:
            _cleanup_owned_stage(stage)
            raise
    return inventory


def extract_test_zip(
    archive_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> Dict[str, object]:
    """Fully validate, then extract NPY bytes under a normalized shapenet root."""

    with _open_owned_archive_source(archive_path) as source:
        inventory, members = _scan_test_source(source)
        stage = _create_owned_stage(output_dir)
        try:
            _extract_test_into(source, stage.path, members)
            _verify_archive_hash(source, inventory)
            _publish_owned_stage(stage)
        except BaseException:
            _cleanup_owned_stage(stage)
            raise
    return inventory
