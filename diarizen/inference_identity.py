"""Build a stable identity for the installed inference engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
from pathlib import Path
from typing import Any


class InferenceIdentityError(RuntimeError):
    """Raised when an inference engine identity cannot be resolved safely."""


REQUIRED_DEPENDENCIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("diarizen", ("diarizen",)),
    ("pyannote.audio", ("pyannote.audio", "pyannote-audio")),
    ("asteroid-filterbanks", ("asteroid-filterbanks",)),
    ("torch", ("torch",)),
    ("torchaudio", ("torchaudio",)),
    ("numpy", ("numpy",)),
    ("scipy", ("scipy",)),
    ("scikit-learn", ("scikit-learn", "sklearn")),
    ("pyannote.core", ("pyannote.core", "pyannote-core")),
    ("pyannote.database", ("pyannote.database", "pyannote-database")),
    ("pyannote.metrics", ("pyannote.metrics", "pyannote-metrics")),
    ("pyannote.pipeline", ("pyannote.pipeline", "pyannote-pipeline")),
    ("pytorch-lightning", ("pytorch-lightning",)),
    ("torchmetrics", ("torchmetrics",)),
    ("omegaconf", ("omegaconf",)),
    ("pytorch-metric-learning", ("pytorch-metric-learning",)),
    ("rich", ("rich",)),
    ("semver", ("semver",)),
    ("tensorboardX", ("tensorboardX", "tensorboardx")),
    ("torch-audiomentations", ("torch-audiomentations", "torch_audiomentations")),
    ("speechbrain", ("speechbrain",)),
    ("soundfile", ("soundfile",)),
    ("einops", ("einops",)),
    ("huggingface_hub", ("huggingface_hub", "huggingface-hub")),
    ("toml", ("toml",)),
)
OPTIONAL_DEPENDENCIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lightning", ("lightning",)),
    ("lightning-fabric", ("lightning-fabric", "lightning_fabric")),
)
# vbx_setup loads these exact files from plda_dir
REQUIRED_PLDA_ASSETS = ("xvec_transform.npz", "plda.npz")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one identity input file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_roots(package_name: str) -> tuple[Path, ...]:
    """Resolve the source roots used by an importable installed package."""

    try:
        spec = importlib.util.find_spec(package_name)
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        raise InferenceIdentityError(f"Cannot resolve package {package_name!r}: {error}") from error
    if spec is None:
        raise InferenceIdentityError(f"Cannot resolve package {package_name!r}")

    if spec.submodule_search_locations:
        candidates = spec.submodule_search_locations
    elif spec.origin and spec.origin not in {"built-in", "frozen"}:
        candidates = (Path(spec.origin).parent.as_posix(),)
    else:
        raise InferenceIdentityError(f"Package {package_name!r} has no inspectable source root")

    roots = []
    for candidate in candidates:
        try:
            root = Path(candidate).expanduser().resolve(strict=True)
        except OSError as error:
            raise InferenceIdentityError(f"Cannot inspect package {package_name!r} at {candidate}: {error}") from error
        if not root.is_dir():
            raise InferenceIdentityError(f"Package source root is not a directory: {root}")
        roots.append(root)

    return tuple(sorted(dict.fromkeys(roots), key=lambda root: root.as_posix()))


def _snapshot_files(root: Path, suffix: str | None = None) -> list[dict[str, Any]]:
    """Snapshot regular files below a root using relative names and content metadata."""

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "name": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return files


def package_source_identity(package_name: str) -> dict[str, Any]:
    """Return the relative source snapshot for an installed package."""

    roots = _package_roots(package_name)
    files = []
    for root_index, root in enumerate(roots):
        for file_record in _snapshot_files(root, suffix=".py"):
            files.append({"root": root_index, **file_record})
    if not files:
        raise InferenceIdentityError(f"Package {package_name!r} has no Python source files")

    return {"roots": len(roots), "files": files}


def installed_dependency_versions() -> dict[str, dict[str, str]]:
    """Return required dependency versions from installed distribution metadata."""

    versions: dict[str, dict[str, str]] = {}
    for identity_name, candidates in REQUIRED_DEPENDENCIES:
        for distribution_name in candidates:
            try:
                version = importlib.metadata.version(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                continue
            except (TypeError, ValueError) as error:
                raise InferenceIdentityError(
                    f"Cannot resolve metadata for dependency {distribution_name!r}: {error}"
                ) from error
            versions[identity_name] = {"distribution": distribution_name, "version": version}
            break
        else:
            names = ", ".join(candidates)
            raise InferenceIdentityError(f"Required dependency metadata is missing: {identity_name} ({names})")

    for identity_name, candidates in OPTIONAL_DEPENDENCIES:
        for distribution_name in candidates:
            try:
                version = importlib.metadata.version(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                continue
            except (TypeError, ValueError) as error:
                raise InferenceIdentityError(
                    f"Cannot resolve metadata for dependency {distribution_name!r}: {error}"
                ) from error
            versions[identity_name] = {"distribution": distribution_name, "version": version}
            break
    return versions


def _plda_identity(diarizen_hub: str | Path | None) -> dict[str, Any]:
    """Return the content identity of the VBx PLDA directory."""

    if diarizen_hub is None or not str(diarizen_hub):
        raise InferenceIdentityError("VBx inference requires --diarizen_hub for PLDA assets")
    try:
        plda_dir = Path(diarizen_hub).expanduser().resolve(strict=True) / "plda"
    except OSError as error:
        raise InferenceIdentityError(f"Cannot resolve VBx hub {diarizen_hub!r}: {error}") from error
    if not plda_dir.is_dir():
        raise InferenceIdentityError(f"VBx PLDA directory is missing: {plda_dir}")

    missing = [name for name in REQUIRED_PLDA_ASSETS if not (plda_dir / name).is_file()]
    if missing:
        raise InferenceIdentityError(f"VBx PLDA assets are missing from {plda_dir}: {', '.join(missing)}")
    empty = [name for name in REQUIRED_PLDA_ASSETS if (plda_dir / name).stat().st_size == 0]
    if empty:
        raise InferenceIdentityError(f"VBx PLDA assets are empty in {plda_dir}: {', '.join(empty)}")

    files = _snapshot_files(plda_dir)
    if not files:
        raise InferenceIdentityError(f"VBx PLDA directory has no assets: {plda_dir}")
    return {"files": files}


def build_engine_identity(clustering_method: str, diarizen_hub: str | Path | None = None) -> dict[str, Any]:
    """Build the installed source, dependency, and selected asset identity."""

    if clustering_method not in {"AgglomerativeClustering", "VBxClustering"}:
        raise InferenceIdentityError(f"Unsupported clustering method: {clustering_method}")

    identity: dict[str, Any] = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": installed_dependency_versions(),
        "sources": {
            package_name: package_source_identity(package_name) for package_name in ("diarizen", "pyannote.audio")
        },
    }
    if clustering_method == "VBxClustering":
        identity["vbx_plda"] = _plda_identity(diarizen_hub)
    else:
        identity["vbx_plda"] = None
    return identity
