"""Typed boundary for trainer validation-mode configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import ClassVar, TypeAlias


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 uses the dependency
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - fallback for the project's ``toml`` dependency
        import toml as tomllib

from .contracts import Sha256Digest, StoppingPolicy


VALIDATION_CONFIG_SCHEMA = "diarizen-trainer-validation-config-v1"
_URL_PATTERN = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*://|//[a-z0-9])")
_DRIVE_PATH_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY_NAMES = frozenset(
    {
        "accesskey",
        "accesskeyid",
        "apikey",
        "authorization",
        "bearer",
        "credential",
        "password",
        "privatekey",
        "secret",
        "secretkey",
        "sessiontoken",
        "signedurl",
        "token",
    }
)


class ValidationConfigError(ValueError):
    """A trainer validation configuration is not a supported contract."""


def _object(value: object, field: str) -> Mapping[str, object]:
    """Require a string-keyed mapping at a configuration boundary."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValidationConfigError(f"{field} must be an object with string fields")
    return value


def _exact_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    field: str,
) -> None:
    """Reject missing and unknown fields in one object."""

    actual = frozenset(value)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        raise ValidationConfigError(f"{field} fields are not exact: missing={sorted(missing)}, extra={sorted(extra)}")


def _scan_for_secrets_or_urls(value: object, location: str = "trainer.validation") -> None:
    """Reject credential fields and URL values recursively without echoing them."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValidationConfigError(f"{location} keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            parts = frozenset(re.findall(r"[a-z0-9]+", key.casefold()))
            if normalized in _SECRET_KEY_NAMES or parts & {
                "credential",
                "secret",
                "token",
                "signed_url",
                "signedurl",
            }:
                raise ValidationConfigError(f"{location} contains a credential field")
            _scan_for_secrets_or_urls(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_for_secrets_or_urls(child, f"{location}[{index}]")
        return
    if isinstance(value, str) and _URL_PATTERN.search(value):
        raise ValidationConfigError(f"{location} must not contain a URL")


def _local_path(value: object, field: str) -> Path:
    """Parse a local absolute or safe repository-relative path."""

    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str) or not value:
        raise ValidationConfigError(f"{field} must be a non-empty local path")
    if any(ord(character) < 32 for character in value):
        raise ValidationConfigError(f"{field} must not contain control characters")
    if "\\" in value or _URL_PATTERN.search(value) or _DRIVE_PATH_PATTERN.match(value) or value.startswith("~"):
        raise ValidationConfigError(f"{field} must be a local absolute or safe repository-relative path")
    parts = value.split("/")
    parts_to_check = parts[1:] if value.startswith("/") else parts
    if any(part in {"", ".", ".."} for part in parts_to_check):
        raise ValidationConfigError(f"{field} contains an unsafe path component")
    parsed = PurePosixPath(value)
    if value.startswith("/") and not parsed.is_absolute():
        raise ValidationConfigError(f"{field} is not a valid local path")
    return Path(parsed.as_posix())


def _string(value: object, field: str, *, maximum: int = 512) -> str:
    """Parse one bounded printable non-URL string."""

    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValidationConfigError(f"{field} must be a non-empty bounded string")
    if any(not character.isprintable() for character in value):
        raise ValidationConfigError(f"{field} must not contain control characters")
    if _URL_PATTERN.search(value):
        raise ValidationConfigError(f"{field} must not contain a URL")
    return value


def _positive_integer(value: object, field: str) -> int:
    """Parse a positive integer without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationConfigError(f"{field} must be a positive integer")
    return value


def _positive_number(value: object, field: str, *, maximum: float = 86_400.0) -> float:
    """Parse one bounded positive finite number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationConfigError(f"{field} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > maximum:
        raise ValidationConfigError(f"{field} must be a positive finite number no greater than {maximum:g}")
    return parsed


def _enum(value: object, enum_type: type[Enum], field: str) -> Enum:
    """Parse one closed enum value."""

    if not isinstance(value, str):
        raise ValidationConfigError(f"{field} is not supported")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValidationConfigError(f"{field} is not supported") from error


class ValidationMode(str, Enum):
    """The two supported trainer validation modes."""

    INLINE = "inline"
    EXTERNAL = "external"


class InlineStoppingPolicy(str, Enum):
    """Inline stopping behavior."""

    LEGACY = "legacy"
    PATIENCE = "patience"

    @property
    def applies_to_fixed_updates(self) -> bool:
        """Return whether this policy intentionally stops fixed-update runs."""

        return self is InlineStoppingPolicy.PATIENCE


@dataclass(frozen=True)
class CampaignManifestReference:
    """Exact local campaign-manifest path bound to one SHA-256 digest."""

    path: Path
    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or _local_path(self.path, "campaign_manifest.path") != self.path:
            raise ValidationConfigError("campaign_manifest.path must be a canonical local path")
        if not isinstance(self.sha256, Sha256Digest):
            raise ValidationConfigError("campaign_manifest.sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        """Return the strict boundary representation."""

        return {"path": self.path.as_posix(), "sha256": self.sha256.value}


@dataclass(frozen=True)
class ImmutableObjectStoreDestination:
    """Credential-free immutable object-store destination identity."""

    provider: str
    bucket: str
    prefix: str

    def __post_init__(self) -> None:
        provider = _string(self.provider, "object_store_destination.provider", maximum=64)
        bucket = _string(self.bucket, "object_store_destination.bucket", maximum=256)
        if not _IDENTIFIER_PATTERN.fullmatch(provider) or "/" in provider:
            raise ValidationConfigError("object_store_destination.provider is not a safe identity")
        if not _IDENTIFIER_PATTERN.fullmatch(bucket) or "/" in bucket:
            raise ValidationConfigError("object_store_destination.bucket is not a safe identity")
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValidationConfigError("object_store_destination.prefix must be non-empty")
        if "\\" in self.prefix or any(part in {"", ".", ".."} for part in self.prefix.split("/")):
            raise ValidationConfigError("object_store_destination.prefix is not a safe identity")
        if _URL_PATTERN.search(self.prefix):
            raise ValidationConfigError("object_store_destination.prefix must not contain a URL")

    def to_dict(self) -> dict[str, str]:
        """Return the strict credential-free representation."""

        return {"provider": self.provider, "bucket": self.bucket, "prefix": self.prefix}


@dataclass(frozen=True)
class PollBackoffBounds:
    """Positive poll and exponential-backoff bounds."""

    initial_seconds: float
    maximum_seconds: float
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        initial = _positive_number(self.initial_seconds, "poll_backoff.initial_seconds")
        maximum = _positive_number(self.maximum_seconds, "poll_backoff.maximum_seconds")
        multiplier = _positive_number(self.multiplier, "poll_backoff.multiplier", maximum=100.0)
        if initial > maximum:
            raise ValidationConfigError("poll_backoff.initial_seconds cannot exceed maximum_seconds")
        if multiplier < 1:
            raise ValidationConfigError("poll_backoff.multiplier must be at least one")

    def to_dict(self) -> dict[str, float]:
        """Return the strict boundary representation."""

        return {
            "initial_seconds": float(self.initial_seconds),
            "maximum_seconds": float(self.maximum_seconds),
            "multiplier": float(self.multiplier),
        }


@dataclass(frozen=True)
class InlineValidationConfig:
    """Inline validation owned by the trainer."""

    stopping_policy: InlineStoppingPolicy = InlineStoppingPolicy.LEGACY
    patience_limit: int | None = None
    mode: ClassVar[ValidationMode] = ValidationMode.INLINE

    def __post_init__(self) -> None:
        if not isinstance(self.stopping_policy, InlineStoppingPolicy):
            raise ValidationConfigError("inline stopping policy is not supported")
        if self.stopping_policy is InlineStoppingPolicy.PATIENCE:
            if self.patience_limit is None:
                raise ValidationConfigError("inline patience policy requires a positive patience limit")
            _positive_integer(self.patience_limit, "inline patience_limit")
        elif self.patience_limit is not None:
            raise ValidationConfigError("legacy inline stopping cannot carry a patience limit")

    def patience_applies(self, *, fixed_update_run: bool) -> bool:
        """Return whether inline patience should control this run."""

        return self.stopping_policy is InlineStoppingPolicy.PATIENCE or not fixed_update_run

    def to_dict(self) -> dict[str, object]:
        """Return the strict boundary representation."""

        result: dict[str, object] = {
            "mode": self.mode.value,
            "stopping_policy": self.stopping_policy.value,
        }
        if self.patience_limit is not None:
            result["patience_limit"] = self.patience_limit
        return result


@dataclass(frozen=True)
class ExternalValidationConfig:
    """External validation coordinated by the trusted selection controller."""

    campaign_manifest: CampaignManifestReference
    trusted_selection_state_path: Path
    publication_transaction_root: Path
    object_store_destination: ImmutableObjectStoreDestination
    poll_backoff: PollBackoffBounds
    stopping_policy: StoppingPolicy
    patience_limit: int
    epoch_zero_gate: ClassVar[bool] = True
    mode: ClassVar[ValidationMode] = ValidationMode.EXTERNAL

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_manifest, CampaignManifestReference):
            raise ValidationConfigError("campaign_manifest must be typed")
        for field, value in (
            ("trusted_selection_state_path", self.trusted_selection_state_path),
            ("publication_transaction_root", self.publication_transaction_root),
        ):
            if not isinstance(value, Path) or _local_path(value, field) != value:
                raise ValidationConfigError(f"{field} must be a canonical local path")
        if not isinstance(self.object_store_destination, ImmutableObjectStoreDestination):
            raise ValidationConfigError("object_store_destination must be typed")
        if not isinstance(self.poll_backoff, PollBackoffBounds):
            raise ValidationConfigError("poll_backoff must be typed")
        if self.stopping_policy is not StoppingPolicy.EXTERNAL_PATIENCE_OR_MAX_UPDATES:
            raise ValidationConfigError("external validation requires external_patience_or_max_updates")
        _positive_integer(self.patience_limit, "external patience_limit")

    def to_dict(self) -> dict[str, object]:
        """Return the strict boundary representation."""

        return {
            "mode": self.mode.value,
            "campaign_manifest": self.campaign_manifest.to_dict(),
            "trusted_selection_state_path": self.trusted_selection_state_path.as_posix(),
            "publication_transaction_root": self.publication_transaction_root.as_posix(),
            "object_store_destination": self.object_store_destination.to_dict(),
            "epoch_zero_gate": self.epoch_zero_gate,
            "poll_backoff": self.poll_backoff.to_dict(),
            "stopping_policy": self.stopping_policy.value,
            "patience_limit": self.patience_limit,
        }


ValidationConfig: TypeAlias = InlineValidationConfig | ExternalValidationConfig


def _parse_manifest(value: object) -> CampaignManifestReference:
    """Parse the exact campaign manifest reference."""

    data = _object(value, "trainer.validation.campaign_manifest")
    _exact_fields(data, required=frozenset({"path", "sha256"}), field="campaign_manifest")
    try:
        digest = (
            data["sha256"]
            if isinstance(data["sha256"], Sha256Digest)
            else Sha256Digest.parse(data["sha256"], "campaign_manifest.sha256")
        )
    except ValueError as error:
        raise ValidationConfigError(str(error)) from error
    return CampaignManifestReference(path=_local_path(data["path"], "campaign_manifest.path"), sha256=digest)


def _parse_destination(value: object) -> ImmutableObjectStoreDestination:
    """Parse the exact credential-free object-store identity."""

    data = _object(value, "trainer.validation.object_store_destination")
    _exact_fields(data, required=frozenset({"provider", "bucket", "prefix"}), field="object_store_destination")
    if not isinstance(data["prefix"], str):
        raise ValidationConfigError("object_store_destination.prefix must be a string")
    return ImmutableObjectStoreDestination(
        provider=_string(data["provider"], "object_store_destination.provider", maximum=64),
        bucket=_string(data["bucket"], "object_store_destination.bucket", maximum=256),
        prefix=data["prefix"],
    )


def _parse_poll_backoff(value: object) -> PollBackoffBounds:
    """Parse exact poll/backoff bounds."""

    data = _object(value, "trainer.validation.poll_backoff")
    _exact_fields(
        data,
        required=frozenset({"initial_seconds", "maximum_seconds"}),
        optional=frozenset({"multiplier"}),
        field="poll_backoff",
    )
    return PollBackoffBounds(
        initial_seconds=_positive_number(data["initial_seconds"], "poll_backoff.initial_seconds"),
        maximum_seconds=_positive_number(data["maximum_seconds"], "poll_backoff.maximum_seconds"),
        multiplier=_positive_number(data.get("multiplier", 2.0), "poll_backoff.multiplier", maximum=100.0),
    )


def _parse_inline(data: Mapping[str, object]) -> InlineValidationConfig:
    """Parse the inline branch of the tagged union."""

    _exact_fields(
        data,
        required=frozenset({"mode"}),
        optional=frozenset({"stopping_policy", "patience_limit"}),
        field="trainer.validation.inline",
    )
    stopping_value = data.get("stopping_policy", InlineStoppingPolicy.LEGACY.value)
    stopping_policy = _enum(stopping_value, InlineStoppingPolicy, "inline stopping_policy")
    return InlineValidationConfig(
        stopping_policy=stopping_policy,
        patience_limit=data.get("patience_limit"),
    )


def _parse_external(data: Mapping[str, object]) -> ExternalValidationConfig:
    """Parse the external branch of the tagged union."""

    _exact_fields(
        data,
        required=frozenset(
            {
                "mode",
                "campaign_manifest",
                "trusted_selection_state_path",
                "publication_transaction_root",
                "object_store_destination",
                "epoch_zero_gate",
                "poll_backoff",
                "stopping_policy",
                "patience_limit",
            }
        ),
        field="trainer.validation.external",
    )
    policy = _enum(data["stopping_policy"], StoppingPolicy, "external stopping_policy")
    if policy is not StoppingPolicy.EXTERNAL_PATIENCE_OR_MAX_UPDATES:
        raise ValidationConfigError("external stopping_policy must be external_patience_or_max_updates")
    gate = data["epoch_zero_gate"]
    if gate is not True:
        raise ValidationConfigError("external validation requires epoch_zero_gate=true")
    return ExternalValidationConfig(
        campaign_manifest=_parse_manifest(data["campaign_manifest"]),
        trusted_selection_state_path=_local_path(data["trusted_selection_state_path"], "trusted_selection_state_path"),
        publication_transaction_root=_local_path(data["publication_transaction_root"], "publication_transaction_root"),
        object_store_destination=_parse_destination(data["object_store_destination"]),
        poll_backoff=_parse_poll_backoff(data["poll_backoff"]),
        stopping_policy=policy,
        patience_limit=_positive_integer(data["patience_limit"], "external patience_limit"),
    )


def parse_validation_config(value: object = None) -> ValidationConfig:
    """Parse a full trainer mapping or a direct ``trainer.validation`` mapping.

    A missing validation table is the explicit compatibility path for existing
    trainer TOML and returns legacy inline behavior.  A present table must use
    the closed ``mode`` tag and its exact mode-specific fields.
    """

    if value is None:
        return InlineValidationConfig()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValidationConfigError("validation configuration must be a mapping")
    root = _object(value, "trainer configuration")
    if "trainer" in root:
        trainer = _object(root["trainer"], "trainer")
        if "validation" not in trainer:
            return InlineValidationConfig()
        candidate = _object(trainer["validation"], "trainer.validation")
    else:
        candidate = root
    _scan_for_secrets_or_urls(candidate)
    if "mode" not in candidate:
        raise ValidationConfigError("trainer.validation requires the mode tag")
    mode = _enum(candidate["mode"], ValidationMode, "trainer.validation.mode")
    if mode is ValidationMode.INLINE:
        return _parse_inline(candidate)
    return _parse_external(candidate)


def parse_validation_toml(value: str | bytes | Mapping[str, object]) -> ValidationConfig:
    """Parse TOML containing ``[trainer.validation]`` or a direct mapping."""

    if isinstance(value, Mapping):
        return parse_validation_config(value)
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        decoded = tomllib.loads(text)
    except (TypeError, UnicodeDecodeError, ValueError) as error:
        raise ValidationConfigError("validation configuration is not valid TOML") from error
    return parse_validation_config(decoded)


__all__ = [
    "CampaignManifestReference",
    "ExternalValidationConfig",
    "ImmutableObjectStoreDestination",
    "InlineStoppingPolicy",
    "InlineValidationConfig",
    "PollBackoffBounds",
    "VALIDATION_CONFIG_SCHEMA",
    "ValidationConfig",
    "ValidationConfigError",
    "ValidationMode",
    "parse_validation_config",
    "parse_validation_toml",
]
