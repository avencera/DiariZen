"""Typed, offline DiariZen validation plans for CloudDeck protocol v3.

The module only builds and validates immutable documents.  It does not create
queues, register workers, issue credentials, or contact a provider
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import ClassVar, Iterable, Mapping, Sequence

from .contracts import (
    Sha256Digest,
    SnapshotPoint,
    ValidationCampaign,
    ValidationContractError,
    ValidationSlot,
    snapshot_point_from_dict,
)
from .preflight import CudaVersion, EvaluatorCapabilityProfile, GpuName
from .slot_plan import build_validation_campaign


WORK_PAYLOAD_SCHEMA = "diarizen-cloudeck-validation-work-payload-v1"
EXTRA_BATCH_SCHEMA = "diarizen-cloudeck-validation-extra-batch-v1"
COST_RECORD_SCHEMA = "diarizen-cloudeck-validation-cost-record-v1"
PROFILE_SCHEMA = "diarizen-validation-evaluator-profile-v1"
DEFAULT_QUEUE_DEADLINE_UNIX_SECONDS = 2_000_000_000
DEFAULT_MAX_PAYLOAD_BYTES = 1_048_576
DEFAULT_LEASE_DURATION_SECONDS = 3_600
DEFAULT_VALIDATION_RESULT_LIMIT_BYTES = 64 * 1024
DEFAULT_PROFILE_ID = "diarizen-validation-rtx-5060-ti"
DEFAULT_ARTIFACT_ISSUER_ID = "diarizen-validation-artifacts"
PUBLISHED_EVALUATOR_IMAGE_REPOSITORY = "docker.io/praveenperera/diarizen-validation-worker"
PUBLISHED_EVALUATOR_IMAGE_DIGEST = Sha256Digest("db656b2639753bbc6f574d4c40a9a70677713e9566d41f325887354172e1dc3b")
# Vast runtype=args keeps the image ENTRYPOINT and appends this command
VALIDATION_WORKER_COMMAND = ("claim",)
DEFAULT_WORKLOAD_FAMILY = "diarizen-validation"
DEFAULT_REQUEST_SCHEMA = 1
DEFAULT_RESULT_SCHEMA = 1
DEFAULT_PROOF_SCHEMA = 1
GIB = 1024**3
MAX_ARTIFACT_BYTES = 1 << 40

MAX_OPAQUE_ID_BYTES = 128
_HTTP_URL_RE = re.compile(r"(?i)(?:https?|ftp)://")
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+|token=|password=|secret=|private[_-]?key=|access[_-]?key=|api[_-]?key=|"
    r"x-amz-signature=|x-amz-credential=|x-goog-signature=|ghp_[a-z0-9]|github_pat_[a-z0-9]|sk-[a-z0-9])"
)
_SECRET_KEY_RE = re.compile(
    r"(?i)(?:credential|password|token|secret|private[_-]?key|access[_-]?key|api[_-]?key|"
    r"authorization|bearer|presigned|signed[_-]?url|github[_-]?(?:token|credential)|push[_-]?credential|"
    r"issuer[_-]?credential|url)"
)


class QueuePlanError(ValidationContractError):
    """A typed queue plan violates its immutable contract."""


def _exact_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    actual = frozenset(value)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        raise QueuePlanError(f"{context} fields are not exact: missing={sorted(missing)}, extra={sorted(extra)}")


def _string(value: object, field: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise QueuePlanError(f"{field} must be a non-empty string of at most {maximum} UTF-8 bytes")
    if any(character.isspace() and character in "\r\n\x00" for character in value):
        raise QueuePlanError(f"{field} cannot contain control characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise QueuePlanError(f"{field} cannot contain control characters")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QueuePlanError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _digest(value: object, field: str) -> Sha256Digest:
    try:
        return Sha256Digest.parse(value, field)
    except ValidationContractError as error:
        raise QueuePlanError(str(error)) from error


def _canonical_digest(value: object) -> Sha256Digest:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def _ordered_digest(value: object) -> Sha256Digest:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def _validate_uuid_v7(value: str, field: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise QueuePlanError(f"{field} must be a canonical UUIDv7") from error
    if str(parsed) != value or parsed.version != 7 or parsed.variant != uuid.RFC_4122:
        raise QueuePlanError(f"{field} must be a canonical UUIDv7")


def _new_uuid_v7() -> str:
    timestamp_ms = int(time.time() * 1_000) & ((1 << 48) - 1)
    random_a = secrets.randbits(12)
    random_b = secrets.randbits(62)
    value = (timestamp_ms << 80) | (7 << 76) | (random_a << 64) | (2 << 62) | random_b
    return str(uuid.UUID(int=value))


def _reject_unsafe(value: object, *, path: str = "root") -> None:
    """Reject URLs and secret-shaped fields before a document leaves the domain."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise QueuePlanError(f"{path} contains a non-string field name")
            if _SECRET_KEY_RE.search(key):
                raise QueuePlanError(f"{path}.{key} is not allowed in a queue plan")
            _reject_unsafe(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_unsafe(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _HTTP_URL_RE.search(value):
            raise QueuePlanError(f"{path} cannot contain a URL")
        if _SECRET_VALUE_RE.search(value):
            raise QueuePlanError(f"{path} cannot contain a secret or signed credential")


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QueuePlanError(f"{field} must be an object")
    return value


@dataclass(frozen=True, order=True)
class OpaqueIdentity:
    """A bounded non-secret identity or opaque object location."""

    value: str

    def __post_init__(self) -> None:
        _string(self.value, "opaque identity", maximum=MAX_OPAQUE_ID_BYTES)
        if _SECRET_VALUE_RE.search(self.value):
            raise QueuePlanError("opaque identity cannot contain a secret")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class QueueId(OpaqueIdentity):
    """Immutable CloudDeck queue identity."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_uuid_v7(self.value, "queue_id")


@dataclass(frozen=True, order=True)
class PoolId(OpaqueIdentity):
    """Immutable managed-pool identity."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_uuid_v7(self.value, "pool_id")


@dataclass(frozen=True, order=True)
class ProfileId(OpaqueIdentity):
    """Immutable evaluator profile identity."""


@dataclass(frozen=True, order=True)
class ArtifactIssuerId(OpaqueIdentity):
    """Configured artifact issuer identity."""


@dataclass(frozen=True, order=True)
class WorkUnitId(OpaqueIdentity):
    """Immutable work-unit identity."""


@dataclass(frozen=True, order=True)
class IdentityText(OpaqueIdentity):
    """Bounded non-secret identity text."""


@dataclass(frozen=True, order=True)
class ImmutableLocation:
    """An exact immutable object location, never a short-lived access URL."""

    value: str

    def __post_init__(self) -> None:
        _string(self.value, "artifact location", maximum=2_048)
        if _HTTP_URL_RE.search(self.value):
            raise QueuePlanError("artifact location cannot be an access URL")
        if _SECRET_VALUE_RE.search(self.value):
            raise QueuePlanError("artifact location cannot contain a credential")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: object, field: str = "artifact location") -> ImmutableLocation:
        return cls(_string(value, field, maximum=2_048))


@dataclass(frozen=True, order=True)
class ArtifactLocation(ImmutableLocation):
    """Typed immutable artifact location."""


@dataclass(frozen=True, order=True)
class PositiveInteger:
    """A positive bounded integer used by a queue contract."""

    value: int

    def __post_init__(self) -> None:
        _integer(self.value, "positive integer", minimum=1)


@dataclass(frozen=True, order=True)
class NonNegativeInteger:
    """A nonnegative bounded integer used by a cost record."""

    value: int

    def __post_init__(self) -> None:
        _integer(self.value, "nonnegative integer", minimum=0)


@dataclass(frozen=True, order=True)
class MaxActiveLeases(PositiveInteger):
    """Validated queue-level concurrent lease capacity."""


@dataclass(frozen=True, order=True)
class PoolCapacity(PositiveInteger):
    """Validated managed-worker pool capacity."""


@dataclass(frozen=True, order=True)
class ValidationResultLimit(PositiveInteger):
    """Validated compact retained-result byte limit."""


@dataclass(frozen=True, order=True)
class UnixDeadline(PositiveInteger):
    """Validated absolute Unix deadline."""


@dataclass(frozen=True, order=True)
class WorkerCount(PositiveInteger):
    """Validated managed-worker count in one cost record."""


@dataclass(frozen=True, order=True)
class ColdDevCacheAdmissions(PositiveInteger):
    """Validated cold development-cache admission count."""


class WorkProtocolVersion(IntEnum):
    """CloudDeck's explicitly selected leased-work protocol version."""

    V3 = 3


class ManagedReusePolicy(str, Enum):
    """Worker reuse policy for a managed CloudDeck pool."""

    QUEUE_LIFETIME = "queue_lifetime"


class EvaluatorNetworkPolicy(str, Enum):
    """Network access required for static evaluator downloads."""

    EGRESS = "egress"


class ArtifactAccessOperation(str, Enum):
    """Artifact operation allowed by a unit declaration."""

    READ = "read"


class ArtifactCompression(str, Enum):
    """Compression identity carried by a complete artifact reference."""

    GZIP = "gzip"
    ZSTD = "zstd"
    BROTLI = "brotli"


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """A positive application schema version."""

    value: int

    def __post_init__(self) -> None:
        _integer(self.value, "schema version", minimum=1)


@dataclass(frozen=True)
class CapabilityIdentity:
    """The v3 workload capability identity accepted by the queue."""

    protocol_version: WorkProtocolVersion = WorkProtocolVersion.V3
    workload_family: IdentityText = IdentityText(DEFAULT_WORKLOAD_FAMILY)
    request_schema: SchemaVersion = SchemaVersion(DEFAULT_REQUEST_SCHEMA)
    result_schema: SchemaVersion = SchemaVersion(DEFAULT_RESULT_SCHEMA)
    proof_schema: SchemaVersion = SchemaVersion(DEFAULT_PROOF_SCHEMA)

    def __post_init__(self) -> None:
        if self.protocol_version is not WorkProtocolVersion.V3:
            raise QueuePlanError("validation workers require protocol v3")
        if not isinstance(self.workload_family, IdentityText):
            raise QueuePlanError("capability workload_family must be an IdentityText")
        for field in ("request_schema", "result_schema", "proof_schema"):
            value = getattr(self, field)
            if not isinstance(value, SchemaVersion):
                raise QueuePlanError(f"capability {field} must be a SchemaVersion")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": int(self.protocol_version),
            "workload_family": self.workload_family.value,
            "request_schema": self.request_schema.value,
            "result_schema": self.result_schema.value,
            "proof_schema": self.proof_schema.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> CapabilityIdentity:
        mapping = _require_mapping(value, "capability identity")
        _exact_fields(
            mapping,
            required=frozenset(
                {"protocol_version", "workload_family", "request_schema", "result_schema", "proof_schema"}
            ),
            context="capability identity",
        )
        try:
            version = WorkProtocolVersion(_integer(mapping["protocol_version"], "protocol_version"))
        except (TypeError, ValueError) as error:
            raise QueuePlanError("capability identity protocol version is not v3") from error
        return cls(
            protocol_version=version,
            workload_family=IdentityText(_string(mapping["workload_family"], "workload_family")),
            request_schema=SchemaVersion(_integer(mapping["request_schema"], "request_schema", minimum=1)),
            result_schema=SchemaVersion(_integer(mapping["result_schema"], "result_schema", minimum=1)),
            proof_schema=SchemaVersion(_integer(mapping["proof_schema"], "proof_schema", minimum=1)),
        )


class CapabilityCancellation(str, Enum):
    """Cancellation behavior published by the validation worker."""

    NONE = "none"
    COOPERATIVE = "cooperative"
    FORCED = "forced"


class CapabilityResume(str, Enum):
    """Lease-loss resume behavior published by the validation worker."""

    NONE = "none"
    UNIT_BOUNDARY = "unit_boundary"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class WorkerCapabilities:
    """Complete capability document used to bind a CloudDeck queue."""

    protocol_versions: tuple[WorkProtocolVersion, ...]
    workload_family: IdentityText
    request_schemas: tuple[SchemaVersion, ...]
    result_schemas: tuple[SchemaVersion, ...]
    proof_schemas: tuple[SchemaVersion, ...]
    max_payload_bytes: PositiveInteger
    max_result_bytes: PositiveInteger
    progress: None = None
    cancellation: CapabilityCancellation = CapabilityCancellation.COOPERATIVE
    resume: CapabilityResume = CapabilityResume.UNIT_BOUNDARY

    def __post_init__(self) -> None:
        if not self.protocol_versions or tuple(sorted(set(self.protocol_versions))) != self.protocol_versions:
            raise QueuePlanError("capability protocol_versions must be a sorted unique tuple")
        if WorkProtocolVersion.V3 not in self.protocol_versions:
            raise QueuePlanError("validation capabilities must include protocol v3")
        if not isinstance(self.workload_family, IdentityText):
            raise QueuePlanError("capability workload_family must be an IdentityText")
        for field in ("request_schemas", "result_schemas", "proof_schemas"):
            versions = getattr(self, field)
            if not versions or tuple(sorted(set(versions))) != versions:
                raise QueuePlanError(f"capability {field} must be a sorted unique tuple")
            if not all(isinstance(version, SchemaVersion) for version in versions):
                raise QueuePlanError(f"capability {field} must contain SchemaVersion values")
        if not isinstance(self.max_payload_bytes, PositiveInteger):
            raise QueuePlanError("capability max_payload_bytes must be a PositiveInteger")
        if not isinstance(self.max_result_bytes, PositiveInteger):
            raise QueuePlanError("capability max_result_bytes must be a PositiveInteger")
        if self.progress is not None:
            raise QueuePlanError("validation capabilities do not publish progress")
        if not isinstance(self.cancellation, CapabilityCancellation):
            raise QueuePlanError("capability cancellation must be typed")
        if not isinstance(self.resume, CapabilityResume):
            raise QueuePlanError("capability resume must be typed")

    def to_dict(self) -> dict[str, object]:
        return {
            "cancellation": self.cancellation.value,
            "max_payload_bytes": self.max_payload_bytes.value,
            "max_result_bytes": self.max_result_bytes.value,
            "progress": None,
            "proof_schemas": [version.value for version in self.proof_schemas],
            "protocol_versions": [int(version) for version in self.protocol_versions],
            "request_schemas": [version.value for version in self.request_schemas],
            "result_schemas": [version.value for version in self.result_schemas],
            "resume": self.resume.value,
            "workload_family": self.workload_family.value,
        }

    def canonical_bytes(self) -> bytes:
        """Return the exact compact capability encoding used by CloudDeck."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def digest(self) -> Sha256Digest:
        """Return the SHA-256 identity of the complete capability document."""

        return Sha256Digest(hashlib.sha256(self.canonical_bytes()).hexdigest())

    def negotiate_v3(self) -> CapabilityIdentity:
        """Return the v3 identity selected from this capability document."""

        return CapabilityIdentity(
            protocol_version=WorkProtocolVersion.V3,
            workload_family=self.workload_family,
            request_schema=self.request_schemas[-1],
            result_schema=self.result_schemas[-1],
            proof_schema=self.proof_schemas[-1],
        )

    @classmethod
    def from_dict(cls, value: object) -> WorkerCapabilities:
        mapping = _require_mapping(value, "worker capabilities")
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "cancellation",
                    "max_payload_bytes",
                    "max_result_bytes",
                    "progress",
                    "proof_schemas",
                    "protocol_versions",
                    "request_schemas",
                    "result_schemas",
                    "resume",
                    "workload_family",
                }
            ),
            context="worker capabilities",
        )
        protocol_values = mapping["protocol_versions"]
        request_values = mapping["request_schemas"]
        result_values = mapping["result_schemas"]
        proof_values = mapping["proof_schemas"]
        if not all(
            isinstance(values, list) for values in (protocol_values, request_values, result_values, proof_values)
        ):
            raise QueuePlanError("worker capability version fields must be arrays")
        try:
            protocols = tuple(
                WorkProtocolVersion(_integer(version, "protocol version")) for version in protocol_values
            )
        except ValueError as error:
            raise QueuePlanError("worker capabilities contain an unsupported protocol") from error
        try:
            cancellation = CapabilityCancellation(_string(mapping["cancellation"], "cancellation"))
            resume = CapabilityResume(_string(mapping["resume"], "resume"))
        except ValueError as error:
            raise QueuePlanError("worker capabilities contain an unsupported policy") from error
        return cls(
            protocol_versions=protocols,
            workload_family=IdentityText(_string(mapping["workload_family"], "workload_family")),
            request_schemas=tuple(
                SchemaVersion(_integer(version, "request schema", minimum=1)) for version in request_values
            ),
            result_schemas=tuple(
                SchemaVersion(_integer(version, "result schema", minimum=1)) for version in result_values
            ),
            proof_schemas=tuple(
                SchemaVersion(_integer(version, "proof schema", minimum=1)) for version in proof_values
            ),
            max_payload_bytes=PositiveInteger(_integer(mapping["max_payload_bytes"], "max_payload_bytes", minimum=1)),
            max_result_bytes=PositiveInteger(_integer(mapping["max_result_bytes"], "max_result_bytes", minimum=1)),
            progress=None if mapping["progress"] is None else _reject_capability_progress(),
            cancellation=cancellation,
            resume=resume,
        )


def _reject_capability_progress() -> None:
    raise QueuePlanError("validation capabilities do not publish progress")


@dataclass(frozen=True)
class RequiredCapability:
    """A digest-bound v3 capability identity required by one queue."""

    capability_digest: Sha256Digest
    identity: CapabilityIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.capability_digest, Sha256Digest):
            raise QueuePlanError("capability_digest must be a Sha256Digest")
        if not isinstance(self.identity, CapabilityIdentity):
            raise QueuePlanError("required capability identity must be a CapabilityIdentity")
        if self.identity.protocol_version is not WorkProtocolVersion.V3:
            raise QueuePlanError("required capability identity must be v3")

    def to_dict(self) -> dict[str, object]:
        return {"capability_digest": self.capability_digest.value, "identity": self.identity.to_dict()}

    @classmethod
    def from_dict(cls, value: object) -> RequiredCapability:
        mapping = _require_mapping(value, "required capability")
        _exact_fields(mapping, required=frozenset({"capability_digest", "identity"}), context="required capability")
        identity = CapabilityIdentity.from_dict(mapping["identity"])
        return cls(_digest(mapping["capability_digest"], "capability_digest"), identity)


@dataclass(frozen=True)
class QueueProtocol:
    """Queue-level protocol-v3 and single artifact-issuer selection."""

    version: WorkProtocolVersion
    artifact_issuer_id: ArtifactIssuerId

    def __post_init__(self) -> None:
        if self.version is not WorkProtocolVersion.V3:
            raise QueuePlanError("validation queues require protocol v3")
        if not isinstance(self.artifact_issuer_id, ArtifactIssuerId):
            raise QueuePlanError("artifact issuer must be an ArtifactIssuerId")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "artifact_access": {"mode": "issuer", "issuer_id": self.artifact_issuer_id.value},
        }

    @classmethod
    def from_dict(cls, value: object) -> QueueProtocol:
        mapping = _require_mapping(value, "queue protocol")
        _exact_fields(mapping, required=frozenset({"version", "artifact_access"}), context="queue protocol")
        try:
            version = WorkProtocolVersion(_integer(mapping["version"], "protocol version"))
        except (TypeError, ValueError) as error:
            raise QueuePlanError("queue protocol must be v3") from error
        access = _require_mapping(mapping["artifact_access"], "artifact access")
        _exact_fields(access, required=frozenset({"mode", "issuer_id"}), context="artifact access")
        if access["mode"] != "issuer":
            raise QueuePlanError("validation queues require issuer-backed artifact access")
        return cls(version, ArtifactIssuerId(_string(access["issuer_id"], "issuer_id")))


@dataclass(frozen=True)
class ResultPolicy:
    """Compact retained JSON policy for one ``ValidationResult``."""

    max_bytes: ValidationResultLimit
    mode: ClassVar[str] = "retain_json"

    def __post_init__(self) -> None:
        if not isinstance(self.max_bytes, ValidationResultLimit):
            raise QueuePlanError("result policy max_bytes must be a ValidationResultLimit")
        if self.max_bytes.value > 1_048_576:
            raise QueuePlanError("validation result limit exceeds the CloudDeck maximum")

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "max_bytes": self.max_bytes.value}

    @classmethod
    def from_dict(cls, value: object) -> ResultPolicy:
        mapping = _require_mapping(value, "result policy")
        _exact_fields(mapping, required=frozenset({"mode", "max_bytes"}), context="result policy")
        if mapping["mode"] != cls.mode:
            raise QueuePlanError("validation queues require retained JSON results")
        return cls(ValidationResultLimit(_integer(mapping["max_bytes"], "result limit", minimum=1)))


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry policy for one immutable validation queue."""

    max_attempts: PositiveInteger = PositiveInteger(1)
    retry_worker_failures: bool = False
    retry_deadline_exceeded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, PositiveInteger):
            raise QueuePlanError("retry policy max_attempts must be a PositiveInteger")
        if not isinstance(self.retry_worker_failures, bool) or not isinstance(self.retry_deadline_exceeded, bool):
            raise QueuePlanError("retry policy switches must be booleans")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_attempts": self.max_attempts.value,
            "retry_worker_failures": self.retry_worker_failures,
            "retry_deadline_exceeded": self.retry_deadline_exceeded,
        }

    @classmethod
    def from_dict(cls, value: object) -> RetryPolicy:
        mapping = _require_mapping(value, "retry policy")
        _exact_fields(
            mapping,
            required=frozenset({"max_attempts", "retry_worker_failures", "retry_deadline_exceeded"}),
            context="retry policy",
        )
        if not isinstance(mapping["retry_worker_failures"], bool) or not isinstance(
            mapping["retry_deadline_exceeded"], bool
        ):
            raise QueuePlanError("retry policy switches must be booleans")
        return cls(
            PositiveInteger(_integer(mapping["max_attempts"], "max_attempts", minimum=1)),
            mapping["retry_worker_failures"],
            mapping["retry_deadline_exceeded"],
        )


@dataclass(frozen=True)
class MediaType:
    """A bounded artifact media type."""

    value: str

    def __post_init__(self) -> None:
        _string(self.value, "media type", maximum=255)
        if "/" not in self.value or any(ord(character) > 127 for character in self.value):
            raise QueuePlanError("media type must use ASCII type/subtype syntax")


@dataclass(frozen=True)
class ArtifactRef:
    """A complete immutable content-addressed development bundle reference."""

    content_digest: Sha256Digest
    byte_length: NonNegativeInteger
    media_type: MediaType
    location: ImmutableLocation
    compression: ArtifactCompression | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content_digest, Sha256Digest):
            raise QueuePlanError("artifact content_digest must be a Sha256Digest")
        if not isinstance(self.byte_length, NonNegativeInteger):
            raise QueuePlanError("artifact byte_length must be a NonNegativeInteger")
        if self.byte_length.value > MAX_ARTIFACT_BYTES:
            raise QueuePlanError("artifact byte_length exceeds the CloudDeck maximum")
        if not isinstance(self.media_type, MediaType):
            raise QueuePlanError("artifact media_type must be a MediaType")
        if not isinstance(self.location, ImmutableLocation):
            raise QueuePlanError("artifact location must be an ImmutableLocation")
        if self.compression is not None and not isinstance(self.compression, ArtifactCompression):
            raise QueuePlanError("artifact compression must be typed")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "content_digest": self.content_digest.value,
            "byte_length": self.byte_length.value,
            "media_type": self.media_type.value,
            "location": self.location.value,
            "compression": self.compression.value if self.compression is not None else None,
        }
        return value

    @classmethod
    def from_dict(cls, value: object) -> ArtifactRef:
        mapping = _require_mapping(value, "artifact reference")
        _exact_fields(
            mapping,
            required=frozenset({"content_digest", "byte_length", "media_type", "location"}),
            optional=frozenset({"compression"}),
            context="artifact reference",
        )
        compression = mapping.get("compression")
        if compression is not None:
            try:
                compression = ArtifactCompression(_string(compression, "artifact compression", maximum=32))
            except ValueError as error:
                raise QueuePlanError("artifact compression is not supported") from error
        return cls(
            content_digest=_digest(mapping["content_digest"], "content_digest"),
            byte_length=NonNegativeInteger(_integer(mapping["byte_length"], "byte_length", minimum=0)),
            media_type=MediaType(_string(mapping["media_type"], "media_type", maximum=255)),
            location=ImmutableLocation.parse(mapping["location"], "location"),
            compression=compression,
        )


@dataclass(frozen=True)
class UncommittedModelArtifact:
    """A future model location with no digest until its manifest commits."""

    location: ImmutableLocation
    state: ClassVar[str] = "uncommitted"

    def __post_init__(self) -> None:
        if not isinstance(self.location, ImmutableLocation):
            raise QueuePlanError("future model location must be an ImmutableLocation")

    def to_dict(self) -> dict[str, object]:
        return {"state": self.state, "location": self.location.value}

    @classmethod
    def from_dict(cls, value: object) -> UncommittedModelArtifact:
        mapping = _require_mapping(value, "future model artifact")
        _exact_fields(mapping, required=frozenset({"state", "location"}), context="future model artifact")
        if mapping["state"] != cls.state:
            raise QueuePlanError("future model artifact must remain uncommitted")
        return cls(ImmutableLocation.parse(mapping["location"], "future model location"))


@dataclass(frozen=True)
class ReadAccessSpec:
    """One exact lease-scoped read declaration."""

    location: ImmutableLocation
    not_after: UnixDeadline
    operation: ArtifactAccessOperation = ArtifactAccessOperation.READ

    def __post_init__(self) -> None:
        if not isinstance(self.location, ImmutableLocation):
            raise QueuePlanError("access location must be an ImmutableLocation")
        if not isinstance(self.not_after, UnixDeadline):
            raise QueuePlanError("access expiry must be a UnixDeadline")
        if self.operation is not ArtifactAccessOperation.READ:
            raise QueuePlanError("validation units only authorize artifact reads")

    def to_dict(self) -> dict[str, object]:
        return {
            "location": self.location.value,
            "operation": self.operation.value,
            "not_after_unix_seconds": self.not_after.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReadAccessSpec:
        mapping = _require_mapping(value, "artifact access specification")
        _exact_fields(
            mapping,
            required=frozenset({"location", "operation", "not_after_unix_seconds"}),
            context="artifact access specification",
        )
        if mapping["operation"] != ArtifactAccessOperation.READ.value:
            raise QueuePlanError("validation units only authorize artifact reads")
        return cls(
            location=ImmutableLocation.parse(mapping["location"], "access location"),
            operation=ArtifactAccessOperation.READ,
            not_after=UnixDeadline(_integer(mapping["not_after_unix_seconds"], "not_after_unix_seconds", minimum=1)),
        )


@dataclass(frozen=True)
class ExpectedIdentities:
    """Campaign and evaluator identities copied into every work payload."""

    campaign_id: Sha256Digest
    slot_id: Sha256Digest
    training_launch_id: IdentityText
    updates: int
    trainer_configuration_digest: Sha256Digest
    evaluator_image_identity: IdentityText
    evaluator_implementation_digest: Sha256Digest
    dev_bundle_digest: Sha256Digest

    def __post_init__(self) -> None:
        for field in (
            "campaign_id",
            "slot_id",
            "trainer_configuration_digest",
            "evaluator_implementation_digest",
            "dev_bundle_digest",
        ):
            if not isinstance(getattr(self, field), Sha256Digest):
                raise QueuePlanError(f"{field} must be a Sha256Digest")
        if not isinstance(self.training_launch_id, IdentityText):
            raise QueuePlanError("training_launch_id must be an IdentityText")
        if not isinstance(self.evaluator_image_identity, IdentityText):
            raise QueuePlanError("evaluator_image_identity must be an IdentityText")
        _integer(self.updates, "updates")

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id.value,
            "slot_id": self.slot_id.value,
            "training_launch_id": self.training_launch_id.value,
            "updates": self.updates,
            "trainer_configuration_digest": self.trainer_configuration_digest.value,
            "evaluator_image_identity": self.evaluator_image_identity.value,
            "evaluator_implementation_digest": self.evaluator_implementation_digest.value,
            "dev_bundle_digest": self.dev_bundle_digest.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExpectedIdentities:
        mapping = _require_mapping(value, "expected identities")
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "campaign_id",
                    "slot_id",
                    "training_launch_id",
                    "updates",
                    "trainer_configuration_digest",
                    "evaluator_image_identity",
                    "evaluator_implementation_digest",
                    "dev_bundle_digest",
                }
            ),
            context="expected identities",
        )
        return cls(
            campaign_id=_digest(mapping["campaign_id"], "campaign_id"),
            slot_id=_digest(mapping["slot_id"], "slot_id"),
            training_launch_id=IdentityText(_string(mapping["training_launch_id"], "training_launch_id")),
            updates=_integer(mapping["updates"], "updates"),
            trainer_configuration_digest=_digest(
                mapping["trainer_configuration_digest"], "trainer_configuration_digest"
            ),
            evaluator_image_identity=IdentityText(
                _string(mapping["evaluator_image_identity"], "evaluator_image_identity")
            ),
            evaluator_implementation_digest=_digest(
                mapping["evaluator_implementation_digest"], "evaluator_implementation_digest"
            ),
            dev_bundle_digest=_digest(mapping["dev_bundle_digest"], "dev_bundle_digest"),
        )


@dataclass(frozen=True)
class ValidationWorkPayload:
    """Typed immutable payload for one validation snapshot slot."""

    identities: ExpectedIdentities
    slot_ordinal: int
    point: SnapshotPoint
    manifest_location: ImmutableLocation
    model_location: ImmutableLocation
    future_model: UncommittedModelArtifact
    frozen_dev_bundle: ArtifactRef
    trainer_configuration: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.identities, ExpectedIdentities):
            raise QueuePlanError("payload identities must be ExpectedIdentities")
        if not isinstance(self.point, SnapshotPoint):
            raise QueuePlanError("payload point must be a SnapshotPoint")
        _integer(self.slot_ordinal, "slot ordinal")
        if not isinstance(self.manifest_location, ImmutableLocation):
            raise QueuePlanError("manifest location must be an ImmutableLocation")
        if not isinstance(self.model_location, ImmutableLocation):
            raise QueuePlanError("model location must be an ImmutableLocation")
        if not isinstance(self.future_model, UncommittedModelArtifact):
            raise QueuePlanError("future model must use explicit uncommitted state")
        if not isinstance(self.frozen_dev_bundle, ArtifactRef):
            raise QueuePlanError("frozen development bundle must be a complete ArtifactRef")
        if not isinstance(self.trainer_configuration, ArtifactRef):
            raise QueuePlanError("trainer configuration must be a complete ArtifactRef")
        if self.identities.dev_bundle_digest != self.frozen_dev_bundle.content_digest:
            raise QueuePlanError("frozen development bundle digest does not match campaign identity")
        if self.identities.trainer_configuration_digest != self.trainer_configuration.content_digest:
            raise QueuePlanError("trainer configuration digest does not match campaign identity")
        if self.future_model.location != self.model_location:
            raise QueuePlanError("future model location must equal the slot model location")
        if self.trainer_configuration.location in (
            self.manifest_location,
            self.model_location,
            self.frozen_dev_bundle.location,
        ):
            raise QueuePlanError("trainer configuration location must be distinct")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": WORK_PAYLOAD_SCHEMA,
            **self.identities.to_dict(),
            "slot_ordinal": self.slot_ordinal,
            "point": self.point.to_dict(),
            "manifest_location": self.manifest_location.value,
            "model_location": self.model_location.value,
            "future_model": self.future_model.to_dict(),
            "frozen_dev_bundle": self.frozen_dev_bundle.to_dict(),
            "trainer_configuration": self.trainer_configuration.to_dict(),
        }

    def digest(self) -> Sha256Digest:
        """Return the digest of this exact canonical payload."""

        return _canonical_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ValidationWorkPayload:
        mapping = _require_mapping(value, "validation work payload")
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "schema",
                    "campaign_id",
                    "slot_id",
                    "training_launch_id",
                    "updates",
                    "trainer_configuration_digest",
                    "evaluator_image_identity",
                    "evaluator_implementation_digest",
                    "dev_bundle_digest",
                    "slot_ordinal",
                    "point",
                    "manifest_location",
                    "model_location",
                    "future_model",
                    "frozen_dev_bundle",
                    "trainer_configuration",
                }
            ),
            context="validation work payload",
        )
        if mapping["schema"] != WORK_PAYLOAD_SCHEMA:
            raise QueuePlanError("validation work payload schema is not supported")
        return cls(
            identities=ExpectedIdentities.from_dict(
                {
                    "campaign_id": mapping["campaign_id"],
                    "slot_id": mapping["slot_id"],
                    "training_launch_id": mapping["training_launch_id"],
                    "updates": mapping["updates"],
                    "trainer_configuration_digest": mapping["trainer_configuration_digest"],
                    "evaluator_image_identity": mapping["evaluator_image_identity"],
                    "evaluator_implementation_digest": mapping["evaluator_implementation_digest"],
                    "dev_bundle_digest": mapping["dev_bundle_digest"],
                }
            ),
            slot_ordinal=_integer(mapping["slot_ordinal"], "slot_ordinal"),
            point=snapshot_point_from_dict(mapping["point"]),
            manifest_location=ImmutableLocation.parse(mapping["manifest_location"], "manifest_location"),
            model_location=ImmutableLocation.parse(mapping["model_location"], "model_location"),
            future_model=UncommittedModelArtifact.from_dict(mapping["future_model"]),
            frozen_dev_bundle=ArtifactRef.from_dict(mapping["frozen_dev_bundle"]),
            trainer_configuration=ArtifactRef.from_dict(mapping["trainer_configuration"]),
        )


@dataclass(frozen=True)
class ValidationWorkUnit:
    """One ordered CloudDeck work unit for one validation slot."""

    unit_id: WorkUnitId
    ordinal: int
    payload: ValidationWorkPayload
    artifact_access_specs: tuple[ReadAccessSpec, ...]
    payload_digest: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, WorkUnitId):
            raise QueuePlanError("unit_id must be a WorkUnitId")
        _integer(self.ordinal, "unit ordinal")
        if not isinstance(self.payload, ValidationWorkPayload):
            raise QueuePlanError("unit payload must be a ValidationWorkPayload")
        if not isinstance(self.artifact_access_specs, tuple):
            raise QueuePlanError("unit artifact_access_specs must be a tuple")
        if len(self.artifact_access_specs) != 4:
            raise QueuePlanError(
                "validation unit must declare manifest, model, development-bundle, and trainer-configuration reads"
            )
        if not all(isinstance(spec, ReadAccessSpec) for spec in self.artifact_access_specs):
            raise QueuePlanError("validation unit access specifications must be typed reads")
        if len({spec.location for spec in self.artifact_access_specs}) != 4:
            raise QueuePlanError("validation unit access locations must be distinct")
        expected_locations = (
            self.payload.manifest_location,
            self.payload.model_location,
            self.payload.frozen_dev_bundle.location,
            self.payload.trainer_configuration.location,
        )
        if tuple(spec.location for spec in self.artifact_access_specs) != expected_locations:
            raise QueuePlanError(
                "validation unit access declarations must be manifest/model/dev bundle/trainer configuration reads"
            )
        if not isinstance(self.payload_digest, Sha256Digest):
            raise QueuePlanError("unit payload_digest must be a Sha256Digest")
        expected = self.payload.digest()
        if self.payload_digest != expected:
            raise QueuePlanError("work-unit payload digest does not match its payload")

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id.value,
            "ordinal": self.ordinal,
            "payload": self.payload.to_dict(),
            "payload_digest": self.payload_digest.value,
            "progress_total": None,
            "artifact_access_specs": [spec.to_dict() for spec in self.artifact_access_specs],
        }

    @classmethod
    def from_dict(cls, value: object) -> ValidationWorkUnit:
        mapping = _require_mapping(value, "validation work unit")
        _reject_unsafe(mapping)
        _exact_fields(
            mapping,
            required=frozenset(
                {"unit_id", "ordinal", "payload", "payload_digest", "progress_total", "artifact_access_specs"}
            ),
            context="validation work unit",
        )
        if mapping["progress_total"] is not None:
            raise QueuePlanError("validation units do not declare progress totals")
        specs_value = mapping["artifact_access_specs"]
        if not isinstance(specs_value, list):
            raise QueuePlanError("artifact_access_specs must be an array")
        return cls(
            unit_id=WorkUnitId(_string(mapping["unit_id"], "unit_id")),
            ordinal=_integer(mapping["ordinal"], "ordinal"),
            payload=ValidationWorkPayload.from_dict(mapping["payload"]),
            artifact_access_specs=tuple(ReadAccessSpec.from_dict(spec) for spec in specs_value),
            payload_digest=_digest(mapping["payload_digest"], "payload_digest"),
        )


@dataclass(frozen=True)
class CampaignWorkloadPayload:
    """Queue-level typed identity copied into every assignment contract."""

    campaign_id: Sha256Digest
    training_launch_id: IdentityText
    trainer_configuration_digest: Sha256Digest
    evaluator_image_identity: IdentityText
    evaluator_implementation_digest: Sha256Digest
    dev_bundle_digest: Sha256Digest

    def __post_init__(self) -> None:
        for field in (
            "campaign_id",
            "trainer_configuration_digest",
            "evaluator_implementation_digest",
            "dev_bundle_digest",
        ):
            if not isinstance(getattr(self, field), Sha256Digest):
                raise QueuePlanError(f"{field} must be a Sha256Digest")
        if not isinstance(self.training_launch_id, IdentityText):
            raise QueuePlanError("training_launch_id must be an IdentityText")
        if not isinstance(self.evaluator_image_identity, IdentityText):
            raise QueuePlanError("evaluator_image_identity must be an IdentityText")

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id.value,
            "training_launch_id": self.training_launch_id.value,
            "trainer_configuration_digest": self.trainer_configuration_digest.value,
            "evaluator_image_identity": self.evaluator_image_identity.value,
            "evaluator_implementation_digest": self.evaluator_implementation_digest.value,
            "dev_bundle_digest": self.dev_bundle_digest.value,
        }

    def digest(self) -> Sha256Digest:
        return _canonical_digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> CampaignWorkloadPayload:
        mapping = _require_mapping(value, "workload payload")
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "campaign_id",
                    "training_launch_id",
                    "trainer_configuration_digest",
                    "evaluator_image_identity",
                    "evaluator_implementation_digest",
                    "dev_bundle_digest",
                }
            ),
            context="workload payload",
        )
        return cls(
            campaign_id=_digest(mapping["campaign_id"], "campaign_id"),
            training_launch_id=IdentityText(_string(mapping["training_launch_id"], "training_launch_id")),
            trainer_configuration_digest=_digest(
                mapping["trainer_configuration_digest"], "trainer_configuration_digest"
            ),
            evaluator_image_identity=IdentityText(
                _string(mapping["evaluator_image_identity"], "evaluator_image_identity")
            ),
            evaluator_implementation_digest=_digest(
                mapping["evaluator_implementation_digest"], "evaluator_implementation_digest"
            ),
            dev_bundle_digest=_digest(mapping["dev_bundle_digest"], "dev_bundle_digest"),
        )


@dataclass(frozen=True)
class QueueRequest:
    """Complete immutable CloudDeck queue request using protocol-v3 fields."""

    queue_id: QueueId
    workload_revision: Sha256Digest
    manifest_digest: Sha256Digest
    payload_schema: IdentityText
    workload_payload: CampaignWorkloadPayload
    workload_payload_digest: Sha256Digest
    max_payload_bytes: PositiveInteger
    lease_duration_seconds: PositiveInteger
    retry_policy: RetryPolicy
    result_policy: ResultPolicy
    deadline_unix_seconds: UnixDeadline
    max_active_leases: MaxActiveLeases
    protocol: QueueProtocol
    required_capability: RequiredCapability
    progress_unit: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.queue_id, QueueId):
            raise QueuePlanError("queue_id must be a QueueId")
        if not isinstance(self.payload_schema, IdentityText):
            raise QueuePlanError("payload_schema must be an IdentityText")
        if not isinstance(self.workload_payload, CampaignWorkloadPayload):
            raise QueuePlanError("workload_payload must be a CampaignWorkloadPayload")
        for field in ("workload_revision", "manifest_digest", "workload_payload_digest"):
            if not isinstance(getattr(self, field), Sha256Digest):
                raise QueuePlanError(f"{field} must be a Sha256Digest")
        if not isinstance(self.max_payload_bytes, PositiveInteger):
            raise QueuePlanError("max_payload_bytes must be a PositiveInteger")
        if self.max_payload_bytes.value > DEFAULT_MAX_PAYLOAD_BYTES:
            raise QueuePlanError("max_payload_bytes exceeds the CloudDeck maximum")
        if not isinstance(self.lease_duration_seconds, PositiveInteger):
            raise QueuePlanError("lease_duration_seconds must be a PositiveInteger")
        if not 5 <= self.lease_duration_seconds.value <= 24 * 60 * 60:
            raise QueuePlanError("lease_duration_seconds must be between 5 seconds and 24 hours")
        if not isinstance(self.deadline_unix_seconds, UnixDeadline):
            raise QueuePlanError("deadline_unix_seconds must be a UnixDeadline")
        if not isinstance(self.max_active_leases, MaxActiveLeases):
            raise QueuePlanError("max_active_leases must be a MaxActiveLeases")
        if not isinstance(self.retry_policy, RetryPolicy) or not isinstance(self.result_policy, ResultPolicy):
            raise QueuePlanError("queue policies must be typed")
        if not isinstance(self.protocol, QueueProtocol) or self.protocol.version is not WorkProtocolVersion.V3:
            raise QueuePlanError("queue request must select protocol v3")
        if self.max_active_leases.value != 1:
            raise QueuePlanError("validation queue max_active_leases must be exactly one")
        if not isinstance(self.required_capability, RequiredCapability):
            raise QueuePlanError("queue request required_capability must be typed")
        if self.progress_unit is not None:
            raise QueuePlanError("validation queue progress_unit must be null")
        if self.workload_payload_digest != self.workload_payload.digest():
            raise QueuePlanError("workload payload digest does not match its payload")
        if self.deadline_unix_seconds.value <= self.lease_duration_seconds.value:
            raise QueuePlanError("queue deadline must leave one lease lifetime")

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_id": self.queue_id.value,
            "workload_revision": self.workload_revision.value,
            "manifest_digest": self.manifest_digest.value,
            "payload_schema": self.payload_schema.value,
            "workload_payload": self.workload_payload.to_dict(),
            "workload_payload_digest": self.workload_payload_digest.value,
            "max_payload_bytes": self.max_payload_bytes.value,
            "lease_duration_seconds": self.lease_duration_seconds.value,
            "retry_policy": self.retry_policy.to_dict(),
            "progress_unit": self.progress_unit,
            "result_policy": self.result_policy.to_dict(),
            "deadline_unix_seconds": self.deadline_unix_seconds.value,
            "max_active_leases": self.max_active_leases.value,
            "protocol": self.protocol.to_dict(),
            "required_capability": self.required_capability.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> QueueRequest:
        mapping = _require_mapping(value, "queue request")
        _reject_unsafe(mapping)
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "queue_id",
                    "workload_revision",
                    "manifest_digest",
                    "payload_schema",
                    "workload_payload",
                    "workload_payload_digest",
                    "max_payload_bytes",
                    "lease_duration_seconds",
                    "retry_policy",
                    "progress_unit",
                    "result_policy",
                    "deadline_unix_seconds",
                    "max_active_leases",
                    "protocol",
                    "required_capability",
                }
            ),
            context="queue request",
        )
        if mapping["progress_unit"] is not None:
            raise QueuePlanError("validation queue progress_unit must be null")
        return cls(
            queue_id=QueueId(_string(mapping["queue_id"], "queue_id")),
            workload_revision=_digest(mapping["workload_revision"], "workload_revision"),
            manifest_digest=_digest(mapping["manifest_digest"], "manifest_digest"),
            payload_schema=IdentityText(_string(mapping["payload_schema"], "payload_schema")),
            workload_payload=CampaignWorkloadPayload.from_dict(mapping["workload_payload"]),
            workload_payload_digest=_digest(mapping["workload_payload_digest"], "workload_payload_digest"),
            max_payload_bytes=PositiveInteger(_integer(mapping["max_payload_bytes"], "max_payload_bytes", minimum=1)),
            lease_duration_seconds=PositiveInteger(
                _integer(mapping["lease_duration_seconds"], "lease_duration_seconds", minimum=1)
            ),
            retry_policy=RetryPolicy.from_dict(mapping["retry_policy"]),
            result_policy=ResultPolicy.from_dict(mapping["result_policy"]),
            deadline_unix_seconds=UnixDeadline(
                _integer(mapping["deadline_unix_seconds"], "deadline_unix_seconds", minimum=1)
            ),
            max_active_leases=MaxActiveLeases(_integer(mapping["max_active_leases"], "max_active_leases", minimum=1)),
            protocol=QueueProtocol.from_dict(mapping["protocol"]),
            required_capability=RequiredCapability.from_dict(mapping["required_capability"]),
        )


@dataclass(frozen=True)
class ManagedPool:
    """One immutable managed worker pool bound to exactly one queue."""

    pool_id: PoolId
    queue_id: QueueId
    profile_id: ProfileId
    profile_digest: Sha256Digest
    capacity: PoolCapacity
    reuse: ManagedReusePolicy
    service: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pool_id, PoolId):
            raise QueuePlanError("pool_id must be a PoolId")
        if not isinstance(self.queue_id, QueueId):
            raise QueuePlanError("queue_id must be a QueueId")
        if not isinstance(self.profile_id, ProfileId):
            raise QueuePlanError("profile_id must be a ProfileId")
        if not isinstance(self.profile_digest, Sha256Digest):
            raise QueuePlanError("profile_digest must be a Sha256Digest")
        if not isinstance(self.capacity, PoolCapacity):
            raise QueuePlanError("capacity must be a PoolCapacity")
        if self.capacity.value != 1:
            raise QueuePlanError("validation managed pool capacity must be exactly one")
        if self.reuse is not ManagedReusePolicy.QUEUE_LIFETIME:
            raise QueuePlanError("validation managed pool reuse must be queue_lifetime")
        if self.service is not None:
            raise QueuePlanError("validation workers use command workers without a managed service")

    def to_dict(self) -> dict[str, object]:
        return {
            "pool_id": self.pool_id.value,
            "queue_id": self.queue_id.value,
            "profile_id": self.profile_id.value,
            "profile_digest": self.profile_digest.value,
            "capacity": self.capacity.value,
            "reuse": self.reuse.value,
            "service": self.service,
        }

    @classmethod
    def from_dict(cls, value: object) -> ManagedPool:
        mapping = _require_mapping(value, "managed pool")
        _exact_fields(
            mapping,
            required=frozenset(
                {"pool_id", "queue_id", "profile_id", "profile_digest", "capacity", "reuse", "service"}
            ),
            context="managed pool",
        )
        if mapping["service"] is not None:
            raise QueuePlanError("validation managed pool service must be null")
        if mapping["reuse"] != ManagedReusePolicy.QUEUE_LIFETIME.value:
            raise QueuePlanError("managed pool reuse must be queue_lifetime")
        return cls(
            pool_id=PoolId(_string(mapping["pool_id"], "pool_id")),
            queue_id=QueueId(_string(mapping["queue_id"], "queue_id")),
            profile_id=ProfileId(_string(mapping["profile_id"], "profile_id")),
            profile_digest=_digest(mapping["profile_digest"], "profile_digest"),
            capacity=PoolCapacity(_integer(mapping["capacity"], "capacity", minimum=1)),
            reuse=ManagedReusePolicy.QUEUE_LIFETIME,
            service=None,
        )


def _host_profile_from_dict(value: object) -> EvaluatorCapabilityProfile:
    try:
        return EvaluatorCapabilityProfile.from_dict(value)
    except (TypeError, ValueError) as error:
        raise QueuePlanError("validation host profile is not a strict preflight profile") from error


def _parse_network_policy(value: object) -> EvaluatorNetworkPolicy:
    try:
        return EvaluatorNetworkPolicy(_string(value, "network_policy"))
    except ValueError as error:
        raise QueuePlanError("validation evaluator network policy must be egress") from error


@dataclass(frozen=True)
class ValidationEvaluatorProfile:
    """Static evaluator-host requirements for protocol-v3 validation workers."""

    profile_id: ProfileId
    host_profile: EvaluatorCapabilityProfile
    network_policy: EvaluatorNetworkPolicy
    download_egress: bool
    reusable_storage_credentials: bool
    github_credentials: bool
    registry_push_credentials: bool
    artifact_issuer_credentials: bool
    capabilities: WorkerCapabilities
    capability_identity: CapabilityIdentity
    capacity_limit: PoolCapacity = PoolCapacity(1)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, ProfileId):
            raise QueuePlanError("profile_id must be a ProfileId")
        if not isinstance(self.host_profile, EvaluatorCapabilityProfile):
            raise QueuePlanError("host_profile must use EvaluatorCapabilityProfile")
        if self.host_profile.accepted_gpu_names != (GpuName("RTX 5060 Ti"),):
            raise QueuePlanError("validation profile GPU name must be exactly `RTX 5060 Ti`")
        if self.host_profile.minimum_gpu_memory_bytes < 16 * GIB:
            raise QueuePlanError("validation profile requires at least 16 GiB GPU memory")
        if self.host_profile.minimum_cuda_version < CudaVersion(12, 8):
            raise QueuePlanError("validation profile requires CUDA >= 12.8")
        if self.host_profile.minimum_system_ram_bytes < 32 * GIB:
            raise QueuePlanError("validation profile requires at least 32 GiB system RAM")
        if self.host_profile.minimum_cpu_cores < 4:
            raise QueuePlanError("validation profile requires at least four CPU cores")
        if self.host_profile.minimum_free_disk_bytes < 1:
            raise QueuePlanError("validation profile requires configurable free disk")
        if not isinstance(self.capacity_limit, PoolCapacity):
            raise QueuePlanError("capacity_limit must be a PoolCapacity")
        if self.network_policy is not EvaluatorNetworkPolicy.EGRESS or self.download_egress is not True:
            raise QueuePlanError("validation profile requires public egress for downloads")
        for field in (
            "reusable_storage_credentials",
            "github_credentials",
            "registry_push_credentials",
            "artifact_issuer_credentials",
        ):
            if getattr(self, field) is not False:
                raise QueuePlanError(f"validation profile cannot carry {field}")
        if not isinstance(self.capabilities, WorkerCapabilities):
            raise QueuePlanError("capabilities must be a WorkerCapabilities document")
        if not isinstance(self.capability_identity, CapabilityIdentity):
            raise QueuePlanError("capability_identity must be a CapabilityIdentity")
        if self.capabilities.negotiate_v3() != self.capability_identity:
            raise QueuePlanError("capability identity must match v3 negotiation")
        if self.capabilities.max_payload_bytes.value < DEFAULT_MAX_PAYLOAD_BYTES:
            raise QueuePlanError("validation capabilities must accept the queue payload limit")
        if self.capabilities.max_result_bytes.value < DEFAULT_VALIDATION_RESULT_LIMIT_BYTES:
            raise QueuePlanError("validation capabilities must accept the result limit")
        if self.capability_identity.protocol_version is not WorkProtocolVersion.V3:
            raise QueuePlanError("validation profile capability identity must be v3")

    def to_dict(self) -> dict[str, object]:
        """Return the strict static fixture representation."""

        return {
            "schema": PROFILE_SCHEMA,
            "profile_id": self.profile_id.value,
            "host_profile": self.host_profile.to_dict(),
            "capacity_limit": self.capacity_limit.value,
            "network_policy": self.network_policy.value,
            "download_egress": self.download_egress,
            "reusable_storage_credentials": self.reusable_storage_credentials,
            "github_credentials": self.github_credentials,
            "registry_push_credentials": self.registry_push_credentials,
            "artifact_issuer_credentials": self.artifact_issuer_credentials,
            "capabilities": self.capabilities.to_dict(),
            "capability_identity": self.capability_identity.to_dict(),
        }

    def digest(self) -> Sha256Digest:
        """Return the identity of this complete static profile."""

        value = {
            "schema": PROFILE_SCHEMA,
            "profile_id": self.profile_id.value,
            "host_profile": self.host_profile.to_dict(),
            "capacity_limit": self.capacity_limit.value,
            "network_policy": self.network_policy.value,
            "download_egress": self.download_egress,
            "reusable_storage_credentials": self.reusable_storage_credentials,
            "github_credentials": self.github_credentials,
            "registry_push_credentials": self.registry_push_credentials,
            "artifact_issuer_credentials": self.artifact_issuer_credentials,
            "capabilities": self.capabilities.to_dict(),
            "capability_identity": self.capability_identity.to_dict(),
        }
        return _canonical_digest(value)

    def to_json(self) -> str:
        """Return deterministic JSON for the static evaluator profile."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: object) -> ValidationEvaluatorProfile:
        mapping = _require_mapping(value, "validation evaluator profile")
        required = frozenset(
            {
                "schema",
                "profile_id",
                "host_profile",
                "capacity_limit",
                "network_policy",
                "download_egress",
                "reusable_storage_credentials",
                "github_credentials",
                "registry_push_credentials",
                "artifact_issuer_credentials",
                "capabilities",
                "capability_identity",
            }
        )
        _exact_fields(
            mapping,
            required=required,
            context="validation evaluator profile",
        )
        if mapping["schema"] != PROFILE_SCHEMA:
            raise QueuePlanError("validation evaluator profile schema is not supported")
        booleans = (
            "download_egress",
            "reusable_storage_credentials",
            "github_credentials",
            "registry_push_credentials",
            "artifact_issuer_credentials",
        )
        if any(not isinstance(mapping[field], bool) for field in booleans):
            raise QueuePlanError("validation evaluator profile flags must be booleans")
        profile = cls(
            profile_id=ProfileId(_string(mapping["profile_id"], "profile_id")),
            host_profile=_host_profile_from_dict(mapping["host_profile"]),
            capacity_limit=PoolCapacity(_integer(mapping["capacity_limit"], "capacity_limit", minimum=1)),
            network_policy=_parse_network_policy(mapping["network_policy"]),
            download_egress=mapping["download_egress"],
            reusable_storage_credentials=mapping["reusable_storage_credentials"],
            github_credentials=mapping["github_credentials"],
            registry_push_credentials=mapping["registry_push_credentials"],
            artifact_issuer_credentials=mapping["artifact_issuer_credentials"],
            capabilities=WorkerCapabilities.from_dict(mapping["capabilities"]),
            capability_identity=CapabilityIdentity.from_dict(mapping["capability_identity"]),
        )
        return profile

    @classmethod
    def from_json(cls, value: str) -> ValidationEvaluatorProfile:
        """Parse one strict JSON evaluator profile."""

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise QueuePlanError("validation evaluator profile JSON is invalid") from error
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class CloudeckWorkerProfile:
    """Exact CloudDeck ``WorkerProfile`` wire document for the validation worker."""

    profile_id: ProfileId
    image_repository: IdentityText
    image_digest: Sha256Digest
    command: tuple[str, ...]
    cpu_millis: PositiveInteger
    memory_mib: PositiveInteger
    timeout_seconds: PositiveInteger
    gpu_name: IdentityText
    num_gpus: PositiveInteger
    min_reliability_percent: NonNegativeInteger
    max_hourly_price_cents: PositiveInteger
    disk_gb: PositiveInteger
    min_cuda: CudaVersion
    verified_only: bool
    network: EvaluatorNetworkPolicy
    capacity_limit: PoolCapacity
    capabilities: WorkerCapabilities
    registry_auth_secret: None = None
    allowed_secrets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, ProfileId):
            raise QueuePlanError("CloudDeck worker profile_id must be a ProfileId")
        if not isinstance(self.image_repository, IdentityText):
            raise QueuePlanError("CloudDeck image repository must be IdentityText")
        if not isinstance(self.image_digest, Sha256Digest):
            raise QueuePlanError("CloudDeck image digest must be a Sha256Digest")
        if self.command != VALIDATION_WORKER_COMMAND:
            raise QueuePlanError("CloudDeck worker command must be the claim subcommand")
        if self.gpu_name.value != "RTX_5060_Ti":
            raise QueuePlanError("CloudDeck worker profile GPU name must be RTX_5060_Ti")
        if self.num_gpus.value != 1:
            raise QueuePlanError("CloudDeck worker profile must request one GPU")
        if self.network is not EvaluatorNetworkPolicy.EGRESS:
            raise QueuePlanError("CloudDeck worker profile must use public egress")
        if self.registry_auth_secret is not None or self.allowed_secrets:
            raise QueuePlanError("CloudDeck worker profile cannot carry reusable secrets")
        if self.capacity_limit.value != 1:
            raise QueuePlanError("CloudDeck worker profile capacity_limit must be one")
        if self.verified_only is not True:
            raise QueuePlanError("CloudDeck worker profile must require verified Vast hosts")
        if not isinstance(self.capabilities, WorkerCapabilities):
            raise QueuePlanError("CloudDeck worker profile must bind a WorkerCapabilities document")
        if self.capabilities.cancellation is not CapabilityCancellation.COOPERATIVE:
            raise QueuePlanError("CloudDeck worker profile must advertise cooperative cancellation")
        if self.capabilities.resume is not CapabilityResume.UNIT_BOUNDARY:
            raise QueuePlanError("CloudDeck worker profile must advertise unit-boundary resume")

    def to_dict(self) -> dict[str, object]:
        """Return the exact CloudDeck WorkerProfile JSON object."""

        return {
            "profile_id": self.profile_id.value,
            "provider": {
                "provider": "vast_ai",
                "gpu_name": self.gpu_name.value,
                "num_gpus": self.num_gpus.value,
                "min_reliability_percent": self.min_reliability_percent.value,
                "max_hourly_price_cents": self.max_hourly_price_cents.value,
                "disk_gb": self.disk_gb.value,
                "min_cuda": str(self.min_cuda),
                "verified_only": self.verified_only,
            },
            "image": {
                "repository": self.image_repository.value,
                "digest": self.image_digest.value,
            },
            "command": list(self.command),
            "cpu_millis": self.cpu_millis.value,
            "memory_mib": self.memory_mib.value,
            "timeout_seconds": self.timeout_seconds.value,
            "network": self.network.value,
            "registry_auth_secret": self.registry_auth_secret,
            "allowed_secrets": list(self.allowed_secrets),
            "capacity_limit": self.capacity_limit.value,
            "capability_digest": self.capabilities.digest().value,
        }

    def canonical_bytes(self) -> bytes:
        """Return compact JSON bytes in CloudDeck struct field order."""

        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def digest(self) -> Sha256Digest:
        """Return SHA-256 of the exact CloudDeck profile JSON bytes."""

        return Sha256Digest(hashlib.sha256(self.canonical_bytes()).hexdigest())

    @classmethod
    def from_dict(cls, value: object) -> CloudeckWorkerProfile:
        mapping = _require_mapping(value, "CloudDeck worker profile")
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "profile_id",
                    "provider",
                    "image",
                    "command",
                    "cpu_millis",
                    "memory_mib",
                    "timeout_seconds",
                    "network",
                    "registry_auth_secret",
                    "allowed_secrets",
                    "capacity_limit",
                    "capability_digest",
                }
            ),
            context="CloudDeck worker profile",
        )
        provider = _require_mapping(mapping["provider"], "CloudDeck provider")
        _exact_fields(
            provider,
            required=frozenset(
                {
                    "provider",
                    "gpu_name",
                    "num_gpus",
                    "min_reliability_percent",
                    "max_hourly_price_cents",
                    "disk_gb",
                    "min_cuda",
                    "verified_only",
                }
            ),
            optional=frozenset({"geolocations"}),
            context="CloudDeck Vast provider",
        )
        if provider["provider"] != "vast_ai":
            raise QueuePlanError("CloudDeck worker profile must use vast_ai")
        if provider.get("geolocations"):
            raise QueuePlanError("CloudDeck worker profile must not restrict geolocations")
        image = _require_mapping(mapping["image"], "CloudDeck image")
        _exact_fields(image, required=frozenset({"repository", "digest"}), context="CloudDeck image")
        command = mapping["command"]
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise QueuePlanError("CloudDeck worker command must be a string array")
        secrets = mapping["allowed_secrets"]
        if not isinstance(secrets, list) or any(not isinstance(item, str) for item in secrets):
            raise QueuePlanError("CloudDeck allowed_secrets must be a string array")
        if mapping["registry_auth_secret"] is not None:
            raise QueuePlanError("CloudDeck worker profile cannot carry registry auth")
        profile = cls(
            profile_id=ProfileId(_string(mapping["profile_id"], "profile_id")),
            image_repository=IdentityText(_string(image["repository"], "image repository")),
            image_digest=_digest(image["digest"], "image digest"),
            command=tuple(command),
            cpu_millis=PositiveInteger(_integer(mapping["cpu_millis"], "cpu_millis", minimum=1)),
            memory_mib=PositiveInteger(_integer(mapping["memory_mib"], "memory_mib", minimum=1)),
            timeout_seconds=PositiveInteger(_integer(mapping["timeout_seconds"], "timeout_seconds", minimum=1)),
            gpu_name=IdentityText(_string(provider["gpu_name"], "gpu_name")),
            num_gpus=PositiveInteger(_integer(provider["num_gpus"], "num_gpus", minimum=1)),
            min_reliability_percent=NonNegativeInteger(
                _integer(provider["min_reliability_percent"], "min_reliability_percent")
            ),
            max_hourly_price_cents=PositiveInteger(
                _integer(provider["max_hourly_price_cents"], "max_hourly_price_cents", minimum=1)
            ),
            disk_gb=PositiveInteger(_integer(provider["disk_gb"], "disk_gb", minimum=1)),
            min_cuda=_parse_profile_cuda(provider["min_cuda"]),
            verified_only=provider["verified_only"] is True,
            network=_parse_network_policy(mapping["network"]),
            capacity_limit=PoolCapacity(_integer(mapping["capacity_limit"], "capacity_limit", minimum=1)),
            capabilities=VALIDATION_WORKER_CAPABILITIES,
        )
        if not isinstance(provider["verified_only"], bool):
            raise QueuePlanError("verified_only must be a boolean")
        if profile.capabilities.digest().value != mapping["capability_digest"]:
            raise QueuePlanError("CloudDeck worker profile capability digest does not match")
        return profile


def _parse_profile_cuda(value: object) -> CudaVersion:
    try:
        return CudaVersion.parse(value)
    except (TypeError, ValueError) as error:
        raise QueuePlanError("CloudDeck worker profile min_cuda must be a major.minor CUDA version") from error


VALIDATION_WORKER_CAPABILITIES = WorkerCapabilities(
    protocol_versions=(WorkProtocolVersion.V3,),
    workload_family=IdentityText(DEFAULT_WORKLOAD_FAMILY),
    request_schemas=(SchemaVersion(DEFAULT_REQUEST_SCHEMA),),
    result_schemas=(SchemaVersion(DEFAULT_RESULT_SCHEMA),),
    proof_schemas=(SchemaVersion(DEFAULT_PROOF_SCHEMA),),
    max_payload_bytes=PositiveInteger(DEFAULT_MAX_PAYLOAD_BYTES),
    max_result_bytes=PositiveInteger(DEFAULT_VALIDATION_RESULT_LIMIT_BYTES),
)

VALIDATION_HOST_PROFILE = EvaluatorCapabilityProfile(
    accepted_gpu_names=(GpuName("RTX 5060 Ti"),),
    minimum_gpu_memory_bytes=16 * GIB,
    minimum_cuda_version=CudaVersion(12, 8),
    minimum_system_ram_bytes=32 * GIB,
    minimum_cpu_cores=4,
    minimum_free_disk_bytes=32 * GIB,
)

VALIDATION_EVALUATOR_PROFILE = ValidationEvaluatorProfile(
    profile_id=ProfileId(DEFAULT_PROFILE_ID),
    host_profile=VALIDATION_HOST_PROFILE,
    network_policy=EvaluatorNetworkPolicy.EGRESS,
    download_egress=True,
    reusable_storage_credentials=False,
    github_credentials=False,
    registry_push_credentials=False,
    artifact_issuer_credentials=False,
    capabilities=VALIDATION_WORKER_CAPABILITIES,
    capability_identity=CapabilityIdentity(),
)

CLOUDECK_WORKER_PROFILE = CloudeckWorkerProfile(
    profile_id=ProfileId(DEFAULT_PROFILE_ID),
    image_repository=IdentityText(PUBLISHED_EVALUATOR_IMAGE_REPOSITORY),
    image_digest=PUBLISHED_EVALUATOR_IMAGE_DIGEST,
    command=VALIDATION_WORKER_COMMAND,
    cpu_millis=PositiveInteger(4_000),
    memory_mib=PositiveInteger(32_768),
    timeout_seconds=PositiveInteger(4 * 60 * 60),
    gpu_name=IdentityText("RTX_5060_Ti"),
    num_gpus=PositiveInteger(1),
    min_reliability_percent=NonNegativeInteger(90),
    max_hourly_price_cents=PositiveInteger(400),
    disk_gb=PositiveInteger(64),
    min_cuda=CudaVersion(12, 8),
    verified_only=True,
    network=EvaluatorNetworkPolicy.EGRESS,
    capacity_limit=PoolCapacity(1),
    capabilities=VALIDATION_WORKER_CAPABILITIES,
)


@dataclass(frozen=True)
class QueuePlan:
    """One complete immutable queue, ordered manifest, and managed pool."""

    campaign: ValidationCampaign
    request: QueueRequest
    units: tuple[ValidationWorkUnit, ...]
    pool: ManagedPool
    evaluator_profile: ValidationEvaluatorProfile
    worker_profile: CloudeckWorkerProfile
    slot_scope: tuple[Sha256Digest, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.campaign, ValidationCampaign):
            raise QueuePlanError("campaign must be a ValidationCampaign")
        if not isinstance(self.request, QueueRequest):
            raise QueuePlanError("request must be a QueueRequest")
        if not isinstance(self.units, tuple) or not self.units:
            raise QueuePlanError("queue plan must contain units")
        if not all(isinstance(unit, ValidationWorkUnit) for unit in self.units):
            raise QueuePlanError("queue plan units must be typed ValidationWorkUnit values")
        if not isinstance(self.pool, ManagedPool):
            raise QueuePlanError("pool must be a ManagedPool")
        if not isinstance(self.evaluator_profile, ValidationEvaluatorProfile):
            raise QueuePlanError("evaluator_profile must be a ValidationEvaluatorProfile")
        if not isinstance(self.worker_profile, CloudeckWorkerProfile):
            raise QueuePlanError("worker_profile must be a CloudeckWorkerProfile")
        if not isinstance(self.slot_scope, tuple):
            raise QueuePlanError("queue slot scope must be a tuple")
        scope = self.slot_scope
        if not all(isinstance(slot_id, Sha256Digest) for slot_id in scope):
            raise QueuePlanError("queue slot scope must contain Sha256Digest values")
        units = self.units
        if len(set(scope)) != len(scope):
            raise QueuePlanError("queue slot scope cannot contain duplicates")
        campaign_slots = self.campaign.slots
        if scope:
            campaign_slot_ids = {slot.slot_id for slot in campaign_slots}
            if not set(scope).issubset(campaign_slot_ids):
                raise QueuePlanError("queue slot scope contains a slot outside the campaign")
            expected_slots = tuple(slot for slot in campaign_slots if slot.slot_id in set(scope))
            if tuple(slot.slot_id for slot in expected_slots) != scope:
                raise QueuePlanError("queue slot scope must follow campaign ordinal order")
        else:
            expected_slots = campaign_slots
        if tuple(unit.ordinal for unit in units) != tuple(range(len(units))):
            raise QueuePlanError("queue unit ordinals must be contiguous from zero")
        if len(units) != len(expected_slots):
            raise QueuePlanError("queue plan units do not match its slot scope")
        if tuple(unit.payload.identities.slot_id for unit in units) != tuple(slot.slot_id for slot in expected_slots):
            raise QueuePlanError("queue units must follow the campaign slot order")
        if self.request.queue_id != self.pool.queue_id:
            raise QueuePlanError("managed pool must remain bound to the immutable queue")
        if self.request.max_active_leases.value != 1 or self.pool.capacity.value != 1:
            raise QueuePlanError("validation queue and pool must each have capacity one")
        if self.pool.reuse is not ManagedReusePolicy.QUEUE_LIFETIME:
            raise QueuePlanError("validation managed pool must use queue_lifetime reuse")
        if self.request.protocol.version is not WorkProtocolVersion.V3:
            raise QueuePlanError("validation queue must use protocol v3")
        if self.pool.profile_id != self.worker_profile.profile_id:
            raise QueuePlanError("managed pool profile does not match the CloudDeck worker profile")
        if self.pool.profile_id != self.evaluator_profile.profile_id:
            raise QueuePlanError("managed pool profile does not match evaluator profile")
        if self.pool.profile_digest != self.worker_profile.digest():
            raise QueuePlanError("managed pool profile digest does not match the CloudDeck worker profile")
        if self.pool.capacity.value > self.evaluator_profile.capacity_limit.value:
            raise QueuePlanError("managed pool capacity exceeds evaluator profile capacity")
        if self.pool.capacity.value > self.worker_profile.capacity_limit.value:
            raise QueuePlanError("managed pool capacity exceeds CloudDeck worker profile capacity")
        if self.worker_profile.capabilities != self.evaluator_profile.capabilities:
            raise QueuePlanError("CloudDeck worker profile capabilities must match the evaluator profile")
        if self.request.required_capability.identity != self.evaluator_profile.capability_identity:
            raise QueuePlanError("queue capability identity does not match evaluator profile")
        if self.request.required_capability.capability_digest != self.worker_profile.capabilities.digest():
            raise QueuePlanError("queue capability digest does not match the CloudDeck worker capability document")
        for slot, unit in zip(expected_slots, units):
            if (
                unit.payload.identities.campaign_id != self.campaign.campaign_id
                or unit.payload.identities.slot_id != slot.slot_id
            ):
                raise QueuePlanError("work-unit identity is not bound to its campaign slot")
            if unit.payload.slot_ordinal != slot.ordinal:
                raise QueuePlanError("work-unit payload ordinal does not match its campaign slot")
            if unit.payload.manifest_location.value != slot.manifest_location.value:
                raise QueuePlanError("work-unit manifest location does not match its slot")
            if unit.payload.model_location.value != slot.model_location.value:
                raise QueuePlanError("work-unit model location does not match its slot")
            if any(
                spec.not_after.value > self.request.deadline_unix_seconds.value for spec in unit.artifact_access_specs
            ):
                raise QueuePlanError("artifact access expiry exceeds the queue deadline")
        manifest_digest = manifest_digest_for_units(units)
        if self.request.manifest_digest != manifest_digest:
            raise QueuePlanError("queue manifest digest does not match its complete unit list")

    def to_dict(self) -> dict[str, object]:
        value = {
            "request": self.request.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
            "pool": self.pool.to_dict(),
        }
        _reject_unsafe(value)
        return value

    def to_json(self) -> str:
        """Return deterministic strict JSON after boundary safety checks."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        campaign: ValidationCampaign,
        evaluator_profile: ValidationEvaluatorProfile = VALIDATION_EVALUATOR_PROFILE,
        worker_profile: CloudeckWorkerProfile = CLOUDECK_WORKER_PROFILE,
        slot_scope: tuple[Sha256Digest, ...] = (),
    ) -> QueuePlan:
        """Parse one strict JSON queue plan."""

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise QueuePlanError("validation queue plan JSON is invalid") from error
        return cls.from_dict(
            decoded,
            campaign=campaign,
            evaluator_profile=evaluator_profile,
            worker_profile=worker_profile,
            slot_scope=slot_scope,
        )

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        campaign: ValidationCampaign,
        evaluator_profile: ValidationEvaluatorProfile = VALIDATION_EVALUATOR_PROFILE,
        worker_profile: CloudeckWorkerProfile = CLOUDECK_WORKER_PROFILE,
        slot_scope: tuple[Sha256Digest, ...] = (),
    ) -> QueuePlan:
        mapping = _require_mapping(value, "validation queue plan")
        _reject_unsafe(mapping)
        _exact_fields(
            mapping,
            required=frozenset({"request", "units", "pool"}),
            context="validation queue plan",
        )
        units_value = mapping["units"]
        if not isinstance(units_value, list):
            raise QueuePlanError("validation queue plan units must be an array")
        return cls(
            campaign=campaign,
            request=QueueRequest.from_dict(mapping["request"]),
            units=tuple(ValidationWorkUnit.from_dict(unit) for unit in units_value),
            pool=ManagedPool.from_dict(mapping["pool"]),
            evaluator_profile=evaluator_profile,
            worker_profile=worker_profile,
            slot_scope=slot_scope,
        )


@dataclass(frozen=True)
class BatchCostRecord:
    """Typed cost accounting attached to a hard-lifetime extra batch."""

    added_worker_count: WorkerCount
    cold_dev_cache_admissions: ColdDevCacheAdmissions
    estimated_worker_seconds: NonNegativeInteger = NonNegativeInteger(0)
    estimated_cost_usd_micros: NonNegativeInteger = NonNegativeInteger(0)

    def __post_init__(self) -> None:
        if not isinstance(self.added_worker_count, WorkerCount):
            raise QueuePlanError("added_worker_count must be a WorkerCount")
        if not isinstance(self.cold_dev_cache_admissions, ColdDevCacheAdmissions):
            raise QueuePlanError("cold_dev_cache_admissions must be ColdDevCacheAdmissions")
        if not isinstance(self.estimated_worker_seconds, NonNegativeInteger):
            raise QueuePlanError("estimated_worker_seconds must be a NonNegativeInteger")
        if not isinstance(self.estimated_cost_usd_micros, NonNegativeInteger):
            raise QueuePlanError("estimated_cost_usd_micros must be a NonNegativeInteger")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": COST_RECORD_SCHEMA,
            "added_worker_count": self.added_worker_count.value,
            "cold_dev_cache_admissions": self.cold_dev_cache_admissions.value,
            "estimated_worker_seconds": self.estimated_worker_seconds.value,
            "estimated_cost_usd_micros": self.estimated_cost_usd_micros.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> BatchCostRecord:
        mapping = _require_mapping(value, "batch cost record")
        _exact_fields(
            mapping,
            required=frozenset(
                {
                    "schema",
                    "added_worker_count",
                    "cold_dev_cache_admissions",
                    "estimated_worker_seconds",
                    "estimated_cost_usd_micros",
                }
            ),
            context="batch cost record",
        )
        if mapping["schema"] != COST_RECORD_SCHEMA:
            raise QueuePlanError("batch cost record schema is not supported")
        return cls(
            added_worker_count=WorkerCount(_integer(mapping["added_worker_count"], "added_worker_count", minimum=1)),
            cold_dev_cache_admissions=ColdDevCacheAdmissions(
                _integer(mapping["cold_dev_cache_admissions"], "cold_dev_cache_admissions", minimum=1)
            ),
            estimated_worker_seconds=NonNegativeInteger(
                _integer(mapping["estimated_worker_seconds"], "estimated_worker_seconds", minimum=0)
            ),
            estimated_cost_usd_micros=NonNegativeInteger(
                _integer(mapping["estimated_cost_usd_micros"], "estimated_cost_usd_micros", minimum=0)
            ),
        )


class ExtraBatchReason(str, Enum):
    """Recoverable terminal condition that permits one unresolved-slot batch."""

    HARD_LIFETIME = "hard_lifetime_recoverable_terminal"


@dataclass(frozen=True)
class ExtraBatchPlan:
    """A new queue for unresolved slots after a recoverable hard lifetime."""

    source_queue_id: QueueId
    queue_plan: QueuePlan
    unresolved_slots: tuple[Sha256Digest, ...]
    reason: ExtraBatchReason
    cost_record: BatchCostRecord

    def __post_init__(self) -> None:
        if not isinstance(self.source_queue_id, QueueId):
            raise QueuePlanError("source_queue_id must be a QueueId")
        if not isinstance(self.queue_plan, QueuePlan):
            raise QueuePlanError("queue_plan must be a QueuePlan")
        if not isinstance(self.unresolved_slots, tuple) or not self.unresolved_slots:
            raise QueuePlanError("extra batch must contain unresolved slots")
        if not all(isinstance(slot_id, Sha256Digest) for slot_id in self.unresolved_slots):
            raise QueuePlanError("extra batch unresolved slots must be Sha256Digest values")
        if len(set(self.unresolved_slots)) != len(self.unresolved_slots):
            raise QueuePlanError("extra batch cannot duplicate unresolved slots")
        if self.reason is not ExtraBatchReason.HARD_LIFETIME:
            raise QueuePlanError("extra batch reason must be hard_lifetime_recoverable_terminal")
        if not isinstance(self.cost_record, BatchCostRecord):
            raise QueuePlanError("cost_record must be a BatchCostRecord")
        source_ids = {unit.payload.identities.slot_id for unit in self.queue_plan.units}
        if not set(self.unresolved_slots).issubset(source_ids):
            raise QueuePlanError("extra batch contains a slot outside the source campaign")
        if tuple(unit.payload.identities.slot_id for unit in self.queue_plan.units) != self.unresolved_slots:
            raise QueuePlanError("extra batch queue units must contain unresolved slots in order")
        if self.queue_plan.request.queue_id == self.source_queue_id:
            raise QueuePlanError("extra batch must use a new immutable queue identity")
        if self.queue_plan.pool.queue_id != self.queue_plan.request.queue_id:
            raise QueuePlanError("extra-batch managed pool cannot move queues")
        if self.cost_record.added_worker_count.value != 1:
            raise QueuePlanError("hard-lifetime extra batch must add exactly one worker")
        if self.cost_record.cold_dev_cache_admissions.value != 1:
            raise QueuePlanError("hard-lifetime extra batch must admit exactly one cold dev cache")

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema": EXTRA_BATCH_SCHEMA,
            "source_queue_id": self.source_queue_id.value,
            "queue_plan": self.queue_plan.to_dict(),
            "unresolved_slots": [slot_id.value for slot_id in self.unresolved_slots],
            "reason": self.reason.value,
            "cost_record": self.cost_record.to_dict(),
        }
        _reject_unsafe(value)
        return value

    def to_json(self) -> str:
        """Return deterministic strict JSON for the extra batch."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        campaign: ValidationCampaign,
        evaluator_profile: ValidationEvaluatorProfile = VALIDATION_EVALUATOR_PROFILE,
        worker_profile: CloudeckWorkerProfile = CLOUDECK_WORKER_PROFILE,
    ) -> ExtraBatchPlan:
        mapping = _require_mapping(value, "extra batch plan")
        _reject_unsafe(mapping)
        _exact_fields(
            mapping,
            required=frozenset(
                {"schema", "source_queue_id", "queue_plan", "unresolved_slots", "reason", "cost_record"}
            ),
            context="extra batch plan",
        )
        if mapping["schema"] != EXTRA_BATCH_SCHEMA or mapping["reason"] != ExtraBatchReason.HARD_LIFETIME.value:
            raise QueuePlanError("extra batch reason or schema is not supported")
        slots_value = mapping["unresolved_slots"]
        if not isinstance(slots_value, list):
            raise QueuePlanError("unresolved_slots must be an array")
        unresolved_slots = tuple(_digest(slot_id, "unresolved slot") for slot_id in slots_value)
        return cls(
            source_queue_id=QueueId(_string(mapping["source_queue_id"], "source_queue_id")),
            queue_plan=QueuePlan.from_dict(
                mapping["queue_plan"],
                campaign=campaign,
                evaluator_profile=evaluator_profile,
                worker_profile=worker_profile,
                slot_scope=unresolved_slots,
            ),
            unresolved_slots=unresolved_slots,
            reason=ExtraBatchReason.HARD_LIFETIME,
            cost_record=BatchCostRecord.from_dict(mapping["cost_record"]),
        )

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        campaign: ValidationCampaign,
        evaluator_profile: ValidationEvaluatorProfile = VALIDATION_EVALUATOR_PROFILE,
        worker_profile: CloudeckWorkerProfile = CLOUDECK_WORKER_PROFILE,
    ) -> ExtraBatchPlan:
        """Parse one strict JSON extra-batch plan."""

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise QueuePlanError("extra batch plan JSON is invalid") from error
        return cls.from_dict(
            decoded,
            campaign=campaign,
            evaluator_profile=evaluator_profile,
            worker_profile=worker_profile,
        )


def manifest_digest_for_units(units: Sequence[ValidationWorkUnit]) -> Sha256Digest:
    """Compute the ordered CloudDeck manifest digest without mutable payload maps."""

    entries = tuple(
        {
            "unit_id": unit.unit_id.value,
            "ordinal": unit.ordinal,
            "payload_digest": unit.payload_digest.value,
            "progress_total": None,
            "artifact_access_specs": tuple(spec.to_dict() for spec in unit.artifact_access_specs),
        }
        for unit in units
    )
    return _ordered_digest(entries)


def _new_queue_id() -> QueueId:
    return QueueId(_new_uuid_v7())


def _new_pool_id() -> PoolId:
    return PoolId(_new_uuid_v7())


def _unit_for_slot(
    campaign: ValidationCampaign,
    slot: ValidationSlot,
    *,
    ordinal: int,
    frozen_dev_bundle: ArtifactRef,
    trainer_configuration: ArtifactRef,
    deadline: UnixDeadline,
) -> ValidationWorkUnit:
    identities = ExpectedIdentities(
        campaign_id=campaign.campaign_id,
        slot_id=slot.slot_id,
        training_launch_id=IdentityText(campaign.training_launch_id),
        updates=slot.updates,
        trainer_configuration_digest=campaign.trainer_configuration_digest,
        evaluator_image_identity=IdentityText(campaign.evaluator_image_identity),
        evaluator_implementation_digest=campaign.evaluator_implementation_digest,
        dev_bundle_digest=campaign.dev_bundle_digest,
    )
    payload = ValidationWorkPayload(
        identities=identities,
        slot_ordinal=slot.ordinal,
        point=slot.point,
        manifest_location=ImmutableLocation(slot.manifest_location.value),
        model_location=ImmutableLocation(slot.model_location.value),
        future_model=UncommittedModelArtifact(ImmutableLocation(slot.model_location.value)),
        frozen_dev_bundle=frozen_dev_bundle,
        trainer_configuration=trainer_configuration,
    )
    specs = (
        ReadAccessSpec(payload.manifest_location, deadline),
        ReadAccessSpec(payload.model_location, deadline),
        ReadAccessSpec(payload.frozen_dev_bundle.location, deadline),
        ReadAccessSpec(payload.trainer_configuration.location, deadline),
    )
    return ValidationWorkUnit(
        unit_id=WorkUnitId(f"validation-slot-{slot.slot_id.value}"),
        ordinal=ordinal,
        payload=payload,
        artifact_access_specs=specs,
        payload_digest=payload.digest(),
    )


def build_validation_queue_plan(
    campaign: ValidationCampaign,
    *,
    frozen_dev_bundle: ArtifactRef,
    trainer_configuration: ArtifactRef,
    artifact_issuer_id: ArtifactIssuerId | str = ArtifactIssuerId(DEFAULT_ARTIFACT_ISSUER_ID),
    campaign_deadline_unix_seconds: int = DEFAULT_QUEUE_DEADLINE_UNIX_SECONDS,
    queue_id: QueueId,
    pool_id: PoolId,
    evaluator_profile: ValidationEvaluatorProfile = VALIDATION_EVALUATOR_PROFILE,
    worker_profile: CloudeckWorkerProfile = CLOUDECK_WORKER_PROFILE,
) -> QueuePlan:
    """Build one complete ordered queue; access-URL lifetime never partitions it."""

    if not isinstance(campaign, ValidationCampaign):
        raise QueuePlanError("campaign must be a ValidationCampaign")
    if not isinstance(frozen_dev_bundle, ArtifactRef):
        raise QueuePlanError("frozen_dev_bundle must be a complete ArtifactRef")
    if frozen_dev_bundle.content_digest != campaign.dev_bundle_digest:
        raise QueuePlanError("frozen development bundle digest does not match the campaign")
    if not isinstance(trainer_configuration, ArtifactRef):
        raise QueuePlanError("trainer_configuration must be a complete ArtifactRef")
    if trainer_configuration.content_digest != campaign.trainer_configuration_digest:
        raise QueuePlanError("trainer configuration digest does not match the campaign")
    if isinstance(artifact_issuer_id, (list, tuple, set, frozenset)):
        raise QueuePlanError("a queue must select exactly one artifact issuer")
    issuer = (
        artifact_issuer_id
        if isinstance(artifact_issuer_id, ArtifactIssuerId)
        else ArtifactIssuerId(artifact_issuer_id)
    )
    deadline = UnixDeadline(_integer(campaign_deadline_unix_seconds, "campaign deadline", minimum=1))
    profile = evaluator_profile
    if not isinstance(profile, ValidationEvaluatorProfile):
        raise QueuePlanError("evaluator_profile must be a ValidationEvaluatorProfile")
    if not isinstance(worker_profile, CloudeckWorkerProfile):
        raise QueuePlanError("worker_profile must be a CloudeckWorkerProfile")
    if not isinstance(queue_id, QueueId):
        raise QueuePlanError("queue_id must be a caller-supplied QueueId UUIDv7")
    if not isinstance(pool_id, PoolId):
        raise QueuePlanError("pool_id must be a caller-supplied PoolId UUIDv7")
    units = tuple(
        _unit_for_slot(
            campaign,
            slot,
            ordinal=index,
            frozen_dev_bundle=frozen_dev_bundle,
            trainer_configuration=trainer_configuration,
            deadline=deadline,
        )
        for index, slot in enumerate(campaign.slots)
    )
    workload_payload = CampaignWorkloadPayload(
        campaign_id=campaign.campaign_id,
        training_launch_id=IdentityText(campaign.training_launch_id),
        trainer_configuration_digest=campaign.trainer_configuration_digest,
        evaluator_image_identity=IdentityText(campaign.evaluator_image_identity),
        evaluator_implementation_digest=campaign.evaluator_implementation_digest,
        dev_bundle_digest=campaign.dev_bundle_digest,
    )
    request = QueueRequest(
        queue_id=queue_id,
        workload_revision=_canonical_digest(
            {
                "campaign_id": campaign.campaign_id.value,
                "evaluator_implementation_digest": campaign.evaluator_implementation_digest.value,
            }
        ),
        manifest_digest=manifest_digest_for_units(units),
        payload_schema=IdentityText(WORK_PAYLOAD_SCHEMA),
        workload_payload=workload_payload,
        workload_payload_digest=workload_payload.digest(),
        max_payload_bytes=PositiveInteger(DEFAULT_MAX_PAYLOAD_BYTES),
        lease_duration_seconds=PositiveInteger(DEFAULT_LEASE_DURATION_SECONDS),
        retry_policy=RetryPolicy(),
        result_policy=ResultPolicy(ValidationResultLimit(DEFAULT_VALIDATION_RESULT_LIMIT_BYTES)),
        deadline_unix_seconds=deadline,
        max_active_leases=MaxActiveLeases(1),
        protocol=QueueProtocol(WorkProtocolVersion.V3, issuer),
        required_capability=RequiredCapability(
            capability_digest=profile.capabilities.digest(),
            identity=profile.capability_identity,
        ),
    )
    pool = ManagedPool(
        pool_id=pool_id,
        queue_id=queue_id,
        profile_id=worker_profile.profile_id,
        profile_digest=worker_profile.digest(),
        capacity=PoolCapacity(1),
        reuse=ManagedReusePolicy.QUEUE_LIFETIME,
        service=None,
    )
    return QueuePlan(
        campaign=campaign,
        request=request,
        units=units,
        pool=pool,
        evaluator_profile=profile,
        worker_profile=worker_profile,
    )


def build_hard_lifetime_extra_batch_plan(
    source_plan: QueuePlan,
    *,
    unresolved_slots: Iterable[ValidationSlot | Sha256Digest],
    accepted_slots: Iterable[ValidationSlot | Sha256Digest] = (),
    cost_record: BatchCostRecord | None = None,
    queue_id: QueueId | None = None,
    pool_id: PoolId | None = None,
) -> ExtraBatchPlan:
    """Build a new queue containing unresolved slots only after hard lifetime."""

    if not isinstance(source_plan, QueuePlan):
        raise QueuePlanError("source_plan must be a QueuePlan")
    if queue_id is not None and not isinstance(queue_id, QueueId):
        raise QueuePlanError("extra queue_id must be a QueueId UUIDv7")
    if pool_id is not None and not isinstance(pool_id, PoolId):
        raise QueuePlanError("extra pool_id must be a PoolId UUIDv7")

    def slot_id(value: ValidationSlot | Sha256Digest) -> Sha256Digest:
        if isinstance(value, ValidationSlot):
            return value.slot_id
        if isinstance(value, Sha256Digest):
            return value
        raise QueuePlanError("accepted and unresolved slots must use typed slots or digests")

    accepted = tuple(slot_id(value) for value in accepted_slots)
    if len(set(accepted)) != len(accepted):
        raise QueuePlanError("accepted slots cannot be duplicated")
    unresolved = tuple(slot_id(value) for value in unresolved_slots)
    if len(set(unresolved)) != len(unresolved):
        raise QueuePlanError("unresolved slots cannot be duplicated")
    overlap = set(accepted) & set(unresolved)
    if overlap:
        raise QueuePlanError("accepted slots cannot be duplicated in an extra batch")
    source_slot_ids = {unit.payload.identities.slot_id for unit in source_plan.units}
    if not set(accepted).issubset(source_slot_ids):
        raise QueuePlanError("accepted slots must belong to the source campaign")
    if not set(unresolved).issubset(source_slot_ids):
        raise QueuePlanError("unresolved slots must belong to the source campaign")
    unresolved_ordered = tuple(slot for slot in source_plan.campaign.slots if slot.slot_id in set(unresolved))
    if len(unresolved_ordered) != len(unresolved):
        raise QueuePlanError("unresolved slot set is not complete")
    extra_queue_id = queue_id or _new_queue_id()
    extra_pool_id = pool_id or _new_pool_id()
    if extra_queue_id == source_plan.request.queue_id:
        raise QueuePlanError("extra batch must use a distinct queue identity")
    if extra_pool_id == source_plan.pool.pool_id:
        raise QueuePlanError("extra batch must use a distinct managed-pool identity")
    # reusing the source builder preserves the source queue policy while creating
    # a distinct immutable queue and managed pool for the unresolved subset
    extra_plan = _build_subset_queue_plan(
        source_plan,
        unresolved_ordered,
        queue_id=extra_queue_id,
        pool_id=extra_pool_id,
    )
    return ExtraBatchPlan(
        source_queue_id=source_plan.request.queue_id,
        queue_plan=extra_plan,
        unresolved_slots=tuple(slot.slot_id for slot in unresolved_ordered),
        reason=ExtraBatchReason.HARD_LIFETIME,
        cost_record=cost_record or BatchCostRecord(WorkerCount(1), ColdDevCacheAdmissions(1)),
    )


def _build_subset_queue_plan(
    source_plan: QueuePlan,
    slots: Sequence[ValidationSlot],
    *,
    queue_id: QueueId,
    pool_id: PoolId,
) -> QueuePlan:
    deadline = source_plan.request.deadline_unix_seconds
    units = tuple(
        _unit_for_slot(
            source_plan.campaign,
            slot,
            ordinal=index,
            frozen_dev_bundle=source_plan.units[slot.ordinal].payload.frozen_dev_bundle,
            trainer_configuration=source_plan.units[slot.ordinal].payload.trainer_configuration,
            deadline=deadline,
        )
        for index, slot in enumerate(slots)
    )
    request = QueueRequest(
        queue_id=queue_id,
        workload_revision=source_plan.request.workload_revision,
        manifest_digest=manifest_digest_for_units(units),
        payload_schema=source_plan.request.payload_schema,
        workload_payload=source_plan.request.workload_payload,
        workload_payload_digest=source_plan.request.workload_payload_digest,
        max_payload_bytes=source_plan.request.max_payload_bytes,
        lease_duration_seconds=source_plan.request.lease_duration_seconds,
        retry_policy=source_plan.request.retry_policy,
        result_policy=source_plan.request.result_policy,
        deadline_unix_seconds=deadline,
        max_active_leases=MaxActiveLeases(1),
        protocol=source_plan.request.protocol,
        required_capability=source_plan.request.required_capability,
    )
    pool = ManagedPool(
        pool_id=pool_id,
        queue_id=queue_id,
        profile_id=source_plan.pool.profile_id,
        profile_digest=source_plan.pool.profile_digest,
        capacity=PoolCapacity(1),
        reuse=ManagedReusePolicy.QUEUE_LIFETIME,
        service=None,
    )
    return QueuePlan(
        campaign=source_plan.campaign,
        request=request,
        units=units,
        pool=pool,
        evaluator_profile=source_plan.evaluator_profile,
        worker_profile=source_plan.worker_profile,
        slot_scope=tuple(slot.slot_id for slot in slots),
    )


def emit_cloudeck_wire_documents() -> dict[str, object]:
    """Return Python-produced CloudDeck v3 documents for cross-language parsing."""

    campaign = build_validation_campaign(
        training_launch_id="launch-four-source-v1",
        max_updates=60_000,
        updates_per_complete_epoch=2_967,
        artifact_prefix="s3://validation/campaigns",
        dev_bundle_digest=Sha256Digest("a" * 64),
        trainer_configuration_digest=Sha256Digest("b" * 64),
        evaluator_image_identity=(
            f"{PUBLISHED_EVALUATOR_IMAGE_REPOSITORY}@sha256:{PUBLISHED_EVALUATOR_IMAGE_DIGEST.value}"
        ),
        evaluator_implementation_digest=Sha256Digest("d" * 64),
    )
    frozen = ArtifactRef(
        content_digest=Sha256Digest("a" * 64),
        byte_length=NonNegativeInteger(123),
        media_type=MediaType("application/zstd"),
        location=ImmutableLocation("s3://validation/frozen/dev-bundle.tar.zst"),
        compression=ArtifactCompression.ZSTD,
    )
    trainer = ArtifactRef(
        content_digest=Sha256Digest("b" * 64),
        byte_length=NonNegativeInteger(64),
        media_type=MediaType("application/toml"),
        location=ImmutableLocation("s3://validation/frozen/trainer-config.toml"),
    )
    plan = build_validation_queue_plan(
        campaign,
        frozen_dev_bundle=frozen,
        trainer_configuration=trainer,
        queue_id=QueueId("018f0d2f-1234-7abc-8def-0123456789ab"),
        pool_id=PoolId("018f0d2f-1234-7abc-8def-0123456789ac"),
    )
    return {
        "capabilities": VALIDATION_WORKER_CAPABILITIES.to_dict(),
        "profile": plan.worker_profile.to_dict(),
        "queue_request": plan.request.to_dict(),
        "units": [unit.to_dict() for unit in plan.units],
        "pool": plan.pool.to_dict(),
    }
