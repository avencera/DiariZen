"""Tests for the typed DiariZen-to-CloudDeck validation queue plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from recipes.speakrs.large.validation.contracts import Sha256Digest
from recipes.speakrs.large.validation.queue_plan import (
    CLOUDECK_WORKER_PROFILE,
    PUBLISHED_EVALUATOR_IMAGE_DIGEST,
    PUBLISHED_EVALUATOR_IMAGE_REPOSITORY,
    VALIDATION_EVALUATOR_PROFILE,
    VALIDATION_WORKER_CAPABILITIES,
    VALIDATION_WORKER_COMMAND,
    ArtifactCompression,
    ArtifactIssuerId,
    ArtifactRef,
    BatchCostRecord,
    CapabilityCancellation,
    CapabilityResume,
    CloudeckWorkerProfile,
    ColdDevCacheAdmissions,
    CudaVersion,
    ExtraBatchPlan,
    ImmutableLocation,
    ManagedReusePolicy,
    MediaType,
    NonNegativeInteger,
    PoolId,
    QueueId,
    QueuePlan,
    QueuePlanError,
    ValidationEvaluatorProfile,
    WorkerCount,
    WorkProtocolVersion,
    build_hard_lifetime_extra_batch_plan,
    build_validation_queue_plan,
    emit_cloudeck_wire_documents,
)
from recipes.speakrs.large.validation.slot_plan import build_validation_campaign


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign(max_updates: int = 60_000, updates_per_complete_epoch: int = 2_967):
    return build_validation_campaign(
        training_launch_id="launch-four-source-v1",
        max_updates=max_updates,
        updates_per_complete_epoch=updates_per_complete_epoch,
        artifact_prefix="s3://validation/campaigns",
        dev_bundle_digest=_digest("a"),
        trainer_configuration_digest=_digest("b"),
        evaluator_image_identity="registry.example/evaluator@sha256:" + "c" * 64,
        evaluator_implementation_digest=_digest("d"),
    )


def _dev_bundle() -> ArtifactRef:
    return ArtifactRef(
        content_digest=_digest("a"),
        byte_length=NonNegativeInteger(123),
        media_type=MediaType("application/zstd"),
        location=ImmutableLocation("s3://validation/frozen/dev-bundle.tar.zst"),
        compression=ArtifactCompression.ZSTD,
    )


def _trainer_configuration() -> ArtifactRef:
    return ArtifactRef(
        content_digest=_digest("b"),
        byte_length=NonNegativeInteger(64),
        media_type=MediaType("application/toml"),
        location=ImmutableLocation("s3://validation/frozen/trainer-config.toml"),
    )


def _plan(
    queue_id: str = "018f0d2f-1234-7abc-8def-0123456789ab", pool_id: str = "018f0d2f-1234-7abc-8def-0123456789ac"
):
    return build_validation_queue_plan(
        _campaign(),
        frozen_dev_bundle=_dev_bundle(),
        trainer_configuration=_trainer_configuration(),
        artifact_issuer_id=ArtifactIssuerId("configured-validation-issuer"),
        campaign_deadline_unix_seconds=1_900_000_000,
        queue_id=QueueId(queue_id),
        pool_id=PoolId(pool_id),
    )


def test_complete_60000_update_campaign_creates_exactly_22_units() -> None:
    plan = _plan()

    assert len(plan.units) == 22
    assert [unit.ordinal for unit in plan.units] == list(range(22))
    assert [unit.payload.slot_ordinal for unit in plan.units] == list(range(22))
    assert plan.units[-1].payload.identities.updates == 60_000
    assert plan.units[-1].payload.point.to_dict() == {
        "kind": "final_partial",
        "completed_epochs": 20,
        "partial_updates": 660,
    }


def test_aligned_target_omits_final_partial_unit() -> None:
    campaign = _campaign(max_updates=2_967 * 20)
    plan = build_validation_queue_plan(
        campaign,
        frozen_dev_bundle=_dev_bundle(),
        trainer_configuration=_trainer_configuration(),
        queue_id=QueueId("018f0d2f-1234-7abc-8def-0123456789ad"),
        pool_id=PoolId("018f0d2f-1234-7abc-8def-0123456789ae"),
    )

    assert len(plan.units) == 21
    assert plan.units[-1].payload.point.to_dict() == {"kind": "complete_epoch", "epoch": 20}


def test_single_queue_and_managed_pool_invariants() -> None:
    plan = _plan()

    assert plan.request.queue_id == plan.pool.queue_id
    assert plan.request.max_active_leases.value == 1
    assert plan.pool.capacity.value == 1
    assert plan.pool.reuse is ManagedReusePolicy.QUEUE_LIFETIME
    assert plan.request.protocol.version is WorkProtocolVersion.V3
    assert plan.request.protocol.artifact_issuer_id.value == "configured-validation-issuer"
    assert plan.request.protocol.to_dict() == {
        "version": 3,
        "artifact_access": {"mode": "issuer", "issuer_id": "configured-validation-issuer"},
    }
    assert plan.request.required_capability.capability_digest == VALIDATION_WORKER_CAPABILITIES.digest()
    assert all(len(unit.artifact_access_specs) == 4 for unit in plan.units)
    assert all(
        unit.artifact_access_specs[3].location == unit.payload.trainer_configuration.location for unit in plan.units
    )


def _walk_strings(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_strings(nested)
    elif isinstance(value, str):
        yield value


def test_serialized_plan_contains_no_urls_or_credentials() -> None:
    encoded = _plan().to_dict()
    strings = tuple(_walk_strings(encoded))

    assert not any("https://" in value or "http://" in value for value in strings)
    assert not any(
        any(term in value.lower() for term in ("token=", "bearer ", "password=", "secret=")) for value in strings
    )
    assert not any("credential" in value.lower() or "access_key" in value.lower() for value in strings)
    assert all(spec.operation.value == "read" for unit in _plan().units for spec in unit.artifact_access_specs)


def test_url_lifetime_never_splits_the_queue() -> None:
    plan = _plan()

    assert len(plan.units) == 22
    assert {spec.not_after.value for unit in plan.units for spec in unit.artifact_access_specs} == {1_900_000_000}


def test_hard_lifetime_extra_batch_contains_only_unresolved_slots_and_cost_record() -> None:
    source = _plan()
    unresolved = (source.campaign.slots[3], source.campaign.slots[9])
    extra = build_hard_lifetime_extra_batch_plan(
        source,
        unresolved_slots=unresolved,
        accepted_slots=(source.campaign.slots[0], source.campaign.slots[1]),
        cost_record=BatchCostRecord(
            WorkerCount(1), ColdDevCacheAdmissions(1), NonNegativeInteger(17), NonNegativeInteger(23)
        ),
    )

    assert tuple(unit.payload.identities.slot_id for unit in extra.queue_plan.units) == tuple(
        slot.slot_id for slot in unresolved
    )
    assert extra.queue_plan.pool.queue_id == extra.queue_plan.request.queue_id
    assert extra.queue_plan.request.queue_id != source.request.queue_id
    assert extra.cost_record.added_worker_count.value == 1
    assert extra.cost_record.cold_dev_cache_admissions.value == 1
    assert extra.cost_record.estimated_worker_seconds.value == 17
    assert extra.cost_record.estimated_cost_usd_micros.value == 23
    assert (
        ExtraBatchPlan.from_dict(extra.to_dict(), campaign=source.campaign, evaluator_profile=source.evaluator_profile)
        == extra
    )

    with pytest.raises(QueuePlanError, match="duplicated"):
        build_hard_lifetime_extra_batch_plan(source, unresolved_slots=(unresolved[0],), accepted_slots=unresolved)


def test_future_model_has_explicit_uncommitted_state_and_dev_ref_is_complete() -> None:
    unit = _plan().units[0]
    model = unit.payload.to_dict()["future_model"]
    dev = unit.payload.to_dict()["frozen_dev_bundle"]

    assert model == {"state": "uncommitted", "location": unit.payload.model_location.value}
    assert "digest" not in model
    assert set(dev) == {"content_digest", "byte_length", "media_type", "location", "compression"}
    assert dev["content_digest"] == _digest("a").value


def test_strict_round_trip_rejects_unknown_fields() -> None:
    plan = _plan()
    encoded = plan.to_dict()
    assert set(encoded) == {"request", "units", "pool"}
    assert set(encoded["request"]) == {
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
    assert set(encoded["units"][0]) == {
        "unit_id",
        "ordinal",
        "payload",
        "payload_digest",
        "progress_total",
        "artifact_access_specs",
    }
    assert set(encoded["pool"]) == {
        "pool_id",
        "queue_id",
        "profile_id",
        "profile_digest",
        "capacity",
        "reuse",
        "service",
    }
    assert QueuePlan.from_dict(encoded, campaign=plan.campaign, evaluator_profile=plan.evaluator_profile) == plan
    encoded["unexpected"] = True

    with pytest.raises(QueuePlanError, match="fields are not exact"):
        QueuePlan.from_dict(encoded, campaign=plan.campaign, evaluator_profile=plan.evaluator_profile)


def test_static_validation_profile_has_exact_gpu_and_v3_identity() -> None:
    profile_path = Path(__file__).parents[1] / "conf" / "validation_evaluator_profile.json"
    encoded = json.loads(profile_path.read_text())
    profile = ValidationEvaluatorProfile.from_dict(encoded)

    assert profile == VALIDATION_EVALUATOR_PROFILE
    assert "profile_digest" not in encoded
    assert profile.host_profile.accepted_gpu_names[0].value == "RTX 5060 Ti"
    assert profile.host_profile.minimum_gpu_memory_bytes >= 16 * 1024**3
    assert profile.host_profile.minimum_cuda_version >= CudaVersion(12, 8)
    assert profile.host_profile.minimum_system_ram_bytes >= 32 * 1024**3
    assert profile.host_profile.minimum_cpu_cores >= 4
    assert profile.network_policy == "egress"
    assert profile.download_egress is True
    assert profile.reusable_storage_credentials is False
    assert profile.github_credentials is False
    assert profile.registry_push_credentials is False
    assert profile.artifact_issuer_credentials is False
    assert profile.capability_identity.protocol_version is WorkProtocolVersion.V3


def test_capability_binding_matches_cloudeck_v3_canonical_bytes() -> None:
    encoded = json.loads(VALIDATION_WORKER_CAPABILITIES.canonical_bytes())

    assert VALIDATION_WORKER_CAPABILITIES.cancellation is CapabilityCancellation.COOPERATIVE
    assert VALIDATION_WORKER_CAPABILITIES.resume is CapabilityResume.UNIT_BOUNDARY
    assert encoded["cancellation"] == "cooperative"
    assert encoded["resume"] == "unit_boundary"
    assert encoded["protocol_versions"] == [3]
    assert encoded["workload_family"] == "diarizen-validation"
    assert (
        VALIDATION_WORKER_CAPABILITIES.digest().value
        == hashlib.sha256(VALIDATION_WORKER_CAPABILITIES.canonical_bytes()).hexdigest()
    )


def test_cloudeck_worker_profile_is_the_exact_vast_wire_document() -> None:
    profile = CLOUDECK_WORKER_PROFILE
    encoded = profile.to_dict()

    assert CloudeckWorkerProfile.from_dict(encoded) == profile
    assert encoded["provider"]["provider"] == "vast_ai"
    assert encoded["provider"]["gpu_name"] == "RTX_5060_Ti"
    assert encoded["provider"]["num_gpus"] == 1
    assert encoded["image"]["repository"] == PUBLISHED_EVALUATOR_IMAGE_REPOSITORY
    assert encoded["image"]["digest"] == PUBLISHED_EVALUATOR_IMAGE_DIGEST.value
    assert encoded["command"] == ["claim"]
    assert CLOUDECK_WORKER_PROFILE.command == VALIDATION_WORKER_COMMAND
    with pytest.raises(QueuePlanError, match="claim subcommand"):
        replace(
            CLOUDECK_WORKER_PROFILE,
            command=("/usr/local/bin/diarizen-validation-worker",),
        )
    assert encoded["network"] == "egress"
    assert encoded["registry_auth_secret"] is None
    assert encoded["allowed_secrets"] == []
    assert encoded["capacity_limit"] == 1
    assert "geolocations" not in encoded["provider"]
    assert _plan().pool.profile_digest == profile.digest()


def test_python_cloudeck_documents_are_strict_wire_objects() -> None:
    documents = emit_cloudeck_wire_documents()

    assert documents["capabilities"]["cancellation"] == "cooperative"
    assert documents["capabilities"]["resume"] == "unit_boundary"
    assert len(documents["units"]) == 22
    assert documents["queue_request"]["max_active_leases"] == 1
    assert documents["queue_request"]["protocol"]["version"] == 3
    assert documents["pool"]["capacity"] == 1
    assert documents["pool"]["reuse"] == "queue_lifetime"
    assert documents["pool"]["profile_digest"] == CLOUDECK_WORKER_PROFILE.digest().value
    assert "https://" not in json.dumps(documents)
