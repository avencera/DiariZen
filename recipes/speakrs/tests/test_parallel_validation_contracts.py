"""Tests for immutable external-validation campaign contracts."""

from __future__ import annotations

import pytest

from recipes.speakrs.large.validation import (
    CompleteEpoch,
    EpochZero,
    FinalPartial,
    Sha256Digest,
    ValidationCampaign,
    ValidationContractError,
    build_validation_campaign,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _campaign(max_updates: int = 60_000, updates_per_complete_epoch: int = 2_967) -> ValidationCampaign:
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


def test_60000_update_campaign_has_all_22_ordered_slots() -> None:
    campaign = _campaign()

    assert len(campaign.slots) == 22
    assert campaign.slots[0].point == EpochZero()
    assert campaign.slots[0].updates == 0
    assert [slot.point for slot in campaign.slots[1:21]] == [CompleteEpoch(epoch) for epoch in range(1, 21)]
    assert [slot.updates for slot in campaign.slots[1:21]] == [epoch * 2_967 for epoch in range(1, 21)]
    assert campaign.slots[21].point == FinalPartial(completed_epochs=20, partial_updates=660)
    assert campaign.slots[21].updates == 60_000


def test_aligned_target_has_no_separate_final_partial_slot() -> None:
    campaign = _campaign(max_updates=2_967 * 20)

    assert len(campaign.slots) == 21
    assert campaign.slots[-1].point == CompleteEpoch(20)
    assert campaign.slots[-1].updates == 2_967 * 20


def test_campaign_and_slot_identities_are_deterministic() -> None:
    first = _campaign()
    second = _campaign()

    assert first == second
    assert len({slot.slot_id for slot in first.slots}) == len(first.slots)


def test_strict_campaign_round_trip_rejects_unknown_fields() -> None:
    campaign = _campaign()
    encoded = campaign.to_dict()

    assert ValidationCampaign.from_dict(encoded) == campaign
    encoded["unexpected"] = True
    with pytest.raises(ValidationContractError, match="fields are not exact"):
        ValidationCampaign.from_dict(encoded)


def test_changed_slot_contract_cannot_reuse_a_slot_identity() -> None:
    encoded = _campaign().to_dict()
    slot = encoded["slots"][1]
    assert isinstance(slot, dict)
    slot["updates"] = 1

    with pytest.raises(ValidationContractError):
        ValidationCampaign.from_dict(encoded)


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "g" * 64, ""])
def test_digest_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValidationContractError):
        Sha256Digest(value)
