from __future__ import annotations

import pytest

from recipes.speakrs.large.commercial_assets import (
    AugmentationCandidate,
    BlockedAugmentationRelease,
    CommercialAssetState,
    GoldComponent,
    GoldComponentRegistry,
    SilverHumanWorkQueue,
    SilverQueueItem,
)


def _hash(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


def test_gold_registry_totals_only_immutable_components() -> None:
    registry = GoldComponentRegistry(
        "commercial-gold-components-v1",
        CommercialAssetState.COMPONENTS_READY,
        (
            GoldComponent("replay", _hash("replay"), _hash("replay-restore"), 1127, 318.721),
            GoldComponent("lotusdis", _hash("lotusdis"), _hash("lotusdis-restore"), 44, 11.361193),
            GoldComponent("icsi", _hash("icsi"), _hash("icsi-restore"), 56, 52.483064),
        ),
    )

    assert registry.recordings == 1227
    assert registry.uem_hours == pytest.approx(382.565257)


def test_blocked_augmentation_release_cannot_contain_objects() -> None:
    candidate = AugmentationCandidate("MUSAN", ("music", "noise"), "CC-BY-4.0", "source objects not bound")

    with pytest.raises(ValueError, match="cannot contain"):
        BlockedAugmentationRelease(
            "commercial-augmentation-v1",
            CommercialAssetState.BLOCKED,
            (candidate,),
            (object(),),
        )


def test_silver_queue_rejects_duplicate_sources() -> None:
    item = SilverQueueItem(
        source="Open-Yap-full",
        activity_method="conservative proposals from separate speaker streams",
        confidence_gate="retain only reviewed activity",
        human_audit="double-check sampled windows",
        parent_provenance="bind publisher parent and both stream identities",
        failure_threshold="reject above 5% miss plus false alarm",
        blocker="human activity review is incomplete",
    )

    with pytest.raises(ValueError, match="unique"):
        SilverHumanWorkQueue("commercial-silver-human-queue-v1", (item, item))
