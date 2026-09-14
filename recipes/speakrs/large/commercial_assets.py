"""Typed manifests for separated commercial data-preparation assets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .contracts import require_content_hash


class CommercialAssetState(str, Enum):
    """A versioned asset state that cannot imply an unproved release seal."""

    COMPONENTS_READY = "components-ready"
    BLOCKED = "blocked"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class GoldComponent:
    """One immutable deterministic-speaker source release."""

    release_id: str
    release_sha256: str
    restore_receipt_sha256: str
    recordings: int
    uem_hours: float

    def __post_init__(self) -> None:
        if not self.release_id:
            raise ValueError("gold component release_id must be non-empty")
        require_content_hash(self.release_sha256, "gold component release")
        require_content_hash(self.restore_receipt_sha256, "gold component restore receipt")
        if self.recordings < 1:
            raise ValueError("gold component recordings must be positive")
        if not math.isfinite(self.uem_hours) or self.uem_hours <= 0:
            raise ValueError("gold component UEM hours must be finite and positive")


@dataclass(frozen=True, slots=True)
class GoldComponentRegistry:
    """Versioned immutable components that precede one final gold union."""

    asset_id: str
    state: CommercialAssetState
    components: tuple[GoldComponent, ...]

    def __post_init__(self) -> None:
        if self.state is not CommercialAssetState.COMPONENTS_READY:
            raise ValueError("gold registry must remain components-ready until a final union is sealed")
        if not self.asset_id or not self.components:
            raise ValueError("gold registry identity and components are required")
        release_ids = [component.release_id for component in self.components]
        if len(set(release_ids)) != len(release_ids):
            raise ValueError("gold component release IDs must be unique")

    @property
    def recordings(self) -> int:
        """Return the total immutable component recording count."""

        return sum(component.recordings for component in self.components)

    @property
    def uem_hours(self) -> float:
        """Return the total immutable component UEM hours."""

        return sum(component.uem_hours for component in self.components)


@dataclass(frozen=True, slots=True)
class AugmentationCandidate:
    """One permitted augmentation source that is not conversation data."""

    source: str
    condition_types: tuple[str, ...]
    permission: str
    blocker: str

    def __post_init__(self) -> None:
        if not self.source or not self.condition_types or not self.permission or not self.blocker:
            raise ValueError("augmentation candidates require source, conditions, permission, and blocker")


@dataclass(frozen=True, slots=True)
class BlockedAugmentationRelease:
    """A versioned augmentation release boundary with no admitted objects."""

    asset_id: str
    state: CommercialAssetState
    candidates: tuple[AugmentationCandidate, ...]
    admitted_objects: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.state is not CommercialAssetState.BLOCKED:
            raise ValueError("a blocked augmentation release must use the blocked state")
        if not self.asset_id or not self.candidates:
            raise ValueError("blocked augmentation release identity and candidates are required")
        if self.admitted_objects:
            raise ValueError("blocked augmentation release cannot contain admitted objects")


@dataclass(frozen=True, slots=True)
class SilverQueueItem:
    """One non-gold source with an exact completion and failure contract."""

    source: str
    activity_method: str
    confidence_gate: str
    human_audit: str
    parent_provenance: str
    failure_threshold: str
    blocker: str

    def __post_init__(self) -> None:
        values = (
            self.source,
            self.activity_method,
            self.confidence_gate,
            self.human_audit,
            self.parent_provenance,
            self.failure_threshold,
            self.blocker,
        )
        if any(not value for value in values):
            raise ValueError("silver queue fields must be non-empty")


@dataclass(frozen=True, slots=True)
class SilverHumanWorkQueue:
    """A versioned queue that remains outside gold supervised training."""

    asset_id: str
    items: tuple[SilverQueueItem, ...]

    def __post_init__(self) -> None:
        if not self.asset_id or not self.items:
            raise ValueError("silver queue identity and items are required")
        sources = [item.source for item in self.items]
        if len(set(sources)) != len(sources):
            raise ValueError("silver queue sources must be unique")
