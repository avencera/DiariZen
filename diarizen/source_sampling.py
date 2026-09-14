"""Typed deterministic source-weighted sampling for diarization training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Sampler


SAMPLING_SCHEMA = "speakrs-source-sampling-v1"
SAMPLING_METHOD = "deterministic-component-weighted-with-replacement"
WEIGHT_SCALE = 1_000_000


@dataclass(frozen=True, slots=True)
class SamplingComponent:
    """One disjoint source group and its exact integer sampling weight."""

    name: str
    sources: tuple[str, ...]
    weight_millionths: int

    def __post_init__(self) -> None:
        if not self.name or not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError("sampling component identity is invalid")
        if self.weight_millionths <= 0 or self.weight_millionths > WEIGHT_SCALE:
            raise ValueError("sampling component weight is outside the valid range")


@dataclass(frozen=True, slots=True)
class SourceSamplingPolicy:
    """A complete sampling contract bound to one immutable data release."""

    release_sha256: str
    seed: int
    samples_per_epoch: int
    components: tuple[SamplingComponent, ...]

    def __post_init__(self) -> None:
        if len(self.release_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.release_sha256
        ):
            raise ValueError("sampling release SHA-256 is invalid")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("sampling seed must be a non-negative integer")
        if isinstance(self.samples_per_epoch, bool) or not isinstance(self.samples_per_epoch, int):
            raise ValueError("samples_per_epoch must be an integer")
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if not self.components or sum(component.weight_millionths for component in self.components) != WEIGHT_SCALE:
            raise ValueError("sampling component weights must sum to one million")
        names = [component.name for component in self.components]
        sources = [source for component in self.components for source in component.sources]
        if len(set(names)) != len(names) or len(set(sources)) != len(sources):
            raise ValueError("sampling component names and source membership must be disjoint")

    @classmethod
    def from_json(cls, payload: object) -> SourceSamplingPolicy:
        """Parse a strict JSON sampling contract."""

        if not isinstance(payload, Mapping):
            raise ValueError("sampling policy must be an object")
        expected = {"schema", "method", "release_sha256", "seed", "samples_per_epoch", "components"}
        if set(payload) != expected or payload.get("schema") != SAMPLING_SCHEMA:
            raise ValueError("sampling policy schema or fields are invalid")
        if payload.get("method") != SAMPLING_METHOD:
            raise ValueError("sampling method is invalid")
        component_rows = payload.get("components")
        if not isinstance(component_rows, list):
            raise ValueError("sampling components must be an array")
        components = []
        for row in component_rows:
            if not isinstance(row, Mapping) or set(row) != {"name", "sources", "weight_millionths"}:
                raise ValueError("sampling component fields are invalid")
            sources = row.get("sources")
            if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
                raise ValueError("sampling component sources are invalid")
            name = row.get("name")
            weight = row.get("weight_millionths")
            if not isinstance(name, str) or isinstance(weight, bool) or not isinstance(weight, int):
                raise ValueError("sampling component values are invalid")
            components.append(SamplingComponent(name, tuple(sources), weight))
        release_sha256 = payload.get("release_sha256")
        seed = payload.get("seed")
        samples_per_epoch = payload.get("samples_per_epoch")
        if not isinstance(release_sha256, str):
            raise ValueError("sampling release SHA-256 is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("sampling seed is invalid")
        if isinstance(samples_per_epoch, bool) or not isinstance(samples_per_epoch, int):
            raise ValueError("sampling samples_per_epoch is invalid")
        return cls(release_sha256, seed, samples_per_epoch, tuple(components))

    @classmethod
    def load(cls, path: str | Path) -> SourceSamplingPolicy:
        """Load a strict sampling contract from disk."""

        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def component_sample_counts(self) -> tuple[int, ...]:
        """Allocate the exact epoch length by largest remainder."""

        scaled = [self.samples_per_epoch * component.weight_millionths for component in self.components]
        counts = [value // WEIGHT_SCALE for value in scaled]
        remaining = self.samples_per_epoch - sum(counts)
        order = sorted(
            range(len(self.components)),
            key=lambda index: (-(scaled[index] % WEIGHT_SCALE), self.components[index].name),
        )
        for index in order[:remaining]:
            counts[index] += 1
        return tuple(counts)


def load_recording_sources(bundle_path: str | Path, release_sha256: str) -> dict[str, str]:
    """Load exact recording-to-source membership from a bound training bundle."""

    payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != "speakrs-training-bundle-v1":
        raise ValueError("training bundle schema is invalid")
    if payload.get("release_sha256") != release_sha256:
        raise ValueError("sampling policy belongs to a different training release")
    rows = payload.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training bundle has no recordings")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("training bundle recording is invalid")
        recording_id = row.get("recording_id")
        source = row.get("source")
        if not isinstance(recording_id, str) or not recording_id or not isinstance(source, str) or not source:
            raise ValueError("training bundle recording identity is invalid")
        if recording_id in result:
            raise ValueError("training bundle recording IDs must be unique")
        result[recording_id] = source
    return result


class DeterministicSourceWeightedSampler(Sampler[int]):
    """Draw exact component quotas and uniform samples within each component."""

    def __init__(
        self,
        policy: SourceSamplingPolicy,
        chunk_recording_ids: Sequence[str],
        recording_sources: Mapping[str, str],
        generator: torch.Generator,
    ) -> None:
        self.policy = policy
        self.generator = generator
        source_to_component = {
            source: index for index, component in enumerate(policy.components) for source in component.sources
        }
        if set(recording_sources.values()) != set(source_to_component):
            raise ValueError("sampling components must cover the exact training source set")
        pools: list[list[int]] = [[] for _ in policy.components]
        for chunk_index, recording_id in enumerate(chunk_recording_ids):
            source = recording_sources.get(recording_id)
            if source is None:
                raise ValueError("training chunk names a recording absent from the bound bundle")
            pools[source_to_component[source]].append(chunk_index)
        if any(not pool for pool in pools):
            raise ValueError("every sampling component must have at least one training chunk")
        self._pools = tuple(torch.tensor(pool, dtype=torch.int64) for pool in pools)

    def __len__(self) -> int:
        """Return the exact number of samples in one epoch."""

        return self.policy.samples_per_epoch

    def __iter__(self):
        """Yield one deterministic, shuffled epoch with exact component quotas."""

        selected = []
        for pool, count in zip(self._pools, self.policy.component_sample_counts()):
            offsets = torch.randint(len(pool), (count,), generator=self.generator)
            selected.append(pool[offsets])
        epoch = torch.cat(selected)
        order = torch.randperm(len(epoch), generator=self.generator)
        yield from epoch[order].tolist()


def build_source_weighted_sampler(
    *,
    policy_path: str | Path,
    bundle_path: str | Path,
    chunk_recording_ids: Sequence[str],
) -> DeterministicSourceWeightedSampler:
    """Build a deterministic sampler from two independently hashed contracts."""

    policy = SourceSamplingPolicy.load(policy_path)
    sources = load_recording_sources(bundle_path, policy.release_sha256)
    generator = torch.Generator().manual_seed(policy.seed)
    return DeterministicSourceWeightedSampler(policy, chunk_recording_ids, sources, generator)
