"""Tests for deterministic source-weighted training sampling."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import torch

from diarizen.source_sampling import (
    DeterministicSourceWeightedSampler,
    SamplingComponent,
    SourceSamplingPolicy,
    build_source_weighted_sampler,
)


DIGEST = "a" * 64


def _policy(replay_weight: int = 900_000) -> SourceSamplingPolicy:
    return SourceSamplingPolicy(
        release_sha256=DIGEST,
        seed=17,
        samples_per_epoch=100,
        components=(
            SamplingComponent("replay", ("AMI", "AliMeeting"), replay_weight),
            SamplingComponent("new", ("LOTUSDIS",), 1_000_000 - replay_weight),
        ),
    )


def test_sampler_has_exact_component_quota_and_is_deterministic() -> None:
    policy = _policy()
    chunks = ("ami", "ami", "ali", "ali", "lotus", "lotus")
    sources = {"ami": "AMI", "ali": "AliMeeting", "lotus": "LOTUSDIS"}
    first = DeterministicSourceWeightedSampler(policy, chunks, sources, torch.Generator().manual_seed(policy.seed))
    second = DeterministicSourceWeightedSampler(policy, chunks, sources, torch.Generator().manual_seed(policy.seed))

    first_epoch = list(first)
    assert first_epoch == list(second)
    assert len(first_epoch) == 100
    counts = Counter(sources[chunks[index]] for index in first_epoch)
    assert counts["LOTUSDIS"] == 10
    assert counts["AMI"] + counts["AliMeeting"] == 90


def test_sampler_generator_state_reproduces_mid_epoch_order() -> None:
    policy = _policy(850_000)
    chunks = ("replay", "ali", "lotus")
    sources = {"replay": "AMI", "ali": "AliMeeting", "lotus": "LOTUSDIS"}
    generator = torch.Generator().manual_seed(policy.seed)
    sampler = DeterministicSourceWeightedSampler(policy, chunks, sources, generator)
    state = generator.get_state()

    expected = list(sampler)
    generator.set_state(state)
    assert list(sampler) == expected


def test_policy_rejects_inexact_or_overlapping_membership() -> None:
    with pytest.raises(ValueError, match="sum to one million"):
        SourceSamplingPolicy(DIGEST, 1, 10, (SamplingComponent("only", ("AMI",), 999_999),))
    with pytest.raises(ValueError, match="disjoint"):
        SourceSamplingPolicy(
            DIGEST,
            1,
            10,
            (
                SamplingComponent("first", ("AMI",), 500_000),
                SamplingComponent("second", ("AMI",), 500_000),
            ),
        )


def test_builder_binds_policy_to_bundle_release(tmp_path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "speakrs-training-bundle-v1",
                "release_sha256": "b" * 64,
                "recordings": [{"recording_id": "meeting", "source": "AMI"}],
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "sampling.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "speakrs-source-sampling-v1",
                "method": "deterministic-component-weighted-with-replacement",
                "release_sha256": DIGEST,
                "seed": 1,
                "samples_per_epoch": 10,
                "components": [{"name": "replay", "sources": ["AMI"], "weight_millionths": 1_000_000}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different training release"):
        build_source_weighted_sampler(
            policy_path=policy,
            bundle_path=bundle,
            chunk_recording_ids=("meeting",),
        )
