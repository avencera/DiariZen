"""Tests for the strict trainer validation-mode configuration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.speakrs.large.validation.config import (
    ExternalValidationConfig,
    InlineStoppingPolicy,
    InlineValidationConfig,
    ValidationConfigError,
    parse_validation_config,
    parse_validation_toml,
)


DIGEST = "a" * 64


def _external() -> dict[str, object]:
    return {
        "mode": "external",
        "campaign_manifest": {"path": "campaign.json", "sha256": DIGEST},
        "trusted_selection_state_path": "selection-state.json",
        "publication_transaction_root": "publication",
        "object_store_destination": {"provider": "r2", "bucket": "models", "prefix": "campaign"},
        "epoch_zero_gate": True,
        "poll_backoff": {"initial_seconds": 1, "maximum_seconds": 30, "multiplier": 2},
        "stopping_policy": "external_patience_or_max_updates",
        "patience_limit": 5,
    }


def test_unknown_fields_are_rejected_in_each_mode() -> None:
    inline = {"mode": "inline", "unknown": True}
    external = _external() | {"unknown": True}

    with pytest.raises(ValidationConfigError):
        parse_validation_config(inline)
    with pytest.raises(ValidationConfigError):
        parse_validation_config(external)


def test_external_requires_every_field() -> None:
    for field in _external():
        candidate = _external()
        candidate.pop(field)
        with pytest.raises(ValidationConfigError):
            parse_validation_config(candidate)


def test_cross_mode_fields_are_rejected() -> None:
    external = _external() | {"validation_dataloader_owner": "trainer"}
    inline = {
        "mode": "inline",
        "campaign_manifest": {"path": "campaign.json", "sha256": DIGEST},
    }

    with pytest.raises(ValidationConfigError):
        parse_validation_config(external)
    with pytest.raises(ValidationConfigError):
        parse_validation_config(inline)


def test_missing_validation_table_is_legacy_inline() -> None:
    legacy = parse_validation_config({"trainer": {"path": "trainer_dual_opt.Trainer"}})
    legacy_toml = parse_validation_toml('[trainer]\npath = "trainer_dual_opt.Trainer"\n')

    assert isinstance(legacy, InlineValidationConfig)
    assert legacy.stopping_policy is InlineStoppingPolicy.LEGACY
    assert isinstance(legacy_toml, InlineValidationConfig)
    assert legacy_toml == legacy


def test_legacy_inline_does_not_honor_patience_for_fixed_updates() -> None:
    config = parse_validation_config({"mode": "inline"})

    assert isinstance(config, InlineValidationConfig)
    assert config.stopping_policy is InlineStoppingPolicy.LEGACY
    assert config.patience_applies(fixed_update_run=True) is False
    assert config.patience_applies(fixed_update_run=False) is True


def test_inline_patience_is_explicit_and_positive() -> None:
    config = parse_validation_config({"mode": "inline", "stopping_policy": "patience", "patience_limit": 3})

    assert isinstance(config, InlineValidationConfig)
    assert config.stopping_policy is InlineStoppingPolicy.PATIENCE
    assert config.patience_limit == 3
    assert config.patience_applies(fixed_update_run=True) is True

    with pytest.raises(ValidationConfigError):
        parse_validation_config({"mode": "inline", "stopping_policy": "patience"})
    with pytest.raises(ValidationConfigError):
        parse_validation_config({"mode": "inline", "stopping_policy": "legacy", "patience_limit": 3})


def test_external_has_typed_required_policy_gate_and_patience() -> None:
    config = parse_validation_config(_external())

    assert isinstance(config, ExternalValidationConfig)
    assert config.campaign_manifest.path == Path("campaign.json")
    assert config.campaign_manifest.sha256.value == DIGEST
    assert config.epoch_zero_gate is True
    assert config.stopping_policy.value == "external_patience_or_max_updates"
    assert config.patience_limit == 5

    for field, replacement in (
        ("epoch_zero_gate", False),
        ("patience_limit", 0),
        ("stopping_policy", "legacy"),
    ):
        candidate = _external()
        candidate[field] = replacement
        with pytest.raises(ValidationConfigError):
            parse_validation_config(candidate)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value | {"nested": {"token": "do-not-store"}},
        lambda value: value | {"nested": {"signed_url": "https://example.invalid/object"}},
        lambda value: (
            value | {"object_store_destination": {"provider": "r2", "bucket": "models", "prefix": "https://bad"}}
        ),
        lambda value: value | {"campaign_manifest": {"path": "https://bad/campaign.json", "sha256": DIGEST}},
    ],
)
def test_recursive_secret_and_url_rejection(mutation) -> None:
    with pytest.raises(ValidationConfigError):
        parse_validation_config(mutation(_external()))


def test_credential_scan_does_not_reject_ordinary_words() -> None:
    config = parse_validation_config(
        _external()
        | {
            "object_store_destination": {
                "provider": "r2",
                "bucket": "models",
                "prefix": "tokenizer/secretary",
            }
        }
    )

    assert isinstance(config, ExternalValidationConfig)
