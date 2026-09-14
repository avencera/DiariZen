"""Pure construction of complete ordered validation slot plans."""

from __future__ import annotations

from .contracts import (
    ArtifactLocation,
    CompleteEpoch,
    EpochZero,
    FinalPartial,
    Sha256Digest,
    SnapshotPoint,
    StoppingPolicy,
    ValidationCampaign,
    ValidationSlot,
    canonical_digest,
)


def build_validation_campaign(
    *,
    training_launch_id: str,
    max_updates: int,
    updates_per_complete_epoch: int,
    artifact_prefix: str,
    dev_bundle_digest: Sha256Digest,
    trainer_configuration_digest: Sha256Digest,
    evaluator_image_identity: str,
    evaluator_implementation_digest: Sha256Digest,
    maximum_validation_lag: int = 1,
) -> ValidationCampaign:
    """Build the complete maximum slot list before queue creation."""

    campaign_fields = {
        "training_launch_id": training_launch_id,
        "max_updates": max_updates,
        "updates_per_complete_epoch": updates_per_complete_epoch,
        "dev_bundle_digest": dev_bundle_digest.value,
        "trainer_configuration_digest": trainer_configuration_digest.value,
        "evaluator_image_identity": evaluator_image_identity,
        "evaluator_implementation_digest": evaluator_implementation_digest.value,
        "maximum_validation_lag": maximum_validation_lag,
        "stopping_policy": StoppingPolicy.EXTERNAL_PATIENCE_OR_MAX_UPDATES.value,
    }
    campaign_id = Sha256Digest(canonical_digest(campaign_fields))
    prefix = ArtifactLocation(artifact_prefix.rstrip("/")).value
    points: list[tuple[SnapshotPoint, int]] = [(EpochZero(), 0)]
    complete_epochs, remainder = divmod(max_updates, updates_per_complete_epoch)
    points.extend(
        (CompleteEpoch(epoch), epoch * updates_per_complete_epoch) for epoch in range(1, complete_epochs + 1)
    )
    if remainder:
        points.append((FinalPartial(complete_epochs, remainder), max_updates))

    slots = []
    for ordinal, (point, updates) in enumerate(points):
        slot_prefix = f"{prefix}/{campaign_id.value}/slots/{ordinal:04d}"
        identity_fields = {
            "ordinal": ordinal,
            "point": point.to_dict(),
            "updates": updates,
            "manifest_location": f"{slot_prefix}/snapshot.json",
            "model_location": f"{slot_prefix}/pytorch_model.bin",
        }
        slots.append(
            ValidationSlot(
                slot_id=Sha256Digest(canonical_digest({"campaign_id": campaign_id.value, **identity_fields})),
                ordinal=ordinal,
                point=point,
                updates=updates,
                manifest_location=ArtifactLocation(identity_fields["manifest_location"]),
                model_location=ArtifactLocation(identity_fields["model_location"]),
            )
        )

    return ValidationCampaign(
        campaign_id=campaign_id,
        training_launch_id=training_launch_id,
        max_updates=max_updates,
        updates_per_complete_epoch=updates_per_complete_epoch,
        slots=tuple(slots),
        dev_bundle_digest=dev_bundle_digest,
        trainer_configuration_digest=trainer_configuration_digest,
        evaluator_image_identity=evaluator_image_identity,
        evaluator_implementation_digest=evaluator_implementation_digest,
        maximum_validation_lag=maximum_validation_lag,
        stopping_policy=StoppingPolicy.EXTERNAL_PATIENCE_OR_MAX_UPDATES,
    )
