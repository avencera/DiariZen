"""Development selection, test isolation, and archive reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import RuntimeGateError
from .hashing import sha256_json


DEV_CORPORA = ("AMI", "AliMeeting", "AISHELL4", "NOTSOFAR_real", "ICSI", "LOTUSDIS")
TEST_CORPORA = DEV_CORPORA + ("VoxConverse",)
POLICIES = ("loss_average", "der_average", "best_loss", "best_der")
COLLAR_SECONDS = 0.0
OVERLAP_INCLUDED = True


@dataclass(frozen=True)
class CorpusScore:
    """Speaker-time DER components for one corpus."""

    corpus: str
    error_seconds: float
    reference_seconds: float

    @property
    def der(self) -> float:
        """Return speaker-time DER."""

        if self.reference_seconds <= 0:
            raise RuntimeGateError("reference time must be positive", {"corpus": self.corpus})
        return self.error_seconds / self.reference_seconds


def macro_der(scores: Sequence[CorpusScore], corpora: Sequence[str] = DEV_CORPORA) -> float:
    """Unweighted mean of per-corpus speaker-time DER."""

    by_name = {score.corpus: score for score in scores}
    missing = [name for name in corpora if name not in by_name]
    if missing:
        raise RuntimeGateError("selection is missing a required corpus", {"missing": missing})
    extra = sorted(set(by_name) - set(corpora))
    if extra:
        raise RuntimeGateError("selection received an unexpected corpus", {"extra": extra})
    return sum(by_name[name].der for name in corpora) / len(corpora)


@dataclass(frozen=True)
class CycleRecord:
    """One trained cycle eligible for ranking."""

    cycle: int
    model_hash: str
    loss: float
    der: float
    scores: tuple[CorpusScore, ...]


def reject_test_ranking(corpora: Sequence[str]) -> None:
    """Fail if VoxConverse published-dev or any test split is used to rank."""

    forbidden = {"VoxConverse", "VoxConverse_dev", "synthetic"}
    if forbidden.intersection(corpora):
        raise RuntimeGateError("test or VoxConverse published-dev cannot enter ranking", {"corpora": list(corpora)})


def top_five(records: Sequence[CycleRecord], key: str) -> list[CycleRecord]:
    """Return up to five trained cycles, excluding epoch zero."""

    eligible = [record for record in records if record.cycle > 0]
    reverse = False
    if key == "der":
        eligible.sort(key=lambda record: (record.der, record.loss, record.cycle))
    elif key == "loss":
        eligible.sort(key=lambda record: (record.loss, record.der, record.cycle))
    else:
        raise RuntimeGateError("unknown ranking key", {"key": key})
    if reverse:
        eligible.reverse()
    return eligible[:5]


def four_policies(records: Sequence[CycleRecord]) -> dict[str, object]:
    """Build the four candidate policies. Fewer than five models is incomplete."""

    reject_test_ranking(tuple(score.corpus for record in records for score in record.scores) or DEV_CORPORA)
    loss_index = top_five(records, "loss")
    der_index = top_five(records, "der")
    complete = len(loss_index) == 5 and len(der_index) == 5
    if not records:
        raise RuntimeGateError("no trained cycles to select")
    result = {
        "loss_index": [record.model_hash for record in loss_index],
        "der_index": [record.model_hash for record in der_index],
        "k": min(len(loss_index), len(der_index)),
        "complete": complete,
        "policies": {},
    }
    if not complete:
        result["policies"] = dict.fromkeys(POLICIES)
        result["incomplete_reason"] = "fewer than five trained models"
        return result

    def average_hashes(index: Sequence[CycleRecord]) -> dict[str, object]:
        hashes = [record.model_hash for record in index]
        if len(set(hashes)) != len(hashes):
            hashes = list(dict.fromkeys(hashes))
        return {
            "model_hashes": hashes,
            "kind": "average",
            "strict_load": True,
        }

    best_loss = loss_index[0]
    best_der = der_index[0]
    result["policies"] = {
        "loss_average": average_hashes(loss_index),
        "der_average": average_hashes(der_index),
        "best_loss": {"model_hashes": [best_loss.model_hash], "kind": "single", "cycle": best_loss.cycle},
        "best_der": {"model_hashes": [best_der.model_hash], "kind": "single", "cycle": best_der.cycle},
    }
    return result


def seal_selection(
    *,
    records: Sequence[CycleRecord],
    chosen_policy: str,
    data_release_hash: str,
    image_digest: str,
    pipeline_hash: str,
    test_manifest_hash: str,
) -> dict[str, object]:
    """Seal development selection before any test path is opened."""

    if chosen_policy not in POLICIES:
        raise RuntimeGateError("unknown selection policy", {"policy": chosen_policy})
    policies = four_policies(records)
    if not policies["complete"]:
        raise RuntimeGateError("four-policy result is incomplete", policies)
    chosen = policies["policies"][chosen_policy]
    if chosen is None:
        raise RuntimeGateError("chosen policy is missing")
    selected_der = None
    if chosen_policy == "best_der":
        selected_der = top_five(records, "der")[0].der
    elif chosen_policy == "best_loss":
        selected_der = top_five(records, "loss")[0].der
    elif chosen_policy == "der_average":
        selected_der = sum(record.der for record in top_five(records, "der")) / 5
    else:
        selected_der = sum(record.der for record in top_five(records, "loss")) / 5
    seal = {
        "collar_seconds": COLLAR_SECONDS,
        "overlap_included": OVERLAP_INCLUDED,
        "dev_corpora": list(DEV_CORPORA),
        "policies": policies,
        "chosen_policy": chosen_policy,
        "chosen": chosen,
        "selected_dev_macro_der": selected_der,
        "data_release_hash": data_release_hash,
        "image_digest": image_digest,
        "pipeline_hash": pipeline_hash,
        "test_manifest_hash": test_manifest_hash,
        "test_access": False,
    }
    seal["selection_hash"] = sha256_json(seal)
    return seal


def open_test_path(selection: Mapping[str, object]) -> dict[str, object]:
    """Permit the seven-corpus test path only after a sealed selection."""

    required = (
        "selection_hash",
        "chosen_policy",
        "data_release_hash",
        "image_digest",
        "pipeline_hash",
        "test_manifest_hash",
    )
    missing = [key for key in required if not selection.get(key)]
    if missing:
        raise RuntimeGateError("test path requires a sealed selection", {"missing": missing})
    if selection.get("test_access") is True:
        raise RuntimeGateError("selection was already opened for test")
    opened = dict(selection)
    opened["test_access"] = True
    opened["test_corpora"] = list(TEST_CORPORA)
    return opened


def panel_from_overlap(
    recordings: Sequence[Mapping[str, object]],
    per_corpus: Mapping[str, int],
) -> list[dict[str, object]]:
    """Select the frozen 12-recording panel from overlap statistics."""

    selected = []
    for corpus, count in per_corpus.items():
        pool = [row for row in recordings if row.get("corpus") == corpus]
        if len(pool) < count:
            raise RuntimeGateError("panel is missing recordings", {"corpus": corpus})
        ranked = sorted(pool, key=lambda row: (float(row["overlap"]), str(row["parent_id"])))
        if count == 1:
            chosen = [ranked[len(ranked) // 2]]
        else:
            chosen = [ranked[0], ranked[-1]]
            if count > 2:
                raise RuntimeGateError("panel helper supports 1 or 2 recordings per corpus")
        for row in chosen:
            selected.append(
                {
                    "corpus": corpus,
                    "parent_id": row["parent_id"],
                    "audio_sha256": row["audio_sha256"],
                    "rttm_sha256": row["rttm_sha256"],
                    "overlap": row["overlap"],
                    "duration": row["duration"],
                }
            )
    if len(selected) != 12:
        raise RuntimeGateError("panel must contain 12 recordings", {"count": len(selected)})
    return selected


def complete_archive(payload: Mapping[str, object]) -> dict[str, object]:
    """Build a complete archive record."""

    required = ("selection", "test_results", "coverage_complete")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeGateError("complete archive is missing outputs", {"missing": missing})
    return {"kind": "complete", "ok": True, **dict(payload)}


def incomplete_archive(payload: Mapping[str, object], missing: Sequence[str]) -> dict[str, object]:
    """Build a failure archive that cannot claim acceptance."""

    if payload.get("accepted") is True:
        raise RuntimeGateError("incomplete archive cannot assert acceptance")
    return {
        "kind": "incomplete",
        "ok": False,
        "missing": list(missing),
        "available": dict(payload),
        "accepted": False,
    }
