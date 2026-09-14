from __future__ import annotations

import pytest

from recipes.speakrs.large.acceptance import RttmInterval
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.lotusdis import (
    LotusdisParentOutcome,
    LotusdisParentRejection,
    LotusdisParseReason,
    LotusdisParseResult,
    LotusdisStrictSubset,
    LotusdisTextGridError,
    LotusdisTrainRejectionReason,
    parse_lotusdis_textgrid,
    select_lotusdis_strict_subset,
)


def _textgrid(marks: list[str], *, duration: float | None = None, tier_name: str = "speaker") -> str:
    end = float(duration if duration is not None else len(marks))
    intervals = []
    for index, mark in enumerate(marks):
        intervals.append(
            f'''        intervals [{index + 1}]:
            xmin = {index}
            xmax = {index + 1}
            text = "{mark}"
'''
        )
    return (
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n'
        "xmin = 0\n"
        f"xmax = {end}\n"
        "tiers? <exists>\n"
        "size = 1\n"
        "item []:\n"
        "    item [1]:\n"
        '        class = "IntervalTier"\n'
        f'        name = "{tier_name}"\n'
        "        xmin = 0\n"
        f"        xmax = {end}\n"
        f"        intervals: size = {len(marks)}\n" + "".join(intervals)
    )


def test_parses_one_two_and_three_speaker_marks_and_preserves_bounds() -> None:
    result = parse_lotusdis_textgrid(
        _textgrid(["M01,one", "F02&M01,two", "M01&F02&F03,three"]),
        "Hijack_S001_T057",
    )

    assert result.duration == 3.0
    assert result.speakers == frozenset({"M01", "F02", "F03"})
    assert result.intervals == (
        RttmInterval("Hijack_S001_T057", 0.0, 1.0, "M01"),
        RttmInterval("Hijack_S001_T057", 1.0, 2.0, "F02"),
        RttmInterval("Hijack_S001_T057", 1.0, 2.0, "M01"),
        RttmInterval("Hijack_S001_T057", 2.0, 3.0, "M01"),
        RttmInterval("Hijack_S001_T057", 2.0, 3.0, "F02"),
        RttmInterval("Hijack_S001_T057", 2.0, 3.0, "F03"),
    )


def test_non_speech_marks_produce_no_intervals() -> None:
    result = parse_lotusdis_textgrid(_textgrid(["", "<n>", "<sil>", "<unk>", "<td>"]), "parent")

    assert result.intervals == ()
    assert result.speakers == frozenset()


def test_one_publisher_tier_can_use_a_con123_name() -> None:
    result = parse_lotusdis_textgrid(_textgrid(["M01,speech"], tier_name="Con123"), "parent")

    assert result.speakers == frozenset({"M01"})


@pytest.mark.parametrize("mark", ["S01,text", "F1,text", "F01 &F02,text", "F01", "F01,"])
def test_malformed_mark_is_sanitized_and_typed(mark: str) -> None:
    with pytest.raises(LotusdisTextGridError) as raised:
        parse_lotusdis_textgrid(_textgrid([mark]), "parent")

    error = raised.value
    assert error.reason is LotusdisParseReason.MALFORMED_MARK
    assert error.details == {"parent_id": "parent", "interval_index": 0}
    assert "text" not in str(error)


@pytest.mark.parametrize(
    ("marks", "duration", "reason"),
    [
        (["M01,a", "M01,b"], 3.0, LotusdisParseReason.GAPPED_INTERVALS),
        (["M01,a", "M01,b"], 1.5, LotusdisParseReason.OUT_OF_BOUNDS_INTERVAL),
        (["M01,a", "M01,b"], 2.0, LotusdisParseReason.OVERLAPPING_INTERVALS),
    ],
)
def test_partition_gaps_and_bounds_are_rejected(
    marks: list[str], duration: float, reason: LotusdisParseReason
) -> None:
    text = _textgrid(marks, duration=duration)
    if reason is LotusdisParseReason.OVERLAPPING_INTERVALS:
        text = text.replace("xmin = 1\n            xmax = 2", "xmin = 0.5\n            xmax = 2")
    with pytest.raises(LotusdisTextGridError) as raised:
        parse_lotusdis_textgrid(text, "parent")
    assert raised.value.reason is reason


@pytest.mark.parametrize(
    ("start", "end", "reason"),
    [
        ("-1", "1", LotusdisParseReason.NEGATIVE_INTERVAL),
        ("1", "1", LotusdisParseReason.REVERSED_INTERVAL),
        ("2", "3", LotusdisParseReason.OUT_OF_BOUNDS_INTERVAL),
    ],
)
def test_invalid_interval_bounds_are_rejected(start: str, end: str, reason: LotusdisParseReason) -> None:
    text = _textgrid(["M01,a"])
    text = text.replace("xmin = 0\n            xmax = 1", f"xmin = {start}\n            xmax = {end}")
    with pytest.raises(LotusdisTextGridError) as raised:
        parse_lotusdis_textgrid(text, "parent")
    assert raised.value.reason is reason


def _result(parent_id: str, *speakers: str) -> LotusdisParseResult:
    return LotusdisParseResult(
        parent_id,
        1.0,
        tuple(RttmInterval(parent_id, 0.0, 1.0, speaker) for speaker in speakers),
    )


def test_strict_subset_is_deterministic_and_preserves_heldout_splits() -> None:
    outcomes = {
        "train-good": _result("train-good", "M03"),
        "train-four": _result("train-four", "M03", "F03", "M04", "F04"),
        "train-leak": _result("train-leak", "M01"),
        "train-malformed": LotusdisTextGridError(LotusdisParseReason.MALFORMED_MARK, "train-malformed", 2),
        "dev": _result("dev", "M01"),
        "test": _result("test", "F01"),
    }

    selected = select_lotusdis_strict_subset(
        ["train-leak", "train-good", "train-four", "train-malformed"],
        ["dev"],
        ["test"],
        outcomes,
        speaker_graph={
            "train-good": ["M03"],
            "train-four": ["M03", "F03", "M04", "F04"],
            "train-leak": ["M01"],
            "train-malformed": ["M05"],
            "dev": ["M01"],
            "test": ["F01"],
        },
    )

    assert isinstance(selected, LotusdisStrictSubset)
    assert selected.accepted_train_ids == ("train-good",)
    assert selected.dev_parent_ids == ("dev",)
    assert selected.test_parent_ids == ("test",)
    assert {item.parent_id: item.reason for item in selected.rejected} == {
        "train-four": LotusdisTrainRejectionReason.TOO_MANY_SPEAKERS,
        "train-leak": LotusdisTrainRejectionReason.HELDOUT_SPEAKER_LEAKAGE,
        "train-malformed": LotusdisTrainRejectionReason.MALFORMED_LABEL,
    }


def test_strict_subset_requires_every_publisher_outcome() -> None:
    with pytest.raises(PreparationError, match="outcomes are incomplete"):
        select_lotusdis_strict_subset(
            ["train"],
            ["dev"],
            ["test"],
            {},
            speaker_graph={"train": ["M01"], "dev": ["F01"], "test": ["F02"]},
        )


@pytest.mark.parametrize(
    ("train", "dev", "test", "split"),
    [
        (["train", "train"], ["dev"], ["test"], "train"),
        (["train"], ["dev", "dev"], ["test"], "dev"),
        (["train"], ["dev"], ["test", "test"], "test"),
    ],
)
def test_strict_subset_rejects_duplicate_publisher_ids(
    train: list[str], dev: list[str], test: list[str], split: str
) -> None:
    with pytest.raises(PreparationError, match="duplicate") as raised:
        select_lotusdis_strict_subset(train, dev, test, {}, speaker_graph={})
    assert raised.value.details == {"split": split}


@pytest.mark.parametrize(
    "intervals",
    [
        (RttmInterval("parent", 0.0, 1.0, "S01"),),
        (RttmInterval("parent", float("nan"), 1.0, "M01"),),
        (RttmInterval("parent", -0.1, 1.0, "M01"),),
        (RttmInterval("parent", 1.0, 1.0, "M01"),),
        (RttmInterval("parent", 0.0, 1.1, "M01"),),
        (
            RttmInterval("parent", 0.0, 1.0, "M01"),
            RttmInterval("parent", 0.0, 1.0, "M01"),
        ),
    ],
)
def test_parse_result_rejects_invalid_public_intervals(intervals: tuple[RttmInterval, ...]) -> None:
    with pytest.raises(ValueError):
        LotusdisParseResult("parent", 1.0, intervals)


def test_strict_subset_rejects_cross_split_and_duplicate_rejections() -> None:
    reason = LotusdisTrainRejectionReason.MALFORMED_LABEL
    with pytest.raises(ValueError, match="disjoint"):
        LotusdisStrictSubset(("parent",), ("parent",), (), ())
    with pytest.raises(ValueError, match="unique"):
        LotusdisStrictSubset(
            ("train",),
            (),
            (),
            (LotusdisParentRejection("bad", reason), LotusdisParentRejection("bad", reason)),
        )
    with pytest.raises(ValueError, match="selected"):
        LotusdisStrictSubset(
            (),
            ("dev",),
            (),
            (LotusdisParentRejection("dev", reason),),
        )


@pytest.mark.parametrize(
    "value",
    [
        _result("wrong", "M01"),
        LotusdisParentOutcome.accepted(_result("wrong", "M01")),
        LotusdisTextGridError(LotusdisParseReason.MALFORMED_MARK, "wrong", 0),
    ],
)
def test_strict_subset_rejects_mismatched_outcome_identity(value: object) -> None:
    with pytest.raises(PreparationError, match="identity mismatch") as raised:
        select_lotusdis_strict_subset(
            ["train"],
            ["dev"],
            ["test"],
            {
                "train": value,
                "dev": _result("dev", "F01"),
                "test": _result("test", "F02"),
            },
            speaker_graph={"train": ["M01"], "dev": ["F01"], "test": ["F02"]},
        )
    assert raised.value.details == {"parent_id": "train"}


def test_strict_subset_uses_identity_proof_when_heldout_activity_is_malformed() -> None:
    selected = select_lotusdis_strict_subset(
        ["train-good", "train-leak"],
        ["dev"],
        ["test"],
        {
            "train-good": _result("train-good", "M02"),
            "train-leak": _result("train-leak", "M01"),
        },
        speaker_graph={
            "train-good": ["M02"],
            "train-leak": ["M01"],
            "dev": ["F01"],
            "test": ["M01"],
        },
    )

    assert selected.accepted_train_ids == ("train-good",)
    assert selected.rejected == (
        LotusdisParentRejection("train-leak", LotusdisTrainRejectionReason.HELDOUT_SPEAKER_LEAKAGE),
    )


def test_strict_subset_rejects_incomplete_or_invalid_speaker_graph() -> None:
    outcomes = {"train": _result("train", "M01")}

    with pytest.raises(PreparationError, match="speaker graph is incomplete"):
        select_lotusdis_strict_subset(
            ["train"],
            ["dev"],
            ["test"],
            outcomes,
            speaker_graph={"train": ["M01"], "dev": ["F01"]},
        )

    with pytest.raises(PreparationError, match="invalid identities"):
        select_lotusdis_strict_subset(
            ["train"],
            ["dev"],
            ["test"],
            outcomes,
            speaker_graph={"train": ["M01"], "dev": ["F01"], "test": ["M4"]},
        )


def test_strict_subset_rejects_train_label_and_identity_disagreement() -> None:
    with pytest.raises(PreparationError, match="differ from the speaker graph"):
        select_lotusdis_strict_subset(
            ["train"],
            ["dev"],
            ["test"],
            {"train": _result("train", "M01")},
            speaker_graph={"train": ["M02"], "dev": ["F01"], "test": ["F02"]},
        )
