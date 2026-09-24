"""Produce tight-v2 label releases for AMI, AliMeeting, and AVA-AVD."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..jsonio import read_json, write_json
from ..storage import require_free_gib
from . import LIBRARY_VERSION
from .domain import RecordingLabels, SpeakerSegment
from .gates import GateResult, evaluate_gates
from .intervals import clip_recording_to_uem
from .listening import write_listening_pack
from .mapping import SpeakerMapping, apply_published_replacement
from .merge import MIN_GAP_NS, MIN_GAP_SECONDS, merge_recording
from .tightness import FORCED_AMI_REFERENCE, TightnessStats
from .time import NANOSECONDS_PER_SECOND, ns_to_seconds
from .writers import (
    parse_speaker_rttm,
    render_speaker_rttm,
    rewrite_uem_file_id,
    uem_bounds,
    write_recording_labels,
)


REPO_DIR = Path(__file__).resolve().parents[4]
DV_ROOT = Path("/Volumes/CacheDisk/dev-cache/diarization-data-verification")
TRAINING_VIEW = DV_ROOT / "assets/commercial-data-preparation-v1/final-union-training-v1"
ANNOTATIONS_PATH = TRAINING_VIEW / "annotations.rttm-uem.txt"
PUBLISHED_ROOT = Path("/tmp/dfa")
PUBLISHED_REPO = "https://github.com/nttcslab-sp/diar-forced-alignment"
PUBLISHED_COMMIT = "9527b7c64846fb38316a610f32e9d3466bd6d8b7"
STAGING_ROOT = DV_ROOT / "staging/tight-labels-v1"
MINIMUM_FREE_GIB = 120.0
SAMPLE_RATE = 16_000
V1_RELEASES = {
    "ami": "ami-tight-v1",
    "alimeeting": "alimeeting-tight-v1",
    "ava-avd": "ava-avd-tight-v1",
}

SOURCE_SPECS = {
    "ami": {
        "component": "ami",
        "release_id": "ami-tight-v2",
        "source_name": "AMI",
        "prefix": "ami::",
        "method": "replace-published-forced-alignment",
        "license": "CC BY 4.0",
        "published_split": "AMI/train",
        "require_subset": False,
        "listening_name": "ami",
    },
    "alimeeting": {
        "component": "alimeeting",
        "release_id": "alimeeting-tight-v2",
        "source_name": "AliMeeting",
        "prefix": "alimeeting::",
        "method": "replace-published-forced-alignment",
        "license": "CC BY-SA 4.0",
        "published_split": "AliMeeting/Train_Ali_far",
        "require_subset": False,
        "listening_name": "alimeeting",
    },
    "ava-avd": {
        "component": "ava-avd",
        "release_id": "ava-avd-tight-v2",
        "source_name": "AVA-AVD",
        "prefix": "ava-avd::",
        "method": "merge-short-gaps",
        "license": "CC BY 4.0",
        "published_split": None,
        "require_subset": False,
        "listening_name": "ava-avd",
    },
}


@dataclass(frozen=True, slots=True)
class RecordingRef:
    """Paths and duration for one training recording."""

    recording_id: str
    audio_path: Path
    uem_path: Path
    duration_ns: int
    uem_start_ns: int
    uem_end_ns: int

    @property
    def duration_seconds(self) -> float:
        """Return audio duration in seconds."""

        return ns_to_seconds(self.duration_ns)


def release_id_with_version(release_id: str, version: str | None) -> str:
    """Replace the trailing tight-vN suffix when a release version is set."""

    if version is None:
        return release_id
    normalized = version.strip()
    if not normalized:
        raise ValueError("release version must be a non-empty string")
    if not normalized.startswith("v"):
        normalized = f"v{normalized}"
    prefix, sep, _current = release_id.rpartition("-")
    if not sep:
        return f"{release_id}-{normalized}"
    return f"{prefix}-{normalized}"


def library_commit(repo: Path = REPO_DIR) -> str:
    """Return the git commit of the recipe worktree."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()


def published_commit(root: Path = PUBLISHED_ROOT) -> str:
    """Return the git commit of the published-label clone."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def published_parent_id(recording_id: str, source: str) -> str:
    """Map a training recording id onto a published RTTM stem."""

    parent = recording_id.split("::", 1)[1]
    if source == "alimeeting":
        parts = parent.split("_")
        if len(parts) < 2:
            raise ValueError(f"AliMeeting recording id has no session stem: {recording_id}")
        return f"{parts[0]}_{parts[1]}"
    return parent


def load_component_refs(source: str) -> tuple[RecordingRef, ...]:
    """Load audio, UEM, and duration from the admitted training component."""

    spec = SOURCE_SPECS[source]
    payload = read_json(TRAINING_VIEW / "components" / f"{spec['component']}.json")
    refs: list[RecordingRef] = []
    for record in payload["records"]:
        recording_id = record["recording_id"]
        uem_path = Path(record["uem"]["source_absolute_path"])
        uem_start_ns, uem_end_ns = uem_bounds(uem_path.read_text(encoding="utf-8"))
        duration_samples = int(record["duration_samples"])
        duration_ns = duration_samples * NANOSECONDS_PER_SECOND // SAMPLE_RATE
        refs.append(
            RecordingRef(
                recording_id=recording_id,
                audio_path=Path(record["audio"]["source_absolute_path"]),
                uem_path=uem_path,
                duration_ns=duration_ns,
                uem_start_ns=uem_start_ns,
                uem_end_ns=uem_end_ns,
            )
        )
    refs.sort(key=lambda item: item.recording_id)
    return tuple(refs)


def load_current_labels(prefixes: Iterable[str]) -> dict[str, list[SpeakerSegment]]:
    """Parse current training-view SPEAKER rows for the given prefixes."""

    wanted = tuple(prefixes)
    grouped: dict[str, list[SpeakerSegment]] = defaultdict(list)
    with ANNOTATIONS_PATH.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.startswith("SPEAKER "):
                continue
            if not any(raw.startswith(f"SPEAKER {prefix}") for prefix in wanted):
                continue
            for segment in parse_speaker_rttm(raw):
                grouped[segment.recording_id].append(segment)
    return grouped


def load_published_labels(path: Path, recording_id: str) -> RecordingLabels:
    """Parse one published RTTM onto the training recording id."""

    text = path.read_text(encoding="utf-8")
    segments = parse_speaker_rttm(text, recording_id=recording_id)
    return RecordingLabels(recording_id, segments)


def current_recording(recording_id: str, segments: Sequence[SpeakerSegment]) -> RecordingLabels:
    """Build original labels for one recording."""

    return RecordingLabels(recording_id, tuple(segments))


def render_gate_markdown(source_name: str, release_id: str, gate: GateResult, *, kept_current: bool) -> str:
    """Render a human-readable gate report."""

    tightness = gate.tightness
    original = gate.original_tightness
    reference = FORCED_AMI_REFERENCE
    status = "PASS" if gate.passed and not kept_current else "FAIL"
    lines = [
        f"# {source_name} tight labels ({release_id})",
        "",
        f"Status: {status}",
        "",
        "## Tightness",
        "",
        "| Measure | Original | Produced | Forced AMI reference |",
        "|---|---:|---:|---:|",
        _tightness_row(
            "Gaps ≤0.3 s / min",
            original.gaps_le_0_3_per_min,
            tightness.gaps_le_0_3_per_min,
            reference.gaps_le_0_3_per_min,
        ),
        _tightness_row(
            "Gaps 0.3–1.0 s / min",
            original.gaps_0_3_to_1_per_min,
            tightness.gaps_0_3_to_1_per_min,
            reference.gaps_0_3_to_1_per_min,
        ),
        _tightness_row("Smallest gap s", original.smallest_gap_s, tightness.smallest_gap_s, reference.smallest_gap_s),
        _tightness_row(
            "Median segment s", original.median_segment_s, tightness.median_segment_s, reference.median_segment_s
        ),
        _tightness_row("p90 segment s", original.p90_segment_s, tightness.p90_segment_s, reference.p90_segment_s),
        "",
        "## Hard gates",
        "",
        "These fail the release. A failed source keeps the current labels only",
        "when a hard gate or the failure-share limit fails.",
        "",
        f"- Rows inside UEM: {_ok(gate.uem_ok)}",
        f"- Subset of 200 ms-closed original: {_ok(gate.subset_ok)}",
        f"- Smallest same-speaker gap ≥ 0.200 s: {_ok(gate.smallest_gap_ok)}",
        f"- Published speakers mapped: {_ok(gate.unmapped_speakers_ok)}",
        f"- UNCERTAIN overlapping same-speaker speech: {_ok(gate.uncertain_overlap_ok)}",
        f"- Failure share ≤ {gate.failure_share_limit:.2f}: {_ok(gate.failure_share_ok)} ({gate.failure_share:.4f})",
        "",
        "## Reported (do not fail)",
        "",
        "Tightness rates are compared with the forced-AMI reference of 4.63",
        "gaps ≤0.3 s and 17.00 gaps 0.3–1.0 s per speaker-minute.",
        "",
        f"- Tightness band: {_ok(gate.tightness_ok)}",
        f"- Speaker retention: {_ok(gate.speaker_retention_ok)}",
        f"- Overlap retention: {_ok(gate.overlap_retention_ok)} ({_fmt(gate.overlap_retention)})",
        f"- UNCERTAIN share of original speech: {_fmt(gate.uncertain_share)}",
        f"- UNCERTAIN share excluding by-design: {_fmt(gate.uncertain_share_excluding_by_design)}",
        f"- By-design UNCERTAIN share of original segments: {_fmt(gate.by_design_share)}",
        "",
        f"Flagged speakers below {gate.speaker_retention_floor:.0%} retention: {len(gate.flagged_speakers)}",
        f"Flagged recordings with >5% new speech outside original: {len(gate.flagged_recordings)}",
        f"Rows outside UEM: {len(gate.rows_outside_uem)}",
        f"Unmapped published speakers: {len(gate.unmapped_published)}",
        f"Dropped unmapped published speech: {gate.dropped_unmapped_seconds:.3f} s",
        "",
    ]
    if gate.uncertain_by_recording:
        lines.append("## UNCERTAIN share by recording")
        lines.append("")
        for item in gate.uncertain_by_recording:
            lines.append(
                f"- `{item.recording_id}` {item.ratio:.4f} "
                f"({item.uncertain_seconds:.3f} s of {item.original_seconds:.3f} s)"
            )
        lines.append("")
    if kept_current:
        lines.append(
            "The produced labels are the current labels because the source "
            "failed a hard gate or the failure-share limit."
        )
        lines.append("")
    if gate.notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in gate.notes)
        lines.append("")
    if gate.flagged_recordings:
        lines.append("## Flagged recordings")
        lines.append("")
        lines.extend(f"- `{item}`" for item in gate.flagged_recordings)
        lines.append("")
    if gate.unmapped_published:
        lines.append("## Unmapped published speakers")
        lines.append("")
        lines.extend(f"- `{item}`" for item in gate.unmapped_published)
        lines.append("")
    if gate.flagged_speakers:
        lines.append("## Flagged speakers")
        lines.append("")
        for item in gate.flagged_speakers:
            lines.append(
                f"- `{item.recording_id}` `{item.speaker}` retention {item.ratio:.3f} "
                f"({item.kept_seconds:.3f} s of {item.original_seconds:.3f} s)"
            )
        lines.append("")
    return "\n".join(lines)


def _tightness_row(name: str, original: float | None, produced: float | None, reference: float | None) -> str:
    return f"| {name} | {_fmt(original)} | {_fmt(produced)} | {_fmt(reference)} |"


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _ok(value: bool | None) -> str:
    if value is None:
        return "not required"
    return "pass" if value else "fail"


def write_release(
    *,
    source: str,
    refs: Sequence[RecordingRef],
    original: Sequence[RecordingLabels],
    produced: Sequence[RecordingLabels],
    mappings: Mapping[str, SpeakerMapping],
    gate: GateResult,
    kept_current: bool,
    published_used: dict[str, Any] | None,
) -> Path:
    """Write one local tight-v2 release directory."""

    spec = SOURCE_SPECS[source]
    release_dir = DV_ROOT / "releases" / spec["release_id"]
    if release_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release {release_dir}")
    labels_dir = release_dir / "labels"
    labels_dir.mkdir(parents=True)
    uem_by_id = {item.recording_id: item.uem_path for item in refs}
    all_rttm: list[str] = []
    all_uem: list[str] = []
    for labels in produced:
        write_recording_labels(labels_dir, labels)
        uem_text = rewrite_uem_file_id(
            uem_by_id[labels.recording_id].read_text(encoding="utf-8"),
            labels.recording_id,
        )
        (labels_dir / f"{labels.recording_id}.uem").write_text(uem_text, encoding="utf-8")
        all_rttm.append(render_speaker_rttm(labels.segments))
        all_uem.append(uem_text if uem_text.endswith("\n") else uem_text + "\n")
    (release_dir / "all.rttm").write_text("".join(all_rttm), encoding="utf-8")
    (release_dir / "all.uem").write_text("".join(all_uem), encoding="utf-8")
    provenance = {
        "schema": "speakrs-tight-labels-release",
        "schema_version": 2,
        "release_id": spec["release_id"],
        "source": spec["source_name"],
        "method": spec["method"] if not kept_current else "keep-current-labels",
        "kept_current_labels": kept_current,
        "library_version": LIBRARY_VERSION,
        "library_commit": library_commit(),
        "min_gap_seconds": MIN_GAP_SECONDS,
        "min_gap_ns": MIN_GAP_NS,
        "supersedes": V1_RELEASES[source],
        "recording_count": len(produced),
        "licenses": [spec["license"]],
        "published_labels": published_used,
        "speaker_mappings": {recording_id: mapping.to_json() for recording_id, mapping in sorted(mappings.items())},
        "gate_results": gate.to_json(),
    }
    write_json(release_dir / "provenance.json", provenance)
    write_json(release_dir / "gate-report.json", gate.to_json())
    (release_dir / "gate-report.md").write_text(
        render_gate_markdown(spec["source_name"], spec["release_id"], gate, kept_current=kept_current),
        encoding="utf-8",
    )
    return release_dir


def produce_source(
    source: str,
    current_by_id: Mapping[str, list[SpeakerSegment]],
) -> dict[str, Any]:
    """Produce one source release, listening pack, and gate report."""

    spec = SOURCE_SPECS[source]
    refs = load_component_refs(source)
    original: list[RecordingLabels] = []
    produced: list[RecordingLabels] = []
    mappings: dict[str, SpeakerMapping] = {}
    uem_by_recording: dict[str, tuple[int, int]] = {}
    published_used: dict[str, Any] | None = None
    if spec["published_split"]:
        commit = published_commit()
        if commit != PUBLISHED_COMMIT:
            raise RuntimeError(f"published-label commit is {commit}, expected {PUBLISHED_COMMIT}")
        published_used = {
            "repository": PUBLISHED_REPO,
            "commit": commit,
            "split": spec["published_split"],
            "license": spec["license"],
        }
    for ref in refs:
        source_labels = current_recording(ref.recording_id, current_by_id[ref.recording_id])
        original.append(source_labels)
        uem = (ref.uem_start_ns, ref.uem_end_ns)
        uem_by_recording[ref.recording_id] = uem
        if spec["method"] == "merge-short-gaps":
            clipped = clip_recording_to_uem(source_labels, ref.uem_start_ns, ref.uem_end_ns)
            produced.append(merge_recording(clipped))
            continue
        parent = published_parent_id(ref.recording_id, source)
        published_path = PUBLISHED_ROOT / spec["published_split"] / f"{parent}.rttm"
        if not published_path.is_file():
            raise FileNotFoundError(f"missing published train labels: {published_path}")
        published = load_published_labels(published_path, ref.recording_id)
        replaced, mapping = apply_published_replacement(source_labels, published, uem=uem)
        produced.append(replaced)
        mappings[ref.recording_id] = mapping

    attempted = list(produced)
    gate = evaluate_gates(
        original,
        attempted,
        require_subset=spec["require_subset"],
        uem_by_recording=uem_by_recording,
        mappings=mappings or None,
    )
    kept_current = not gate.passed
    written = list(original) if kept_current else attempted

    release_dir = write_release(
        source=source,
        refs=refs,
        original=original,
        produced=written,
        mappings=mappings,
        gate=gate,
        kept_current=kept_current,
        published_used=published_used,
    )
    listening_dir = STAGING_ROOT / "listening-v2" / spec["listening_name"]
    listening = write_listening_pack(
        listening_dir,
        original=original,
        new=attempted,
        audio_by_recording={item.recording_id: item.audio_path for item in refs},
        duration_by_recording={item.recording_id: item.duration_ns for item in refs},
    )
    return {
        "source": source,
        "release_id": spec["release_id"],
        "release_dir": str(release_dir),
        "recording_count": len(produced),
        "kept_current_labels": kept_current,
        "gate": gate,
        "listening": listening,
        "original": original,
        "produced": produced,
    }


def render_core_report(results: Sequence[Mapping[str, Any]]) -> str:
    """Render the cross-source CORE-REPORT for v2."""

    lines = [
        "# Tight labels v2 core report",
        "",
        "v1 is superseded. The v1 releases kept published rows that ran past",
        "the UEM, and float `start + duration` merged same-speaker gaps of",
        "exactly 200 ms. v2 clips every produced row to the UEM before merge,",
        "does interval arithmetic in nanoseconds, and fails a release when a",
        "row falls outside the UEM or a published speaker is unmapped.",
        "",
        "v1 paths remain on disk and are not modified:",
        "",
        "- `/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/ami-tight-v1`",
        "- `/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/alimeeting-tight-v1`",
        "- `/Volumes/CacheDisk/dev-cache/diarization-data-verification/releases/ava-avd-tight-v1`",
        "- `/Volumes/CacheDisk/dev-cache/diarization-data-verification/staging/tight-labels-v1/listening/`",
        "",
        "This report records tightness before and after the v2 tight-label",
        "wave (AMI, AliMeeting, AVA-AVD) next to the published forced-AMI",
        "reference. Human listening of the clip packs is still required.",
        "",
        f"Library version: {LIBRARY_VERSION}",
        f"Library commit: {library_commit()}",
        f"Published labels: {PUBLISHED_REPO} `{PUBLISHED_COMMIT}`",
        "",
        "## Tightness",
        "",
        "| Source | Recs | Gaps ≤0.3 s/min orig | Gaps ≤0.3 s/min new | Gaps 0.3–1 s/min orig | Gaps 0.3–1 s/min new | Smallest gap orig | Smallest gap new | Median seg orig | Median seg new | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        _core_row(
            "Forced AMI reference",
            FORCED_AMI_REFERENCE.recording_count,
            None,
            FORCED_AMI_REFERENCE,
            None,
            "reference",
        ),
    ]
    for result in results:
        gate: GateResult = result["gate"]
        status = "PASS" if gate.passed and not result["kept_current_labels"] else "FAIL"
        lines.append(
            _core_row(
                SOURCE_SPECS[result["source"]]["source_name"],
                result["recording_count"],
                gate.original_tightness,
                gate.tightness,
                result["kept_current_labels"],
                status,
            )
        )
    lines.extend(["", "## Releases and listening packs", ""])
    for result in results:
        spec = SOURCE_SPECS[result["source"]]
        gate = result["gate"]
        lines.append(f"### {spec['source_name']}")
        lines.append("")
        lines.append(f"- Release: `{result['release_dir']}`")
        lines.append(f"- Supersedes: `{(DV_ROOT / 'releases' / V1_RELEASES[result['source']])}`")
        lines.append(f"- Listening pack: `{result['listening']['directory']}`")
        lines.append(f"- Listening clips: {result['listening']['clip_count']}")
        lines.append(f"- Listening seconds: {result['listening']['total_seconds']:.1f}")
        lines.append(f"- Kept current labels: {result['kept_current_labels']}")
        lines.append(f"- Flagged recordings: {len(gate.flagged_recordings)}")
        lines.append(
            f"- Flagged speakers below {gate.speaker_retention_floor:.0%} retention: {len(gate.flagged_speakers)}"
        )
        lines.append(f"- Rows outside UEM: {len(gate.rows_outside_uem)}")
        lines.append(f"- Unmapped published speakers: {len(gate.unmapped_published)}")
        lines.append(f"- Dropped unmapped published speech: {gate.dropped_unmapped_seconds:.3f} s")
        if gate.flagged_recordings:
            lines.append("- Flagged recording ids:")
            lines.extend(f"  - `{item}`" for item in gate.flagged_recordings)
        if gate.notes:
            lines.append("- Notes:")
            lines.extend(f"  - {note}" for note in gate.notes)
        lines.append("")
    return "\n".join(lines) + "\n"


def _core_row(
    name: str,
    recs: int,
    original: TightnessStats | None,
    produced: TightnessStats,
    kept_current: bool | None,
    status: str,
) -> str:
    orig_short = "—" if original is None else f"{original.gaps_le_0_3_per_min:.2f}"
    orig_mid = "—" if original is None else f"{original.gaps_0_3_to_1_per_min:.2f}"
    orig_gap = "—" if original is None else _fmt(original.smallest_gap_s)
    orig_med = "—" if original is None else _fmt(original.median_segment_s)
    return (
        f"| {name} | {recs} | {orig_short} | {produced.gaps_le_0_3_per_min:.2f} | "
        f"{orig_mid} | {produced.gaps_0_3_to_1_per_min:.2f} | {orig_gap} | "
        f"{_fmt(produced.smallest_gap_s)} | {orig_med} | {_fmt(produced.median_segment_s)} | {status} |"
    )


def produce(sources: Sequence[str] | None = None) -> dict[str, Any]:
    """Produce the requested source releases and the core report."""

    require_free_gib(DV_ROOT, MINIMUM_FREE_GIB)
    selected = list(sources or SOURCE_SPECS)
    for source in selected:
        if source not in SOURCE_SPECS:
            raise ValueError(f"unknown source {source}")
    prefixes = [SOURCE_SPECS[source]["prefix"] for source in selected]
    current_by_id = load_current_labels(prefixes)
    results = [produce_source(source, current_by_id) for source in selected]
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    report = render_core_report(results)
    (STAGING_ROOT / "CORE-REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "ok": True,
        "library_version": LIBRARY_VERSION,
        "library_commit": library_commit(),
        "core_report": str(STAGING_ROOT / "CORE-REPORT.md"),
        "sources": [
            {
                "source": item["source"],
                "release_id": item["release_id"],
                "release_dir": item["release_dir"],
                "passed": item["gate"].passed and not item["kept_current_labels"],
                "kept_current_labels": item["kept_current_labels"],
                "listening": item["listening"],
            }
            for item in results
        ],
    }
    write_json(STAGING_ROOT / "produce-summary-v2.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run production from the command line."""

    parser = argparse.ArgumentParser(description="Produce tight-v2 label releases")
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_SPECS),
        help="Produce one source; repeat to select several. Default: all three",
    )
    args = parser.parse_args(argv)
    summary = produce(args.source)
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
