"""Protect the portable cold-read and deletion boundaries end to end"""

from pathlib import Path

import pytest
import soundfile as sf

import recipes.speakrs.large.data as data_module
import recipes.speakrs.large.storage as storage_module
from recipes.speakrs.large.acceptance import parse_rttm
from recipes.speakrs.large.contracts import parse_data_preparation_spec
from recipes.speakrs.large.data import evict_data, restore_check, upload_data, verify_data, verify_remote
from recipes.speakrs.large.errors import PreparationError
from recipes.speakrs.large.hashing import sha256_file
from recipes.speakrs.large.jsonio import write_json
from recipes.speakrs.large.prepare import prepare_parent
from recipes.speakrs.large.storage import MemoryBackend, directory_bytes
from recipes.speakrs.tests.test_selection_verification import selection as selection_fixture


@pytest.fixture
def pipeline(tmp_path):
    _, raw, release, manifest = selection_fixture.__wrapped__(tmp_path)
    raw["disk"]["max_cache_bytes"] = 16 * 1024 * 1024
    raw["disk"]["max_staging_bytes"] = 16 * 1024 * 1024
    raw["disk"]["staging_root"] = str(release)
    spec = parse_data_preparation_spec(raw)
    return spec, raw, release, manifest, MemoryBackend()


def _commit(pipeline):
    spec, _, release, _, store = pipeline
    verify_data(spec, release, release / "acceptance.json")
    upload_data(spec, release, release / "upload.json", backend=store)
    verify_remote(
        spec,
        release / "upload.json",
        release / "remote.json",
        backend=store,
        label_policy_id="human-gold-v1",
        split_id="published-v1",
    )
    return restore_check(spec, release / "remote.json", release / "restore.json", backend=store)


def test_cold_restore_and_receipted_cleanup(pipeline):
    spec, _, release, _, store = pipeline
    restored = _commit(pipeline)
    assert {sample["chunk_seconds"] for sample in restored["samples"]} == {8, 16}
    assert all(Path(sample["source_path_used"]).is_relative_to(spec.disk.cache_root) for sample in restored["samples"])
    result = evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert len(result["evicted"]) == 8
    assert all(not Path(item["path"]).exists() for item in result["evicted"])
    assert evict_data(spec, release / "restore.json", release / "evict.json", backend=store) == result


def test_cold_restore_preserves_uem_gaps(pipeline):
    _, _, release, manifest, _ = pipeline
    row = manifest["recordings"][0]
    label = Path(row["rttm"]["path"])
    label.write_text(
        "SPEAKER ES2002a 1 1 17 <NA> <NA> speaker-1 <NA> <NA>\nSPEAKER ES2002a 1 22 16 <NA> <NA> speaker-1 <NA> <NA>\n"
    )
    uem = Path(row["uem"]["path"])
    uem.write_text("ES2002a 1 0 20\nES2002a 1 21 40\n")
    row["rttm"]["sha256"] = sha256_file(label)
    row["uem"]["sha256"] = sha256_file(uem)
    write_json(release / "selection.json", manifest)
    restored = _commit(pipeline)
    assert all(sample["uem_regions"] == [(0.0, 20.0), (21.0, 40.0)] for sample in restored["samples"])
    import json

    accepted = json.loads((release / "acceptance.json").read_text())
    assert accepted["hours"]["unknown_hours"] == pytest.approx(1 / 3600)
    assert {row["chunks"] for row in accepted["capacity"] if row["chunk_seconds"] == 8} == {4}


def test_cold_restore_resumes_one_owned_cache_after_mid_restore_interruption(pipeline, monkeypatch):
    spec, _, release, _, store = pipeline
    verify_data(spec, release, release / "acceptance.json")
    upload_data(spec, release, release / "upload.json", backend=store)
    verify_remote(
        spec,
        release / "upload.json",
        release / "remote.json",
        backend=store,
        label_policy_id="human-gold-v1",
        split_id="published-v1",
    )
    output = release / "interrupted-restore.json"
    attempt_path = data_module._restore_attempt_path(output)
    existing_caches = set(spec.disk.cache_root.glob("cold-*"))
    real_restore_object = data_module.restore_object
    calls = {"count": 0}

    def interrupt_after_one(*args, **kwargs):
        calls["count"] += 1
        assert attempt_path.is_file()
        if calls["count"] == 2:
            raise PreparationError("simulated mid-restore interruption")
        return real_restore_object(*args, **kwargs)

    monkeypatch.setattr(data_module, "restore_object", interrupt_after_one)
    with pytest.raises(PreparationError, match="simulated mid-restore interruption"):
        restore_check(spec, release / "remote.json", output, backend=store)

    assert not output.exists()
    assert attempt_path.is_file()
    attempt = data_module.read_json(attempt_path)
    cache = Path(attempt["cache_path"])
    assert cache.is_dir()
    assert set(spec.disk.cache_root.glob("cold-*")) - existing_caches == {cache}
    assert directory_bytes(spec.disk.cache_root) <= spec.disk.max_cache_bytes

    monkeypatch.setattr(data_module, "restore_object", real_restore_object)
    resumed = restore_check(spec, release / "remote.json", output, backend=store)
    assert output.is_file()
    assert not attempt_path.exists()
    assert all(Path(sample["source_path_used"]).is_relative_to(cache) for sample in resumed["samples"])
    assert set(spec.disk.cache_root.glob("cold-*")) - existing_caches == {cache}
    assert directory_bytes(spec.disk.cache_root) <= spec.disk.max_cache_bytes


def test_changed_source_configuration_cannot_restore_old_batch(pipeline):
    _, raw, release, _, store = pipeline
    _commit(pipeline)
    raw["release_id"] = "different-release"
    changed = parse_data_preparation_spec(raw)
    with pytest.raises(PreparationError, match="accepted source/profile"):
        restore_check(changed, release / "remote.json", release / "changed.json", backend=store)


def test_unreceipted_missing_copy_cannot_pass_cleanup(pipeline):
    spec, _, release, manifest, store = pipeline
    _commit(pipeline)
    Path(manifest["recordings"][0]["audio"]["path"]).unlink()
    with pytest.raises(PreparationError, match="no matching completed deletion"):
        evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert Path(manifest["recordings"][0]["rttm"]["path"]).is_file()


def test_eviction_wrapper_recovers_unlink_before_completion(pipeline, monkeypatch):
    spec, _, release, _, store = pipeline
    _commit(pipeline)
    complete = storage_module._persist_deletion_completion

    def interrupt_completion(intent, receipt):
        raise RuntimeError("interrupted after unlink")

    monkeypatch.setattr(storage_module, "_persist_deletion_completion", interrupt_completion)
    with pytest.raises(RuntimeError, match="interrupted after unlink"):
        evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert not (release / "evict.json").exists()
    monkeypatch.setattr(storage_module, "_persist_deletion_completion", complete)
    resumed = evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert len(resumed["evicted"]) == 8
    assert all(not Path(item["path"]).exists() for item in resumed["evicted"])
    assert evict_data(spec, release / "restore.json", release / "evict.json", backend=store) == resumed


def test_directory_cannot_replace_an_eviction_candidate(pipeline):
    spec, _, release, manifest, store = pipeline
    _commit(pipeline)
    path = Path(manifest["recordings"][0]["audio"]["path"])
    path.unlink()
    path.mkdir()
    with pytest.raises(PreparationError, match="changed from a regular file"):
        evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert Path(manifest["recordings"][0]["rttm"]["path"]).is_file()


def test_real_transform_proof_resumes_raw_and_canonical_cleanup(pipeline, monkeypatch):
    spec, _, release, manifest, store = pipeline
    old = manifest["recordings"][0]
    canonical = Path(old["audio"]["path"])
    waveform, rate = sf.read(canonical, dtype="int16")
    raw = release / "original.wav"
    sf.write(raw, waveform, rate, subtype="PCM_16")
    intervals = parse_rttm(Path(old["rttm"]["path"]).read_text())
    for field in ("audio", "rttm", "uem"):
        Path(old[field]["path"]).unlink()
    prepared = prepare_parent(
        raw,
        0,
        sha256_file(raw),
        intervals,
        [{"start": 0.0, "end": 40.0}],
        canonical,
        spec.disk,
        parent_id=old["recording_id"],
    )
    row = prepared["recording"]
    row["known_speakers"] = old["known_speakers"]
    manifest["recordings"] = [row]
    transform = release / "time-transform.json"
    write_json(
        transform, {"schema": "speakrs-source-transforms-v1", "source": "AMI", "parents": [prepared["receipt"]]}
    )
    manifest["evidence"]["time_transform"] = {"path": str(transform), "sha256": sha256_file(transform)}
    write_json(release / "selection.json", manifest)
    _commit(pipeline)
    assert raw.is_file()
    complete = storage_module._persist_deletion_completion

    def interrupt_completion(intent, receipt):
        raise RuntimeError("interrupted after raw unlink")

    monkeypatch.setattr(storage_module, "_persist_deletion_completion", interrupt_completion)
    with pytest.raises(RuntimeError, match="interrupted after raw unlink"):
        evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert not raw.exists()
    assert canonical.is_file()
    monkeypatch.setattr(storage_module, "_persist_deletion_completion", complete)
    evicted = evict_data(spec, release / "restore.json", release / "evict.json", backend=store)
    assert not raw.exists()
    assert Path(evicted["source_cleanup"]["path"]).is_file()
    assert evict_data(spec, release / "restore.json", release / "evict.json", backend=store) == evicted
