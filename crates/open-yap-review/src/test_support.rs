//! Synthetic review sessions for unit tests

use std::{fs, path::Path};

use serde_json::{Value, json};

use crate::{
    canonical_json::{sha256_hex, sha256_json},
    proposal::GridProposal,
    text::Sha256Hex,
};

/// Window ids created by [`write_session`], in manifest order
pub const WINDOW_IDS: [&str; 2] = ["w1", "w2"];

fn write_json(path: &Path, value: &Value) -> String {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let text = serde_json::to_string_pretty(value).unwrap();
    fs::write(path, &text).unwrap();
    sha256_hex(text.as_bytes())
}

/// Write a packet, overlay, and session with two windows and return the session root
pub fn write_session(root: &Path) -> std::path::PathBuf {
    let packet = root.join("packet");
    let session = root.join("session");
    let overlay = session.join("overlay");

    let mut packet_windows = Vec::new();
    let mut overlay_windows = Vec::new();
    for (index, window_id) in WINDOW_IDS.into_iter().enumerate() {
        let window_dir = format!("windows/{window_id}");
        let candidate = json!({
            "window_id": window_id,
            "speaker_activity": [
                {"speaker_role": "speaker_a", "intervals": [{"start_seconds": 0.1, "end_seconds": 0.5}]},
                {"speaker_role": "speaker_b", "intervals": [{"start_seconds": 0.4, "end_seconds": 0.9}]},
            ],
        });
        let candidate_relative = format!("{window_dir}/candidate-annotation.json");
        let candidate_sha256 = write_json(&packet.join(&candidate_relative), &candidate);
        for name in [
            "emitted.flac",
            "reference-speaker_a.flac",
            "reference-speaker_b.flac",
        ] {
            fs::write(packet.join(&window_dir).join(name), b"fLaC").unwrap();
        }
        packet_windows.push(json!({
            "window": {"window_id": window_id},
            "files": {"candidate_annotation": candidate_relative},
        }));

        let digest = Sha256Hex::parse_str(&candidate_sha256, "candidate").unwrap();
        let proposal = GridProposal::from_candidate(&candidate, window_id, &digest).unwrap();
        let transcript_relative = format!("{window_dir}/transcript.json");
        let proposal_relative = format!("{window_dir}/proposal.json");
        let transcript_sha256 = write_json(
            &overlay.join(&transcript_relative),
            &json!({"display_only": true, "not_human_activity": true, "words": []}),
        );
        let proposal_sha256 = write_json(
            &overlay.join(&proposal_relative),
            &json!({"intervals": proposal.activity.to_json(), "content_sha256": proposal.content_sha256.as_str()}),
        );
        overlay_windows.push(json!({
            "window_id": window_id,
            "parent_id": format!("conv_{index}"),
            "selection_kind": if index == 0 { "uniform" } else { "targeted" },
            "stratum": if index == 0 { Value::Null } else { json!("overlap") },
            "source_clock": {"start_seconds": 10.0, "end_seconds": 40.0},
            "packet_files": {
                "candidate_annotation": candidate_relative,
                "emitted": format!("{window_dir}/emitted.flac"),
                "reference_speaker_a": format!("{window_dir}/reference-speaker_a.flac"),
                "reference_speaker_b": format!("{window_dir}/reference-speaker_b.flac"),
            },
            "candidate_sha256": candidate_sha256,
            "transcript": {"path": transcript_relative, "sha256": transcript_sha256},
            "proposal": {"path": proposal_relative, "sha256": proposal_sha256},
        }));
    }

    let packet_manifest_sha256 = write_json(
        &packet.join("manifest.json"),
        &json!({"windows": packet_windows}),
    );
    let mut manifest = json!({
        "schema": "speakrs-open-yap-review-overlay",
        "schema_version": 1,
        "frame_seconds": 0.02,
        "packet": {"path": packet.canonicalize().unwrap().display().to_string(), "manifest_sha256": packet_manifest_sha256},
        "windows": overlay_windows,
    });
    let overlay_sha256 = sha256_json(&manifest);
    manifest["overlay_sha256"] = json!(overlay_sha256);
    write_json(&overlay.join("overlay-manifest.json"), &manifest);
    write_json(
        &session.join("session.json"),
        &json!({
            "schema": "speakrs-open-yap-review-session",
            "schema_version": 1,
            "packet_path": packet.canonicalize().unwrap().display().to_string(),
            "overlay_path": "overlay",
            "event_store_path": "events",
            "packet_manifest_sha256": packet_manifest_sha256,
            "overlay_sha256": overlay_sha256,
        }),
    );
    session
}
