//! Review session and overlay loading with identity checks
//!
//! Deep packet validation stays in the Python `prepare` and `validate`
//! commands. This loader binds the packet through its manifest hash and
//! re-derives every overlay identity it relies on

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    canonical_json::sha256_json,
    domain::Activity,
    error::{ReviewError, ReviewResult},
    media::{MediaFiles, MediaKind},
    proposal::GridProposal,
    record::Record,
    text::{NonEmptyText, Sha256Hex},
};

const SESSION_SCHEMA: &str = "speakrs-open-yap-review-session";
const OVERLAY_SCHEMA: &str = "speakrs-open-yap-review-overlay";
const OVERLAY_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "conversion_policy",
    "frame_seconds",
    "packet",
    "archive",
    "policy",
    "privacy",
    "membership",
    "source_transcript_members",
    "windows",
    "overlay_sha256",
];

/// How a window was selected for review
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionKind {
    Uniform,
    Targeted,
}

impl SelectionKind {
    /// Return the wire name
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Targeted => "targeted",
        }
    }
}

/// Source-clock bounds of a window inside its parent conversation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceClock {
    pub start_seconds: f64,
    pub end_seconds: f64,
}

/// One validated overlay window
#[derive(Debug, Clone)]
pub struct OverlayWindow {
    pub window_id: String,
    pub parent_id: String,
    pub selection_kind: SelectionKind,
    pub stratum: Option<String>,
    pub source_clock: SourceClock,
    pub transcript_path: PathBuf,
    pub proposal_path: PathBuf,
    pub media: MediaFiles,
    pub proposal: GridProposal,
}

impl OverlayWindow {
    /// Serialize the window header sent to the UI
    pub fn header_json(&self) -> Value {
        json!({
            "window_id": self.window_id,
            "parent_id": self.parent_id,
            "selection_kind": self.selection_kind.as_str(),
            "stratum": self.stratum,
            "source_clock": {
                "start_seconds": self.source_clock.start_seconds,
                "end_seconds": self.source_clock.end_seconds,
            },
        })
    }

    /// The machine proposal activity
    pub fn proposal_activity(&self) -> &Activity {
        &self.proposal.activity
    }
}

/// A loaded review session bound to one packet and overlay
#[derive(Debug, Clone)]
pub struct ReviewSession {
    pub root: PathBuf,
    pub packet_root: PathBuf,
    pub events_dir: PathBuf,
    pub packet_hash: Sha256Hex,
    pub overlay_hash: Sha256Hex,
    windows: Vec<OverlayWindow>,
    index: HashMap<String, usize>,
}

impl ReviewSession {
    /// Load `session.json`, the overlay manifest, and verify identities
    pub fn load(path: &Path) -> ReviewResult<Self> {
        let root = canonical(path, "review session is missing")?;
        let session_value = read_json(&root.join("session.json"), "review session is unreadable")?;
        let session = Record::object(&session_value, "review session")?;
        let schema_valid = session.get("schema").as_str() == Some(SESSION_SCHEMA)
            && session.get("schema_version").as_u64() == Some(1);
        if !schema_valid {
            return Err(ReviewError::preparation("review session schema is invalid"));
        }

        let overlay_root = canonical(
            &root.join(relative_path(session.get("overlay_path"), "overlay_path")?),
            "review overlay is missing",
        )?;
        let events_dir = root.join(relative_path(
            session.get("event_store_path"),
            "event_store_path",
        )?);
        let packet_root = canonical(
            Path::new(required_str(session.get("packet_path"), "packet_path")?),
            "review packet is missing",
        )?;

        let overlay = load_overlay(&overlay_root, &packet_root)?;
        if Some(overlay.overlay_hash.as_str()) != session.get("overlay_sha256").as_str() {
            return Err(ReviewError::preparation(
                "session overlay hash does not match the overlay manifest",
            ));
        }
        if Some(overlay.packet_hash.as_str()) != session.get("packet_manifest_sha256").as_str() {
            return Err(ReviewError::preparation("session packet manifest changed"));
        }

        let index = overlay
            .windows
            .iter()
            .enumerate()
            .map(|(position, window)| (window.window_id.clone(), position))
            .collect();
        Ok(Self {
            root,
            packet_root,
            events_dir,
            packet_hash: overlay.packet_hash,
            overlay_hash: overlay.overlay_hash,
            windows: overlay.windows,
            index,
        })
    }

    /// Windows in overlay manifest order
    pub fn windows(&self) -> &[OverlayWindow] {
        &self.windows
    }

    /// Return the position of a window in manifest order
    pub fn position(&self, window_id: &str) -> Option<usize> {
        self.index.get(window_id).copied()
    }

    /// Return one window or an unknown-window error
    pub fn window(&self, window_id: &str) -> ReviewResult<&OverlayWindow> {
        self.position(window_id)
            .map(|position| &self.windows[position])
            .ok_or_else(|| ReviewError::UnknownWindow {
                window_id: window_id.to_owned(),
            })
    }
}

struct LoadedOverlay {
    overlay_hash: Sha256Hex,
    packet_hash: Sha256Hex,
    windows: Vec<OverlayWindow>,
}

fn load_overlay(overlay_root: &Path, session_packet: &Path) -> ReviewResult<LoadedOverlay> {
    let manifest_value = read_json(
        &overlay_root.join("overlay-manifest.json"),
        "review overlay manifest is unreadable",
    )?;
    let manifest = Record::object(&manifest_value, "overlay manifest")?;
    let schema_valid = manifest.get("schema").as_str() == Some(OVERLAY_SCHEMA)
        && manifest.get("schema_version").as_u64() == Some(1);
    if !schema_valid {
        return Err(ReviewError::preparation("review overlay schema is invalid"));
    }
    manifest.exact(OVERLAY_FIELDS, &[])?;

    let Some(windows) = manifest
        .get("windows")
        .as_array()
        .filter(|items| !items.is_empty())
    else {
        return Err(ReviewError::preparation("review overlay has no windows"));
    };
    let packet_info = Record::object(manifest.get("packet"), "overlay packet")?;
    let bound_packet = canonical(
        Path::new(required_str(packet_info.get("path"), "packet.path")?),
        "overlay packet is missing",
    )?;
    if bound_packet != session_packet {
        return Err(ReviewError::preparation(
            "overlay packet path does not match the session packet",
        ));
    }
    let packet_hash = Sha256Hex::from_digest(sha256_file(&bound_packet.join("manifest.json"))?);
    if Some(packet_hash.as_str()) != packet_info.get("manifest_sha256").as_str() {
        return Err(ReviewError::preparation(
            "original packet manifest changed after overlay preparation",
        ));
    }

    let candidates = packet_candidates(&bound_packet)?;
    if windows.len() != candidates.len() {
        return Err(ReviewError::preparation(
            "overlay window membership does not match the packet",
        ));
    }

    let mut seen = HashSet::new();
    let mut loaded = Vec::with_capacity(windows.len());
    for value in windows {
        let window = load_window(value, overlay_root, &bound_packet, &candidates)?;
        if !seen.insert(window.window_id.clone()) {
            return Err(ReviewError::preparation(
                "overlay window IDs must be unique",
            ));
        }
        loaded.push(window);
    }

    let mut identity = manifest.map().clone();
    let stored_hash = identity.remove("overlay_sha256");
    let overlay_hash = Sha256Hex::from_digest(sha256_json(&Value::Object(identity)));
    if let Some(stored) = stored_hash
        && stored.as_str() != Some(overlay_hash.as_str())
    {
        return Err(ReviewError::preparation(
            "overlay identity hash does not match",
        ));
    }

    Ok(LoadedOverlay {
        overlay_hash,
        packet_hash,
        windows: loaded,
    })
}

fn packet_candidates(packet_root: &Path) -> ReviewResult<HashMap<String, String>> {
    let manifest = read_json(
        &packet_root.join("manifest.json"),
        "review packet manifest is unreadable",
    )?;
    let Some(windows) = manifest
        .get("windows")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
    else {
        return Err(ReviewError::preparation("review packet has no windows"));
    };
    let mut candidates = HashMap::with_capacity(windows.len());
    for item in windows {
        let record = Record::object(item, "packet window")?;
        let window = Record::object(record.get("window"), "packet window.window")?;
        let files = Record::object(record.get("files"), "packet window.files")?;
        let window_id = required_str(window.get("window_id"), "packet window_id")?;
        let candidate = required_str(
            files.get("candidate_annotation"),
            "packet candidate_annotation",
        )?;
        candidates.insert(window_id.to_owned(), candidate.to_owned());
    }
    Ok(candidates)
}

fn load_window(
    value: &Value,
    overlay_root: &Path,
    packet_root: &Path,
    candidates: &HashMap<String, String>,
) -> ReviewResult<OverlayWindow> {
    let window = Record::object(value, "overlay window")?;
    let window_id = NonEmptyText::parse(window.get("window_id"), "window_id")?
        .as_str()
        .to_owned();
    let Some(candidate_relative) = candidates.get(&window_id) else {
        return Err(ReviewError::preparation_with(
            "overlay contains a window that is not in the packet",
            json!({ "window_id": window_id }),
        ));
    };

    let transcript = Record::object(window.get("transcript"), "overlay transcript")?;
    let proposal = Record::object(window.get("proposal"), "overlay proposal")?;
    let transcript_path = contained(
        overlay_root,
        transcript.get("path"),
        "overlay artifact path is unsafe",
    )?;
    let proposal_path = contained(
        overlay_root,
        proposal.get("path"),
        "overlay artifact path is unsafe",
    )?;
    if Some(sha256_file(&transcript_path)?.as_str()) != transcript.get("sha256").as_str() {
        return Err(ReviewError::preparation(
            "overlay transcript hash does not match",
        ));
    }
    if Some(sha256_file(&proposal_path)?.as_str()) != proposal.get("sha256").as_str() {
        return Err(ReviewError::preparation(
            "overlay proposal hash does not match",
        ));
    }

    let transcript_value = read_json(&transcript_path, "overlay transcript is unreadable")?;
    let display_only = transcript_value.get("display_only") == Some(&Value::Bool(true))
        && transcript_value.get("not_human_activity") == Some(&Value::Bool(true));
    if !display_only {
        return Err(ReviewError::preparation(
            "overlay transcript must remain a display aid",
        ));
    }

    let candidate_path = contained(
        packet_root,
        &Value::String(candidate_relative.clone()),
        "candidate annotation path is unsafe",
    )?;
    let candidate_sha256 = Sha256Hex::from_digest(sha256_file(&candidate_path)?);
    if Some(candidate_sha256.as_str()) != window.get("candidate_sha256").as_str() {
        return Err(ReviewError::preparation(
            "overlay candidate hash does not match the packet file",
        ));
    }
    let candidate = read_json(&candidate_path, "candidate annotation is unreadable")?;
    let rebuilt = GridProposal::from_candidate(&candidate, &window_id, &candidate_sha256)?;
    let stored_proposal = read_json(&proposal_path, "overlay proposal is unreadable")?;
    if stored_proposal
        .get("content_sha256")
        .and_then(Value::as_str)
        != Some(rebuilt.content_sha256.as_str())
    {
        return Err(ReviewError::preparation(
            "overlay proposal is not a deterministic conversion of the candidate",
        ));
    }

    let packet_files = Record::object(window.get("packet_files"), "overlay packet_files")?;
    let media_path = |kind: MediaKind| {
        let key = kind.packet_file_key();
        if !packet_files.get(key).is_string() {
            return Err(ReviewError::preparation_with(
                "packet media path is missing",
                json!({ "window_id": window_id, "kind": kind.as_str() }),
            ));
        }
        contained(
            packet_root,
            packet_files.get(key),
            "packet media path is unsafe or missing",
        )
    };
    let media = MediaFiles {
        emitted: media_path(MediaKind::Emitted)?,
        speaker_a: media_path(MediaKind::SpeakerA)?,
        speaker_b: media_path(MediaKind::SpeakerB)?,
    };

    let clock = Record::object(window.get("source_clock"), "overlay source_clock")?;
    let source_clock = SourceClock {
        start_seconds: finite(clock.get("start_seconds"), "source_clock.start_seconds")?,
        end_seconds: finite(clock.get("end_seconds"), "source_clock.end_seconds")?,
    };
    let selection_kind = match window.get("selection_kind").as_str() {
        Some("uniform") => SelectionKind::Uniform,
        Some("targeted") => SelectionKind::Targeted,
        _ => {
            return Err(ReviewError::preparation_with(
                "overlay window selection_kind is unknown",
                json!({ "window_id": window_id }),
            ));
        }
    };
    let stratum = match window.get("stratum") {
        Value::Null => None,
        Value::String(stratum) => Some(stratum.clone()),
        _ => {
            return Err(ReviewError::preparation(
                "overlay window stratum must be a string or null",
            ));
        }
    };

    Ok(OverlayWindow {
        parent_id: required_str(window.get("parent_id"), "parent_id")?.to_owned(),
        window_id,
        selection_kind,
        stratum,
        source_clock,
        transcript_path,
        proposal_path,
        media,
        proposal: rebuilt,
    })
}

/// Return the lowercase SHA-256 hex digest of a file
pub fn sha256_file(path: &Path) -> ReviewResult<String> {
    let mut file = File::open(path).map_err(|error| ReviewError::io("open", path, error))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| ReviewError::io("read", path, error))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex::encode(digest.finalize()))
}

/// Read and parse a JSON file, reporting failures as preparation errors
pub fn read_json(path: &Path, message: &str) -> ReviewResult<Value> {
    let bytes = fs::read(path).map_err(|error| unreadable(message, path, &error.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|error| unreadable(message, path, &error.to_string()))
}

fn unreadable(message: &str, path: &Path, reason: &str) -> ReviewError {
    ReviewError::preparation_with(
        message,
        json!({ "path": path.display().to_string(), "reason": reason }),
    )
}

fn canonical(path: &Path, message: &str) -> ReviewResult<PathBuf> {
    path.canonicalize()
        .map_err(|error| unreadable(message, path, &error.to_string()))
}

/// Resolve a relative path strictly inside a root and require an existing file
fn contained(root: &Path, relative: &Value, message: &str) -> ReviewResult<PathBuf> {
    let unsafe_path = || ReviewError::preparation_with(message, json!({ "path": relative }));
    let relative = relative.as_str().ok_or_else(unsafe_path)?;
    let resolved = root
        .join(relative)
        .canonicalize()
        .map_err(|_| unsafe_path())?;
    if !resolved.starts_with(root) || resolved == root || !resolved.is_file() {
        return Err(unsafe_path());
    }
    Ok(resolved)
}

fn relative_path(value: &Value, label: &str) -> ReviewResult<PathBuf> {
    let text = required_str(value, label)?;
    let path = PathBuf::from(text);
    if !path
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(ReviewError::preparation_with(
            format!("session {label} must be a plain relative path"),
            json!({ "path": text }),
        ));
    }
    Ok(path)
}

fn required_str<'a>(value: &'a Value, label: &str) -> ReviewResult<&'a str> {
    value
        .as_str()
        .ok_or_else(|| ReviewError::preparation(format!("{label} must be a string")))
}

fn finite(value: &Value, label: &str) -> ReviewResult<f64> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| ReviewError::preparation(format!("{label} must be a finite number")))
}
