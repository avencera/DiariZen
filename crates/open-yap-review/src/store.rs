//! Append-only hash-chained event store for one review session

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions, TryLockError},
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde_json::{Value, json};

use crate::{
    canonical_json::canonical_json,
    domain::{Activity, Decision, ReviewAction},
    error::{ReviewError, ReviewResult},
    event::{EventBody, ReviewEvent},
    session::ReviewSession,
    state::{ProgressEntry, ReviewProgress, WindowReviewState},
    text::{NonEmptyText, Sha256Hex},
};

/// File name of the exclusive session lock inside the session root
///
/// The lock lives outside `events/`, where the Python reader would treat it as an event file
pub const LOCK_FILE_NAME: &str = "review.lock";

const TEMPORARY_SUFFIXES: &[&str] = &[".partial", ".tmp", ".temp"];

/// An exclusive advisory lock held for the lifetime of a store
#[derive(Debug)]
pub struct SessionLock {
    _file: File,
}

impl SessionLock {
    /// Take the lock or fail when another process holds it
    pub fn acquire(session_root: &Path) -> ReviewResult<Self> {
        let path = session_root.join(LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|error| ReviewError::io("open", &path, error))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(ReviewError::SessionLocked {
                lock_path: path.display().to_string(),
            }),
            Err(TryLockError::Error(error)) => Err(ReviewError::io("lock", &path, error)),
        }
    }
}

/// One validated append request
#[derive(Debug, Clone)]
pub struct AppendRequest {
    pub window_id: String,
    pub actor: NonEmptyText,
    pub request_id: NonEmptyText,
    pub base_revision: u64,
    pub action: ReviewAction,
    /// Fixed timestamp for deterministic callers, otherwise the current UTC second
    pub timestamp_utc: Option<NonEmptyText>,
}

#[derive(Debug, Clone)]
struct HistoryEntry {
    event_index: usize,
    state: WindowReviewState,
}

#[derive(Debug, Clone, Default)]
struct WindowHistory {
    entries: Vec<HistoryEntry>,
}

impl WindowHistory {
    fn current(&self) -> WindowReviewState {
        self.entries
            .last()
            .map(|entry| entry.state.clone())
            .unwrap_or_default()
    }

    fn previous(&self) -> Option<&WindowReviewState> {
        let len = self.entries.len();
        (len >= 2).then(|| &self.entries[len - 2].state)
    }
}

/// The replayed event log of one session and the derived window states
#[derive(Debug)]
pub struct ReviewStore {
    session: Arc<ReviewSession>,
    events: Vec<ReviewEvent>,
    by_request: HashMap<String, usize>,
    histories: Vec<WindowHistory>,
    quarantined: Vec<String>,
    _lock: SessionLock,
}

impl ReviewStore {
    /// Load a session, take its lock, and replay durable events
    pub fn open(session_root: &Path) -> ReviewResult<Self> {
        let session = Arc::new(ReviewSession::load(session_root)?);
        let lock = SessionLock::acquire(&session.root)?;
        fs::create_dir_all(&session.events_dir)
            .map_err(|error| ReviewError::io("create", &session.events_dir, error))?;

        let mut store = Self {
            histories: vec![WindowHistory::default(); session.windows().len()],
            session,
            events: Vec::new(),
            by_request: HashMap::new(),
            quarantined: Vec::new(),
            _lock: lock,
        };
        store.replay()?;
        Ok(store)
    }

    fn replay(&mut self) -> ReviewResult<()> {
        let events_dir = self.session.events_dir.clone();
        let mut names: Vec<(String, PathBuf)> = fs::read_dir(&events_dir)
            .map_err(|error| ReviewError::io("list", &events_dir, error))?
            .map(|entry| {
                let entry = entry.map_err(|error| ReviewError::io("list", &events_dir, error))?;
                Ok((
                    entry.file_name().to_string_lossy().into_owned(),
                    entry.path(),
                ))
            })
            .collect::<ReviewResult<_>>()?;
        names.sort_by(|left, right| left.0.cmp(&right.0));

        let mut loaded = Vec::new();
        for (name, path) in names {
            if !path.is_file() {
                continue;
            }
            if name.starts_with('.')
                || TEMPORARY_SUFFIXES
                    .iter()
                    .any(|suffix| name.ends_with(suffix))
            {
                self.quarantined.push(name);
                continue;
            }
            loaded.push(read_event(&path)?);
        }
        loaded.sort_by_key(|event| event.body.sequence);

        let contiguous = loaded
            .iter()
            .enumerate()
            .all(|(index, event)| event.body.sequence == index as u64 + 1);
        if !contiguous {
            let sequences: Vec<u64> = loaded.iter().map(|event| event.body.sequence).collect();
            return Err(ReviewError::contract_with(
                "review event sequences are not contiguous",
                json!({ "sequences": sequences }),
            ));
        }

        for event in loaded {
            let prior = self.head_hash();
            if event.body.prior_hash != prior {
                return Err(ReviewError::contract_with(
                    "review event chain is broken",
                    json!({ "sequence": event.body.sequence }),
                ));
            }
            if event.body.packet_hash != self.session.packet_hash
                || event.body.overlay_hash != self.session.overlay_hash
            {
                return Err(ReviewError::contract(
                    "review event is bound to a different packet or overlay",
                ));
            }
            let Some(position) = self.session.position(event.body.window_id.as_str()) else {
                return Err(ReviewError::contract_with(
                    "review event window is not in the overlay",
                    json!({ "window_id": event.body.window_id.as_str() }),
                ));
            };
            let current = self.histories[position].current();
            if event.body.base_revision != current.revision {
                return Err(ReviewError::contract(
                    "stored event base_revision does not replay",
                ));
            }
            let next = self.next_state(
                position,
                &current,
                &event.body.action,
                &event.body.actor,
                &event.event_hash,
            )?;
            self.record(position, event, next);
        }
        Ok(())
    }

    /// Create one event, or return the stored event for an identical retry
    pub fn append(&mut self, request: AppendRequest) -> ReviewResult<ReviewEvent> {
        let position = self.require_window(&request.window_id)?;

        if let Some(&index) = self.by_request.get(request.request_id.as_str()) {
            let existing = &self.events[index];
            let identical = existing.body.window_id.as_str() == request.window_id
                && existing.body.actor == request.actor
                && existing.body.base_revision == request.base_revision
                && existing.body.action == request.action;
            if !identical {
                return Err(ReviewError::RequestReused {
                    request_id: request.request_id.to_string(),
                });
            }
            return Ok(existing.clone());
        }

        let current = self.histories[position].current();
        if request.base_revision != current.revision {
            return Err(ReviewError::StaleRevision {
                base_revision: request.base_revision,
                current_revision: current.revision,
            });
        }

        let timestamp_utc = match request.timestamp_utc {
            Some(timestamp) => timestamp,
            None => NonEmptyText::parse_str(&utc_now(), "timestamp_utc")?,
        };
        let event = EventBody {
            sequence: self.events.len() as u64 + 1,
            prior_hash: self.head_hash(),
            packet_hash: self.session.packet_hash.clone(),
            overlay_hash: self.session.overlay_hash.clone(),
            window_id: NonEmptyText::parse_str(&request.window_id, "window_id")?,
            actor: request.actor,
            base_revision: request.base_revision,
            request_id: request.request_id,
            timestamp_utc,
            action: request.action,
        }
        .seal()?;
        let next = self.next_state(
            position,
            &current,
            &event.body.action,
            &event.body.actor,
            &event.event_hash,
        )?;

        write_event(&self.session.events_dir, &event)?;
        self.record(position, event.clone(), next);
        Ok(event)
    }

    fn next_state(
        &self,
        position: usize,
        current: &WindowReviewState,
        action: &ReviewAction,
        actor: &NonEmptyText,
        event_hash: &Sha256Hex,
    ) -> ReviewResult<WindowReviewState> {
        match action {
            ReviewAction::Transition(transition) => current.apply(transition, actor, event_hash),
            ReviewAction::Undo {
                reverted_event_hash,
            } => {
                let history = &self.histories[position];
                let latest = history
                    .entries
                    .last()
                    .map(|entry| &self.events[entry.event_index].event_hash);
                if latest != Some(reverted_event_hash) {
                    return Err(ReviewError::contract(
                        "undo must name the latest event for this window",
                    ));
                }
                let restored = history.previous().cloned().unwrap_or_default();
                Ok(restored.restored_at(current.revision + 1))
            }
        }
    }

    fn record(&mut self, position: usize, event: ReviewEvent, state: WindowReviewState) {
        let event_index = self.events.len();
        self.by_request
            .insert(event.body.request_id.to_string(), event_index);
        self.events.push(event);
        self.histories[position]
            .entries
            .push(HistoryEntry { event_index, state });
    }

    fn head_hash(&self) -> Sha256Hex {
        self.events
            .last()
            .map(|event| event.event_hash.clone())
            .unwrap_or_else(Sha256Hex::genesis)
    }

    /// Return the window position or an unknown-window error
    pub fn require_window(&self, window_id: &str) -> ReviewResult<usize> {
        self.session
            .position(window_id)
            .ok_or_else(|| ReviewError::UnknownWindow {
                window_id: window_id.to_owned(),
            })
    }

    /// The loaded session
    pub fn session(&self) -> &Arc<ReviewSession> {
        &self.session
    }

    /// All events in sequence order
    pub fn events(&self) -> &[ReviewEvent] {
        &self.events
    }

    /// Names of crash residue files ignored during replay
    pub fn quarantined(&self) -> &[String] {
        &self.quarantined
    }

    /// The current state of a window
    pub fn window_state(&self, window_id: &str) -> ReviewResult<WindowReviewState> {
        Ok(self.histories[self.require_window(window_id)?].current())
    }

    /// The latest event recorded for a window
    pub fn latest_event(&self, window_id: &str) -> ReviewResult<Option<&ReviewEvent>> {
        let position = self.require_window(window_id)?;
        Ok(self.histories[position]
            .entries
            .last()
            .map(|entry| &self.events[entry.event_index]))
    }

    /// The activity implied by the current decision
    ///
    /// Signed windows report the activity of the state that was signed. Pending,
    /// returned, uncertain, and follow-up windows have no reviewed activity
    pub fn reviewed_activity(&self, window_id: &str) -> ReviewResult<Option<Activity>> {
        let position = self.require_window(window_id)?;
        let proposal = self.session.windows()[position].proposal_activity();
        let history = &self.histories[position];
        let current = history.current();
        if !matches!(current.decision, Decision::SignedOff { .. }) {
            return Ok(current.decision.stated_activity(proposal));
        }
        Ok(history
            .previous()
            .and_then(|previous| previous.decision.stated_activity(proposal)))
    }

    /// Session progress in overlay order
    pub fn progress(&self) -> ReviewProgress {
        let states: Vec<WindowReviewState> =
            self.histories.iter().map(WindowHistory::current).collect();
        let entries: Vec<ProgressEntry<'_>> = self
            .session
            .windows()
            .iter()
            .zip(&states)
            .map(|(window, state)| ProgressEntry {
                window_id: &window.window_id,
                uniform: window.selection_kind == crate::session::SelectionKind::Uniform,
                decision: &state.decision,
            })
            .collect();
        ReviewProgress::from_entries(&entries)
    }
}

fn read_event(path: &Path) -> ReviewResult<ReviewEvent> {
    let parsed = fs::read(path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| {
            serde_json::from_slice::<Value>(&bytes).map_err(|error| error.to_string())
        })
        .and_then(|value| ReviewEvent::from_json(&value).map_err(|error| error.to_string()));
    parsed.map_err(|reason| {
        ReviewError::preparation_with(
            "review event file is corrupt",
            json!({ "path": path.display().to_string(), "reason": reason }),
        )
    })
}

fn write_event(events_dir: &Path, event: &ReviewEvent) -> ReviewResult<()> {
    let destination = events_dir.join(event.file_name());
    let temporary = events_dir.join(format!("{}.partial", event.file_name()));
    let text = canonical_json(&event.to_json());

    let mut file =
        File::create(&temporary).map_err(|error| ReviewError::io("create", &temporary, error))?;
    file.write_all(text.as_bytes())
        .map_err(|error| ReviewError::io("write", &temporary, error))?;
    file.sync_all()
        .map_err(|error| ReviewError::io("sync", &temporary, error))?;
    drop(file);
    fs::rename(&temporary, &destination)
        .map_err(|error| ReviewError::io("rename", &destination, error))?;
    // directory fsync is best effort, matching the Python store
    if let Ok(directory) = File::open(events_dir) {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn utc_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::{Value, json};

    use super::{AppendRequest, LOCK_FILE_NAME, ReviewStore};
    use crate::{
        domain::ReviewAction,
        error::ReviewError,
        test_support::{WINDOW_IDS, write_session},
        text::NonEmptyText,
    };

    fn text(value: &str) -> NonEmptyText {
        NonEmptyText::parse_str(value, "text").unwrap()
    }

    fn request(
        window: usize,
        actor: &str,
        request_id: &str,
        base_revision: u64,
        action: Value,
    ) -> AppendRequest {
        AppendRequest {
            window_id: WINDOW_IDS[window].to_owned(),
            actor: text(actor),
            request_id: text(request_id),
            base_revision,
            action: ReviewAction::parse(&action).unwrap(),
            timestamp_utc: Some(text("2026-09-14T00:00:00Z")),
        }
    }

    fn clear_defects() -> Value {
        json!({"kind": "set_defects", "defects": {
            "identity": {"kind": "clear"},
            "clock": {"kind": "clear", "measurement": {"kind": "absent"}},
            "synchronization": {"kind": "clear"},
            "redaction": {"kind": "clear"},
        }})
    }

    #[test]
    fn append_replay_duplicate_stale_undo_and_restart() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let mut store = ReviewStore::open(&session).unwrap();

        let first = store
            .append(request(
                0,
                "rev-1",
                "req-confirm",
                0,
                json!({"kind": "confirm"}),
            ))
            .unwrap();
        let duplicate = store
            .append(request(
                0,
                "rev-1",
                "req-confirm",
                0,
                json!({"kind": "confirm"}),
            ))
            .unwrap();
        assert_eq!(duplicate, first);
        assert_eq!(store.events().len(), 1);

        let reused = store
            .append(request(
                0,
                "rev-1",
                "req-confirm",
                0,
                json!({"kind": "no_speech"}),
            ))
            .unwrap_err();
        assert!(matches!(reused, ReviewError::RequestReused { .. }));
        let other_window = store
            .append(request(
                1,
                "rev-1",
                "req-confirm",
                0,
                json!({"kind": "confirm"}),
            ))
            .unwrap_err();
        assert!(matches!(other_window, ReviewError::RequestReused { .. }));
        let stale = store
            .append(request(
                0,
                "rev-1",
                "req-stale",
                0,
                json!({"kind": "no_speech"}),
            ))
            .unwrap_err();
        assert!(matches!(
            stale,
            ReviewError::StaleRevision {
                base_revision: 0,
                current_revision: 1
            }
        ));

        let undo = json!({"kind": "undo", "reverted_event_hash": first.event_hash.as_str()});
        let undone = store
            .append(request(0, "rev-1", "req-undo", 1, undo))
            .unwrap();
        assert_eq!(store.window_state("w1").unwrap().decision.kind(), "pending");
        assert_eq!(store.window_state("w1").unwrap().revision, 2);
        assert_eq!(store.progress().reviewed_count, 0);
        assert_eq!(undone.body.prior_hash, first.event_hash);
        drop(store);

        let restarted = ReviewStore::open(&session).unwrap();
        let hashes: Vec<_> = restarted
            .events()
            .iter()
            .map(|event| event.event_hash.clone())
            .collect();
        assert_eq!(
            hashes,
            vec![first.event_hash.clone(), undone.event_hash.clone()]
        );
        assert_eq!(
            restarted.window_state("w1").unwrap().decision.kind(),
            "pending"
        );
        assert_eq!(restarted.progress().next_window_id.as_deref(), Some("w1"));
        let file = session.join("events").join(first.file_name());
        let stored = fs::read_to_string(file).unwrap();
        assert!(stored.ends_with("}\n"));
        assert!(!stored.contains(": "));
    }

    #[test]
    fn unknown_window_is_reported_before_request_checks() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let mut store = ReviewStore::open(&session).unwrap();
        let mut unknown = request(0, "rev-1", "req", 0, json!({"kind": "confirm"}));
        unknown.window_id = "missing".to_owned();

        assert!(matches!(
            store.append(unknown),
            Err(ReviewError::UnknownWindow { .. })
        ));
    }

    #[test]
    fn undo_names_only_the_latest_event_of_the_window() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let mut store = ReviewStore::open(&session).unwrap();

        let confirm = store
            .append(request(0, "rev-1", "c1", 0, json!({"kind": "confirm"})))
            .unwrap();
        let defects = store
            .append(request(0, "rev-1", "d1", 1, clear_defects()))
            .unwrap();
        // another window's event does not change which event this window may undo
        store
            .append(request(
                1,
                "rev-1",
                "other",
                0,
                json!({"kind": "no_speech"}),
            ))
            .unwrap();

        let old = json!({"kind": "undo", "reverted_event_hash": confirm.event_hash.as_str()});
        let error = store
            .append(request(0, "rev-1", "u-old", 2, old))
            .unwrap_err();
        assert!(error.to_string().contains("latest event"));

        let undo_defects =
            json!({"kind": "undo", "reverted_event_hash": defects.event_hash.as_str()});
        let undo = store
            .append(request(0, "rev-1", "u1", 2, undo_defects))
            .unwrap();
        let state = store.window_state("w1").unwrap();
        assert_eq!(state.decision.kind(), "confirmed_proposal");
        assert!(state.defects.has_unresolved());
        assert_eq!(state.revision, 3);
        assert_eq!(
            store.latest_event("w1").unwrap().unwrap().event_hash,
            undo.event_hash
        );
        assert_eq!(
            store
                .latest_event("w1")
                .unwrap()
                .unwrap()
                .body
                .action
                .kind(),
            "undo"
        );

        // undoing the undo restores the defects, which acts as redo
        let redo = json!({"kind": "undo", "reverted_event_hash": undo.event_hash.as_str()});
        store.append(request(0, "rev-1", "u2", 3, redo)).unwrap();
        let state = store.window_state("w1").unwrap();
        assert!(!state.defects.has_unresolved());
        assert_eq!(
            state.to_json()["last_review_event_hash"],
            defects.event_hash.as_str()
        );
        assert_eq!(state.revision, 4);
        drop(store);

        let restarted = ReviewStore::open(&session).unwrap();
        assert_eq!(restarted.window_state("w1").unwrap().revision, 4);
        assert!(
            !restarted
                .window_state("w1")
                .unwrap()
                .defects
                .has_unresolved()
        );
    }

    #[test]
    fn reviewed_activity_follows_the_decision_and_the_signed_state() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let mut store = ReviewStore::open(&session).unwrap();
        assert_eq!(store.reviewed_activity("w1").unwrap(), None);

        store
            .append(request(0, "rev-1", "c1", 0, json!({"kind": "confirm"})))
            .unwrap();
        let proposal = store.session().windows()[0].proposal_activity().clone();
        assert_eq!(store.reviewed_activity("w1").unwrap(), Some(proposal));

        let correction = json!({"kind": "correct", "activity": [{"speaker": "speaker_b", "start_frame": 5, "end_frame": 9}]});
        store
            .append(request(0, "rev-1", "c2", 1, correction))
            .unwrap();
        store
            .append(request(0, "rev-1", "d1", 2, clear_defects()))
            .unwrap();
        store
            .append(request(
                0,
                "signer",
                "s1",
                3,
                json!({"kind": "sign_off_accept"}),
            ))
            .unwrap();
        let signed = store.reviewed_activity("w1").unwrap().unwrap();
        assert_eq!(
            signed.to_json(),
            json!([{"speaker": "speaker_b", "start_frame": 5, "end_frame": 9}])
        );

        store
            .append(request(1, "rev-1", "n1", 0, json!({"kind": "no_speech"})))
            .unwrap();
        assert_eq!(
            store
                .reviewed_activity("w2")
                .unwrap()
                .map(|activity| activity.is_empty()),
            Some(true)
        );
        store
            .append(request(
                1,
                "signer",
                "r1",
                1,
                json!({"kind": "sign_off_return", "reason": "check"}),
            ))
            .unwrap();
        assert_eq!(store.reviewed_activity("w2").unwrap(), None);
        assert_eq!(store.progress().next_window_id.as_deref(), Some("w2"));
    }

    #[test]
    fn crash_residue_is_quarantined_and_corrupt_events_fail() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let events = session.join("events");
        fs::create_dir_all(&events).unwrap();
        fs::write(events.join("event-00000001-deadbeef.json.partial"), "{").unwrap();
        fs::write(events.join(".DS_Store"), "x").unwrap();
        fs::create_dir_all(events.join("nested")).unwrap();

        let store = ReviewStore::open(&session).unwrap();
        assert_eq!(
            store.quarantined(),
            &[
                ".DS_Store".to_owned(),
                "event-00000001-deadbeef.json.partial".to_owned()
            ]
        );
        assert!(store.events().is_empty());
        drop(store);

        fs::write(events.join("event-00000001-bad.json"), "{").unwrap();
        let error = ReviewStore::open(&session).unwrap_err();
        assert!(error.to_string().contains("corrupt"));
    }

    #[test]
    fn tampered_and_gapped_chains_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let mut store = ReviewStore::open(&session).unwrap();
        let first = store
            .append(request(0, "rev-1", "c1", 0, json!({"kind": "confirm"})))
            .unwrap();
        let second = store
            .append(request(1, "rev-1", "c2", 0, json!({"kind": "confirm"})))
            .unwrap();
        drop(store);

        let first_path = session.join("events").join(first.file_name());
        fs::rename(&first_path, temp.path().join("moved.json")).unwrap();
        let gap = ReviewStore::open(&session).unwrap_err();
        assert!(gap.to_string().contains("not contiguous"));

        let mut tampered: Value =
            serde_json::from_str(&fs::read_to_string(temp.path().join("moved.json")).unwrap())
                .unwrap();
        tampered["actor"] = json!("tampered");
        fs::write(&first_path, tampered.to_string()).unwrap();
        let corrupt = ReviewStore::open(&session).unwrap_err();
        assert!(corrupt.to_string().contains("corrupt"));
        assert!(second.body.sequence == 2);
    }

    #[test]
    fn lock_blocks_a_second_store() {
        let temp = tempfile::tempdir().unwrap();
        let session = write_session(temp.path());
        let store = ReviewStore::open(&session).unwrap();

        let second = ReviewStore::open(&session).unwrap_err();
        assert!(matches!(second, ReviewError::SessionLocked { .. }));
        assert!(session.join(LOCK_FILE_NAME).is_file());
        assert!(!session.join("events").join(LOCK_FILE_NAME).exists());

        drop(store);
        assert!(ReviewStore::open(&session).is_ok());
    }
}
