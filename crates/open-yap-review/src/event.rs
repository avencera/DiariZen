//! Immutable hash-chained review events

use serde_json::{Map, Value, json};

use crate::{
    canonical_json::sha256_json,
    domain::ReviewAction,
    error::{ReviewError, ReviewResult},
    record::Record,
    text::{NonEmptyText, Sha256Hex, parse_non_negative_integer},
};

/// Event schema name
pub const EVENT_SCHEMA: &str = "speakrs-open-yap-review-event";

/// Event schema version
pub const EVENT_SCHEMA_VERSION: u64 = 1;

const EVENT_FIELDS: &[&str] = &[
    "schema",
    "schema_version",
    "sequence",
    "event_hash",
    "prior_hash",
    "packet_hash",
    "overlay_hash",
    "window_id",
    "actor",
    "base_revision",
    "request_id",
    "timestamp_utc",
    "action",
];

/// One accepted immutable review event
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewEvent {
    pub event_hash: Sha256Hex,
    pub body: EventBody,
}

/// The hashed fields of an event
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventBody {
    pub sequence: u64,
    pub prior_hash: Sha256Hex,
    pub packet_hash: Sha256Hex,
    pub overlay_hash: Sha256Hex,
    pub window_id: NonEmptyText,
    pub actor: NonEmptyText,
    pub base_revision: u64,
    pub request_id: NonEmptyText,
    pub timestamp_utc: NonEmptyText,
    pub action: ReviewAction,
}

impl EventBody {
    fn to_map(&self) -> Map<String, Value> {
        let mut map = Map::new();
        let mut insert = |key: &str, value: Value| {
            map.insert(key.to_owned(), value);
        };
        insert("schema", json!(EVENT_SCHEMA));
        insert("schema_version", json!(EVENT_SCHEMA_VERSION));
        insert("sequence", json!(self.sequence));
        insert("prior_hash", json!(self.prior_hash.as_str()));
        insert("packet_hash", json!(self.packet_hash.as_str()));
        insert("overlay_hash", json!(self.overlay_hash.as_str()));
        insert("window_id", json!(self.window_id.as_str()));
        insert("actor", json!(self.actor.as_str()));
        insert("base_revision", json!(self.base_revision));
        insert("request_id", json!(self.request_id.as_str()));
        insert("timestamp_utc", json!(self.timestamp_utc.as_str()));
        insert("action", self.action.to_json());
        map
    }

    /// Hash the canonical body and parse the result back through the stored-event checks
    pub fn seal(self) -> ReviewResult<ReviewEvent> {
        let mut record = self.to_map();
        let event_hash = sha256_json(&Value::Object(record.clone()));
        record.insert("event_hash".to_owned(), Value::String(event_hash));
        ReviewEvent::from_json(&Value::Object(record))
    }
}

impl ReviewEvent {
    /// Parse and verify one stored event
    pub fn from_json(value: &Value) -> ReviewResult<Self> {
        let record = Record::object(value, "review event")?.exact(EVENT_FIELDS, EVENT_FIELDS)?;
        let schema_valid = record.get("schema").as_str() == Some(EVENT_SCHEMA)
            && parse_non_negative_integer(record.get("schema_version"))
                == Some(EVENT_SCHEMA_VERSION);
        if !schema_valid {
            return Err(ReviewError::contract("review event schema is invalid"));
        }

        let mut unhashed = record.map().clone();
        unhashed.remove("event_hash");
        let event_hash = Sha256Hex::parse(record.get("event_hash"), "event_hash")?;
        if sha256_json(&Value::Object(unhashed)) != event_hash.as_str() {
            return Err(ReviewError::contract(
                "review event hash does not match its body",
            ));
        }

        let sequence = parse_non_negative_integer(record.get("sequence"))
            .filter(|sequence| *sequence > 0)
            .ok_or_else(|| {
                ReviewError::contract("review event sequence must be a positive integer")
            })?;
        let base_revision =
            parse_non_negative_integer(record.get("base_revision")).ok_or_else(|| {
                ReviewError::contract("review event base_revision must be a non-negative integer")
            })?;

        let body = EventBody {
            sequence,
            prior_hash: Sha256Hex::parse(record.get("prior_hash"), "prior_hash")?,
            packet_hash: Sha256Hex::parse(record.get("packet_hash"), "packet_hash")?,
            overlay_hash: Sha256Hex::parse(record.get("overlay_hash"), "overlay_hash")?,
            window_id: NonEmptyText::parse(record.get("window_id"), "window_id")?,
            actor: NonEmptyText::parse(record.get("actor"), "actor")?,
            base_revision,
            request_id: NonEmptyText::parse(record.get("request_id"), "request_id")?,
            timestamp_utc: NonEmptyText::parse(record.get("timestamp_utc"), "timestamp_utc")?,
            action: ReviewAction::parse(record.get("action"))?,
        };
        Ok(Self { event_hash, body })
    }

    /// Serialize the event with its hash
    pub fn to_json(&self) -> Value {
        let mut record = self.body.to_map();
        record.insert(
            "event_hash".to_owned(),
            Value::String(self.event_hash.to_string()),
        );
        Value::Object(record)
    }

    /// Return the file name `event-{sequence:08}-{event_hash}.json`
    pub fn file_name(&self) -> String {
        format!("event-{:08}-{}.json", self.body.sequence, self.event_hash)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{EventBody, ReviewEvent};
    use crate::{
        canonical_json::canonical_json,
        domain::ReviewAction,
        text::{NonEmptyText, Sha256Hex},
    };

    const GOLDEN: &str = r#"{"action":{"defects":{"clock":{"kind":"unresolved","measurement":{"drift_seconds_per_second":1e-05,"kind":"entered","offset_seconds":0.0},"reason":"drift"},"identity":{"kind":"clear"},"redaction":{"kind":"unresolved","reason":"cut"},"synchronization":{"kind":"not_reviewed"}},"kind":"set_defects"},"actor":"rev-1","base_revision":0,"event_hash":"5a8782947b49b66b41dd7fdd5116e5b451cd010efaf037eb3cab20632c3c3d90","overlay_hash":"f283505b9243390e21419c241f09d3fe20369f367de56a4ac181b2c158f4a321","packet_hash":"a3b5d49d641ea1a0a552851ef45db1df8ae03568b8dd14c74e2a90e92e2eb238","prior_hash":"0000000000000000000000000000000000000000000000000000000000000000","request_id":"req-1","schema":"speakrs-open-yap-review-event","schema_version":1,"sequence":1,"timestamp_utc":"2026-09-14T00:00:00Z","window_id":"oy-00bb969a2902aed5-uniform-000240402"}"#;

    fn text(value: &str) -> NonEmptyText {
        NonEmptyText::parse_str(value, "text").unwrap()
    }

    fn sha(value: &str) -> Sha256Hex {
        Sha256Hex::parse_str(value, "hash").unwrap()
    }

    #[test]
    fn golden_event_round_trips_byte_for_byte() {
        let stored: Value = serde_json::from_str(GOLDEN).unwrap();
        let event = ReviewEvent::from_json(&stored).unwrap();

        assert_eq!(canonical_json(&event.to_json()), format!("{GOLDEN}\n"));
        assert_eq!(
            event.file_name(),
            "event-00000001-5a8782947b49b66b41dd7fdd5116e5b451cd010efaf037eb3cab20632c3c3d90.json"
        );
    }

    #[test]
    fn sealing_a_draft_reproduces_the_golden_hash() {
        let action = ReviewAction::parse(&json!({
            "kind": "set_defects",
            "defects": {
                "identity": {"kind": "clear"},
                "clock": {"kind": "unresolved", "reason": " drift\u{1f}", "measurement": {"kind": "entered", "offset_seconds": 0, "drift_seconds_per_second": 0.00001}},
                "synchronization": {"kind": "not_reviewed"},
                "redaction": {"kind": "unresolved", "reason": "cut "},
            }
        }))
        .unwrap();
        let event = EventBody {
            sequence: 1,
            prior_hash: Sha256Hex::genesis(),
            packet_hash: sha("a3b5d49d641ea1a0a552851ef45db1df8ae03568b8dd14c74e2a90e92e2eb238"),
            overlay_hash: sha("f283505b9243390e21419c241f09d3fe20369f367de56a4ac181b2c158f4a321"),
            window_id: text("oy-00bb969a2902aed5-uniform-000240402"),
            actor: text("rev-1"),
            base_revision: 0,
            request_id: text("req-1"),
            timestamp_utc: text("2026-09-14T00:00:00Z"),
            action,
        }
        .seal()
        .unwrap();

        assert_eq!(
            event.event_hash.as_str(),
            "5a8782947b49b66b41dd7fdd5116e5b451cd010efaf037eb3cab20632c3c3d90"
        );
    }

    #[test]
    fn tampered_or_malformed_events_are_rejected() {
        let mut tampered: Value = serde_json::from_str(GOLDEN).unwrap();
        tampered["actor"] = json!("tampered");
        let error = ReviewEvent::from_json(&tampered).unwrap_err();
        assert!(error.to_string().contains("hash does not match"));

        let mut extra: Value = serde_json::from_str(GOLDEN).unwrap();
        extra["note"] = json!("x");
        assert!(
            ReviewEvent::from_json(&extra)
                .unwrap_err()
                .to_string()
                .contains("unknown fields")
        );

        let mut schema: Value = serde_json::from_str(GOLDEN).unwrap();
        schema["schema_version"] = json!(true);
        assert!(
            ReviewEvent::from_json(&schema)
                .unwrap_err()
                .to_string()
                .contains("schema")
        );
    }
}
