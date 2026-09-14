//! Strict JSON object access with the Python schema error messages

use serde_json::{Map, Value, json};

use crate::error::{ReviewError, ReviewResult};

static NULL: Value = Value::Null;

/// A borrowed JSON object with a label used in error messages
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    map: &'a Map<String, Value>,
    label: &'a str,
}

impl<'a> Record<'a> {
    /// Require a JSON object
    pub fn object(value: &'a Value, label: &'a str) -> ReviewResult<Self> {
        match value {
            Value::Object(map) => Ok(Self { map, label }),
            _ => Err(ReviewError::contract(format!("{label} must be an object"))),
        }
    }

    /// Reject keys outside the allowed set and require the required set
    pub fn exact(self, allowed: &[&str], required: &[&str]) -> ReviewResult<Self> {
        let mut unknown: Vec<&str> = self
            .map
            .keys()
            .map(String::as_str)
            .filter(|key| !allowed.contains(key))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(ReviewError::contract_with(
                format!("{} has unknown fields", self.label),
                json!({ "unknown": unknown }),
            ));
        }
        let mut missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|key| !self.map.contains_key(*key))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            return Err(ReviewError::contract_with(
                format!("{} is missing fields", self.label),
                json!({ "missing": missing }),
            ));
        }
        Ok(self)
    }

    /// Return a field or JSON null when it is absent
    pub fn get(self, key: &str) -> &'a Value {
        self.map.get(key).unwrap_or(&NULL)
    }

    /// Return the `kind` tag when it is a string
    pub fn kind(self) -> Option<&'a str> {
        self.get("kind").as_str()
    }

    /// Return the underlying map
    pub fn map(self) -> &'a Map<String, Value> {
        self.map
    }

    /// Build the error for an unknown `kind` tag
    pub fn unknown_kind(self, label: &str) -> ReviewError {
        ReviewError::contract_with(
            format!("{label} kind is unknown"),
            json!({ "kind": self.get("kind") }),
        )
    }
}
