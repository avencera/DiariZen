//! Text and digest newtypes parsed with the Python review rules

use std::fmt;

use serde_json::Value;

use crate::error::{ReviewError, ReviewResult};

/// SHA-256 digest of 64 zeros that starts every event chain
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Strip leading and trailing whitespace with the Python `str.strip` set
///
/// Python also treats the information separators U+001C..U+001F as
/// whitespace, while Rust `char::is_whitespace` does not
pub fn py_strip(text: &str) -> &str {
    text.trim_matches(is_python_whitespace)
}

fn is_python_whitespace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

/// A string that is non-empty after Python whitespace stripping
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NonEmptyText(String);

impl NonEmptyText {
    /// Parse a JSON string, strip it, and require content
    pub fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        let Value::String(text) = value else {
            return Err(ReviewError::contract(format!("{label} must be a string")));
        };
        Self::parse_str(text, label)
    }

    /// Strip a string and require content
    pub fn parse_str(text: &str, label: &str) -> ReviewResult<Self> {
        let stripped = py_strip(text);
        if stripped.is_empty() {
            return Err(ReviewError::contract(format!("{label} must be non-empty")));
        }
        Ok(Self(stripped.to_owned()))
    }

    /// Borrow the stripped text
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NonEmptyText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A lowercase 64-character SHA-256 hex digest
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sha256Hex(String);

impl Sha256Hex {
    /// Parse a JSON string as a strict lowercase digest
    pub fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        let Value::String(text) = value else {
            return Err(ReviewError::contract(format!(
                "{label} must be a SHA-256 hex digest"
            )));
        };
        Self::parse_str(text, label)
    }

    /// Parse text as a strict lowercase digest
    pub fn parse_str(text: &str, label: &str) -> ReviewResult<Self> {
        let valid = text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if !valid {
            return Err(ReviewError::contract(format!(
                "{label} must be a SHA-256 hex digest"
            )));
        }
        Ok(Self(text.to_owned()))
    }

    /// Wrap a digest produced by this crate's hashing helpers
    pub(crate) fn from_digest(digest: String) -> Self {
        debug_assert!(Self::parse_str(&digest, "digest").is_ok());
        Self(digest)
    }

    /// The genesis prior hash
    pub fn genesis() -> Self {
        Self(GENESIS_HASH.to_owned())
    }

    /// Borrow the hex text
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256Hex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Parse a JSON integer that is not a bool or float
pub fn parse_non_negative_integer(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{NonEmptyText, Sha256Hex, py_strip};

    #[test]
    fn strip_removes_python_information_separators() {
        assert_eq!(py_strip("\u{1c}\u{1f} rev-1 \u{3000}\u{85}"), "rev-1");
        assert_eq!(py_strip("a\u{1d}b"), "a\u{1d}b");
    }

    #[test]
    fn non_empty_text_rejects_blank_and_non_string() {
        assert!(NonEmptyText::parse(&json!(" \u{1e} "), "actor").is_err());
        assert!(NonEmptyText::parse(&json!(null), "reason").is_err());
        assert_eq!(
            NonEmptyText::parse(&json!(" x "), "actor")
                .unwrap()
                .as_str(),
            "x"
        );
    }

    #[test]
    fn sha256_requires_lowercase_hex() {
        assert!(Sha256Hex::parse_str(&"a".repeat(64), "hash").is_ok());
        assert!(Sha256Hex::parse_str(&"A".repeat(64), "hash").is_err());
        assert!(Sha256Hex::parse_str(&"g".repeat(64), "hash").is_err());
        assert!(Sha256Hex::parse_str(&"a".repeat(63), "hash").is_err());
    }
}
