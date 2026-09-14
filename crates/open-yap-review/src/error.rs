//! Typed review failures with the Python-compatible JSON error body

use std::{io, path::Path};

use serde_json::{Map, Value, json};

/// Result alias for review operations
pub type ReviewResult<T> = Result<T, ReviewError>;

/// A review failure that maps to one machine-readable error body
#[derive(Debug, thiserror::Error)]
pub enum ReviewError {
    /// A schema, identity, or state-machine violation
    #[error("{message}")]
    Contract {
        message: String,
        details: Map<String, Value>,
    },

    /// A session, overlay, or event store that cannot be loaded
    #[error("{message}")]
    Preparation {
        message: String,
        details: Map<String, Value>,
    },

    /// The request names a window that is not in the overlay
    #[error("unknown window")]
    UnknownWindow { window_id: String },

    /// The request was built on an older window revision
    #[error("stale revision")]
    StaleRevision {
        base_revision: u64,
        current_revision: u64,
    },

    /// A request id was sent again with a different window, actor, revision, or action
    #[error("request_id was reused with different content")]
    RequestReused { request_id: String },

    /// Another process holds the exclusive session lock
    #[error("review session is locked by another server")]
    SessionLocked { lock_path: String },

    /// A file system failure outside the typed contract
    #[error("{context}: {source}")]
    Io { context: String, source: io::Error },
}

impl ReviewError {
    /// Build a contract error without details
    pub fn contract(message: impl Into<String>) -> Self {
        Self::Contract {
            message: message.into(),
            details: Map::new(),
        }
    }

    /// Build a contract error with a details object
    pub fn contract_with(message: impl Into<String>, details: Value) -> Self {
        Self::Contract {
            message: message.into(),
            details: object(details),
        }
    }

    /// Build a preparation error without details
    pub fn preparation(message: impl Into<String>) -> Self {
        Self::Preparation {
            message: message.into(),
            details: Map::new(),
        }
    }

    /// Build a preparation error with a details object
    pub fn preparation_with(message: impl Into<String>, details: Value) -> Self {
        Self::Preparation {
            message: message.into(),
            details: object(details),
        }
    }

    /// Wrap an IO failure on one path
    pub fn io(action: &str, path: &Path, source: io::Error) -> Self {
        Self::Io {
            context: format!("{action} {}", path.display()),
            source,
        }
    }

    /// Return the stable machine-readable error code
    pub fn code(&self) -> &'static str {
        match self {
            Self::Contract { .. }
            | Self::UnknownWindow { .. }
            | Self::StaleRevision { .. }
            | Self::RequestReused { .. } => "contract",
            Self::Preparation { .. } | Self::SessionLocked { .. } => "preparation",
            Self::Io { .. } => "io",
        }
    }

    /// Return the details object sent with the error body
    pub fn details(&self) -> Value {
        match self {
            Self::Contract { details, .. } | Self::Preparation { details, .. } => {
                Value::Object(details.clone())
            }
            Self::UnknownWindow { window_id } => json!({ "window_id": window_id }),
            Self::StaleRevision {
                base_revision,
                current_revision,
            } => json!({
                "base_revision": base_revision,
                "current_revision": current_revision,
            }),
            Self::RequestReused { request_id } => json!({ "request_id": request_id }),
            Self::SessionLocked { lock_path } => json!({ "path": lock_path }),
            // io details stay out of the HTTP body because they carry local paths
            Self::Io { .. } => json!({}),
        }
    }

    /// Return the `{"ok":false,"error":{...}}` body
    pub fn to_json(&self) -> Value {
        json!({
            "ok": false,
            "error": {
                "code": self.code(),
                "message": self.to_string(),
                "details": self.details(),
            }
        })
    }
}

fn object(details: Value) -> Map<String, Value> {
    match details {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}
