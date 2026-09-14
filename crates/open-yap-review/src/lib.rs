//! Open Yap local review server, event store, and domain models
//!
//! Events written here are byte-compatible with the Python review store, so the
//! Python `validate` and `export` commands replay them unchanged

pub mod canonical_json;
pub mod domain;
pub mod error;
pub mod event;
pub mod http;
pub mod media;
pub mod proposal;
pub mod record;
pub mod session;
pub mod state;
pub mod store;
pub mod text;

#[cfg(test)]
mod test_support;
