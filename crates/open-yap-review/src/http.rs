//! Localhost review HTTP server: router, shared state, and serving

pub mod guard;
pub mod routes;
pub mod ui;

use std::{
    future::Future,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use axum::{
    Router,
    body::Body,
    http::{HeaderValue, StatusCode, header},
    middleware,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use tokio::net::TcpListener;

use crate::{
    error::ReviewError, media::WavCache, session::ReviewSession, store::ReviewStore,
    text::NonEmptyText,
};

pub use ui::UiDist;

/// Whether the server accepts writes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    ReadWrite,
    ReadOnly,
}

/// Which POST route records actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Reviewer events through `/events`
    Reviewer,
    /// Independent sign-off through `/signoff`
    Signer,
}

/// Shared state for all requests
#[derive(Debug)]
pub struct AppState {
    session: Arc<ReviewSession>,
    store: Mutex<ReviewStore>,
    actor: NonEmptyText,
    access: Access,
    role: Role,
    ui: UiDist,
    wav_cache: WavCache,
}

impl AppState {
    /// Build the shared state around an opened store
    pub fn new(
        store: ReviewStore,
        actor: NonEmptyText,
        access: Access,
        role: Role,
        ui: UiDist,
    ) -> Self {
        Self {
            session: Arc::clone(store.session()),
            store: Mutex::new(store),
            actor,
            access,
            role,
            ui,
            wav_cache: WavCache::default(),
        }
    }

    fn store(&self) -> MutexGuard<'_, ReviewStore> {
        // a panic mid-append cannot leave partial state because appends record only after the write
        self.store.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Build the review router with the host and origin guard
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .fallback(routes::dispatch)
        .with_state(state)
        .layer(middleware::from_fn(guard::guard))
}

/// Serve requests until `shutdown` resolves
pub async fn serve(
    listener: TcpListener,
    state: Arc<AppState>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}

/// A failed request with its status and JSON body
#[derive(Debug)]
pub struct ApiFailure {
    status: StatusCode,
    body: Value,
}

impl ApiFailure {
    /// A failure body without details, used for transport-level rejections
    pub fn simple(status: StatusCode, code: &str, message: &str) -> Self {
        Self {
            status,
            body: json!({ "ok": false, "error": { "code": code, "message": message } }),
        }
    }

    fn internal(message: &str) -> Self {
        Self::simple(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
    }
}

impl From<ReviewError> for ApiFailure {
    fn from(error: ReviewError) -> Self {
        let status = match &error {
            ReviewError::StaleRevision { .. } | ReviewError::RequestReused { .. } => {
                StatusCode::CONFLICT
            }
            ReviewError::UnknownWindow { .. } => StatusCode::NOT_FOUND,
            ReviewError::Io { .. } | ReviewError::SessionLocked { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ReviewError::Contract { .. } | ReviewError::Preparation { .. } => {
                StatusCode::BAD_REQUEST
            }
        };
        if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!("request failed error={error}");
        }
        Self {
            status,
            body: error.to_json(),
        }
    }
}

impl IntoResponse for ApiFailure {
    fn into_response(self) -> Response {
        json_response(self.status, &self.body)
    }
}

/// Serialize a JSON body with the review content type
pub fn json_response(status: StatusCode, body: &Value) -> Response {
    let mut bytes = serde_json::to_vec_pretty(body).unwrap_or_default();
    bytes.push(b'\n');
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    response
}
