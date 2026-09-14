//! Request dispatch for the session, window, media, event, sign-off, and UI routes

use std::{
    borrow::Cow,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};

use crate::{
    domain::{ReviewAction, TransitionAction},
    error::{ReviewError, ReviewResult},
    http::{Access, ApiFailure, AppState, Role, json_response, ui},
    media::{ByteRange, MAX_FULL_MEDIA_BYTES, MediaKind, parse_byte_range, parse_python_int},
    session::read_json,
    store::{AppendRequest, ReviewStore},
    text::{NonEmptyText, parse_non_negative_integer},
};

/// Largest accepted JSON request body
pub const MAX_JSON_BYTES: u64 = 1024 * 1024;

type ApiResult = Result<Response, ApiFailure>;

/// A parsed request path with percent-decoded window ids
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route<'a> {
    Session,
    Window(Cow<'a, str>),
    Media {
        window_id: Cow<'a, str>,
        kind: Option<MediaKind>,
    },
    Events(Cow<'a, str>),
    Signoff(Cow<'a, str>),
    Other,
}

impl<'a> Route<'a> {
    /// Match a URL path against the API routes
    ///
    /// The UI builds window paths with `encodeURIComponent`, so the window id
    /// segment is percent-decoded. A malformed encoding matches no route
    pub fn parse(path: &'a str) -> Self {
        if path == "/api/v1/session" {
            return Self::Session;
        }
        let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
        let (window_segment, route): (&str, fn(Cow<'a, str>) -> Self) = match parts.as_slice() {
            ["api", "v1", "windows", window_id] => (window_id, Self::Window),
            ["api", "v1", "windows", window_id, "media", kind] => {
                let kind = MediaKind::from_segment(kind);
                let Some(window_id) = percent_decode(window_id) else {
                    return Self::Other;
                };
                return Self::Media { window_id, kind };
            }
            ["api", "v1", "windows", window_id, "events"] => (window_id, Self::Events),
            ["api", "v1", "windows", window_id, "signoff"] => (window_id, Self::Signoff),
            _ => return Self::Other,
        };
        percent_decode(window_segment).map_or(Self::Other, route)
    }
}

/// Decode `%XX` escapes in one path segment, requiring valid UTF-8
pub fn percent_decode(segment: &str) -> Option<Cow<'_, str>> {
    if !segment.contains('%') {
        return Some(Cow::Borrowed(segment));
    }
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let hex = segment.get(index + 1..index + 3)?;
        // from_str_radix alone would accept a sign such as `%+1`
        if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return None;
        }
        decoded.push(u8::from_str_radix(hex, 16).ok()?);
        index += 3;
    }
    String::from_utf8(decoded).ok().map(Cow::Owned)
}

/// Percent-encode a window id for use as one URL path segment
pub fn percent_encode(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// Route one request after the guard accepted it
pub async fn dispatch(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let result = match *request.method() {
        Method::OPTIONS => Ok(preflight()),
        Method::GET => get(state, request).await,
        Method::POST => post(state, request).await,
        _ => Err(ApiFailure::simple(
            StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method is not allowed",
        )),
    };
    result.unwrap_or_else(IntoResponse::into_response)
}

fn preflight() -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Origin"),
    );
    response
}

async fn get(state: Arc<AppState>, request: Request) -> ApiResult {
    let path = request.uri().path().to_owned();
    let wants_wav = query_has(request.uri().query(), "container", "wav");
    let range = request
        .headers()
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);

    match Route::parse(&path) {
        Route::Session => blocking(state, session_payload).await,
        Route::Window(window_id) => {
            let window_id = window_id.into_owned();
            blocking(state, move |state| window_payload(state, &window_id)).await
        }
        Route::Media { window_id, kind } => {
            let Some(kind) = kind else {
                return Err(media_not_found());
            };
            let window = state
                .session
                .window(&window_id)
                .map_err(|_| media_not_found())?;
            let path = window.media.path(kind).to_path_buf();
            blocking(state, move |state| {
                media(state, &path, wants_wav, range.as_deref())
            })
            .await
        }
        Route::Events(_) | Route::Signoff(_) => Err(unknown_path()),
        Route::Other if path.starts_with("/api/") => Err(unknown_path()),
        Route::Other => serve_ui(&state.ui, &path).await,
    }
}

async fn post(state: Arc<AppState>, request: Request) -> ApiResult {
    if state.access == Access::ReadOnly {
        return Err(ApiFailure::simple(
            StatusCode::FORBIDDEN,
            "read_only",
            "review actions are disabled",
        ));
    }
    let path = request.uri().path().to_owned();
    let payload = read_json_body(request).await?;

    match (Route::parse(&path), state.role) {
        (Route::Events(window_id), Role::Reviewer) => {
            let window_id = window_id.into_owned();
            blocking(state, move |state| {
                create_event(state, &window_id, &payload)
            })
            .await
        }
        (Route::Signoff(window_id), Role::Signer) => {
            let window_id = window_id.into_owned();
            blocking(state, move |state| {
                create_signoff(state, &window_id, &payload)
            })
            .await
        }
        (Route::Events(_), Role::Signer) => {
            Err(ReviewError::contract("reviewer events are disabled in sign-off mode").into())
        }
        (Route::Signoff(_), Role::Reviewer) => {
            Err(ReviewError::contract("sign-off is disabled in reviewer mode").into())
        }
        _ => Err(ReviewError::contract("unknown path").into()),
    }
}

async fn read_json_body(request: Request) -> Result<Value, ApiFailure> {
    let declared = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_python_int);
    let Some(length) = declared else {
        return Err(ApiFailure::simple(
            StatusCode::LENGTH_REQUIRED,
            "length",
            "Content-Length is required",
        ));
    };

    let mut body = request.into_body();
    if length < 0 || length > i128::from(MAX_JSON_BYTES) {
        // drain so the client finishes sending and reads the 413 instead of a reset
        while let Some(Ok(_)) = body.frame().await {}
        return Err(ApiFailure::simple(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "JSON payload exceeds the bound",
        ));
    }

    let invalid =
        || ApiFailure::simple(StatusCode::BAD_REQUEST, "invalid_json", "body is not JSON");
    let bytes = axum::body::to_bytes(body, MAX_JSON_BYTES as usize)
        .await
        .map_err(|_| invalid())?;
    serde_json::from_slice(&bytes).map_err(|_| invalid())
}

async fn blocking<F>(state: Arc<AppState>, work: F) -> ApiResult
where
    F: FnOnce(&AppState) -> ApiResult + Send + 'static,
{
    tokio::task::spawn_blocking(move || work(&state))
        .await
        .unwrap_or_else(|_| Err(ApiFailure::internal("request worker failed")))
}

fn session_payload(state: &AppState) -> ApiResult {
    let store = state.store();
    let windows: Vec<Value> = state
        .session
        .windows()
        .iter()
        .map(|window| {
            let window_state = store.window_state(&window.window_id)?;
            Ok(json!({
                "window_id": window.window_id,
                "parent_id": window.parent_id,
                "selection_kind": window.selection_kind.as_str(),
                "stratum": window.stratum,
                "revision": window_state.revision,
                "decision": window_state.decision.to_json(),
            }))
        })
        .collect::<ReviewResult<_>>()?;

    Ok(json_response(
        StatusCode::OK,
        &json!({
            "ok": true,
            "actor_id": state.actor.as_str(),
            "read_only": state.access == Access::ReadOnly,
            "signoff_mode": state.role == Role::Signer,
            "overlay_sha256": state.session.overlay_hash.as_str(),
            "packet_manifest_sha256": state.session.packet_hash.as_str(),
            "progress": store.progress(),
            "quarantined": store.quarantined(),
            "windows": windows,
        }),
    ))
}

fn window_payload(state: &AppState, window_id: &str) -> ApiResult {
    let window = state.session.window(window_id)?;
    let encoded = percent_encode(window_id);
    // overlay files are immutable and hash-checked at startup, so read them without the store lock
    let transcript = read_json(&window.transcript_path, "overlay transcript is unreadable")?;
    let proposal = read_json(&window.proposal_path, "overlay proposal is unreadable")?;

    let store = state.store();
    let mut payload = window_activity(&store, window_id)?;
    payload.insert("ok".to_owned(), json!(true));
    payload.insert("window".to_owned(), window.header_json());
    payload.insert("transcript".to_owned(), transcript);
    payload.insert("proposal".to_owned(), proposal);
    payload.insert(
        "media".to_owned(),
        json!({
            "emitted": format!("/api/v1/windows/{encoded}/media/emitted"),
            "speaker_a": format!("/api/v1/windows/{encoded}/media/speaker-a"),
            "speaker_b": format!("/api/v1/windows/{encoded}/media/speaker-b"),
        }),
    );
    Ok(json_response(StatusCode::OK, &Value::Object(payload)))
}

/// State, latest event, and reviewed activity of one window
fn window_activity(
    store: &ReviewStore,
    window_id: &str,
) -> ReviewResult<serde_json::Map<String, Value>> {
    let latest = store.latest_event(window_id)?;
    let fields = json!({
        "state": store.window_state(window_id)?.to_json(),
        "latest_event_hash": latest.map(|event| event.event_hash.as_str()),
        "latest_event_kind": latest.map(|event| event.body.action.kind()),
        "reviewed_activity": store.reviewed_activity(window_id)?.map(|activity| activity.to_json()),
    });
    match fields {
        Value::Object(map) => Ok(map),
        _ => unreachable!("json object literal"),
    }
}

fn create_event(state: &AppState, window_id: &str, payload: &Value) -> ApiResult {
    let mut store = state.store();
    store.require_window(window_id)?;
    let Value::Object(body) = payload else {
        return Err(ReviewError::contract("event body must be an object").into());
    };
    let request_id = parse_request_id(body.get("request_id"))?;
    let base_revision = parse_base_revision(body.get("base_revision"))?;
    let action = ReviewAction::parse(body.get("action").unwrap_or(&Value::Null))?;
    append(
        state,
        &mut store,
        window_id,
        request_id,
        base_revision,
        action,
    )
}

fn create_signoff(state: &AppState, window_id: &str, payload: &Value) -> ApiResult {
    let mut store = state.store();
    store.require_window(window_id)?;
    let Value::Object(body) = payload else {
        return Err(ReviewError::contract("sign-off body must be an object").into());
    };
    let request_id = parse_request_id(body.get("request_id"))?;
    let base_revision = parse_base_revision(body.get("base_revision"))?;
    let action = match body.get("decision").and_then(Value::as_str) {
        Some("accept") => TransitionAction::SignOffAccept,
        Some("return") => TransitionAction::SignOffReturn {
            reason: NonEmptyText::parse(body.get("reason").unwrap_or(&Value::Null), "reason")?,
        },
        _ => {
            return Err(ReviewError::contract("sign-off decision must be accept or return").into());
        }
    };
    append(
        state,
        &mut store,
        window_id,
        request_id,
        base_revision,
        ReviewAction::Transition(action),
    )
}

fn append(
    state: &AppState,
    store: &mut ReviewStore,
    window_id: &str,
    request_id: NonEmptyText,
    base_revision: u64,
    action: ReviewAction,
) -> ApiResult {
    let event = store.append(AppendRequest {
        window_id: window_id.to_owned(),
        actor: state.actor.clone(),
        request_id,
        base_revision,
        action,
        timestamp_utc: None,
    })?;
    let mut payload = window_activity(store, window_id)?;
    payload.insert("ok".to_owned(), json!(true));
    payload.insert("event".to_owned(), event.to_json());
    payload.insert("progress".to_owned(), json!(store.progress()));
    Ok(json_response(StatusCode::OK, &Value::Object(payload)))
}

fn parse_request_id(value: Option<&Value>) -> ReviewResult<NonEmptyText> {
    NonEmptyText::parse(value.unwrap_or(&Value::Null), "request_id")
}

fn parse_base_revision(value: Option<&Value>) -> ReviewResult<u64> {
    value
        .and_then(parse_non_negative_integer)
        .ok_or_else(|| ReviewError::contract("base_revision must be a non-negative integer"))
}

fn media(state: &AppState, path: &Path, wants_wav: bool, range: Option<&str>) -> ApiResult {
    if wants_wav {
        let wav = state.wav_cache.get_or_decode(path)?;
        return Ok(ranged_bytes(Bytes::from_owner(wav), "audio/wav", range));
    }

    let content_type = match path.extension().and_then(|extension| extension.to_str()) {
        Some("flac") => "audio/flac",
        _ => "audio/wav",
    };
    let size = path
        .metadata()
        .map_err(|error| ReviewError::io("stat", path, error))?
        .len();
    let Some(range) = range else {
        if size > MAX_FULL_MEDIA_BYTES {
            return Err(range_required());
        }
        let bytes = std::fs::read(path).map_err(|error| ReviewError::io("read", path, error))?;
        return Ok(media_response(
            StatusCode::OK,
            Bytes::from(bytes),
            content_type,
            None,
            size,
        ));
    };
    let Some(byte_range) = parse_byte_range(range, size) else {
        return Ok(unsatisfiable(content_type, size));
    };
    let bytes = read_range(path, byte_range)?;
    Ok(media_response(
        StatusCode::PARTIAL_CONTENT,
        Bytes::from(bytes),
        content_type,
        Some(byte_range),
        size,
    ))
}

fn ranged_bytes(bytes: Bytes, content_type: &'static str, range: Option<&str>) -> Response {
    let size = bytes.len() as u64;
    let Some(range) = range else {
        if size > MAX_FULL_MEDIA_BYTES {
            return range_required().into_response();
        }
        return media_response(StatusCode::OK, bytes, content_type, None, size);
    };
    let Some(byte_range) = parse_byte_range(range, size) else {
        return unsatisfiable(content_type, size);
    };
    let slice = bytes.slice(byte_range.start as usize..=byte_range.end as usize);
    media_response(
        StatusCode::PARTIAL_CONTENT,
        slice,
        content_type,
        Some(byte_range),
        size,
    )
}

fn read_range(path: &Path, range: ByteRange) -> ReviewResult<Vec<u8>> {
    let mut file = File::open(path).map_err(|error| ReviewError::io("open", path, error))?;
    file.seek(SeekFrom::Start(range.start))
        .map_err(|error| ReviewError::io("seek", path, error))?;
    let mut bytes = Vec::with_capacity(range.byte_count() as usize);
    file.take(range.byte_count())
        .read_to_end(&mut bytes)
        .map_err(|error| ReviewError::io("read", path, error))?;
    Ok(bytes)
}

fn media_response(
    status: StatusCode,
    bytes: Bytes,
    content_type: &'static str,
    range: Option<ByteRange>,
    size: u64,
) -> Response {
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    if let Some(range) = range {
        insert_text(
            headers,
            header::CONTENT_RANGE,
            &format!("bytes {}-{}/{size}", range.start, range.end),
        );
    }
    response
}

fn unsatisfiable(content_type: &'static str, size: u64) -> Response {
    let mut response = StatusCode::RANGE_NOT_SATISFIABLE.into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    insert_text(headers, header::CONTENT_RANGE, &format!("bytes */{size}"));
    response
}

fn insert_text(headers: &mut HeaderMap, name: header::HeaderName, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

async fn serve_ui(dist: &ui::UiDist, path: &str) -> ApiResult {
    let Some(file) = dist.resolve(path) else {
        return Err(ApiFailure::simple(
            StatusCode::NOT_FOUND,
            "not_found",
            "unknown path",
        ));
    };
    let bytes = tokio::fs::read(&file)
        .await
        .map_err(|error| ReviewError::io("read", &file, error))?;
    let mut response = Response::new(Body::from(bytes));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(ui::content_type(&file)),
    );
    Ok(response)
}

fn query_has(query: Option<&str>, key: &str, value: &str) -> bool {
    query.is_some_and(|query| {
        query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .any(|pair| pair == (key, value))
    })
}

fn unknown_path() -> ApiFailure {
    ApiFailure::simple(StatusCode::NOT_FOUND, "not_found", "unknown path")
}

fn media_not_found() -> ApiFailure {
    ApiFailure::simple(
        StatusCode::NOT_FOUND,
        "not_found",
        "media path is not allow-listed",
    )
}

fn range_required() -> ApiFailure {
    ApiFailure::simple(
        StatusCode::RANGE_NOT_SATISFIABLE,
        "range_required",
        "media requires a byte range",
    )
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{Route, percent_decode, percent_encode, query_has};
    use crate::media::MediaKind;

    #[test]
    fn routes_match_exact_segment_counts() {
        assert_eq!(Route::parse("/api/v1/session"), Route::Session);
        assert_eq!(
            Route::parse("/api/v1/windows/w1"),
            Route::Window(Cow::Borrowed("w1"))
        );
        assert_eq!(
            Route::parse("/api/v1/windows/w1/media/speaker-a"),
            Route::Media {
                window_id: Cow::Borrowed("w1"),
                kind: Some(MediaKind::SpeakerA)
            }
        );
        assert_eq!(
            Route::parse("/api/v1/windows/w1/events"),
            Route::Events(Cow::Borrowed("w1"))
        );
        assert_eq!(
            Route::parse("/api/v1/windows/w1/signoff"),
            Route::Signoff(Cow::Borrowed("w1"))
        );
        assert_eq!(Route::parse("/api/v1/windows/../secret"), Route::Other);
        assert_eq!(Route::parse("/assets/app.js"), Route::Other);
    }

    #[test]
    fn window_ids_are_percent_decoded() {
        assert_eq!(
            Route::parse("/api/v1/windows/oy%2D1%20a/events"),
            Route::Events(Cow::Owned("oy-1 a".to_owned()))
        );
        assert_eq!(
            Route::parse("/api/v1/windows/w%C3%A9/media/emitted"),
            Route::Media {
                window_id: Cow::Owned("wé".to_owned()),
                kind: Some(crate::media::MediaKind::Emitted)
            }
        );
        assert_eq!(Route::parse("/api/v1/windows/w%2"), Route::Other);
        assert_eq!(Route::parse("/api/v1/windows/w%zz"), Route::Other);
        assert_eq!(percent_decode("%FF"), None);
        assert_eq!(percent_decode("%+1"), None);
        assert_eq!(percent_encode("oy-1 a/é"), "oy-1%20a%2F%C3%A9");
        assert_eq!(
            percent_decode(&percent_encode("oy-1 a/é")).as_deref(),
            Some("oy-1 a/é")
        );
    }

    #[test]
    fn query_matching_is_exact() {
        assert!(query_has(Some("container=wav"), "container", "wav"));
        assert!(query_has(Some("a=1&container=wav"), "container", "wav"));
        assert!(!query_has(Some("container=flac"), "container", "wav"));
        assert!(!query_has(None, "container", "wav"));
    }
}
