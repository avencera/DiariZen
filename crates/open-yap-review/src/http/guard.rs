//! Host and origin checks plus security headers on every response

use axum::{
    extract::Request,
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::http::ApiFailure;

/// Content security policy for the bundled UI and API
pub const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; connect-src 'self'; media-src 'self' blob:; \
     img-src 'self' data:; style-src 'self'; script-src 'self'; \
     object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'";

/// Reject foreign hosts and origins, then add security headers
///
/// Binding to 127.0.0.1 is not enough on its own: a DNS-rebinding page reaches
/// the port with a foreign Host header, and a cross-site page sends a foreign Origin
pub async fn guard(request: Request, next: Next) -> Response {
    let host = header_text(request.headers(), header::HOST)
        .unwrap_or_default()
        .to_owned();
    let origin = header_text(request.headers(), header::ORIGIN).map(str::to_owned);
    let origin_present = request.headers().contains_key(header::ORIGIN);
    let origin_ok = origin_allowed(origin.as_deref(), origin_present, &host);

    let mut response = if !host_allowed(&host) {
        ApiFailure::simple(StatusCode::FORBIDDEN, "forbidden", "host is not 127.0.0.1")
            .into_response()
    } else if request.method() != Method::OPTIONS && !origin_ok {
        ApiFailure::simple(StatusCode::FORBIDDEN, "forbidden", "origin is not allowed")
            .into_response()
    } else {
        next.run(request).await
    };

    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    if let Some(origin) = origin.filter(|origin| !origin.is_empty() && origin_ok)
        && let Ok(value) = HeaderValue::from_str(&origin)
    {
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        headers.insert(header::VARY, HeaderValue::from_static("Origin"));
    }
    response
}

fn header_text(headers: &HeaderMap, name: header::HeaderName) -> Option<&str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

/// Accept `127.0.0.1` with or without a port
pub fn host_allowed(host: &str) -> bool {
    host == "127.0.0.1" || host.starts_with("127.0.0.1:")
}

/// Accept a missing, empty, or `null` origin, or exactly `http://{host}`
pub fn origin_allowed(origin: Option<&str>, present: bool, host: &str) -> bool {
    match origin {
        None => !present,
        Some("" | "null") => true,
        Some(origin) => origin.strip_prefix("http://") == Some(host),
    }
}

#[cfg(test)]
mod tests {
    use super::{host_allowed, origin_allowed};

    #[test]
    fn host_must_be_loopback_ipv4() {
        assert!(host_allowed("127.0.0.1"));
        assert!(host_allowed("127.0.0.1:8765"));
        assert!(!host_allowed("localhost:8765"));
        assert!(!host_allowed("127.0.0.10:8765"));
        assert!(!host_allowed("evil.example"));
        assert!(!host_allowed(""));
    }

    #[test]
    fn origin_must_match_the_host() {
        assert!(origin_allowed(None, false, "127.0.0.1:1"));
        assert!(origin_allowed(Some(""), true, "127.0.0.1:1"));
        assert!(origin_allowed(Some("null"), true, "127.0.0.1:1"));
        assert!(origin_allowed(
            Some("http://127.0.0.1:1"),
            true,
            "127.0.0.1:1"
        ));
        assert!(!origin_allowed(
            Some("http://example.com"),
            true,
            "127.0.0.1:1"
        ));
        assert!(!origin_allowed(
            Some("https://127.0.0.1:1"),
            true,
            "127.0.0.1:1"
        ));
        // an origin header that is not valid text is never allowed
        assert!(!origin_allowed(None, true, "127.0.0.1:1"));
    }
}
