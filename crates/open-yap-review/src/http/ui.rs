//! Built UI files served from the Vite output directory

use std::path::{Path, PathBuf};

/// Command that builds the UI output directory
pub const UI_BUILD_COMMAND: &str =
    "pnpm --dir recipes/speakrs/review-ui install && pnpm --dir recipes/speakrs/review-ui build";

/// Relative location of the UI output directory inside the repository
pub const UI_DIST_RELATIVE: &str = "recipes/speakrs/review-ui/dist";

/// Failure to open the UI output directory
#[derive(Debug, thiserror::Error)]
#[error("review UI build is missing: {index} does not exist; build it with `{UI_BUILD_COMMAND}`")]
pub struct UiDistMissing {
    index: String,
}

/// A validated UI output directory containing `index.html`
#[derive(Debug, Clone)]
pub struct UiDist {
    root: PathBuf,
}

impl UiDist {
    /// Open a UI output directory and require `index.html`
    pub fn open(path: &Path) -> Result<Self, UiDistMissing> {
        let index = path.join("index.html");
        let missing = || UiDistMissing {
            index: index.display().to_string(),
        };
        let root = path.canonicalize().map_err(|_| missing())?;
        if !root.join("index.html").is_file() {
            return Err(missing());
        }
        Ok(Self { root })
    }

    /// Resolve a request path to a file inside the output directory
    ///
    /// Only plain segments are accepted, and the canonical result must stay
    /// inside the root so symlinks cannot point outside it
    pub fn resolve(&self, request_path: &str) -> Option<PathBuf> {
        let relative = match request_path {
            "/" => "index.html",
            path => path.strip_prefix('/')?,
        };
        let safe_segments = relative.split('/').all(|segment| {
            !segment.is_empty()
                && segment != "."
                && segment != ".."
                && !segment.contains(['\\', '\0'])
        });
        if !safe_segments {
            return None;
        }
        let resolved = self.root.join(relative).canonicalize().ok()?;
        (resolved.starts_with(&self.root) && resolved.is_file()).then_some(resolved)
    }
}

/// Content type for a UI file by extension
pub fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("webp") => "image/webp",
        Some("woff2") => "font/woff2",
        Some("wasm") => "application/wasm",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Find the default UI output directory from the repository root
///
/// The repository root is the nearest ancestor of the current directory with a
/// workspace `Cargo.toml`, falling back to this crate's workspace
pub fn default_dist_dir() -> PathBuf {
    let from_cwd = std::env::current_dir().ok().and_then(|cwd| {
        cwd.ancestors()
            .find(|dir| {
                std::fs::read_to_string(dir.join("Cargo.toml"))
                    .is_ok_and(|text| text.contains("[workspace]"))
            })
            .map(Path::to_path_buf)
    });
    let root = from_cwd.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
    root.join(UI_DIST_RELATIVE)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{UiDist, content_type};

    #[test]
    fn resolves_only_files_inside_the_root() {
        let temp = tempfile::tempdir().unwrap();
        let dist = temp.path().join("dist");
        fs::create_dir_all(dist.join("assets")).unwrap();
        fs::write(dist.join("index.html"), "<main></main>").unwrap();
        fs::write(dist.join("assets/app.js"), "export {}").unwrap();
        fs::write(temp.path().join("secret.txt"), "secret").unwrap();
        let ui = UiDist::open(&dist).unwrap();

        assert!(ui.resolve("/").unwrap().ends_with("index.html"));
        assert!(ui.resolve("/assets/app.js").is_some());
        assert!(ui.resolve("/assets").is_none());
        assert!(ui.resolve("/../secret.txt").is_none());
        assert!(ui.resolve("/assets/../../secret.txt").is_none());
        assert!(ui.resolve("//secret.txt").is_none());
        assert!(ui.resolve("/missing.js").is_none());
        assert_eq!(
            content_type(&dist.join("assets/app.js")),
            "text/javascript; charset=utf-8"
        );
    }

    #[test]
    fn missing_index_names_the_build_command() {
        let temp = tempfile::tempdir().unwrap();

        let error = UiDist::open(temp.path()).unwrap_err().to_string();

        assert!(error.contains("pnpm --dir recipes/speakrs/review-ui install && pnpm --dir recipes/speakrs/review-ui build"));
    }
}
