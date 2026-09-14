//! Content-addressed artifact cache with verify-on-hit admission

use std::{
    fs,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
};

use cloudeck_core::Sha256Digest;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Cache admission or verification failure
#[derive(Debug, Error)]
pub enum CacheError {
    /// Filesystem failure while staging or admitting an object
    #[error("artifact cache I/O failed")]
    Io,
    /// Cached or downloaded bytes did not match the declared digest and length
    #[error("artifact cache entry failed digest or length verification")]
    Identity,
}

impl From<io::Error> for CacheError {
    fn from(_: io::Error) -> Self {
        Self::Io
    }
}

/// Directory-backed cache keyed by SHA-256 digest
#[derive(Debug, Clone)]
pub struct ArtifactCache {
    root: PathBuf,
}

impl ArtifactCache {
    /// Opens a cache directory, creating it when absent
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, CacheError> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Returns the admitted path for a digest if present and valid
    pub fn verified_path(
        &self,
        digest: &Sha256Digest,
        byte_length: u64,
    ) -> Result<Option<PathBuf>, CacheError> {
        let path = self.path_for(digest);
        if !path.exists() {
            return Ok(None);
        }
        match verify_file(&path, digest, byte_length) {
            Ok(()) => Ok(Some(path)),
            Err(CacheError::Identity) => {
                let _ = fs::remove_file(&path);
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Stages bytes, verifies them, then atomically admits the object
    pub fn admit(
        &self,
        digest: &Sha256Digest,
        byte_length: u64,
        staged: &Path,
    ) -> Result<PathBuf, CacheError> {
        if let Err(error) = verify_file(staged, digest, byte_length) {
            let _ = fs::remove_file(staged);
            return Err(error);
        }
        let admitted = self.path_for(digest);
        if let Some(existing) = self.verified_path(digest, byte_length)? {
            let _ = fs::remove_file(staged);
            return Ok(existing);
        }
        fs::rename(staged, &admitted)?;
        verify_file(&admitted, digest, byte_length)?;
        Ok(admitted)
    }

    /// Unique staging path for an in-flight download
    #[must_use]
    pub fn staging_path(&self, digest: &Sha256Digest, suffix: &str) -> PathBuf {
        self.root.join(format!("{}.part.{suffix}", digest.as_str()))
    }

    fn path_for(&self, digest: &Sha256Digest) -> PathBuf {
        self.root.join(digest.as_str())
    }
}

/// Verifies one file's SHA-256 digest and exact byte length
pub fn verify_file(path: &Path, digest: &Sha256Digest, byte_length: u64) -> Result<(), CacheError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() != byte_length {
        return Err(CacheError::Identity);
    }
    let mut file = BufReader::new(fs::File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes_read = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }

        bytes_read = bytes_read
            .checked_add(read as u64)
            .ok_or(CacheError::Identity)?;
        if bytes_read > byte_length {
            return Err(CacheError::Identity);
        }
        hasher.update(&buffer[..read]);
    }
    if bytes_read != byte_length {
        return Err(CacheError::Identity);
    }

    let actual = Sha256Digest::from_bytes(hasher.finalize().into());
    if actual != *digest {
        return Err(CacheError::Identity);
    }
    Ok(())
}
