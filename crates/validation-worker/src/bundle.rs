//! Bounded frozen development bundle extraction
//!
//! The extractor accepts USTAR, plain GNU, or legacy regular files and zero-size directories
//! It rejects links, devices, sparse entries, and GNU or PAX extension records

use std::{
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
};

use crate::{ArtifactCompression, WorkerError, payload::FrozenBundleRef};

const EXTRACTION_READ_BYTES: usize = 64 * 1024;
const BUNDLE_EXTRACTION_LIMITS: ExtractionLimits = ExtractionLimits {
    // keep one extraction below one quarter of the 64 GiB worker disk
    expanded_bytes: 16 * 1024 * 1024 * 1024,
    entries: 20_000,
    file_bytes: 4 * 1024 * 1024 * 1024,
    path_bytes: 512,
};

#[derive(Debug, Clone, Copy)]
struct ExtractionLimits {
    expanded_bytes: u64,
    entries: usize,
    file_bytes: u64,
    path_bytes: usize,
}

pub(crate) struct ExtractedBundle(PathBuf);

impl ExtractedBundle {
    pub(crate) fn extract(
        object: &Path,
        bundle: &FrozenBundleRef,
        cancelled: impl Fn() -> bool,
    ) -> Result<Self, WorkerError> {
        let parent = object.parent().ok_or(WorkerError::Layout)?;
        let root = parent.join(format!(".bundle.dev.{}", uuid::Uuid::now_v7()));
        fs::create_dir(&root).map_err(|_| WorkerError::Layout)?;
        let extracted = Self(root);

        let extraction = (|| {
            unpack_frozen_dev(
                object,
                extracted.root(),
                bundle,
                BUNDLE_EXTRACTION_LIMITS,
                &cancelled,
            )?;

            let manifest_type = fs::symlink_metadata(extracted.manifest())
                .map_err(|_| WorkerError::Layout)?
                .file_type();
            manifest_type
                .is_file()
                .then_some(())
                .ok_or(WorkerError::Layout)
        })();
        if let Err(error) = extraction {
            fs::remove_dir_all(extracted.root()).map_err(|_| WorkerError::Layout)?;

            return Err(error);
        }

        Ok(extracted)
    }

    pub(crate) fn root(&self) -> &Path {
        &self.0
    }

    pub(crate) fn manifest(&self) -> PathBuf {
        self.0.join("bundle.json")
    }
}

impl Drop for ExtractedBundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn unpack_frozen_dev<F>(
    object: &Path,
    root: &Path,
    bundle: &FrozenBundleRef,
    limits: ExtractionLimits,
    cancelled: &F,
) -> Result<(), WorkerError>
where
    F: Fn() -> bool,
{
    if cancelled() {
        return Err(WorkerError::Process);
    }

    let file = fs::File::open(object).map_err(|_| WorkerError::Layout)?;
    if is_zstd_archive(object, bundle) {
        let source = CancellableReader::new(file, cancelled);
        let decoder = zstd::Decoder::new(source).map_err(|_| extraction_error(cancelled))?;

        return unpack_tar(decoder, root, limits, cancelled);
    }

    unpack_tar(file, root, limits, cancelled)
}

fn is_zstd_archive(object: &Path, bundle: &FrozenBundleRef) -> bool {
    if bundle.compression == Some(ArtifactCompression::Zstd) {
        return true;
    }
    if bundle.media_type.contains("zstd") || bundle.media_type.contains("zst") {
        return true;
    }

    let Ok(mut file) = fs::File::open(object) else {
        return false;
    };
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic).is_ok() && magic == [0x28, 0xB5, 0x2F, 0xFD]
}

struct CancellableReader<'a, R, F> {
    inner: R,
    cancelled: &'a F,
}

impl<'a, R, F> CancellableReader<'a, R, F> {
    fn new(inner: R, cancelled: &'a F) -> Self {
        Self { inner, cancelled }
    }
}

impl<R: Read, F: Fn() -> bool> Read for CancellableReader<'_, R, F> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if (self.cancelled)() {
            return Err(io::Error::other("bundle extraction cancelled"));
        }

        let bounded = buffer.len().min(EXTRACTION_READ_BYTES);
        self.inner.read(&mut buffer[..bounded])
    }
}

fn unpack_tar<R, F>(
    reader: R,
    root: &Path,
    limits: ExtractionLimits,
    cancelled: &F,
) -> Result<(), WorkerError>
where
    R: Read,
    F: Fn() -> bool,
{
    let reader = CancellableReader::new(reader, cancelled);
    let mut archive = tar::Archive::new(reader);

    // raw iteration prevents tar from allocating unbounded GNU or PAX metadata
    let entries = archive
        .entries()
        .map_err(|_| extraction_error(cancelled))?
        .raw(true);
    let mut entry_count = 0_usize;
    let mut expanded_bytes = 0_u64;

    for entry in entries {
        if cancelled() {
            return Err(WorkerError::Process);
        }

        entry_count = entry_count.checked_add(1).ok_or(WorkerError::Layout)?;
        if entry_count > limits.entries {
            return Err(WorkerError::Layout);
        }

        let mut entry = entry.map_err(|_| extraction_error(cancelled))?;
        let entry_type = entry.header().entry_type();
        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(WorkerError::Layout);
        }

        let declared_size = entry.header().size().map_err(|_| WorkerError::Layout)?;
        if entry_type.is_dir() {
            if declared_size != 0 {
                return Err(WorkerError::Layout);
            }

            let destination = root.join(safe_archive_path(&entry, limits.path_bytes)?);
            fs::create_dir_all(destination).map_err(|_| WorkerError::Layout)?;
            continue;
        }
        if declared_size > limits.file_bytes {
            return Err(WorkerError::Layout);
        }

        expanded_bytes = expanded_bytes
            .checked_add(declared_size)
            .ok_or(WorkerError::Layout)?;
        if expanded_bytes > limits.expanded_bytes {
            return Err(WorkerError::Layout);
        }

        let destination = root.join(safe_archive_path(&entry, limits.path_bytes)?);
        write_archive_file(&mut entry, &destination, declared_size, cancelled)?;
    }

    Ok(())
}

fn safe_archive_path<R: Read>(
    entry: &tar::Entry<'_, R>,
    max_bytes: usize,
) -> Result<PathBuf, WorkerError> {
    let path = entry.path().map_err(|_| WorkerError::Layout)?;
    let encoded = path.to_str().ok_or(WorkerError::Layout)?;
    if encoded.len() > max_bytes {
        return Err(WorkerError::Layout);
    }

    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(segment) => relative.push(segment),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(WorkerError::Layout);
            }
        }
    }
    if relative.as_os_str().is_empty() && entry.header().entry_type().is_file() {
        return Err(WorkerError::Layout);
    }

    Ok(relative)
}

fn write_archive_file<R: Read, F: Fn() -> bool>(
    entry: &mut tar::Entry<'_, R>,
    destination: &Path,
    declared_size: u64,
    cancelled: &F,
) -> Result<(), WorkerError> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|_| WorkerError::Layout)?;
    }

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|_| WorkerError::Layout)?;
    let mut written = 0_u64;
    let mut buffer = [0_u8; EXTRACTION_READ_BYTES];
    loop {
        if cancelled() {
            return Err(WorkerError::Process);
        }

        let read = entry
            .read(&mut buffer)
            .map_err(|_| extraction_error(cancelled))?;
        if read == 0 {
            break;
        }
        written = written
            .checked_add(read as u64)
            .ok_or(WorkerError::Layout)?;
        if written > declared_size {
            return Err(WorkerError::Layout);
        }
        file.write_all(&buffer[..read])
            .map_err(|_| WorkerError::Layout)?;
    }
    if written != declared_size {
        return Err(WorkerError::Layout);
    }

    file.flush().map_err(|_| WorkerError::Layout)
}

fn extraction_error(cancelled: &impl Fn() -> bool) -> WorkerError {
    if cancelled() {
        WorkerError::Process
    } else {
        WorkerError::Layout
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::Cursor,
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use cloudeck_core::{ArtifactLocation, Sha256Digest};
    use sha2::{Digest, Sha256};
    use tar::{Builder, EntryType, Header};
    use tempfile::tempdir;

    use super::{ExtractedBundle, ExtractionLimits, unpack_tar};
    use crate::{WorkerError, payload::FrozenBundleRef};

    const TEST_LIMITS: ExtractionLimits = ExtractionLimits {
        expanded_bytes: 32,
        entries: 4,
        file_bytes: 16,
        path_bytes: 32,
    };

    #[test]
    fn extraction_accepts_a_python_ustar_bundle() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source");
        let object = directory.path().join("bundle.tar");
        fs::create_dir_all(source.join("audio/AMI")).unwrap();
        fs::write(source.join("bundle.json"), b"{}").unwrap();
        fs::write(source.join("audio/AMI/sample.flac"), b"audio").unwrap();
        let script = r#"import pathlib, sys, tarfile
source = pathlib.Path(sys.argv[1])
def supported(info):
    if not (info.isfile() or info.isdir()):
        raise ValueError("unsupported source type")
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info
with tarfile.open(sys.argv[2], "w", format=tarfile.USTAR_FORMAT) as archive:
    archive.add(source, arcname=".", recursive=True, filter=supported)
"#;
        let output = Command::new("python3")
            .args([
                "-c",
                script,
                source.to_str().unwrap(),
                object.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = fs::read(&object).unwrap();
        let bundle = bundle_ref(&bytes);

        let extracted = ExtractedBundle::extract(&object, &bundle, || false).unwrap();
        assert_eq!(fs::read(extracted.manifest()).unwrap(), b"{}");
        assert_eq!(
            fs::read(extracted.root().join("audio/AMI/sample.flac")).unwrap(),
            b"audio"
        );
    }

    #[test]
    fn extraction_accepts_a_legacy_regular_file_header() {
        let mut builder = Builder::new(Vec::new());
        let mut header = Header::new_old();
        header.set_size(2);
        header.set_mode(0o644);
        header.set_entry_type(EntryType::file());
        header.set_cksum();
        builder
            .append_data(&mut header, "bundle.json", b"{}".as_slice())
            .unwrap();
        let archive = builder.into_inner().unwrap();
        let root = tempdir().unwrap();

        unpack_tar(Cursor::new(archive), root.path(), TEST_LIMITS, &|| false).unwrap();
        assert_eq!(fs::read(root.path().join("bundle.json")).unwrap(), b"{}");
    }

    #[test]
    fn extraction_rejects_expanded_bytes_over_the_limit() {
        let archive = archive_with_file("bundle.json", b"12345");
        let root = tempdir().unwrap();
        let limits = ExtractionLimits {
            expanded_bytes: 4,
            ..TEST_LIMITS
        };

        assert!(matches!(
            unpack_tar(Cursor::new(archive), root.path(), limits, &|| false),
            Err(WorkerError::Layout)
        ));
        assert!(!root.path().join("bundle.json").exists());
    }

    #[test]
    fn extraction_rejects_too_many_entries() {
        let mut builder = Builder::new(Vec::new());
        append_file(&mut builder, "one", b"1");
        append_file(&mut builder, "two", b"2");
        let archive = builder.into_inner().unwrap();
        let root = tempdir().unwrap();
        let limits = ExtractionLimits {
            entries: 1,
            ..TEST_LIMITS
        };

        assert!(matches!(
            unpack_tar(Cursor::new(archive), root.path(), limits, &|| false),
            Err(WorkerError::Layout)
        ));
    }

    #[test]
    fn extraction_rejects_links_devices_fifos_and_extension_records() {
        for entry_type in *b"12346LKxg" {
            let archive = archive_with_type(EntryType::new(entry_type), 0);
            let root = tempdir().unwrap();

            assert!(matches!(
                unpack_tar(Cursor::new(archive), root.path(), TEST_LIMITS, &|| false),
                Err(WorkerError::Layout)
            ));
        }
    }

    #[test]
    fn extraction_rejects_unsafe_paths() {
        let mut header = Header::new_gnu();
        header.set_size(0);
        header.set_entry_type(EntryType::file());
        header.as_mut_bytes()[..9].copy_from_slice(b"../escape");
        header.set_cksum();
        let root = tempdir().unwrap();

        assert!(matches!(
            unpack_tar(
                Cursor::new(header.as_bytes().to_vec()),
                root.path(),
                TEST_LIMITS,
                &|| false,
            ),
            Err(WorkerError::Layout)
        ));
    }

    #[test]
    fn oversized_extension_metadata_is_rejected_from_its_header() {
        let archive = archive_with_type(EntryType::new(b'x'), u64::MAX / 2);
        let root = tempdir().unwrap();

        assert!(matches!(
            unpack_tar(Cursor::new(archive), root.path(), TEST_LIMITS, &|| false),
            Err(WorkerError::Layout)
        ));
    }

    #[test]
    fn cancellation_removes_the_independent_partial_root() {
        let directory = tempdir().unwrap();
        let object = directory.path().join("bundle.tar");
        let bytes = archive_with_file("bundle.json", b"{}");
        fs::write(&object, &bytes).unwrap();
        let bundle = bundle_ref(&bytes);

        assert!(matches!(
            ExtractedBundle::extract(&object, &bundle, || true),
            Err(WorkerError::Process)
        ));
        let leftovers: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".bundle.dev.")
            })
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn cancellation_during_archive_reads_removes_the_partial_root() {
        let directory = tempdir().unwrap();
        let object = directory.path().join("bundle.tar");
        let bytes = archive_with_file("bundle.json", &[b'x'; 128 * 1024]);
        fs::write(&object, &bytes).unwrap();
        let bundle = bundle_ref(&bytes);
        let polls = AtomicUsize::new(0);

        let result = ExtractedBundle::extract(&object, &bundle, || {
            polls.fetch_add(1, Ordering::Relaxed) >= 4
        });
        assert!(matches!(result, Err(WorkerError::Process)));
        assert!(polls.load(Ordering::Relaxed) > 4);
        let leftovers: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".bundle.dev.")
            })
            .collect();
        assert!(leftovers.is_empty());
    }

    fn archive_with_file(path: &str, contents: &[u8]) -> Vec<u8> {
        let mut builder = Builder::new(Vec::new());
        append_file(&mut builder, path, contents);
        builder.into_inner().unwrap()
    }

    fn append_file(builder: &mut Builder<Vec<u8>>, path: &str, contents: &[u8]) {
        let mut header = Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_entry_type(EntryType::file());
        header.set_cksum();
        builder.append_data(&mut header, path, contents).unwrap();
    }

    fn archive_with_type(entry_type: EntryType, declared_size: u64) -> Vec<u8> {
        let mut header = Header::new_gnu();
        header.set_path("metadata").unwrap();
        header.set_size(declared_size);
        header.set_entry_type(entry_type);
        header.set_cksum();
        header.as_bytes().to_vec()
    }

    fn bundle_ref(bytes: &[u8]) -> FrozenBundleRef {
        FrozenBundleRef {
            content_digest: Sha256Digest::from_bytes(Sha256::digest(bytes).into()),
            byte_length: bytes.len() as u64,
            media_type: "application/x-tar".to_owned(),
            location: ArtifactLocation::new("r2://validation/bundle.tar").unwrap(),
            compression: None,
        }
    }
}
