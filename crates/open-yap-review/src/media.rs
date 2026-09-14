//! Allow-listed packet media, FLAC to WAV conversion, and byte ranges

use std::{
    collections::VecDeque,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde_json::json;

use crate::{
    error::{ReviewError, ReviewResult},
    text::py_strip,
};

/// Number of decoded WAV files kept in memory, three tracks for three windows
pub const WAV_CACHE_CAPACITY: usize = 9;

/// Largest media body served without a byte range
pub const MAX_FULL_MEDIA_BYTES: u64 = 32 * 1024 * 1024;

/// One of the three audio tracks of a window
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Emitted,
    SpeakerA,
    SpeakerB,
}

impl MediaKind {
    /// Parse the URL segment `emitted`, `speaker-a`, or `speaker-b`
    pub fn from_segment(segment: &str) -> Option<Self> {
        match segment {
            "emitted" => Some(Self::Emitted),
            "speaker-a" => Some(Self::SpeakerA),
            "speaker-b" => Some(Self::SpeakerB),
            _ => None,
        }
    }

    /// Return the URL segment
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Emitted => "emitted",
            Self::SpeakerA => "speaker-a",
            Self::SpeakerB => "speaker-b",
        }
    }

    /// Return the overlay `packet_files` key
    pub fn packet_file_key(self) -> &'static str {
        match self {
            Self::Emitted => "emitted",
            Self::SpeakerA => "reference_speaker_a",
            Self::SpeakerB => "reference_speaker_b",
        }
    }
}

/// Resolved media files of one window, each inside the packet root
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaFiles {
    pub emitted: PathBuf,
    pub speaker_a: PathBuf,
    pub speaker_b: PathBuf,
}

impl MediaFiles {
    /// Return the file for one track
    pub fn path(&self, kind: MediaKind) -> &Path {
        match kind {
            MediaKind::Emitted => &self.emitted,
            MediaKind::SpeakerA => &self.speaker_a,
            MediaKind::SpeakerB => &self.speaker_b,
        }
    }
}

/// Decode a FLAC file into a PCM16 WAV byte buffer with the same rate and channels
pub fn flac_to_wav(path: &Path) -> ReviewResult<Vec<u8>> {
    let file = File::open(path).map_err(|error| ReviewError::io("open", path, error))?;
    let mut reader = claxon::FlacReader::new(BufReader::new(file))
        .map_err(|error| decode_error(path, &error))?;
    let info = reader.streaminfo();
    let bits = info.bits_per_sample;
    let channels = u16::try_from(info.channels)
        .map_err(|_| ReviewError::preparation("media channel count is invalid"))?;
    let expected_samples = info
        .samples
        .unwrap_or(0)
        .saturating_mul(u64::from(channels));

    let mut pcm = Vec::with_capacity(usize::try_from(expected_samples * 2).unwrap_or(0));
    for sample in reader.samples() {
        let sample = sample.map_err(|error| decode_error(path, &error))?;
        pcm.extend_from_slice(&to_pcm16(sample, bits).to_le_bytes());
    }
    Ok(wav_bytes(info.sample_rate, channels, &pcm))
}

fn to_pcm16(sample: i32, bits: u32) -> i16 {
    // packet media is PCM_16, other depths are rescaled rather than rejected
    let scaled = match bits {
        16 => sample,
        bits if bits > 16 => sample >> (bits - 16),
        bits => sample << (16 - bits),
    };
    scaled.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn decode_error(path: &Path, error: &claxon::Error) -> ReviewError {
    ReviewError::preparation_with(
        "media file cannot be decoded as FLAC",
        json!({ "path": path.display().to_string(), "reason": error.to_string() }),
    )
}

/// Wrap little-endian PCM16 samples in a 44-byte RIFF WAVE header
pub fn wav_bytes(sample_rate: u32, channels: u16, pcm: &[u8]) -> Vec<u8> {
    let data_len = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let block_align = channels * 2;
    let byte_rate = sample_rate * u32::from(block_align);

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// A small least-recently-used cache of decoded WAV buffers
#[derive(Debug, Default)]
pub struct WavCache {
    entries: Mutex<VecDeque<(PathBuf, Arc<[u8]>)>>,
}

impl WavCache {
    /// Return the cached WAV for a path, decoding and inserting it on a miss
    pub fn get_or_decode(&self, path: &Path) -> ReviewResult<Arc<[u8]>> {
        if let Some(hit) = self.take(path) {
            return Ok(hit);
        }
        // decode outside the lock so one slow file does not block other tracks
        let decoded: Arc<[u8]> = flac_to_wav(path)?.into();
        let mut entries = self.lock();
        entries.retain(|(cached, _)| cached != path);
        entries.push_back((path.to_path_buf(), Arc::clone(&decoded)));
        while entries.len() > WAV_CACHE_CAPACITY {
            entries.pop_front();
        }
        Ok(decoded)
    }

    fn take(&self, path: &Path) -> Option<Arc<[u8]>> {
        let mut entries = self.lock();
        let position = entries.iter().position(|(cached, _)| cached == path)?;
        let entry = entries.remove(position)?;
        let bytes = Arc::clone(&entry.1);
        entries.push_back(entry);
        Some(bytes)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(PathBuf, Arc<[u8]>)>> {
        // a poisoned cache only holds immutable buffers, so keep using it
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// An inclusive byte range inside a body of known size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

impl ByteRange {
    /// Number of bytes in the range
    pub fn byte_count(self) -> u64 {
        self.end - self.start + 1
    }
}

/// Parse a single `bytes=` range, or return `None` when it is unsatisfiable
pub fn parse_byte_range(header: &str, size: u64) -> Option<ByteRange> {
    let spec = header.strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (start_text, end_text) = spec.split_once('-')?;
    let size = i128::from(size);
    let (start, end) = if start_text.is_empty() {
        let suffix = parse_python_int(end_text)?;
        if suffix <= 0 {
            return None;
        }
        ((size - suffix).max(0), size - 1)
    } else {
        let start = parse_python_int(start_text)?;
        let end = if end_text.is_empty() {
            size - 1
        } else {
            parse_python_int(end_text)?
        };
        (start, end)
    };
    if start < 0 || end < start || start >= size {
        return None;
    }
    let end = end.min(size - 1);
    Some(ByteRange {
        start: u64::try_from(start).ok()?,
        end: u64::try_from(end).ok()?,
    })
}

/// Parse an integer with the Python `int()` text rules for base 10
pub fn parse_python_int(text: &str) -> Option<i128> {
    let text = py_strip(text);
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    // python allows single underscores between digits
    let valid = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits
            .chars()
            .all(|character| character.is_ascii_digit() || character == '_');
    if !valid {
        return None;
    }
    let value: i128 = digits.replace('_', "").parse().ok()?;
    Some(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::{ByteRange, parse_byte_range, parse_python_int, wav_bytes};

    #[test]
    fn byte_ranges_follow_the_single_range_rules() {
        let range = |start, end| Some(ByteRange { start, end });
        assert_eq!(parse_byte_range("bytes=0-15", 100), range(0, 15));
        assert_eq!(parse_byte_range("bytes=10-", 100), range(10, 99));
        assert_eq!(parse_byte_range("bytes=-10", 100), range(90, 99));
        assert_eq!(parse_byte_range("bytes=-500", 100), range(0, 99));
        assert_eq!(parse_byte_range("bytes=90-500", 100), range(90, 99));
        assert_eq!(parse_byte_range("bytes= 1 - 2 ", 100), range(1, 2));
        assert_eq!(parse_byte_range("bytes=100-", 100), None);
        assert_eq!(parse_byte_range("bytes=5-4", 100), None);
        assert_eq!(parse_byte_range("bytes=-0", 100), None);
        assert_eq!(parse_byte_range("bytes=0-1,3-4", 100), None);
        assert_eq!(parse_byte_range("bytes=abc", 100), None);
        assert_eq!(parse_byte_range("bytes=x-1", 100), None);
        assert_eq!(parse_byte_range("items=0-1", 100), None);
        assert_eq!(parse_byte_range("bytes=--5", 100), None);
        assert_eq!(parse_byte_range("bytes=0-0", 0), None);
    }

    #[test]
    fn python_int_accepts_sign_whitespace_and_underscores() {
        assert_eq!(parse_python_int(" +1_000 "), Some(1000));
        assert_eq!(parse_python_int("-7"), Some(-7));
        assert_eq!(parse_python_int("1__0"), None);
        assert_eq!(parse_python_int("_1"), None);
        assert_eq!(parse_python_int(""), None);
        assert_eq!(parse_python_int("1.0"), None);
    }

    #[test]
    fn wav_header_describes_pcm16() {
        let bytes = wav_bytes(16_000, 1, &[1, 0, 2, 0]);

        assert_eq!(bytes.len(), 48);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 40);
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            16_000
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            32_000
        );
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 4);
    }
}
