//! Integer 20 ms frame coordinates on the 30-second window grid

use serde_json::{Value, json};

use crate::error::{ReviewError, ReviewResult};

/// Frames in one 30-second review window
pub const WINDOW_FRAME_COUNT: u16 = 1500;

/// Duration of one frame in seconds
pub const FRAME_SECONDS: f64 = 0.02;

/// An inclusive frame index in `0..=1499`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameIndex(u16);

/// An exclusive end frame in `0..=1500`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EndFrame(u16);

impl FrameIndex {
    /// Accept a start frame inside the grid
    pub fn new(value: u16) -> ReviewResult<Self> {
        check_range(i128::from(value), WINDOW_FRAME_COUNT - 1, "start_frame")?;
        Ok(Self(value))
    }

    /// Parse a JSON integer start frame
    pub fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        let frame = parse_integer(value, label)?;
        let frame = check_range(frame, WINDOW_FRAME_COUNT - 1, label)?;
        Ok(Self(frame))
    }

    /// Return the frame number
    pub fn get(self) -> u16 {
        self.0
    }
}

impl EndFrame {
    /// Accept an end frame inside the grid
    pub fn new(value: u16) -> ReviewResult<Self> {
        check_range(i128::from(value), WINDOW_FRAME_COUNT, "end_frame")?;
        Ok(Self(value))
    }

    /// Parse a JSON integer end frame
    pub fn parse(value: &Value, label: &str) -> ReviewResult<Self> {
        let frame = parse_integer(value, label)?;
        let frame = check_range(frame, WINDOW_FRAME_COUNT, label)?;
        Ok(Self(frame))
    }

    /// Return the frame number
    pub fn get(self) -> u16 {
        self.0
    }
}

fn parse_integer(value: &Value, label: &str) -> ReviewResult<i128> {
    let integer = match value {
        Value::Number(number) => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from)),
        _ => None,
    };
    integer.ok_or_else(|| ReviewError::contract(format!("{label} must be an integer frame index")))
}

fn check_range(value: i128, maximum: u16, label: &str) -> ReviewResult<u16> {
    if value < 0 || value > i128::from(maximum) {
        return Err(ReviewError::contract_with(
            format!("{label} is outside the 20 ms window grid"),
            json!({ "value": value as i64, "maximum": maximum }),
        ));
    }
    Ok(value as u16)
}

/// Return the half-open frame range a source interval in seconds touches
///
/// Any intersecting part of a 20 ms frame marks the whole frame. The epsilon
/// matches the Python conversion so proposal content hashes agree
pub fn frames_intersecting_seconds(start_seconds: f64, end_seconds: f64) -> Option<(u16, u16)> {
    if end_seconds <= start_seconds {
        return None;
    }
    let window_end = f64::from(WINDOW_FRAME_COUNT) * FRAME_SECONDS;
    if end_seconds <= 0.0 || start_seconds >= window_end {
        return None;
    }
    let start_frame = (start_seconds / FRAME_SECONDS + 1e-9).floor().max(0.0);
    let end_frame = (end_seconds / FRAME_SECONDS - 1e-9)
        .ceil()
        .min(f64::from(WINDOW_FRAME_COUNT));
    if end_frame <= start_frame {
        return None;
    }
    Some((start_frame as u16, end_frame as u16))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EndFrame, FrameIndex, frames_intersecting_seconds};

    #[test]
    fn frame_occupancy_marks_any_intersecting_part_of_a_frame() {
        assert_eq!(frames_intersecting_seconds(0.0, 0.02), Some((0, 1)));
        assert_eq!(frames_intersecting_seconds(0.0, 0.001), Some((0, 1)));
        assert_eq!(frames_intersecting_seconds(0.02, 0.04), Some((1, 2)));
        assert_eq!(frames_intersecting_seconds(0.019, 0.021), Some((0, 2)));
        assert_eq!(frames_intersecting_seconds(29.99, 30.0), Some((1499, 1500)));
        assert_eq!(frames_intersecting_seconds(30.0, 30.2), None);
        assert_eq!(frames_intersecting_seconds(1.0, 1.0), None);
        assert_eq!(frames_intersecting_seconds(-1.0, 0.0), None);
        assert_eq!(frames_intersecting_seconds(-1.0, 0.03), Some((0, 2)));
    }

    #[test]
    fn frames_require_integers_inside_the_grid() {
        assert!(FrameIndex::parse(&json!(1499), "start_frame").is_ok());
        assert!(FrameIndex::parse(&json!(1500), "start_frame").is_err());
        assert!(EndFrame::parse(&json!(1500), "end_frame").is_ok());
        assert!(EndFrame::parse(&json!(1501), "end_frame").is_err());
        assert!(FrameIndex::parse(&json!(-1), "start_frame").is_err());
        let float = FrameIndex::parse(&json!(5.0), "start_frame").unwrap_err();
        assert!(float.to_string().contains("integer"));
        assert!(FrameIndex::parse(&json!(true), "start_frame").is_err());
    }
}
