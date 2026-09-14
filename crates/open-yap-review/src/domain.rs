//! Typed review domain: frames, activity, scopes, defects, decisions, and actions

pub mod action;
pub mod activity;
pub mod decision;
pub mod defects;
pub mod frames;
pub mod scope;

pub use action::{ReviewAction, TransitionAction};
pub use activity::{Activity, ActivityInterval, NonEmptyActivity, Speaker};
pub use decision::Decision;
pub use defects::{ClockAssessment, ClockMeasurement, DefectAssessment, DefectSet};
pub use frames::{
    EndFrame, FRAME_SECONDS, FrameIndex, WINDOW_FRAME_COUNT, frames_intersecting_seconds,
};
pub use scope::{FrameRange, ReviewScope};
