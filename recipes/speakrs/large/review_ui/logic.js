const FRAME_SECONDS = 0.02;
const WINDOW_FRAME_COUNT = 1500;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function snapFrame(seconds) {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) {
    return 0;
  }
  return clamp(Math.round(seconds / FRAME_SECONDS), 0, WINDOW_FRAME_COUNT);
}

function frameToSeconds(frame) {
  return clamp(frame, 0, WINDOW_FRAME_COUNT) * FRAME_SECONDS;
}

function normalizeIntervals(intervals) {
  const bySpeaker = { speaker_a: [], speaker_b: [] };
  for (const item of intervals) {
    if (!bySpeaker[item.speaker]) {
      continue;
    }
    const start = clamp(item.start_frame, 0, WINDOW_FRAME_COUNT - 1);
    const end = clamp(item.end_frame, start + 1, WINDOW_FRAME_COUNT);
    if (end <= start) {
      continue;
    }
    bySpeaker[item.speaker].push({ speaker: item.speaker, start_frame: start, end_frame: end });
  }
  const merged = [];
  for (const speaker of ["speaker_a", "speaker_b"]) {
    const items = bySpeaker[speaker].sort((a, b) => a.start_frame - b.start_frame || a.end_frame - b.end_frame);
    let current = null;
    for (const item of items) {
      if (!current) {
        current = { ...item };
        continue;
      }
      if (item.start_frame <= current.end_frame) {
        current.end_frame = Math.max(current.end_frame, item.end_frame);
        continue;
      }
      merged.push(current);
      current = { ...item };
    }
    if (current) {
      merged.push(current);
    }
  }
  merged.sort((a, b) => a.start_frame - b.start_frame || a.speaker.localeCompare(b.speaker));
  return merged;
}

function seekWord(word) {
  return snapFrame(word.window_start_seconds);
}

function applyKey(state, key, options) {
  const next = { ...state };
  const shift = Boolean(options && options.shift);
  const readOnly = Boolean(options && options.readOnly);
  const step = shift ? 10 : 1;
  if (key === " " || key === "Space") {
    next.playing = !state.playing;
    return next;
  }
  if (key === "ArrowLeft") {
    next.playheadFrame = clamp(state.playheadFrame - step, 0, WINDOW_FRAME_COUNT);
    return next;
  }
  if (key === "ArrowRight") {
    next.playheadFrame = clamp(state.playheadFrame + step, 0, WINDOW_FRAME_COUNT);
    return next;
  }
  if (key === "[" || key === "PageUp") {
    next.navigate = "previous";
    return next;
  }
  if (key === "]" || key === "PageDown") {
    next.navigate = "next";
    return next;
  }
  if (readOnly) {
    next.blocked = true;
    return next;
  }
  if (key === "c" || key === "C") {
    next.action = "confirm";
    return next;
  }
  if (key === "d" || key === "D") {
    next.action = "deny";
    return next;
  }
  if (key === "0") {
    next.action = "no_speech";
    return next;
  }
  if (key === "u" || key === "U") {
    next.action = "uncertain";
    return next;
  }
  if (key === "z" || key === "Z") {
    next.action = "undo";
    return next;
  }
  return next;
}

function resizeInterval(interval, edge, frame) {
  const snapped = clamp(frame, 0, WINDOW_FRAME_COUNT);
  if (edge === "start") {
    const start = clamp(snapped, 0, interval.end_frame - 1);
    return { ...interval, start_frame: start };
  }
  const end = clamp(snapped, interval.start_frame + 1, WINDOW_FRAME_COUNT);
  return { ...interval, end_frame: end };
}

function moveInterval(interval, deltaFrames) {
  const width = interval.end_frame - interval.start_frame;
  const start = clamp(interval.start_frame + deltaFrames, 0, WINDOW_FRAME_COUNT - width);
  return { ...interval, start_frame: start, end_frame: start + width };
}

const ReviewLogic = {
  FRAME_SECONDS,
  WINDOW_FRAME_COUNT,
  snapFrame,
  frameToSeconds,
  normalizeIntervals,
  seekWord,
  applyKey,
  resizeInterval,
  moveInterval,
};

if (typeof window !== "undefined") {
  window.ReviewLogic = ReviewLogic;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = ReviewLogic;
}
