(() => {
  const logic = window.ReviewLogic;
  const headers = { Accept: "application/json" };
  const state = {
    session: null,
    windows: [],
    index: 0,
    current: null,
    playheadFrame: 0,
    playing: false,
    solo: "all",
    draft: [],
    saveStatus: "saved",
    buffers: {},
    raw: {},
    gains: {},
    context: null,
    sources: {},
    startedAt: 0,
    startFrame: 0,
    readOnly: false,
    signoffMode: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setSave(status, detail) {
    state.saveStatus = status;
    const node = $("save-status");
    node.textContent = detail || status;
    node.className = status;
  }

  function setAudioStatus(text, bad) {
    const node = $("audio-status");
    node.textContent = text;
    node.className = bad ? "audio-status is-bad" : "audio-status";
  }

  function markSolo() {
    ["solo-emitted", "solo-a", "solo-b", "solo-ab"].forEach((id) => $(id).classList.remove("is-on"));
    const map = { all: "solo-emitted", a: "solo-a", b: "solo-b", ab: "solo-ab" };
    const node = $(map[state.solo] || "solo-emitted");
    if (node) node.classList.add("is-on");
  }

  async function api(path, options) {
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: { ...headers, ...(options && options.headers ? options.headers : {}) },
    });
    const text = await response.text();
    let payload = {};
    if (text) {
      payload = JSON.parse(text);
    }
    if (!response.ok) {
      const error = new Error((payload.error && payload.error.message) || response.statusText);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function ensureContext() {
    if (!state.context) {
      state.context = new AudioContext();
    }
    if (state.context.state === "suspended") {
      await state.context.resume();
    }
    return state.context;
  }

  async function loadSession() {
    const session = await api("/api/v1/session");
    state.session = session;
    state.windows = session.windows;
    state.readOnly = Boolean(session.read_only);
    state.signoffMode = Boolean(session.signoff_mode);
    $("readonly-banner").hidden = !state.readOnly;
    const nextId = session.progress && session.progress.next_window_id;
    const idx = state.windows.findIndex((item) => item.window_id === nextId);
    state.index = idx >= 0 ? idx : 0;
    renderProgress();
    setActionEnabled(!state.readOnly);
    await loadWindow(state.index);
  }

  function renderProgress() {
    const progress = state.session.progress;
    $("progress-summary").textContent =
      `${progress.reviewed_count + 1} of ${progress.window_count}` +
      (progress.next_window_id ? "" : ", complete");
    $("progress-split").textContent =
      `${progress.reviewed_count} reviewed · ${progress.signed_count} signed · uniform ${progress.uniform.reviewed}/${progress.uniform.total} · targeted ${progress.targeted.reviewed}/${progress.targeted.total}`;
  }

  function setActionEnabled(enabled) {
    [
      "confirm",
      "deny",
      "no-speech",
      "uncertain",
      "undo",
      "save-defects",
      "add-a",
      "add-b",
      "correct-now",
      "needs-follow-up",
      "save-uncertain",
    ].forEach((id) => {
      $(id).disabled = !enabled;
    });
  }

  async function loadWindow(index) {
    stopPlayback();
    state.index = clampIndex(index);
    const meta = state.windows[state.index];
    const payload = await api(`/api/v1/windows/${encodeURIComponent(meta.window_id)}`);
    state.current = payload;
    state.playheadFrame = 0;
    state.draft = logic.normalizeIntervals(payload.proposal.intervals || []);
    $("window-id").textContent = payload.window.window_id;
    $("window-kind").textContent = payload.window.selection_kind || "—";
    $("window-stratum").textContent = payload.window.stratum || "none";
    renderTranscript(payload.transcript.words || []);
    renderIntervals();
    restoreDefects(payload.state.defects);
    draw();
    setSave("saved", `revision ${payload.state.revision}`);
    setAudioStatus("Loading audio");
    try {
      await loadAudio(payload.media);
      setAudioStatus("Audio ready. Press Play.");
    } catch (error) {
      setAudioStatus(`Audio failed: ${error.message}`, true);
    }
    draw();
  }

  function clampIndex(index) {
    if (!state.windows.length) {
      return 0;
    }
    return Math.min(state.windows.length - 1, Math.max(0, index));
  }

  function renderTranscript(words) {
    const root = $("transcript-words");
    root.replaceChildren();
    words.forEach((word, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `word ${word.speaker}`;
      button.textContent = word.text;
      button.dataset.index = String(index);
      button.addEventListener("click", () => {
        state.playheadFrame = logic.seekWord(word);
        if (state.playing) {
          play();
        } else {
          draw();
        }
      });
      root.appendChild(button);
    });
  }

  function renderIntervals() {
    const root = $("interval-list");
    root.replaceChildren();
    state.draft.forEach((interval, index) => {
      const row = document.createElement("div");
      row.className = "interval";
      const speaker = document.createElement("select");
      speaker.innerHTML = '<option value="speaker_a">Speaker A</option><option value="speaker_b">Speaker B</option>';
      speaker.value = interval.speaker;
      speaker.addEventListener("change", () => {
        state.draft[index] = { ...interval, speaker: speaker.value };
        state.draft = logic.normalizeIntervals(state.draft);
        renderIntervals();
        draw();
      });
      const start = document.createElement("input");
      start.type = "number";
      start.value = String(interval.start_frame);
      start.addEventListener("change", () => {
        state.draft[index] = logic.resizeInterval(interval, "start", Number(start.value));
        state.draft = logic.normalizeIntervals(state.draft);
        renderIntervals();
        draw();
      });
      const end = document.createElement("input");
      end.type = "number";
      end.value = String(interval.end_frame);
      end.addEventListener("change", () => {
        state.draft[index] = logic.resizeInterval(interval, "end", Number(end.value));
        state.draft = logic.normalizeIntervals(state.draft);
        renderIntervals();
        draw();
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        state.draft.splice(index, 1);
        renderIntervals();
        draw();
      });
      row.append(speaker, start, end, remove);
      root.appendChild(row);
    });
  }

  function restoreDefects(defects) {
    $("defect-identity").value = defects.identity.kind;
    $("defect-clock").value = defects.clock.kind;
    $("defect-sync").value = defects.synchronization.kind;
    $("defect-redaction").value = defects.redaction.kind;
    $("defect-reason").value = defects.identity.reason || defects.clock.reason || "";
  }

  async function fetchWav(url) {
    const response = await fetch(`${url}?container=wav`, { credentials: "same-origin", headers });
    if (!response.ok) {
      throw new Error(`media ${response.status}`);
    }
    return response.arrayBuffer();
  }

  async function decodeBuffers() {
    const context = await ensureContext();
    if (state.buffers.emitted || !state.raw.emitted) {
      return;
    }
    const [emitted, speaker_a, speaker_b] = await Promise.all([
      context.decodeAudioData(state.raw.emitted.slice(0)),
      context.decodeAudioData(state.raw.speaker_a.slice(0)),
      context.decodeAudioData(state.raw.speaker_b.slice(0)),
    ]);
    state.buffers = { emitted, speaker_a, speaker_b };
  }

  async function loadAudio(media) {
    stopPlayback();
    state.buffers = {};
    const [emitted, speaker_a, speaker_b] = await Promise.all([
      fetchWav(media.emitted),
      fetchWav(media.speaker_a),
      fetchWav(media.speaker_b),
    ]);
    state.raw = { emitted, speaker_a, speaker_b };
    await decodeBuffers();
  }

  function stopPlayback() {
    Object.values(state.sources).forEach((source) => {
      try {
        source.stop();
      } catch (error) {
        return;
      }
    });
    state.sources = {};
    state.gains = {};
    state.playing = false;
    $("play-toggle").textContent = "Play";
  }

  function currentTimeSeconds() {
    if (!state.playing || !state.context) {
      return logic.frameToSeconds(state.playheadFrame);
    }
    return logic.frameToSeconds(state.startFrame) + (state.context.currentTime - state.startedAt);
  }

  async function play() {
    try {
      await ensureContext();
    } catch (error) {
      setAudioStatus(`Playback failed: ${error.message}`, true);
      return;
    }
    try {
      await decodeBuffers();
    } catch (error) {
      setAudioStatus(`Decode failed: ${error.message}`, true);
      return;
    }
    if (!state.buffers.emitted) {
      setAudioStatus("Audio is not loaded yet.", true);
      return;
    }
    stopPlayback();
    const offset = Math.min(logic.frameToSeconds(state.playheadFrame), state.buffers.emitted.duration - 0.01);
    const names = ["emitted", "speaker_a", "speaker_b"];
    names.forEach((name) => {
      const source = state.context.createBufferSource();
      const gain = state.context.createGain();
      source.buffer = state.buffers[name];
      gain.gain.value = gainFor(name);
      source.connect(gain);
      gain.connect(state.context.destination);
      source.start(0, Math.max(0, offset));
      state.sources[name] = source;
      state.gains[name] = gain;
    });
    state.startedAt = state.context.currentTime;
    state.startFrame = state.playheadFrame;
    state.playing = true;
    $("play-toggle").textContent = "Pause";
    setAudioStatus("");
    requestAnimationFrame(tick);
  }

  function gainFor(name) {
    if (state.solo === "all") {
      return 1;
    }
    if (state.solo === "ab") {
      return name === "emitted" ? 0 : 1;
    }
    if (state.solo === "emitted") {
      return name === "emitted" ? 1 : 0;
    }
    if (state.solo === "a") {
      return name === "speaker_a" ? 1 : 0;
    }
    if (state.solo === "b") {
      return name === "speaker_b" ? 1 : 0;
    }
    return 1;
  }

  function applySolo(value) {
    state.solo = value;
    markSolo();
    Object.keys(state.gains).forEach((name) => {
      state.gains[name].gain.value = gainFor(name);
    });
  }

  function tick() {
    if (!state.playing) {
      return;
    }
    const seconds = currentTimeSeconds();
    state.playheadFrame = logic.snapFrame(seconds);
    if (state.playheadFrame >= logic.WINDOW_FRAME_COUNT) {
      stopPlayback();
      state.playheadFrame = logic.WINDOW_FRAME_COUNT;
    }
    draw();
    if (state.playing) {
      requestAnimationFrame(tick);
    }
  }

  function draw() {
    const canvas = $("waveform");
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.fillStyle = "#0a0d12";
    ctx.fillRect(0, 0, width, height);
    if (state.buffers.emitted) {
      drawWave(ctx, state.buffers.emitted, 10, 84, "#9aa7bb");
    }
    if (state.buffers.speaker_a) {
      drawWave(ctx, state.buffers.speaker_a, 10, 36, "#6ea8ff");
    }
    if (state.buffers.speaker_b) {
      drawWave(ctx, state.buffers.speaker_b, 52, 36, "#ff8a6b");
    }
    drawLanes(ctx, state.current ? state.current.proposal.intervals : [], 110, "#3a465c");
    drawLanes(ctx, state.draft, 150, null);
    const x = (state.playheadFrame / logic.WINDOW_FRAME_COUNT) * width;
    ctx.strokeStyle = "#f4d35e";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    $("clock").textContent = `${logic.frameToSeconds(state.playheadFrame).toFixed(2)} / 30.00`;
    highlightWords();
  }

  function drawWave(ctx, buffer, top, laneHeight, color) {
    const data = buffer.getChannelData(0);
    const width = ctx.canvas.width;
    const step = Math.max(1, Math.floor(data.length / width));
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (let x = 0; x < width; x += 1) {
      let min = 1;
      let max = -1;
      const start = x * step;
      for (let i = 0; i < step && start + i < data.length; i += 1) {
        const value = data[start + i];
        if (value < min) min = value;
        if (value > max) max = value;
      }
      const y1 = top + (1 - (max + 1) / 2) * laneHeight;
      const y2 = top + (1 - (min + 1) / 2) * laneHeight;
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y2);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawLanes(ctx, intervals, top, color) {
    const width = ctx.canvas.width;
    (intervals || []).forEach((interval) => {
      ctx.fillStyle = color || (interval.speaker === "speaker_a" ? "#6ea8ff" : "#ff8a6b");
      const x = (interval.start_frame / logic.WINDOW_FRAME_COUNT) * width;
      const w = ((interval.end_frame - interval.start_frame) / logic.WINDOW_FRAME_COUNT) * width;
      const y = interval.speaker === "speaker_a" ? top : top + 22;
      ctx.fillRect(x, y, Math.max(w, 1), 16);
    });
  }

  function highlightWords() {
    const seconds = logic.frameToSeconds(state.playheadFrame);
    const words = (state.current && state.current.transcript.words) || [];
    document.querySelectorAll(".word").forEach((node, index) => {
      const word = words[index];
      const active = word && seconds >= word.window_start_seconds && seconds < word.window_end_seconds;
      node.classList.toggle("active", Boolean(active));
    });
  }

  async function postEvent(action) {
    if (state.readOnly) {
      setSave("failed", "read-only");
      return;
    }
    const windowId = state.current.window.window_id;
    const revision = state.current.state.revision;
    const requestId = `${windowId}:${revision}:${action.kind}:${Date.now()}`;
    setSave("saving", "saving");
    try {
      const result = await api(`/api/v1/windows/${encodeURIComponent(windowId)}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: window.location.origin },
        body: JSON.stringify({
          request_id: requestId,
          base_revision: revision,
          action,
        }),
      });
      state.current.state = result.state;
      state.session.progress = result.progress;
      renderProgress();
      setSave("saved", `revision ${result.state.revision}`);
    } catch (error) {
      setSave("failed", "save failed — retry");
      throw error;
    }
  }

  function defectPayload(kind, reason) {
    if (kind === "unresolved") {
      return { kind, reason: reason || "unresolved" };
    }
    return { kind };
  }

  function clockPayload(kind, reason) {
    if (kind === "not_reviewed") {
      return { kind };
    }
    if (kind === "clear") {
      return { kind, measurement: { kind: "absent" } };
    }
    return { kind, reason: reason || "unresolved", measurement: { kind: "absent" } };
  }

  $("play-toggle").addEventListener("click", () => {
    if (state.playing) {
      const seconds = currentTimeSeconds();
      stopPlayback();
      state.playheadFrame = logic.snapFrame(seconds);
      draw();
      return;
    }
    play();
  });
  $("solo-emitted").addEventListener("click", () => applySolo("all"));
  $("solo-a").addEventListener("click", () => applySolo("a"));
  $("solo-b").addEventListener("click", () => applySolo("b"));
  $("solo-ab").addEventListener("click", () => applySolo("ab"));
  $("confirm").addEventListener("click", () => postEvent({ kind: "confirm" }));
  $("no-speech").addEventListener("click", () => postEvent({ kind: "no_speech" }));
  $("undo").addEventListener("click", () => {
    const hash = state.current.state.last_review_event_hash;
    if (!hash) {
      return;
    }
    postEvent({ kind: "undo", reverted_event_hash: hash });
  });
  $("deny").addEventListener("click", () => {
    $("deny-panel").hidden = false;
  });
  $("cancel-deny").addEventListener("click", () => {
    $("deny-panel").hidden = true;
  });
  $("correct-now").addEventListener("click", () => {
    $("deny-panel").hidden = true;
    postEvent({ kind: "correct", activity: logic.normalizeIntervals(state.draft) });
  });
  $("needs-follow-up").addEventListener("click", () => {
    postEvent({
      kind: "needs_follow_up",
      reason: $("follow-reason").value,
      scope: { kind: "whole_window" },
    });
    $("deny-panel").hidden = true;
  });
  $("uncertain").addEventListener("click", () => {
    $("uncertain-panel").hidden = false;
  });
  $("cancel-uncertain").addEventListener("click", () => {
    $("uncertain-panel").hidden = true;
  });
  $("save-uncertain").addEventListener("click", () => {
    postEvent({
      kind: "uncertain",
      reason: $("uncertain-reason").value,
      scope: { kind: "whole_window" },
    });
    $("uncertain-panel").hidden = true;
  });
  $("previous").addEventListener("click", () => loadWindow(state.index - 1));
  $("next").addEventListener("click", () => loadWindow(state.index + 1));
  $("add-a").addEventListener("click", () => {
    state.draft.push({
      speaker: "speaker_a",
      start_frame: state.playheadFrame,
      end_frame: Math.min(state.playheadFrame + 5, logic.WINDOW_FRAME_COUNT),
    });
    state.draft = logic.normalizeIntervals(state.draft);
    renderIntervals();
    draw();
  });
  $("add-b").addEventListener("click", () => {
    state.draft.push({
      speaker: "speaker_b",
      start_frame: state.playheadFrame,
      end_frame: Math.min(state.playheadFrame + 5, logic.WINDOW_FRAME_COUNT),
    });
    state.draft = logic.normalizeIntervals(state.draft);
    renderIntervals();
    draw();
  });
  $("save-defects").addEventListener("click", () => {
    const reason = $("defect-reason").value;
    postEvent({
      kind: "set_defects",
      defects: {
        identity: defectPayload($("defect-identity").value, reason),
        clock: clockPayload($("defect-clock").value, reason),
        synchronization: defectPayload($("defect-sync").value, reason),
        redaction: defectPayload($("defect-redaction").value, reason),
      },
    });
  });
  $("waveform").addEventListener("click", (event) => {
    const bounds = event.target.getBoundingClientRect();
    const ratio = (event.clientX - bounds.left) / bounds.width;
    state.playheadFrame = logic.snapFrame(ratio * 30);
    if (state.playing) {
      play();
    } else {
      draw();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
      return;
    }
    const result = logic.applyKey(
      { playheadFrame: state.playheadFrame, playing: state.playing },
      event.key,
      { shift: event.shiftKey, readOnly: state.readOnly },
    );
    if (result.navigate === "previous") {
      event.preventDefault();
      loadWindow(state.index - 1);
      return;
    }
    if (result.navigate === "next") {
      event.preventDefault();
      loadWindow(state.index + 1);
      return;
    }
    if (result.action === "confirm") $("confirm").click();
    if (result.action === "deny") $("deny").click();
    if (result.action === "no_speech") $("no-speech").click();
    if (result.action === "uncertain") $("uncertain").click();
    if (result.action === "undo") $("undo").click();
    if (result.playing !== state.playing && (event.key === " " || event.key === "Space")) {
      event.preventDefault();
      $("play-toggle").click();
    }
    if (result.playheadFrame !== state.playheadFrame) {
      event.preventDefault();
      state.playheadFrame = result.playheadFrame;
      draw();
    }
  });

  markSolo();
  loadSession().catch((error) => {
    setSave("failed", error.message);
    setAudioStatus(error.message, true);
  });
})();
