<script lang="ts">
  import Clock from "./Clock.svelte";
  import { LISTEN_LABELS, type ListenMode } from "../lib/audio";
  import type { Desk } from "../lib/desk.svelte";
  import { ZOOM_LEVELS } from "../lib/view";

  let { desk }: { desk: Desk } = $props();

  const listenModes: readonly ListenMode[] = ["mix", "speaker_a", "speaker_b", "both"];
  const canGoBack = $derived(desk.windowIndex > 0);
  const canGoForward = $derived(desk.windowIndex >= 0 && desk.windowIndex < desk.windows.length - 1);
</script>

<div class="transport">
  <button
    type="button"
    class="play"
    aria-pressed={desk.playing}
    disabled={desk.audioStatus.kind !== "ready"}
    onclick={() => desk.togglePlay()}
  >
    {desk.playing ? "Pause" : "Play"}
    <kbd>Space</kbd>
  </button>
  <Clock frame={desk.playhead} />
  <div class="segmented" role="group" aria-label="Listen">
    <span class="segmented-label">Listen</span>
    {#each listenModes as mode, index (mode)}
      <button type="button" aria-pressed={desk.listen === mode} onclick={() => desk.setListen(mode)}>
        {LISTEN_LABELS[mode]}
        <kbd>{index + 1}</kbd>
      </button>
    {/each}
  </div>
  <div class="segmented" role="group" aria-label="Zoom">
    <span class="segmented-label">Zoom</span>
    {#each ZOOM_LEVELS as zoom (zoom)}
      <button type="button" aria-pressed={desk.zoom === zoom} onclick={() => desk.setZoom(zoom)}>{zoom} s</button>
    {/each}
  </div>
  {#if desk.audioStatus.kind === "loading"}
    <span class="audio-status">Loading audio…</span>
  {:else if desk.audioStatus.kind === "failed"}
    <span class="audio-status failed" role="alert">{desk.audioStatus.message}</span>
  {/if}
  <div class="window-nav">
    <button type="button" disabled={!canGoBack} onclick={() => desk.stepWindow(-1)}>◀ Prev <kbd>[</kbd></button>
    <button type="button" disabled={!canGoForward} onclick={() => desk.stepWindow(1)}>Next ▶ <kbd>]</kbd></button>
  </div>
</div>
