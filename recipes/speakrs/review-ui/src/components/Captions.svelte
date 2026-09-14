<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { SPEAKERS, type Speaker, type Word } from "../lib/domain";
  import { frameToSeconds, snapFrame } from "../lib/frames";

  let { desk }: { desk: Desk } = $props();

  // enough context to follow a sentence without turning the strip into a transcript page
  const CONTEXT_SECONDS = 4;

  const seconds = $derived(frameToSeconds(desk.playhead));
  const words = $derived(desk.current?.transcript.words ?? []);

  function nearby(speaker: Speaker): Word[] {
    return words.filter(
      (word) =>
        word.speaker === speaker &&
        word.window_end_seconds > seconds - CONTEXT_SECONDS &&
        word.window_start_seconds < seconds + CONTEXT_SECONDS,
    );
  }

  function isCurrent(word: Word): boolean {
    return word.window_start_seconds <= seconds && seconds < word.window_end_seconds;
  }

  function display(word: Word): string {
    return word.type === "word" ? word.text : `[${word.type}]`;
  }
</script>

<section class="captions" aria-labelledby="captions-title">
  <h2 id="captions-title">Transcript (ASR, not a label)</h2>
  {#each SPEAKERS as speaker (speaker)}
    {@const line = nearby(speaker)}
    <div class="caption-line">
      <span class="caption-who presence-{speaker}">{speaker === "speaker_a" ? "A" : "B"}</span>
      {#each line as word (word)}
        <button
          type="button"
          class="word"
          class:word-current={isCurrent(word)}
          class:word-sound={word.type !== "word"}
          title="Go to this word"
          onclick={() => desk.seek(snapFrame(word.window_start_seconds))}>{display(word)}</button
        >
      {:else}
        <span class="caption-empty">no transcript words near the playhead</span>
      {/each}
    </div>
  {/each}
</section>
