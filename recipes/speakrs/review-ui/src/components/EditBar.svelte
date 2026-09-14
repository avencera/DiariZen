<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { SPEAKER_LABELS } from "../lib/domain";
  import { formatRange } from "../lib/frames";

  let { desk }: { desk: Desk } = $props();
</script>

<div class="tool-group" role="group" aria-label="Edits">
  <button type="button" aria-pressed={desk.tool === "erase"} onclick={() => desk.toggleErase()}>
    Erase tool <kbd>E</kbd>
  </button>
  <button type="button" disabled={desk.buffers === null} onclick={() => desk.draftFromAudio()}>
    Draft from audio <kbd>D</kbd>
  </button>
  <button type="button" disabled={!desk.canUndoEdit} onclick={() => desk.undoEdit()}>Undo edit</button>
  <button type="button" disabled={!desk.canRedoEdit} onclick={() => desk.redoEdit()}>Redo</button>
  <button type="button" onclick={() => desk.resetToProposal()}>Reset to proposal</button>
  <button type="button" disabled={desk.draft.length === 0} onclick={() => desk.clearAll()}>Clear all</button>
  {#if desk.selectedRange}
    <button type="button" onclick={() => desk.deleteSelectedRange()}>
      Delete range <kbd>Del</kbd>
    </button>
    <span class="tool-note">{SPEAKER_LABELS[desk.selectedRange.speaker]} {formatRange(desk.selectedRange)}</span>
  {/if}
  {#if desk.draftSource !== "window" && desk.dirty}
    <span class="tool-note draft-note">Draft from audio — listen and fix, then save</span>
  {/if}
  <span class="tool-note">
    {desk.changedRanges === 0
      ? "Same as proposal"
      : `${desk.changedRanges} ${desk.changedRanges === 1 ? "range" : "ranges"} changed vs proposal`}
  </span>
</div>
