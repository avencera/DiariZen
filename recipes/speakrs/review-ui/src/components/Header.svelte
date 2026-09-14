<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { DECISION_LABELS } from "../lib/domain";

  let { desk }: { desk: Desk } = $props();

  const status = $derived(desk.saveStatusText());
  const current = $derived(desk.current);
  const progress = $derived(desk.session?.progress);
</script>

<header class="masthead">
  <span class="brand">Open Yap review</span>
  {#if current}
    <span class="where">
      <span>Window <strong>{desk.windowIndex + 1}</strong> of {desk.windows.length}</span>
      <span>{current.window.selection_kind}{current.window.stratum ? ` / ${current.window.stratum}` : ""}</span>
      <span class="chip chip-{current.state.decision.kind}">{DECISION_LABELS[current.state.decision.kind]}</span>
    </span>
  {/if}
  <span class="save-status tone-{status.tone}" role="status">
    {status.text}
    {#if status.retry}
      <button type="button" onclick={() => desk.retry()}>Retry</button>
    {/if}
  </span>
  {#if desk.mode !== "read_only"}
    <label class="toggle">
      <input type="checkbox" bind:checked={desk.autoAdvance} />
      Next window after save
    </label>
  {/if}
  <button type="button" onclick={() => (desk.dialog = { kind: "windows" })}>
    Windows
    {#if progress}
      <span class="tool-note">{progress.reviewed_count}/{progress.window_count} reviewed</span>
    {/if}
  </button>
</header>
