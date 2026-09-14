<script lang="ts">
  import type { MarkTarget } from "../lib/activity";
  import type { Desk } from "../lib/desk.svelte";
  import { formatRange } from "../lib/frames";

  let { desk }: { desk: Desk } = $props();

  const disabled = $derived(!desk.editable || desk.selection === null);

  const marks: readonly { target: MarkTarget; label: string; key: string; tone: string }[] = [
    { target: "speaker_a", label: "Mark A", key: "A", tone: "mark-a" },
    { target: "speaker_b", label: "Mark B", key: "B", tone: "mark-b" },
    { target: "both", label: "Mark both", key: "S", tone: "" },
  ];
</script>

<div class="tool-group" role="group" aria-label="Selection">
  <span class="tool-note">
    {#if desk.selection}
      Selection {formatRange(desk.selection)}
    {:else}
      No selection. Drag on the ruler or Mix, or press I and O.
    {/if}
  </span>
  {#each marks as mark (mark.target)}
    <button type="button" class={mark.tone} {disabled} onclick={() => desk.mark(mark.target)}>
      {mark.label} <kbd>{mark.key}</kbd>
    </button>
  {/each}
  {#each marks as mark (mark.target)}
    <button type="button" {disabled} onclick={() => desk.clear(mark.target)}>
      {mark.label.replace("Mark", "Clear")}
    </button>
  {/each}
</div>
