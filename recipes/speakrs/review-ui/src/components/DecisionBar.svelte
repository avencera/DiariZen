<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { revertLabel, SAVE_BUTTON_ID, saveLabel } from "../lib/save";

  let { desk }: { desk: Desk } = $props();

  const writable = $derived(desk.mode === "review" && !desk.saving);
</script>

<div class="decide" role="group" aria-label="Decision">
  <button type="button" id={SAVE_BUTTON_ID} class="save" disabled={!writable} onclick={() => desk.save()}>
    {saveLabel(desk.saveAction)}
    <kbd>Enter</kbd>
  </button>
  <button type="button" disabled={!writable} onclick={() => desk.askReason("uncertain")}>
    Not sure… <kbd>N</kbd>
  </button>
  <button type="button" disabled={!writable} onclick={() => desk.askReason("needs_follow_up")}>
    Needs follow-up… <kbd>F</kbd>
  </button>
  <button
    type="button"
    disabled={!writable || !desk.current?.latest_event_hash}
    onclick={() => desk.revertLastSave()}
  >
    {revertLabel(desk.current?.latest_event_kind ?? null)}
  </button>
</div>
