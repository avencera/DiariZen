<script lang="ts">
  import Modal from "./Modal.svelte";
  import type { Desk } from "../lib/desk.svelte";

  let { desk }: { desk: Desk } = $props();

  const open = $derived(desk.dialog.kind === "discard");
</script>

<Modal {open} label="Unsaved edits" onclose={() => desk.closeDialog()}>
  <h2>Unsaved edits</h2>
  <p>This window has edits or channel checks that are not saved. Leaving the window discards them.</p>
  <div class="dialog-actions">
    <button type="button" class="save" onclick={() => desk.closeDialog()}>Keep editing</button>
    <button type="button" onclick={() => desk.confirmDiscard()}>Discard and leave</button>
  </div>
</Modal>
