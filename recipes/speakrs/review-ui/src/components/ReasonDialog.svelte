<script lang="ts">
  import Modal from "./Modal.svelte";
  import type { Desk, ReasonPurpose } from "../lib/desk.svelte";

  let { desk }: { desk: Desk } = $props();

  const TITLES: Readonly<Record<ReasonPurpose, string>> = {
    uncertain: "Not sure",
    needs_follow_up: "Needs follow-up",
    return: "Return to reviewer",
  };

  const PROMPTS: Readonly<Record<ReasonPurpose, string>> = {
    uncertain: "What makes this window hard to label?",
    needs_follow_up: "What needs to happen before this window can be labeled?",
    return: "What should the reviewer check again?",
  };

  let reason = $state("");
  const purpose = $derived(desk.dialog.kind === "reason" ? desk.dialog.purpose : null);

  $effect(() => {
    if (purpose) reason = "";
  });

  function submit(event: SubmitEvent): void {
    event.preventDefault();
    if (purpose) desk.saveWithReason(purpose, reason);
  }

  // plain Enter adds a line to the reason, so the modifier form saves from the keyboard
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    (event.currentTarget as HTMLTextAreaElement).form?.requestSubmit();
  }
</script>

<Modal open={purpose !== null} label={purpose ? TITLES[purpose] : "Reason"} onclose={() => desk.closeDialog()}>
  {#if purpose}
    <form onsubmit={submit}>
      <h2>{TITLES[purpose]}</h2>
      <label for="reason-text">{PROMPTS[purpose]}</label>
      <!-- svelte-ignore a11y_autofocus -->
      <textarea id="reason-text" bind:value={reason} onkeydown={onKeydown} autofocus required
      ></textarea>
      <div class="dialog-actions">
        <button type="button" onclick={() => desk.closeDialog()}>Cancel</button>
        <button type="submit" class="save" disabled={reason.trim() === ""}>
          Save: {TITLES[purpose]} <kbd>Ctrl/⌘ Enter</kbd>
        </button>
      </div>
    </form>
  {/if}
</Modal>
