<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { DECISION_LABELS } from "../lib/domain";

  let { desk }: { desk: Desk } = $props();

  const state = $derived(desk.current?.state);
  const decision = $derived(state?.decision.kind ?? "pending");
  const eligible = $derived(decision !== "pending" && decision !== "signed_off" && decision !== "returned");
  const disabled = $derived(!eligible || desk.saving);
</script>

<div class="decide" role="group" aria-label="Sign-off">
  <span class="signoff-note">
    {#if decision === "pending"}
      This window has no saved review to sign off yet.
    {:else}
      {DECISION_LABELS[decision]}{state?.last_review_actor ? ` by ${state.last_review_actor}` : ""}
    {/if}
  </span>
  <button type="button" class="save" {disabled} onclick={() => desk.signoff({ decision: "accept" })}>
    Accept review <kbd>Enter</kbd>
  </button>
  <button type="button" {disabled} onclick={() => desk.askReason("return")}>Return to reviewer…</button>
</div>
