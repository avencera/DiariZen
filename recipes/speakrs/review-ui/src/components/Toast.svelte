<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";

  let { desk }: { desk: Desk } = $props();
</script>

<div class="toast" role="status">
  {#if desk.toast}
    <span>{desk.toast.message}</span>
    {#if desk.toast.revert}
      <button type="button" onclick={() => desk.revertFromToast()}>{desk.toast.revert}</button>
    {/if}
    <button type="button" aria-label="Dismiss" onclick={() => desk.dismissToast()}>✕</button>
  {/if}
</div>
