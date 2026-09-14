<script lang="ts">
  import Modal from "./Modal.svelte";
  import type { Desk } from "../lib/desk.svelte";
  import { DECISION_LABELS } from "../lib/domain";

  let { desk }: { desk: Desk } = $props();

  const open = $derived(desk.dialog.kind === "windows");
  const progress = $derived(desk.session?.progress);
</script>

<Modal {open} label="Windows" variant="drawer" onclose={() => desk.closeDialog()}>
  <div class="drawer-head">
    <h2>Windows</h2>
    <button type="button" onclick={() => desk.closeDialog()}>Close</button>
  </div>
  {#if progress}
    <p class="drawer-progress">
      {progress.reviewed_count} of {progress.window_count} reviewed, {progress.signed_count} signed off. Uniform {progress
        .uniform.reviewed}/{progress.uniform.total}, targeted {progress.targeted.reviewed}/{progress.targeted.total}.
    </p>
  {/if}
  <ol class="window-list">
    {#each desk.windows as item, index (item.window_id)}
      {@const isCurrent = item.window_id === desk.current?.window.window_id}
      <li>
        <button type="button" aria-current={isCurrent} onclick={() => desk.requestWindow(item.window_id)}>
          <span>{index + 1}</span>
          <span class="window-meta">
            {item.selection_kind}{item.stratum ? ` / ${item.stratum}` : ""} · {item.window_id}
          </span>
          <span class="chip chip-{item.decision.kind}">{DECISION_LABELS[item.decision.kind]}</span>
        </button>
      </li>
    {/each}
  </ol>
</Modal>
