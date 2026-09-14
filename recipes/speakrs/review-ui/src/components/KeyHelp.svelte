<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { KEY_HELP } from "../lib/keys";

  let { desk }: { desk: Desk } = $props();
</script>

<details class="key-help" bind:open={desk.helpOpen}>
  <summary>Keyboard shortcuts <kbd>?</kbd></summary>
  <dl>
    {#each KEY_HELP as entry (entry.keys)}
      <div>
        <dt>{entry.keys}</dt>
        <dd>{entry.action}</dd>
      </div>
    {/each}
  </dl>
</details>
