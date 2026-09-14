<script lang="ts">
  import type { Desk } from "../lib/desk.svelte";
  import { CHANNELS, type Channel } from "../lib/domain";
  import { CHANNEL_LABELS, SAVE_BUTTON_ID, type ChannelCheck } from "../lib/save";

  let { desk }: { desk: Desk } = $props();

  const disabled = $derived(desk.mode !== "review" || desk.saving);

  // a focused toggle would take the next Enter, so focus moves on to Save once every check is OK
  function focusSave(): void {
    document.getElementById(SAVE_BUTTON_ID)?.focus();
  }

  function toggleOk(channel: Channel, check: ChannelCheck): void {
    desk.setCheck(channel, check.kind === "ok" ? { kind: "not_reviewed" } : { kind: "ok" });
    if (CHANNELS.every((item) => desk.checks[item].kind === "ok")) focusSave();
  }

  function setAllOk(): void {
    desk.setAllChannelsOk();
    focusSave();
  }

  function toggleProblem(channel: Channel, check: ChannelCheck): void {
    desk.setCheck(channel, check.kind === "problem" ? { kind: "not_reviewed" } : { kind: "problem", reason: "" });
  }
</script>

<fieldset class="channels" {disabled}>
  <legend>Channel checks</legend>
  {#each CHANNELS as channel (channel)}
    {@const check = desk.checks[channel]}
    {@const label = CHANNEL_LABELS[channel]}
    {@const missing = desk.missingReasons.includes(channel)}
    <div class="channel">
      <div class="channel-row" role="group" aria-label="{label} check">
        <span class="channel-name">{label}</span>
        <button type="button" aria-pressed={check.kind === "ok"} onclick={() => toggleOk(channel, check)}>OK</button>
        <button type="button" aria-pressed={check.kind === "problem"} onclick={() => toggleProblem(channel, check)}>
          Problem
        </button>
        {#if check.kind === "not_reviewed"}
          <span class="channel-unset">not checked</span>
        {/if}
      </div>
      {#if check.kind === "problem"}
        <input
          type="text"
          class:missing
          aria-label="{label} problem reason"
          aria-invalid={missing}
          placeholder="What is wrong?"
          value={check.reason}
          oninput={(event) => desk.setCheck(channel, { kind: "problem", reason: event.currentTarget.value })}
        />
        {#if missing}
          <span class="channel-error">Add a reason to save a problem</span>
        {/if}
      {/if}
    </div>
  {/each}
  <button type="button" onclick={setAllOk}>All channels OK</button>
</fieldset>
