<script lang="ts">
  import { PRESENCE_PROPOSAL, PRESENCE_READOUT } from "../lib/activity";
  import type { Desk } from "../lib/desk.svelte";
  import { formatClock } from "../lib/frames";

  let { desk }: { desk: Desk } = $props();
</script>

<!-- announcements would flood a screen reader during playback, so they pause while playing -->
<div class="readout" aria-live={desk.playing ? "off" : "polite"}>
  <p class="readout-now">
    At {formatClock(desk.playhead)} — <strong class="presence-{desk.presence}">{PRESENCE_READOUT[desk.presence]}</strong>
  </p>
  <p class="readout-machine">machine proposal: {PRESENCE_PROPOSAL[desk.proposalPresence]}</p>
  {#if desk.draftSource === "audio_auto" && desk.dirty}
    <p class="readout-warning" role="alert">
      Machine proposal covers under 1 s (transcript likely missing speech). Draft made from audio — check it.
    </p>
  {/if}
</div>
