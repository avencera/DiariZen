<script lang="ts">
  import { formatClock } from "../lib/frames";

  // kept apart so playback only re-renders these two text nodes
  let { frame }: { frame: number } = $props();
</script>

<span class="clock">
  <span class="clock-time">{formatClock(frame)}</span>
  <span class="clock-frame">frame {frame}</span>
</span>
