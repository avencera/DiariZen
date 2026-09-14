<script lang="ts">
  import { onMount } from "svelte";
  import Captions from "./components/Captions.svelte";
  import ChannelChecks from "./components/ChannelChecks.svelte";
  import DecisionBar from "./components/DecisionBar.svelte";
  import DiscardDialog from "./components/DiscardDialog.svelte";
  import EditBar from "./components/EditBar.svelte";
  import Header from "./components/Header.svelte";
  import KeyHelp from "./components/KeyHelp.svelte";
  import Legend from "./components/Legend.svelte";
  import Readout from "./components/Readout.svelte";
  import ReasonDialog from "./components/ReasonDialog.svelte";
  import SelectionTools from "./components/SelectionTools.svelte";
  import SignoffBar from "./components/SignoffBar.svelte";
  import Timeline from "./components/Timeline.svelte";
  import Toast from "./components/Toast.svelte";
  import Transport from "./components/Transport.svelte";
  import WindowList from "./components/WindowList.svelte";
  import { Desk } from "./lib/desk.svelte";
  import { commandForKey, focusTargetOf } from "./lib/keys";

  const desk = new Desk();

  onMount(() => {
    void desk.start();
  });

  function onKeydown(event: KeyboardEvent): void {
    // open dialogs own the keyboard, including Escape to close
    if (desk.dialog.kind !== "none") return;
    const command = commandForKey(event, focusTargetOf(event.target));
    if (command && desk.runCommand(command, event.repeat)) event.preventDefault();
  }

  function onKeyup(event: KeyboardEvent): void {
    desk.releaseKey(event.key);
  }

  function onBeforeUnload(event: BeforeUnloadEvent): void {
    if (desk.dirty) event.preventDefault();
  }
</script>

<svelte:window onkeydown={onKeydown} onkeyup={onKeyup} onbeforeunload={onBeforeUnload} />

<div class="page">
  <a class="skip-link" href="#timeline">Skip to timeline</a>
  {#if desk.load.kind === "loading"}
    <p class="state-message">Loading the review session…</p>
  {:else if desk.load.kind === "failed"}
    <p class="state-message tone-bad">Could not load the review session. {desk.load.message}</p>
  {:else}
    <Header {desk} />
    {#if desk.mode === "read_only"}
      <p class="banner banner-readonly">Listening only. This session cannot save decisions.</p>
    {:else if desk.mode === "signoff"}
      <p class="banner banner-signoff">Sign-off mode. Check the saved review, then accept it or return it.</p>
    {/if}
    {#if desk.load.session.quarantined.length > 0}
      <p class="banner banner-warning">
        {desk.load.session.quarantined.length} unfinished event files were set aside when the session loaded.
      </p>
    {/if}
    {#if desk.windowError}
      <p class="banner banner-warning">Could not open the window. {desk.windowError}</p>
    {/if}
    {#if desk.current}
      <main class="desk">
        <Transport {desk} />
        <div class="readout-row">
          <Readout {desk} />
          <!-- the readout bar has free space on the right, so the toast hides no controls -->
          <Toast {desk} />
        </div>
        <Timeline {desk} />
        <div class="under-timeline">
          <Legend />
          <Captions {desk} />
        </div>
        {#if desk.mode === "review"}
          <div class="tools">
            <SelectionTools {desk} />
            <EditBar {desk} />
          </div>
        {/if}
        <KeyHelp {desk} />
      </main>
      <footer class="dock">
        {#if desk.mode === "signoff"}
          <SignoffBar {desk} />
        {:else}
          <ChannelChecks {desk} />
          <DecisionBar {desk} />
        {/if}
      </footer>
    {/if}
    <WindowList {desk} />
    <ReasonDialog {desk} />
    <DiscardDialog {desk} />
  {/if}
</div>
