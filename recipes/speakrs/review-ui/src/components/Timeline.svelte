<script lang="ts">
  import { onMount } from "svelte";
  import type { Desk } from "../lib/desk.svelte";
  import { SPEAKER_LABELS } from "../lib/domain";
  import { formatRange } from "../lib/frames";
  import {
    edgeAt,
    frameAtX,
    layoutFor,
    rangeUnderPointer,
    readPalette,
    rowAt,
    TimelineRenderer,
    type Scene,
    type View,
  } from "../lib/timeline";

  let { desk }: { desk: Desk } = $props();

  type Cursor = "seek" | "paint" | "erase" | "resize" | "range";

  let canvas: HTMLCanvasElement | undefined = $state();
  let section: HTMLElement | undefined = $state();
  let tip: HTMLDivElement | undefined = $state();
  let width = $state(0);
  let height = $state(0);
  let cursor = $state<Cursor>("seek");
  let renderer: TimelineRenderer | null = null;
  let pendingScene: Scene | null = null;
  let drawRequest: number | null = null;

  const view = $derived<View>({ start: desk.viewStart, span: desk.span });
  const layout = $derived(layoutFor(height));

  // draws at most once per animation frame, and only when something changed
  function scheduleDraw(scene: Scene): void {
    pendingScene = scene;
    if (drawRequest !== null) return;
    drawRequest = requestAnimationFrame(() => {
      drawRequest = null;
      if (!renderer || !pendingScene || !canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const scene = pendingScene;
      const sized = canvas.width === Math.round(scene.width * dpr) && canvas.height === Math.round(scene.layout.height * dpr);
      if (!sized) renderer.resize(scene.width, scene.layout.height, dpr);
      renderer.draw(scene);
    });
  }

  $effect(() => {
    if (width <= 0) return;
    scheduleDraw({
      width,
      layout,
      view,
      buffers: desk.buffers,
      draft: desk.draft,
      proposal: desk.proposal,
      playhead: desk.playhead,
      selection: desk.selection,
      selectedRange: desk.selectedRange,
    });
  });

  // CSSOM custom properties position the tooltip, because the CSP blocks style attributes
  $effect(() => {
    const hover = desk.hover;
    if (!tip || !hover) return;
    tip.style.setProperty("--tip-x", `${hover.x}px`);
    tip.style.setProperty("--tip-y", `${hover.y}px`);
  });

  function localPoint(event: PointerEvent): { x: number; y: number } {
    const bounds = (event.currentTarget as HTMLCanvasElement).getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  function cursorFor(x: number, y: number): Cursor {
    const hit = rowAt(layout, x, y);
    if (hit.kind !== "speaker" || !desk.editable) return "seek";
    if (edgeAt(desk.draft, hit.speaker, x, width, view)) return "resize";
    if (desk.tool === "erase") return "erase";
    return rangeUnderPointer(desk.draft, hit, frameAtX(x, width, view)) ? "range" : "paint";
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button !== 0) return;
    const { x, y } = localPoint(event);
    const hit = rowAt(layout, x, y);
    const frame = frameAtX(x, width, view);
    const edge = hit.kind === "speaker" ? edgeAt(desk.draft, hit.speaker, x, width, view) : null;
    (event.currentTarget as HTMLCanvasElement).setPointerCapture(event.pointerId);
    desk.hover = null;
    desk.pointerDown(hit, frame, x, event.altKey, edge);
  }

  function onPointerMove(event: PointerEvent): void {
    const { x, y } = localPoint(event);
    const target = event.currentTarget as HTMLCanvasElement;
    if (target.hasPointerCapture(event.pointerId)) {
      desk.pointerMove(frameAtX(x, width, view), x);
      return;
    }
    cursor = event.altKey && desk.editable && rowAt(layout, x, y).kind === "speaker" ? "erase" : cursorFor(x, y);
    const interval = rangeUnderPointer(desk.draft, rowAt(layout, x, y), frameAtX(x, width, view));
    desk.hover = interval ? { interval, x, y } : null;
  }

  function onPointerUp(event: PointerEvent): void {
    const { x } = localPoint(event);
    desk.pointerUp(frameAtX(x, width, view));
  }

  onMount(() => {
    const element = canvas;
    if (!element) return;
    renderer = new TimelineRenderer(element, readPalette(element));
    // the section sizes from the page flex layout, and the canvas fills it
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      width = entry.contentRect.width;
      height = entry.contentRect.height;
    });
    observer.observe(section ?? element);
    const clearHover = (): void => {
      desk.hover = null;
    };
    const cancel = (): void => desk.pointerCancel();
    // listeners are attached here because the canvas is not an interactive element to assistive tech
    element.addEventListener("pointerdown", onPointerDown);
    element.addEventListener("pointermove", onPointerMove);
    element.addEventListener("pointerup", onPointerUp);
    element.addEventListener("pointercancel", cancel);
    element.addEventListener("pointerleave", clearHover);
    return () => {
      observer.disconnect();
      element.removeEventListener("pointerdown", onPointerDown);
      element.removeEventListener("pointermove", onPointerMove);
      element.removeEventListener("pointerup", onPointerUp);
      element.removeEventListener("pointercancel", cancel);
      element.removeEventListener("pointerleave", clearHover);
      if (drawRequest !== null) cancelAnimationFrame(drawRequest);
    };
  });
</script>

<section bind:this={section} id="timeline" class="timeline" tabindex="-1" aria-label="Timeline">
  <canvas
    bind:this={canvas}
    class="cursor-{cursor}"
    aria-label="Time ruler, Mix waveform, and Speaker A and Speaker B lanes with your activity and the machine proposal. Drag on a speaker lane to paint, or use the keyboard shortcuts."
  ></canvas>
  {#if desk.hover}
    <div bind:this={tip} class="range-tip">
      {SPEAKER_LABELS[desk.hover.interval.speaker]}: {formatRange(desk.hover.interval)}
    </div>
  {/if}
</section>
