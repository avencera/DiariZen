<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    open: boolean;
    label: string;
    variant?: "dialog" | "drawer";
    onclose: () => void;
    children: Snippet;
  }

  let { open, label, variant = "dialog", onclose, children }: Props = $props();
  let element: HTMLDialogElement | undefined = $state();

  // the native modal gives focus trapping and Escape handling for free
  $effect(() => {
    if (!element) return;
    if (open && !element.open) element.showModal();
    if (!open && element.open) element.close();
  });
</script>

<dialog bind:this={element} class={variant} aria-label={label} {onclose}>
  {#if open}
    {@render children()}
  {/if}
</dialog>
