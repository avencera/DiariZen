/// <reference types="vitest/config" />
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// the review server listens here during development
const reviewServer = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // the server CSP forbids inline scripts, so keep every script and style in files
    assetsInlineLimit: 0,
    cssCodeSplit: false,
    modulePreload: { polyfill: false },
  },
  server: {
    proxy: {
      "/api": {
        target: reviewServer,
        changeOrigin: true,
        // the server only accepts writes whose origin matches its own host
        headers: { origin: reviewServer },
      },
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
