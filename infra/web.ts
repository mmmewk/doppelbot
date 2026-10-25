import { modelBucket } from "./storage.ts";

// onnxruntime-web and Stockfish need SharedArrayBuffer (multithreaded WASM), which requires
// cross-origin isolation. apps/web/vite.config.ts sets the same headers in dev.
const crossOriginIsolation = [
  `event.response.headers["cross-origin-opener-policy"] = { value: "same-origin" };`,
  `event.response.headers["cross-origin-embedder-policy"] = { value: "require-corp" };`,
].join("\n");

export const router = new sst.aws.Router("Router", {
  domain: $app.stage === "production" ? "doppelbot.mekoppe.com" : undefined,
  edge: {
    viewerResponse: { injection: crossOriginIsolation },
  },
});

router.routeBucket("/models", modelBucket, {
  rewrite: { regex: "^/models/(.*)$", to: "/$1" },
});

export const site = new sst.aws.StaticSite("Web", {
  path: "apps/web",
  build: {
    command: "pnpm run build",
    output: "dist",
  },
  router: { instance: router },
  dev: {
    command: "pnpm run dev",
  },
});
