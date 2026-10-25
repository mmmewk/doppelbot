import { createReadStream, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite-plus";
import { lazyPlugins } from "vite-plus";

// onnxruntime-web and Stockfish need SharedArrayBuffer (multithreaded WASM), which requires
// cross-origin isolation. Production sets the same headers in infra/web.ts.
const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// In dev, serve /models/<id>/<file> straight from the Python export. Production serves the same
// paths from the ModelBucket (infra/web.ts); scripts/models.sh uploads them.
const artifactsDir = resolve(import.meta.dirname, "../../python/artifacts");
const MODEL_FILES = new Set(["model.onnx", "moves.json", "elo_emb.json"]);

function devModels(): Plugin {
  return {
    name: "doppelbot-dev-models",
    configureServer(server) {
      server.middlewares.use("/models", (request, response, next) => {
        const [id, file] = (request.url ?? "").replace(/^\//, "").split("?")[0]!.split("/");
        const path = id && file && MODEL_FILES.has(file) ? join(artifactsDir, id, file) : undefined;
        if (!path || !id!.match(/^[\w.-]+$/) || !existsSync(path)) return next();
        response.setHeader(
          "Content-Type",
          file!.endsWith(".json") ? "application/json" : "application/octet-stream",
        );
        createReadStream(path).pipe(response);
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  lint: {
    plugins: ["react", "typescript", "oxc"],
    rules: {
      "react/rules-of-hooks": "error",
      "react/only-export-components": [
        "warn",
        {
          allowConstantExport: true,
        },
      ],
      "vite-plus/prefer-vite-plus-imports": "error",
    },
    options: {
      typeAware: true,
      typeCheck: true,
    },
    jsPlugins: [
      {
        name: "vite-plus",
        specifier: "vite-plus/oxlint-plugin",
      },
    ],
  },
  plugins: lazyPlugins(() => [react(), devModels()]),
});
