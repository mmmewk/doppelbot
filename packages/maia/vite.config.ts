import { defineConfig } from "vite-plus";
import { playwright } from "vite-plus/test/browser-playwright";
import { resolve } from "node:path";

const fixturesPath = resolve(import.meta.dirname, "../../python/data/fixtures/maia3-5m.json");
const modelPath = resolve(import.meta.dirname, "../../python/artifacts/maia3-5m/model.onnx");
const embeddingsPath = resolve(import.meta.dirname, "../../python/artifacts/maia3-5m/elo_emb.json");

export default defineConfig({
  lint: {
    options: {
      typeAware: true,
      typeCheck: true,
    },
  },
  fmt: {},
  test: {
    browser: {
      enabled: true,
      provider: playwright(),
      instances: [{ browser: "chromium" }],
    },
  },
  define: {
    __MAIA_FIXTURES_URL__: JSON.stringify(`/@fs${fixturesPath}`),
    __MAIA_MODEL_URL__: JSON.stringify(`/@fs${modelPath}`),
    __MAIA_EMBEDDINGS_URL__: JSON.stringify(`/@fs${embeddingsPath}`),
  },
});
