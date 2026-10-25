// Writes defaults.json so Python (experiments, Fargate fit job) uses the same defaults.
import { writeFileSync } from "node:fs";
import { DEFAULT_CONFIG } from "../src/index.ts";

const out = new URL("../defaults.json", import.meta.url);
writeFileSync(out, JSON.stringify(DEFAULT_CONFIG, null, 2) + "\n");
console.log(`wrote ${out.pathname}`);
