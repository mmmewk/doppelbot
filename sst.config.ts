// oxlint-disable-next-line typescript/triple-slash-reference -- required by SST
/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "doppelbot",
      removal: input?.stage === "production" ? "retain" : "remove",
      protect: input?.stage === "production",
      home: "aws",
      providers: {
        aws: { region: "us-west-1", profile: "personal" },
      },
    };
  },
  async run() {
    const storage = await import("./infra/storage.ts");
    const web = await import("./infra/web.ts");
    const ingest = await import("./infra/ingest.ts");

    return {
      url: web.router.url,
      table: storage.table.name,
      dataBucket: storage.dataBucket.name,
      modelBucket: storage.modelBucket.name,
      ingestQueue: ingest.queue.url,
    };
  },
});
