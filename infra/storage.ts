// Single-table design; see docs/PLAN.md "Data and storage".
export const table = new sst.aws.Dynamo("Data", {
  fields: {
    pk: "string",
    sk: "string",
  },
  primaryIndex: { hashKey: "pk", rangeKey: "sk" },
  deletionProtection: $app.stage === "production",
});

// Normalized games, move occurrences, LoRA adapters. Private; read through the API.
export const dataBucket = new sst.aws.Bucket("DataBucket");

// ONNX models and embedding JSON, served at /models through the router.
// Objects are versioned by key and uploaded with immutable cache headers.
export const modelBucket = new sst.aws.Bucket("ModelBucket", {
  access: "cloudfront",
});
