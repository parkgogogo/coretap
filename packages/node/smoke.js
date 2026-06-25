const { Coretap } = require("./index.js");

async function main() {
  const client = await Coretap.connect({
    command: process.env.CORETAP_BIN
      ? [process.env.CORETAP_BIN]
      : ["python3", "-m", "coretap"],
    backend: process.env.CORETAP_BACKEND || "simulator",
    device: process.env.CORETAP_DEVICE || "booted",
  });

  const model = await client.model.status();
  if (model.profile !== "builtin:mai-ui-2b-mlx-6bit@1") {
    throw new Error("model status did not return the built-in profile");
  }

  const run = await client.openRun({ name: "node-smoke" });
  await run.test("wait works", async (ui) => {
    const result = await ui.wait(1);
    if (result.waitedMs !== 1) {
      throw new Error("wait result did not round-trip");
    }
  });
  await run.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
