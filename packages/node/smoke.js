const { Coretap } = require("./index.js");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const process = require("node:process");

const REPO_ROOT = path.resolve(__dirname, "../..");

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function pythonCoretapCommand() {
  return process.env.CORETAP_BIN
    ? [process.env.CORETAP_BIN]
    : ["python3", "-m", "coretap"];
}

function fakeCoretapCommand() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "coretap-node-smoke-"));
  const fake = path.join(dir, "fake-coretap.js");
  fs.writeFileSync(
    fake,
    [
      "console.log(JSON.stringify({",
      "  schema: 'coretap.response.v1',",
      "  ok: true,",
      "  command: 'fake',",
      "  requestId: 'req_fake',",
      "  durationMs: 0,",
      "  result: { argv: process.argv.slice(2) },",
      "  artifacts: [],",
      "  warnings: []",
      "}));",
    ].join("\n"),
    "utf8",
  );
  return [process.execPath, fake];
}

function missingModuleCommand() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "coretap-node-missing-module-"));
  const fake = path.join(dir, "missing-module.js");
  fs.writeFileSync(
    fake,
    "process.stderr.write('/usr/bin/python3: No module named coretap\\n'); process.exit(1);\n",
    "utf8",
  );
  return [process.execPath, fake];
}

function argvIncludesInOrder(argv, expected) {
  let cursor = 0;
  for (const token of expected) {
    const index = argv.indexOf(token, cursor);
    if (index === -1) return false;
    cursor = index + 1;
  }
  return true;
}

async function main() {
  const client = await Coretap.connect({
    command: pythonCoretapCommand(),
    backend: process.env.CORETAP_BACKEND || "simulator",
    device: process.env.CORETAP_DEVICE || "booted",
    daemon: "off",
    cwd: REPO_ROOT,
  });

  const model = await client.model.status();
  assert(model.profile === "builtin:mai-ui-2b-mlx-6bit@1", "model status did not return the built-in profile");

  const run = await client.openRun({ name: "node-smoke" });
  await run.test("wait works", async (ui) => {
    const result = await ui.wait(1);
    assert(result.waitedMs === 1, "wait result did not round-trip");
  });
  await run.close();

  const dryRunDevice = await Coretap.connect({
    command: pythonCoretapCommand(),
    backend: "device",
    device: "device-udid",
    daemon: "off",
    cwd: REPO_ROOT,
  });
  const dryRun = await dryRunDevice.openRun({ name: "node-device-dry-run" });
  await dryRun.test("device dry-run gestures work", async (ui) => {
    const tap = await ui.tapAt({ space: "normalized", x: 0.5, y: 0.5 }, { dryRun: true });
    assert(tap.tap.dryRun === true, "tapAt dry-run did not round-trip");
    const press = await ui.pressHome({ dryRun: true });
    assert(press.dryRun === true && press.button === "home", "pressHome dry-run did not round-trip");
    const typed = await ui.typeText("hello@example.com", { dryRun: true, charDelayMs: 0, interDelayMs: 0, pasteAt: { x: 0.2, y: 0.54 } });
    assert(typed.dryRun === true && typed.text.length === "hello@example.com".length, "typeText dry-run did not round-trip");
    const drag = await ui.drag({ x: 0.5, y: 0.8 }, { x: 0.5, y: 0.2 }, { dryRun: true });
    assert(drag.drag.dryRun === true, "drag dry-run did not round-trip");
    const scroll = await ui.scroll("down", { dryRun: true });
    assert(scroll.drag.dryRun === true, "scroll dry-run did not round-trip");
  });
  await dryRun.close();

  const fake = await Coretap.connect({
    command: fakeCoretapCommand(),
    backend: "device",
    device: "device-udid",
    coredeviceTunnelMode: "userspace",
    profile: "builtin:mai-ui-2b-mlx-6bit@1",
    daemon: "off",
  });
  const fakeRun = await fake.openRun({ name: "node-argv" });
  await fakeRun.test("tapText args include OCR options", async (ui) => {
    const result = await ui.tapText("搜索", {
      dryRun: true,
      lang: "chi_sim+eng",
      psm: 11,
      minConfidence: 50,
      caseSensitive: true,
    });
    assert(
      argvIncludesInOrder(result.argv, [
        "--coredevice-tunnel-mode",
        "userspace",
        "tap",
        "text",
        "搜索",
        "--dry-run",
        "--lang",
        "chi_sim+eng",
        "--psm",
        "11",
        "--min-confidence",
        "50",
        "--case-sensitive",
      ]),
      `tapText argv was not built as expected: ${result.argv.join(" ")}`,
    );
  });
  await fakeRun.close();

  const missingBinary = await Coretap.connect({
    binary: "coretap-definitely-not-installed",
    daemon: "off",
  });
  try {
    await missingBinary.checkEnvironment();
    throw new Error("missing binary unexpectedly passed environment check");
  } catch (error) {
    assert(error.code === "CORETAP_CLI_NOT_INSTALLED", `unexpected missing binary error code: ${error.code}`);
    assert(/install\.sh/.test(error.message), "missing binary error did not include install hint");
  }

  const missingModule = await Coretap.connect({
    command: missingModuleCommand(),
    daemon: "off",
  });
  try {
    await missingModule.checkEnvironment();
    throw new Error("missing python module unexpectedly passed environment check");
  } catch (error) {
    assert(error.code === "CORETAP_CLI_NOT_INSTALLED", `unexpected missing module error code: ${error.code}`);
    assert(/CORETAP_BIN/.test(error.message), "missing module error did not include binary override hint");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
