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

  const fake = await Coretap.connect({
    command: fakeCoretapCommand(),
    backend: "device",
    device: "device-udid",
    coredeviceTunnelMode: "userspace",
    profile: "builtin:mai-ui-2b-mlx-6bit@1",
    daemon: "off",
  });
  const fakeRun = await fake.openRun({ name: "node-argv" });
  await fakeRun.test("mobile-use helpers route through step", async (ui) => {
    const tap = await ui.tap("the App Store search field", { dryRun: true, expectChange: true });
    assert(
      argvIncludesInOrder(tap.argv, [
        "--coredevice-tunnel-mode",
        "userspace",
        "step",
        "--action",
        '{"schema":"coretap.action.v2","type":"tap","target":"the App Store search field"}',
        "--expect-change",
        "--dry-run",
      ]),
      `tap argv was not built as expected: ${tap.argv.join(" ")}`,
    );

    const typed = await ui.typeText("hello@example.com", { dryRun: true, charDelayMs: 0, interDelayMs: 0, pasteAt: { x: 0.2, y: 0.54 } });
    assert(
      argvIncludesInOrder(typed.argv, [
        "step",
        "--action",
        '{"schema":"coretap.action.v2","type":"typeText","text":"hello@example.com","charDelayMs":0,"interDelayMs":0,"pasteAt":{"x":0.2,"y":0.54}}',
        "--dry-run",
      ]),
      `typeText argv was not built as expected: ${typed.argv.join(" ")}`,
    );

    const opened = await ui.openApp("App Store", { dryRun: true });
    assert(
      argvIncludesInOrder(opened.argv, [
        "step",
        "--action",
        '{"schema":"coretap.action.v2","type":"openApp","name":"App Store"}',
        "--dry-run",
      ]),
      `openApp argv was not built as expected: ${opened.argv.join(" ")}`,
    );
  });
  await fakeRun.test("observe args use default JSON output", async (ui) => {
    const result = await ui.observe({
      label: "agent-eyes",
      maxLongSide: 1368,
      lang: "chi_sim+eng",
      psm: 11,
      ocrEngine: "vision",
      minConfidence: 10,
    });
    assert(!result.argv.includes("--format"), `observe argv should not include --format: ${result.argv.join(" ")}`);
    assert(
      argvIncludesInOrder(result.argv, [
        "--backend",
        "device",
        "--device",
        "device-udid",
        "observe",
        "--label",
        "agent-eyes",
        "--max-long-side",
        "1368",
        "--lang",
        "chi_sim+eng",
        "--psm",
        "11",
        "--ocr-engine",
        "vision",
        "--min-confidence",
        "10",
      ]),
      `observe argv was not built as expected: ${result.argv.join(" ")}`,
    );
  });
  await fakeRun.test("step args expose single-action mobile-use runtime", async (ui) => {
    const result = await ui.step(
      { schema: "coretap.action.v2", type: "tap", target: "the App Store search field" },
      {
        postWaitMs: 500,
        postTimeoutMs: 1500,
        pollIntervalMs: 250,
        expectChange: true,
        expectText: "搜索",
        ocrEngine: "vision",
        maxLongSide: 1368,
      },
    );
    assert(
      argvIncludesInOrder(result.argv, [
        "step",
        "--action",
        '{"schema":"coretap.action.v2","type":"tap","target":"the App Store search field"}',
        "--post-wait-ms",
        "500",
        "--post-timeout-ms",
        "1500",
        "--poll-interval-ms",
        "250",
        "--expect-text",
        "搜索",
        "--expect-change",
        "--ocr-engine",
        "vision",
        "--max-long-side",
        "1368",
      ]),
      `step argv was not built as expected: ${result.argv.join(" ")}`,
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
