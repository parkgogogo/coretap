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
  const artifactFake = await Coretap.connect({
    command: fakeCoretapCommand(),
    backend: "device",
    device: "device-udid",
    daemon: "off",
    keepArtifacts: true,
  });
  const artifactObserve = await artifactFake.observe({ noVlm: true, noArtifacts: true });
  assert(
    argvIncludesInOrder(artifactObserve.argv, ["--keep-artifacts", "--no-artifacts", "observe", "--no-vlm"]),
    `artifact argv was not built as expected: ${artifactObserve.argv.join(" ")}`,
  );
  const screenshot = await artifactFake.screenshot({ label: "raw", out: "/tmp/coretap-node-shot.png" });
  assert(
    argvIncludesInOrder(screenshot.argv, ["--keep-artifacts", "screenshot", "--label", "raw", "--out", "/tmp/coretap-node-shot.png"]),
    `screenshot argv was not built as expected: ${screenshot.argv.join(" ")}`,
  );

  const fakeRun = await fake.openRun({ name: "node-argv" });
  await fakeRun.test("mobile-use helpers route through step", async (ui) => {
    const tap = await ui.tap("the App Store search field", { dryRun: true });
    const traceIndex = tap.argv.indexOf("--trace-id");
    assert(traceIndex >= 0, `tap argv should include trace id: ${tap.argv.join(" ")}`);
    assert(/^node-argv-/.test(tap.argv[traceIndex + 1]), `trace id should be based on run name: ${tap.argv.join(" ")}`);
    assert(argvIncludesInOrder(tap.argv, ["--trace-title", "node-argv"]), `tap argv should include trace title: ${tap.argv.join(" ")}`);
    assert(
      argvIncludesInOrder(tap.argv, [
        "--coredevice-tunnel-mode",
        "userspace",
        "step",
        "--action",
        '{"type":"tap","target":"the App Store search field"}',
        "--dry-run",
      ]),
      `tap argv was not built as expected: ${tap.argv.join(" ")}`,
    );

    const pointTap = await ui.tapPoint({ x: 0.25, y: 0.5 }, { dryRun: true });
    assert(
      argvIncludesInOrder(pointTap.argv, [
        "step",
        "--action",
        '{"type":"tapPoint","point":{"x":0.25,"y":0.5}}',
        "--dry-run",
      ]),
      `tapPoint argv was not built as expected: ${pointTap.argv.join(" ")}`,
    );

    const held = await ui.longPress({ x: 0.4, y: 0.6 }, { dryRun: true, durationMs: 1500, steps: 16 });
    assert(
      argvIncludesInOrder(held.argv, [
        "step",
        "--action",
        '{"type":"longPress","point":{"x":0.4,"y":0.6},"durationMs":1500,"steps":16}',
        "--dry-run",
      ]),
      `longPress argv was not built as expected: ${held.argv.join(" ")}`,
    );

    const typed = await ui.typeText("hello@example.com", { dryRun: true, charDelayMs: 0, interDelayMs: 0, pasteAt: { x: 0.2, y: 0.54 } });
    assert(
      argvIncludesInOrder(typed.argv, [
        "step",
        "--action",
        '{"type":"typeText","text":"hello@example.com","charDelayMs":0,"interDelayMs":0,"pasteAt":{"x":0.2,"y":0.54}}',
        "--dry-run",
      ]),
      `typeText argv was not built as expected: ${typed.argv.join(" ")}`,
    );

    const opened = await ui.openApp("App Store", { dryRun: true });
    assert(
      argvIncludesInOrder(opened.argv, [
        "step",
        "--action",
        '{"type":"openApp","name":"App Store"}',
        "--dry-run",
      ]),
      `openApp argv was not built as expected: ${opened.argv.join(" ")}`,
    );

    const switcher = await ui.appSwitcher({ dryRun: true });
    assert(
      argvIncludesInOrder(switcher.argv, [
        "step",
        "--action",
        '{"type":"appSwitcher"}',
        "--dry-run",
      ]),
      `appSwitcher argv was not built as expected: ${switcher.argv.join(" ")}`,
    );

    const terminated = await ui.terminateApp("com.apple.AppStore", { dryRun: true });
    assert(
      argvIncludesInOrder(terminated.argv, [
        "step",
        "--action",
        '{"type":"terminateApp","bundleId":"com.apple.AppStore"}',
        "--dry-run",
      ]),
      `terminateApp argv was not built as expected: ${terminated.argv.join(" ")}`,
    );

    const uninstalled = await ui.uninstallApp("小红书", { dryRun: true });
    assert(
      argvIncludesInOrder(uninstalled.argv, [
        "step",
        "--action",
        '{"type":"uninstallApp","name":"小红书"}',
        "--dry-run",
      ]),
      `uninstallApp argv was not built as expected: ${uninstalled.argv.join(" ")}`,
    );
  });
  await fakeRun.test("observe args use default JSON output", async (ui) => {
    const result = await ui.observe({
      label: "agent-eyes",
      maxLongSide: 1368,
      minConfidence: 10,
      noVlm: true,
    });
    assert(!result.argv.includes("--format"), `observe argv should use the single JSON stdout form: ${result.argv.join(" ")}`);
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
        "--min-confidence",
        "10",
        "--no-vlm",
      ]),
      `observe argv was not built as expected: ${result.argv.join(" ")}`,
    );
  });
  await fakeRun.test("step args expose single-action mobile-use runtime", async (ui) => {
    const result = await ui.step(
      { type: "tap", target: "the App Store search field" },
      {
        pageWaitMs: 2000,
        maxLongSide: 1368,
        noRefine: true,
        refineCropRatio: 0.42,
        noVlm: true,
      },
    );
    assert(
      argvIncludesInOrder(result.argv, [
        "step",
        "--action",
        '{"type":"tap","target":"the App Store search field"}',
        "--page-wait-ms",
        "2000",
        "--max-long-side",
        "1368",
        "--no-refine",
        "--refine-crop-ratio",
        "0.42",
        "--no-vlm",
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
