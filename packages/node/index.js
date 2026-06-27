const { spawn } = require("node:child_process");

const INSTALL_COMMAND = "curl -fsSL https://raw.githubusercontent.com/parkgogogo/coretap/main/install.sh | bash";

class CoretapError extends Error {
  constructor(message, response) {
    super(message);
    this.name = "CoretapError";
    this.response = response;
    this.code = response && response.error ? response.error.code : "CORETAP_ERROR";
    this.category = response && response.error ? response.error.category : "internal";
    this.stage = response && response.error ? response.error.stage : "client";
  }
}

class Coretap {
  constructor(options = {}) {
    this.binary = options.binary || process.env.CORETAP_BIN || "coretap";
    this.command = options.command || null;
    this.backend = options.backend || process.env.CORETAP_BACKEND || "simulator";
    this.device = options.device || process.env.CORETAP_DEVICE || "booted";
    this.developerDir = options.developerDir || process.env.DEVELOPER_DIR || undefined;
    this.coredeviceTunnelMode = options.coredeviceTunnelMode || process.env.CORETAP_COREDEVICE_TUNNEL_MODE || undefined;
    this.artifactRoot = options.artifactRoot || process.env.CORETAP_ARTIFACT_ROOT || undefined;
    this.profile = options.profile || process.env.CORETAP_PROFILE || undefined;
    this.daemonMode = options.daemon || process.env.CORETAP_DAEMON || undefined;
    this.cwd = options.cwd || process.env.CORETAP_CWD || undefined;
  }

  static async connect(options = {}) {
    return new Coretap(options);
  }

  static async checkEnvironment(options = {}) {
    const client = new Coretap(options);
    return client.checkEnvironment();
  }

  static async attachFromEnvironment() {
    return new CoretapRun(new Coretap({}));
  }

  async openRun() {
    return new CoretapRun(this);
  }

  async withSession(options, body) {
    const run = await this.openRun(options);
    try {
      return await body(new IosVisualUi(this));
    } finally {
      await run.close();
    }
  }

  async checkEnvironment(options = {}) {
    return this.status({ ...options, daemon: "off" });
  }

  get model() {
    return {
      install: (options = {}) => {
        const args = ["model", "install"];
        if (options.force) args.push("--force");
        if (options.dryRun) args.push("--dry-run");
        return this._run(args, options);
      },
      check: (options = {}) => {
        const args = ["model", "check"];
        if (options.deep) args.push("--deep");
        return this._run(args, options);
      },
      warm: (options = {}) => this._run(["model", "warm"], options),
      status: (options = {}) => this._run(["model", "status"], options),
      stop: (options = {}) => this._run(["model", "stop"], options),
      cache: (options = {}) => this._run(["model", "cache"], options),
      run: (options = {}) => {
        const args = ["model", "run"];
        if (options.image) args.push("--image", options.image);
        if (options.target) args.push("--target", options.target);
        return this._run(args, options);
      },
      gc: (options = {}) => {
        const args = ["model", "gc"];
        if (options.dryRun) args.push("--dry-run");
        return this._run(args, options);
      },
    };
  }

  get ocr() {
    return {
      status: (options = {}) => this._run(["ocr", "status"], options),
      check: (options = {}) => this._run(["ocr", "check"], options),
    };
  }

  get daemon() {
    return {
      start: (options = {}) => this._run(["daemon", "start", ...daemonOptions(options)], { ...options, daemon: "off" }),
      status: (options = {}) => this._run(["daemon", "status", ...daemonOptions(options)], { ...options, daemon: "off" }),
      stop: (options = {}) => this._run(["daemon", "stop", ...daemonOptions(options)], { ...options, daemon: "off" }),
    };
  }

  setup(options = {}) {
    return this._run(["setup"], options);
  }

  status(options = {}) {
    return this._run(["status"], options);
  }

  discover(options = {}) {
    return this._run(["discover"], options);
  }

  doctor(options = {}) {
    return this._run(["doctor"], options);
  }

  runFlow(flow, options = {}) {
    const args = ["run", flow];
    if (options.dryRun) args.push("--dry-run");
    if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
    if (hasOption(options, "pollIntervalMs")) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this._run(args, options);
  }

  replay(path, options = {}) {
    return this._run(["replay", path], options);
  }

  raw(args, options = {}) {
    return this._run(args, options);
  }

  _run(args, options = {}) {
    const command = this.command || [this.binary];
    const full = this._globalArgs(options);
    full.push(...args);
    const cwd = options.cwd || this.cwd;
    return new Promise((resolve, reject) => {
      const proc = spawn(command[0], [...command.slice(1), ...full], { cwd, encoding: "utf8" });
      let stdout = "";
      let stderr = "";
      proc.stdout.on("data", (chunk) => {
        stdout += String(chunk);
      });
      proc.stderr.on("data", (chunk) => {
        stderr += String(chunk);
      });
      proc.on("error", (error) => {
        if (error && error.code === "ENOENT") {
          reject(coretapNotInstalledError(command, full, error));
          return;
        }
        reject(new CoretapError(error.message, spawnFailedResponse(command, full, error)));
      });
      proc.on("close", (status) => {
        let parsed;
        try {
          parsed = JSON.parse(stdout);
        } catch (error) {
          if (looksLikeMissingPythonModule(stderr || stdout)) {
            reject(coretapNotInstalledError(command, full, error, { stdout, stderr, status }));
            return;
          }
          reject(new CoretapError(`Coretap did not return JSON: ${stderr || stdout}`, invalidJsonResponse(command, full, error, { stdout, stderr, status })));
          return;
        }
        if (!parsed.ok || status !== 0) {
          reject(new CoretapError(parsed.error ? parsed.error.message : "Coretap command failed", parsed));
          return;
        }
        resolve(parsed.result);
      });
    });
  }

  _globalArgs(options = {}) {
    const backend = options.backend || this.backend;
    const device = options.device || this.device;
    const developerDir = options.developerDir || this.developerDir;
    const coredeviceTunnelMode = options.coredeviceTunnelMode || this.coredeviceTunnelMode;
    const artifactRoot = options.artifactRoot || this.artifactRoot;
    const profile = options.profile || this.profile;
    const daemon = options.daemon || this.daemonMode;
    const full = ["--format", "json", "--backend", backend, "--device", device];
    if (developerDir) full.push("--developer-dir", developerDir);
    if (coredeviceTunnelMode) full.push("--coredevice-tunnel-mode", coredeviceTunnelMode);
    if (artifactRoot) full.push("--artifact-root", artifactRoot);
    if (profile) full.push("--profile", profile);
    if (daemon) full.push("--daemon", daemon);
    return full;
  }
}

class CoretapRun {
  constructor(client) {
    this.client = client;
    this.runId = process.env.CORETAP_RUN_ID || "local";
    this.artifactDir = process.env.CORETAP_ARTIFACT_DIR || null;
  }

  async test(_name, body) {
    return body(new IosVisualUi(this.client));
  }

  async close() {
    return { runId: this.runId, status: "closed" };
  }

  async detach() {
    return { runId: this.runId, status: "detached" };
  }
}

class IosVisualUi {
  constructor(client) {
    this.client = client;
  }

  async screenshot(options = {}) {
    const args = ["screenshot", "--label", options.label || "screenshot"];
    if (options.out) args.push("--out", options.out);
    if (hasOption(options, "maxLongSide")) args.push("--max-long-side", String(options.maxLongSide));
    if (options.fullSize) args.push("--full-size");
    return this.client._run(args, options);
  }

  async locate(target, options = {}) {
    return this.client._run(["locate", "--target", target], options);
  }

  async tap(target, options = {}) {
    return this.tapTarget(target, options);
  }

  async tapTarget(target, options = {}) {
    const args = ["tap", "target", "--target", target];
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args, options);
  }

  async tapText(text, options = {}) {
    const args = ["tap", "text", text];
    if (options.dryRun) args.push("--dry-run");
    if (options.lang) args.push("--lang", options.lang);
    if (hasOption(options, "psm")) args.push("--psm", String(options.psm));
    if (hasOption(options, "minConfidence")) args.push("--min-confidence", String(options.minConfidence));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args, options);
  }

  async tapAt(point, options = {}) {
    const args = [
      "tap",
      "point",
      "--space",
      point.space,
      "--x",
      String(point.x),
      "--y",
      String(point.y),
    ];
    if (options.frame) args.push("--frame", options.frame);
    if (hasOption(options, "width")) args.push("--width", String(options.width));
    if (hasOption(options, "height")) args.push("--height", String(options.height));
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args, options);
  }

  async press(button, options = {}) {
    const args = ["press", button];
    if (options.state) args.push("--state", options.state);
    if (hasOption(options, "holdMs")) args.push("--hold-ms", String(options.holdMs));
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args, options);
  }

  async pressHome(options = {}) {
    return this.press("home", options);
  }

  async typeText(text, options = {}) {
    const args = ["type", text];
    if (hasOption(options, "charDelayMs")) args.push("--char-delay-ms", String(options.charDelayMs));
    if (hasOption(options, "interDelayMs")) args.push("--inter-delay-ms", String(options.interDelayMs));
    if (options.pasteAt) args.push("--paste-at", pointToPair(options.pasteAt));
    if (hasOption(options, "pasteHoldMs")) args.push("--paste-hold-ms", String(options.pasteHoldMs));
    if (hasOption(options, "verifyTimeoutMs")) args.push("--verify-timeout-ms", String(options.verifyTimeoutMs));
    if (options.noVerify) args.push("--no-verify");
    if (options.replace) args.push("--replace");
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args, options);
  }

  async lock(options = {}) {
    return this.press("lock", options);
  }

  async volumeUp(options = {}) {
    return this.press("volume-up", options);
  }

  async volumeDown(options = {}) {
    return this.press("volume-down", options);
  }

  async drag(from, to, options = {}) {
    const space = options.space || from.space || to.space || "normalized";
    const args = [
      "drag",
      "--space",
      space,
      "--from",
      formatPointPair(from),
      "--to",
      formatPointPair(to),
    ];
    if (options.frame) args.push("--frame", options.frame);
    if (hasOption(options, "width")) args.push("--width", String(options.width));
    if (hasOption(options, "height")) args.push("--height", String(options.height));
    if (hasOption(options, "steps")) args.push("--steps", String(options.steps));
    if (hasOption(options, "durationMs")) args.push("--duration-ms", String(options.durationMs));
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args, options);
  }

  async scroll(direction, options = {}) {
    const args = ["scroll", direction];
    if (hasOption(options, "distance")) args.push("--distance", String(options.distance));
    if (hasOption(options, "anchorX")) args.push("--anchor-x", String(options.anchorX));
    if (hasOption(options, "anchorY")) args.push("--anchor-y", String(options.anchorY));
    if (hasOption(options, "steps")) args.push("--steps", String(options.steps));
    if (hasOption(options, "durationMs")) args.push("--duration-ms", String(options.durationMs));
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args, options);
  }

  async expectText(expected, options = {}) {
    const args = ["assert", "text", "--text", expected];
    if (options.image) args.push("--image", options.image);
    if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
    if (hasOption(options, "pollIntervalMs")) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.lang) args.push("--lang", options.lang);
    if (hasOption(options, "psm")) args.push("--psm", String(options.psm));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args, options);
  }

  async waitForText(expected, options = {}) {
    const args = ["wait", "text", "--text", expected];
    if (options.image) args.push("--image", options.image);
    if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
    if (hasOption(options, "pollIntervalMs")) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.lang) args.push("--lang", options.lang);
    if (hasOption(options, "psm")) args.push("--psm", String(options.psm));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args, options);
  }

  async wait(ms, options = {}) {
    return this.client._run(["wait", "--ms", String(ms)], options);
  }

  async waitForStable() {
    return this.screenshot({ label: "stable" });
  }

  async snapshot(name) {
    return this.screenshot({ label: name });
  }
}

function daemonOptions(options = {}) {
  const args = [];
  if (options.socket) args.push("--socket", options.socket);
  if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
  return args;
}

function coretapNotInstalledError(command, args, cause, extraDetails = {}) {
  const executable = command[0];
  const message = [
    "Coretap CLI is not installed or is not reachable from this Node test process.",
    `Install it with: ${INSTALL_COMMAND}`,
    "If Coretap is already installed, pass { binary: '/path/to/coretap' }, { command: [...] }, or set CORETAP_BIN.",
  ].join(" ");
  return new CoretapError(message, {
    ok: false,
    error: {
      code: "CORETAP_CLI_NOT_INSTALLED",
      category: "environment",
      stage: "client",
      message,
      details: {
        executable,
        argv: [...command, ...args],
        installCommand: INSTALL_COMMAND,
        env: {
          CORETAP_BIN: process.env.CORETAP_BIN || null,
          PATH: process.env.PATH || null,
        },
        cause: cause ? { code: cause.code || null, message: cause.message } : null,
        ...extraDetails,
      },
    },
  });
}

function spawnFailedResponse(command, args, error) {
  return {
    ok: false,
    error: {
      code: "SPAWN_FAILED",
      category: "internal",
      stage: "client",
      message: error.message,
      details: {
        argv: [...command, ...args],
        cause: { code: error.code || null, message: error.message },
      },
    },
  };
}

function invalidJsonResponse(command, args, error, details) {
  return {
    ok: false,
    error: {
      code: "INVALID_JSON",
      category: "internal",
      stage: "client",
      message: error.message,
      details: {
        argv: [...command, ...args],
        ...details,
      },
    },
  };
}

function looksLikeMissingPythonModule(output) {
  return /No module named coretap/.test(output || "");
}

function formatPointPair(point) {
  return `${point.x},${point.y}`;
}

function hasOption(options, key) {
  return Object.prototype.hasOwnProperty.call(options, key) && options[key] !== undefined && options[key] !== null;
}

function pointToPair(point) {
  if (typeof point === "string") return point;
  return `${point.x},${point.y}`;
}

module.exports = {
  Coretap,
  CoretapClient: Coretap,
  CoretapError,
  CoretapRun,
  IosVisualUi,
};
