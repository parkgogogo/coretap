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
    this.keepArtifacts = hasOption(options, "keepArtifacts") ? Boolean(options.keepArtifacts) : truthyEnv("CORETAP_KEEP_ARTIFACTS");
    this.noArtifacts = hasOption(options, "noArtifacts") ? Boolean(options.noArtifacts) : truthyEnv("CORETAP_NO_ARTIFACTS");
    this.profile = options.profile || process.env.CORETAP_PROFILE || undefined;
    this.daemonMode = options.daemon || process.env.CORETAP_DAEMON || undefined;
    this.traceId = options.traceId || process.env.CORETAP_TRACE_ID || undefined;
    this.traceTitle = options.traceTitle || process.env.CORETAP_TRACE_TITLE || undefined;
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

  async openRun(options = {}) {
    return new CoretapRun(this, options);
  }

  async withSession(options, body) {
    const run = await this.openRun(options);
    try {
      return await body(new IosVisualUi(this, run.commandOptions()));
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

  observe(options = {}) {
    return this._run(observeArgs(options), options);
  }

  screenshot(options = {}) {
    return this._run(screenshotArgs(options), options);
  }

  step(action, options = {}) {
    return this._run(stepArgs(action, options), options);
  }

  discover(options = {}) {
    return this._run(["discover"], options);
  }

  doctor(options = {}) {
    return this._run(["doctor"], options);
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
    const keepArtifacts = hasOption(options, "keepArtifacts") ? Boolean(options.keepArtifacts) : this.keepArtifacts;
    const noArtifacts = hasOption(options, "noArtifacts") ? Boolean(options.noArtifacts) : this.noArtifacts;
    const profile = options.profile || this.profile;
    const daemon = options.daemon || this.daemonMode;
    const traceId = options.traceId || this.traceId;
    const traceTitle = options.traceTitle || this.traceTitle;
    const full = ["--backend", backend, "--device", device];
    if (developerDir) full.push("--developer-dir", developerDir);
    if (coredeviceTunnelMode) full.push("--coredevice-tunnel-mode", coredeviceTunnelMode);
    if (artifactRoot) full.push("--artifact-root", artifactRoot);
    if (keepArtifacts) full.push("--keep-artifacts");
    if (noArtifacts) full.push("--no-artifacts");
    if (profile) full.push("--profile", profile);
    if (daemon) full.push("--daemon", daemon);
    if (traceId) full.push("--trace-id", traceId);
    if (traceTitle) full.push("--trace-title", traceTitle);
    return full;
  }
}

class CoretapRun {
  constructor(client, options = {}) {
    this.client = client;
    const explicitTraceId = options.traceId || process.env.CORETAP_TRACE_ID || client.traceId;
    const name = options.name || options.traceTitle || process.env.CORETAP_TRACE_TITLE || client.traceTitle || "node-run";
    this.runId = explicitTraceId || makeTraceId(name);
    this.traceId = this.runId;
    this.traceTitle = options.traceTitle || options.name || process.env.CORETAP_TRACE_TITLE || client.traceTitle || null;
    this.artifactDir = process.env.CORETAP_ARTIFACT_DIR || null;
  }

  async test(_name, body) {
    return body(new IosVisualUi(this.client, this.commandOptions()));
  }

  async close() {
    return { runId: this.runId, traceId: this.traceId, status: "closed" };
  }

  async detach() {
    return { runId: this.runId, traceId: this.traceId, status: "detached" };
  }

  commandOptions() {
    return {
      traceId: this.traceId,
      traceTitle: this.traceTitle || undefined,
    };
  }
}

class IosVisualUi {
  constructor(client, defaults = {}) {
    this.client = client;
    this.defaults = defaults || {};
  }

  _options(options = {}) {
    return { ...this.defaults, ...options };
  }

  async observe(options = {}) {
    return this.client._run(observeArgs(options), this._options(options));
  }

  async screenshot(options = {}) {
    return this.client._run(screenshotArgs(options), this._options(options));
  }

  async step(action, options = {}) {
    return this.client.step(action, this._options(options));
  }

  async tap(target, options = {}) {
    return this.step({ type: "tap", target }, options);
  }

  async tapPoint(point, options = {}) {
    return this.step({ type: "tapPoint", point: pointPayload(point, options) }, options);
  }

  async longPress(point, options = {}) {
    return this.step(
      {
        type: "longPress",
        point: pointPayload(point, options),
        durationMs: options.durationMs,
        steps: options.steps,
      },
      options,
    );
  }

  async openApp(name, options = {}) {
    return this.step(
      {
        type: "openApp",
        name,
        bundleId: options.bundleId,
        strategy: options.strategy,
        killExisting: options.killExisting,
        searchTarget: options.searchTarget,
        resultTarget: options.resultTarget,
      },
      options,
    );
  }

  async press(button, options = {}) {
    return this.step(
      {
        type: "press",
        button,
        state: options.state,
        holdMs: hasOption(options, "holdMs") ? options.holdMs : null,
      },
      options,
    );
  }

  async pressHome(options = {}) {
    return this.press("home", options);
  }

  async typeText(text, options = {}) {
    return this.step(
      {
        type: "typeText",
        text,
        charDelayMs: options.charDelayMs,
        interDelayMs: options.interDelayMs,
        pasteAt: options.pasteAt,
        pasteHoldMs: options.pasteHoldMs,
        verifyTimeoutMs: options.verifyTimeoutMs,
        noVerify: options.noVerify,
        replace: options.replace,
      },
      options,
    );
  }

  async key(key, options = {}) {
    return this.step(
      {
        type: "key",
        key,
        count: options.count,
        interDelayMs: options.interDelayMs,
      },
      options,
    );
  }

  async clearText(options = {}) {
    return this.step(
      {
        type: "clear",
        count: options.count,
        interDelayMs: options.interDelayMs,
      },
      options,
    );
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

  async scroll(direction, options = {}) {
    return this.step(
      {
        type: "scroll",
        direction,
        distance: options.distance,
        anchorX: options.anchorX,
        anchorY: options.anchorY,
        steps: options.steps,
        durationMs: options.durationMs,
      },
      options,
    );
  }

  async appSwitcher(options = {}) {
    return this.step(
      {
        type: "appSwitcher",
        start: options.start,
        end: options.end,
        startX: options.startX,
        startY: options.startY,
        endX: options.endX,
        endY: options.endY,
        steps: options.steps,
        durationMs: options.durationMs,
      },
      options,
    );
  }

  async terminateApp(bundleId, options = {}) {
    return this.step(
      {
        type: "terminateApp",
        bundleId,
        signal: options.signal,
      },
      options,
    );
  }

  async uninstallApp(app, options = {}) {
    const action = {
      type: "uninstallApp",
      ignoreMissing: options.ignoreMissing,
    };
    if (options.bundleId || looksLikeBundleId(app)) {
      action.bundleId = options.bundleId || app;
      if (options.name) action.name = options.name;
    } else {
      action.name = app;
    }
    return this.step(action, options);
  }

  async expectText(expected, options = {}) {
    const args = ["assert", "text", "--text", expected];
    if (options.image) args.push("--image", options.image);
    if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
    if (hasOption(options, "pollIntervalMs")) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args, this._options(options));
  }

  async waitForText(expected, options = {}) {
    const args = ["wait", "text", "--text", expected];
    if (options.image) args.push("--image", options.image);
    if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
    if (hasOption(options, "pollIntervalMs")) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args, this._options(options));
  }

  async wait(ms, options = {}) {
    return this.step({ type: "wait", ms }, options);
  }
}

function daemonOptions(options = {}) {
  const args = [];
  if (options.socket) args.push("--socket", options.socket);
  if (hasOption(options, "timeoutMs")) args.push("--timeout-ms", String(options.timeoutMs));
  return args;
}

function observeArgs(options = {}) {
  const args = ["observe"];
  if (options.label) args.push("--label", options.label);
  if (options.out) args.push("--out", options.out);
  if (hasOption(options, "maxLongSide")) args.push("--max-long-side", String(options.maxLongSide));
  if (options.fullSize) args.push("--full-size");
  if (hasOption(options, "minConfidence")) args.push("--min-confidence", String(options.minConfidence));
  if (options.noOcr) args.push("--no-ocr");
  if (options.noVlm) args.push("--no-vlm");
  return args;
}

function screenshotArgs(options = {}) {
  const args = ["screenshot"];
  if (options.label) args.push("--label", options.label);
  if (options.out) args.push("--out", options.out);
  return args;
}

function stepArgs(action, options = {}) {
  const args = ["step"];
  if (options.actionFile) {
    args.push("--action-file", options.actionFile);
  } else {
    if (action === undefined || action === null) {
      throw new TypeError("Coretap step requires an action object, JSON string, or { actionFile } option.");
    }
    const payload = typeof action === "string" ? action : JSON.stringify(action);
    args.push("--action", payload);
  }
  if (options.dryRun) args.push("--dry-run");
  if (hasOption(options, "pageWaitMs")) args.push("--page-wait-ms", String(options.pageWaitMs));
  if (options.noPage) args.push("--no-page");
  if (hasOption(options, "minConfidence")) args.push("--min-confidence", String(options.minConfidence));
  if (hasOption(options, "maxLongSide")) args.push("--max-long-side", String(options.maxLongSide));
  if (options.noRefine) args.push("--no-refine");
  if (hasOption(options, "refineCropRatio")) args.push("--refine-crop-ratio", String(options.refineCropRatio));
  if (options.fullSize) args.push("--full-size");
  if (options.noOcr) args.push("--no-ocr");
  if (options.noVlm) args.push("--no-vlm");
  return args;
}

function pointPayload(point, options = {}) {
  if (!point || typeof point !== "object") {
    throw new TypeError("Coretap point must be an object with x and y.");
  }
  return {
    x: point.x,
    y: point.y,
    space: point.space || options.space,
    reference: point.reference || options.reference,
  };
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

function makeTraceId(name) {
  const base = String(name || "node-run")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "")
    .slice(0, 80) || "node-run";
  const stamp = new Date().toISOString().replace(/[-:.]/g, "").slice(0, 15);
  const suffix = Math.random().toString(16).slice(2, 8);
  return `${base}-${stamp}-${suffix}`;
}

function hasOption(options, key) {
  return Object.prototype.hasOwnProperty.call(options, key) && options[key] !== undefined && options[key] !== null;
}

function truthyEnv(name) {
  const value = process.env[name];
  return typeof value === "string" && ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function looksLikeBundleId(value) {
  return /^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$/.test(String(value || ""));
}

module.exports = {
  Coretap,
  CoretapClient: Coretap,
  CoretapError,
  CoretapRun,
  IosVisualUi,
};
