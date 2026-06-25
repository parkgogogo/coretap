const { spawn } = require("node:child_process");

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
    this.artifactRoot = options.artifactRoot || process.env.CORETAP_ARTIFACT_ROOT || undefined;
  }

  static async connect(options = {}) {
    return new Coretap(options);
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

  get model() {
    return {
      install: (options = {}) => {
        const args = ["model", "install"];
        if (options.force) args.push("--force");
        return this._run(args);
      },
      check: (options = {}) => {
        const args = ["model", "check"];
        if (options.deep) args.push("--deep");
        return this._run(args);
      },
      warm: () => this._run(["model", "warm"]),
      status: () => this._run(["model", "status"]),
      stop: () => this._run(["model", "stop"]),
      cache: () => this._run(["model", "cache"]),
      gc: (options = {}) => {
        const args = ["model", "gc"];
        if (options.dryRun) args.push("--dry-run");
        return this._run(args);
      },
    };
  }

  _run(args) {
    const command = this.command || [this.binary];
    const full = [
      "--format",
      "json",
      "--backend",
      this.backend,
      "--device",
      this.device,
    ];
    if (this.artifactRoot) {
      full.push("--artifact-root", this.artifactRoot);
    }
    full.push(...args);
    return new Promise((resolve, reject) => {
      const proc = spawn(command[0], [...command.slice(1), ...full], { encoding: "utf8" });
      let stdout = "";
      let stderr = "";
      proc.stdout.on("data", (chunk) => {
        stdout += String(chunk);
      });
      proc.stderr.on("data", (chunk) => {
        stderr += String(chunk);
      });
      proc.on("error", (error) => {
        reject(new CoretapError(error.message, {
          ok: false,
          error: {
            code: "SPAWN_FAILED",
            category: "internal",
            stage: "client",
            message: error.message,
          },
        }));
      });
      proc.on("close", (status) => {
        let parsed;
        try {
          parsed = JSON.parse(stdout);
        } catch (error) {
          reject(new CoretapError(`Coretap did not return JSON: ${stderr || stdout}`, {
            ok: false,
            error: {
              code: "INVALID_JSON",
              category: "internal",
              stage: "client",
              message: error.message,
              details: { stdout, stderr, status },
            },
          }));
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
    return this.client._run(["screenshot", "--label", options.label || "screenshot"]);
  }

  async locate(target) {
    return this.client._run(["locate", "--target", target]);
  }

  async tap(target, options = {}) {
    const args = ["tap", "target", "--target", target];
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args);
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
    if (options.dryRun) args.push("--dry-run");
    return this.client._run(args);
  }

  async expectText(expected, options = {}) {
    const args = ["assert", "text", "--text", expected];
    if (options.image) args.push("--image", options.image);
    if (options.timeoutMs) args.push("--timeout-ms", String(options.timeoutMs));
    if (options.pollIntervalMs) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args);
  }

  async waitForText(expected, options = {}) {
    const args = ["wait", "text", "--text", expected];
    if (options.image) args.push("--image", options.image);
    if (options.timeoutMs) args.push("--timeout-ms", String(options.timeoutMs));
    if (options.pollIntervalMs) args.push("--poll-interval-ms", String(options.pollIntervalMs));
    if (options.caseSensitive) args.push("--case-sensitive");
    return this.client._run(args);
  }

  async wait(ms) {
    return this.client._run(["wait", "--ms", String(ms)]);
  }

  async waitForStable() {
    return this.screenshot({ label: "stable" });
  }

  async snapshot(name) {
    return this.screenshot({ label: name });
  }
}

module.exports = {
  Coretap,
  CoretapClient: Coretap,
  CoretapError,
};
