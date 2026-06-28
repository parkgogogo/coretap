export type CoretapBackend = "simulator" | "device";
export type CoretapFormat = "text" | "json" | "ndjson";
export type CoretapDaemonMode = "off" | "auto" | "on";
export type CoretapTunnelMode = "userspace" | "tunneld";
export type CoordinateSpace = "px" | "normalized" | "hid";
export type ButtonState = "press" | "down" | "up" | "canceled";
export type CoretapButton = "home" | "lock" | "power" | "volume-up" | "volume-down" | "mute" | "siri";
export type ScrollDirection = "down" | "up";
export type ObserveOcrEngine = "auto" | "vision" | "tesseract" | "all";
export type KeyboardKey = "backspace" | "delete" | "enter" | "return" | "tab" | "escape" | "esc" | "space" | "left" | "right" | "up" | "down";

export type StepPostcondition =
  | { type: "textVisible"; text: string; caseSensitive?: boolean; minConfidence?: number }
  | { type: "textAbsent"; text: string; caseSensitive?: boolean; minConfidence?: number }
  | { type: "screenChanged" };

export type CoretapAction =
  | { schema: "coretap.action.v2"; type: "tap"; target: string; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "openApp"; name: string; searchTarget?: string; resultTarget?: string; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "typeText"; text: string; charDelayMs?: number; interDelayMs?: number; pasteAt?: Point | string; pasteHoldMs?: number; verifyTimeoutMs?: number; noVerify?: boolean; replace?: boolean; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "key"; key: KeyboardKey | string; count?: number; interDelayMs?: number; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "clear"; count?: number; interDelayMs?: number; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "press"; button: CoretapButton | string; state?: ButtonState; holdMs?: number | null; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "scroll"; direction: ScrollDirection; distance?: number; anchorX?: number; anchorY?: number; steps?: number; durationMs?: number; postconditions?: StepPostcondition[] }
  | { schema: "coretap.action.v2"; type: "wait"; ms?: number; postconditions?: StepPostcondition[] };

export interface CoretapOptions {
  binary?: string;
  command?: string[];
  backend?: CoretapBackend;
  device?: string;
  developerDir?: string;
  coredeviceTunnelMode?: CoretapTunnelMode;
  artifactRoot?: string;
  profile?: string;
  daemon?: CoretapDaemonMode;
  cwd?: string;
}

export interface CommandOptions {
  backend?: CoretapBackend;
  device?: string;
  developerDir?: string;
  coredeviceTunnelMode?: CoretapTunnelMode;
  artifactRoot?: string;
  profile?: string;
  daemon?: CoretapDaemonMode;
  cwd?: string;
}

export class CoretapError extends Error {
  code: string;
  category: string;
  stage: string;
  response: unknown;
}

export class Coretap {
  static connect(options?: CoretapOptions): Promise<Coretap>;
  static checkEnvironment(options?: CoretapOptions): Promise<unknown>;
  static attachFromEnvironment(): Promise<CoretapRun>;
  model: {
    install(options?: CommandOptions & { force?: boolean; dryRun?: boolean }): Promise<unknown>;
    check(options?: CommandOptions & { deep?: boolean }): Promise<unknown>;
    warm(options?: CommandOptions): Promise<unknown>;
    status(options?: CommandOptions): Promise<unknown>;
  };
  daemon: {
    start(options?: DaemonOptions): Promise<unknown>;
    status(options?: DaemonOptions): Promise<unknown>;
    stop(options?: DaemonOptions): Promise<unknown>;
  };
  setup(options?: CommandOptions): Promise<unknown>;
  checkEnvironment(options?: CommandOptions): Promise<unknown>;
  status(options?: CommandOptions): Promise<unknown>;
  observe(options?: ObserveOptions): Promise<unknown>;
  step(action?: CoretapAction | string | null, options?: StepOptions): Promise<unknown>;
  discover(options?: CommandOptions): Promise<unknown>;
  doctor(options?: CommandOptions): Promise<unknown>;
  openRun(options?: unknown): Promise<CoretapRun>;
  withSession<T>(options: unknown, body: (ui: IosVisualUi) => Promise<T>): Promise<T>;
}

export class CoretapRun {
  runId: string;
  artifactDir: string | null;
  test<T>(name: string, body: (ui: IosVisualUi) => Promise<T>): Promise<T>;
  close(): Promise<unknown>;
  detach(): Promise<unknown>;
}

export interface Point {
  space?: CoordinateSpace;
  x: number;
  y: number;
}

export interface TextOptions extends CommandOptions {
  image?: string;
  timeoutMs?: number;
  pollIntervalMs?: number;
  lang?: string;
  psm?: number;
  caseSensitive?: boolean;
}

export interface TapOptions extends CommandOptions {
  dryRun?: boolean;
}

export interface PressOptions extends CommandOptions {
  state?: ButtonState;
  holdMs?: number;
  dryRun?: boolean;
}

export interface TypeTextOptions extends CommandOptions {
  charDelayMs?: number;
  interDelayMs?: number;
  pasteAt?: Point | string;
  pasteHoldMs?: number;
  verifyTimeoutMs?: number;
  noVerify?: boolean;
  replace?: boolean;
  dryRun?: boolean;
}

export interface KeyOptions extends CommandOptions {
  count?: number;
  interDelayMs?: number;
  dryRun?: boolean;
}

export interface ScrollOptions extends CommandOptions {
  distance?: number;
  anchorX?: number;
  anchorY?: number;
  steps?: number;
  durationMs?: number;
  dryRun?: boolean;
}

export interface ObserveOptions extends CommandOptions {
  label?: string;
  out?: string;
  maxLongSide?: number;
  fullSize?: boolean;
  lang?: string;
  psm?: number;
  ocrEngine?: ObserveOcrEngine;
  minConfidence?: number;
  noOcr?: boolean;
}

export interface StepOptions extends CommandOptions {
  actionFile?: string;
  postWaitMs?: number;
  postTimeoutMs?: number;
  pollIntervalMs?: number;
  expectText?: string | string[];
  expectNoText?: string | string[];
  expectChange?: boolean;
  failOnPostcondition?: boolean;
  dryRun?: boolean;
  lang?: string;
  psm?: number;
  ocrEngine?: ObserveOcrEngine;
  minConfidence?: number;
  maxLongSide?: number;
  fullSize?: boolean;
  noOcr?: boolean;
}

export interface DaemonOptions extends CommandOptions {
  socket?: string;
  timeoutMs?: number;
}

export class IosVisualUi {
  observe(options?: ObserveOptions): Promise<unknown>;
  step(action?: CoretapAction | string | null, options?: StepOptions): Promise<unknown>;
  tap(target: string, options?: TapOptions): Promise<unknown>;
  openApp(name: string, options?: StepOptions & { searchTarget?: string; resultTarget?: string }): Promise<unknown>;
  press(button: CoretapButton, options?: PressOptions): Promise<unknown>;
  pressHome(options?: PressOptions): Promise<unknown>;
  typeText(text: string, options?: TypeTextOptions): Promise<unknown>;
  key(key: KeyboardKey, options?: KeyOptions): Promise<unknown>;
  clearText(options?: KeyOptions): Promise<unknown>;
  lock(options?: PressOptions): Promise<unknown>;
  volumeUp(options?: PressOptions): Promise<unknown>;
  volumeDown(options?: PressOptions): Promise<unknown>;
  scroll(direction: ScrollDirection, options?: ScrollOptions): Promise<unknown>;
  expectText(expected: string, options?: TextOptions): Promise<unknown>;
  waitForText(expected: string, options?: TextOptions): Promise<unknown>;
  wait(ms: number, options?: CommandOptions): Promise<unknown>;
}

export { Coretap as CoretapClient };
