export type CoretapBackend = "simulator" | "device";
export type CoretapDaemonMode = "off" | "auto" | "on";
export type CoretapTunnelMode = "userspace" | "tunneld";
export type CoordinateSpace = "px" | "normalized" | "hid";
export type ButtonState = "press" | "down" | "up" | "canceled";
export type CoretapButton = "home" | "lock" | "power" | "volume-up" | "volume-down" | "mute" | "siri";
export type ScrollDirection = "down" | "up";
export type KeyboardKey = "backspace" | "delete" | "enter" | "return" | "tab" | "escape" | "esc" | "space" | "left" | "right" | "up" | "down";
export type OpenAppStrategy = "auto" | "bundle" | "spotlight";

export type CoretapAction =
  | { type: "tap"; target: string }
  | { type: "tapPoint"; point?: Point; x?: number; y?: number; space?: CoordinateSpace; reference?: PointReference }
  | { type: "longPress"; point?: Point; x?: number; y?: number; space?: CoordinateSpace; reference?: PointReference; durationMs?: number; steps?: number }
  | { type: "openApp"; name: string; bundleId?: string; strategy?: OpenAppStrategy; killExisting?: boolean; searchTarget?: string; resultTarget?: string }
  | { type: "openUrl"; url: string; timeoutSec?: number; timeout?: number }
  | { type: "typeText"; text: string; charDelayMs?: number; interDelayMs?: number; pasteAt?: Point | string; pasteHoldMs?: number; verifyTimeoutMs?: number; noVerify?: boolean; replace?: boolean }
  | { type: "key"; key: KeyboardKey | string; count?: number; interDelayMs?: number }
  | { type: "clear"; count?: number; interDelayMs?: number }
  | { type: "press"; button: CoretapButton | string; state?: ButtonState; holdMs?: number | null }
  | { type: "scroll"; direction: ScrollDirection; distance?: number; anchorX?: number; anchorY?: number; steps?: number; durationMs?: number }
  | { type: "appSwitcher"; start?: Point; end?: Point; startX?: number; startY?: number; endX?: number; endY?: number; steps?: number; durationMs?: number }
  | { type: "terminateApp"; bundleId: string; signal?: number }
  | { type: "uninstallApp"; bundleId?: string; name?: string; ignoreMissing?: boolean }
  | { type: "wait"; ms?: number };

export interface CoretapOptions {
  binary?: string;
  command?: string[];
  backend?: CoretapBackend;
  device?: string;
  developerDir?: string;
  coredeviceTunnelMode?: CoretapTunnelMode;
  artifactRoot?: string;
  keepArtifacts?: boolean;
  noArtifacts?: boolean;
  profile?: string;
  daemon?: CoretapDaemonMode;
  traceId?: string;
  traceTitle?: string;
  cwd?: string;
}

export interface CommandOptions {
  backend?: CoretapBackend;
  device?: string;
  developerDir?: string;
  coredeviceTunnelMode?: CoretapTunnelMode;
  artifactRoot?: string;
  keepArtifacts?: boolean;
  noArtifacts?: boolean;
  profile?: string;
  daemon?: CoretapDaemonMode;
  traceId?: string;
  traceTitle?: string;
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
  screenshot(options?: ScreenshotOptions): Promise<unknown>;
  observe(options?: ObserveOptions): Promise<unknown>;
  step(action?: CoretapAction | string | null, options?: StepOptions): Promise<unknown>;
  discover(options?: CommandOptions): Promise<unknown>;
  doctor(options?: CommandOptions): Promise<unknown>;
  openRun(options?: CoretapRunOptions): Promise<CoretapRun>;
  withSession<T>(options: CoretapRunOptions, body: (ui: IosVisualUi) => Promise<T>): Promise<T>;
}

export class CoretapRun {
  runId: string;
  traceId: string;
  traceTitle: string | null;
  artifactDir: string | null;
  test<T>(name: string, body: (ui: IosVisualUi) => Promise<T>): Promise<T>;
  close(): Promise<unknown>;
  detach(): Promise<unknown>;
}

export interface CoretapRunOptions extends CommandOptions {
  name?: string;
}

export interface Point {
  space?: CoordinateSpace;
  reference?: PointReference;
  x: number;
  y: number;
}

export type PointReference = "source" | "preview";

export interface TextOptions extends CommandOptions {
  image?: string;
  timeoutMs?: number;
  pollIntervalMs?: number;
  caseSensitive?: boolean;
}

export interface TapOptions extends CommandOptions {
  dryRun?: boolean;
}

export interface TapPointOptions extends CommandOptions {
  space?: CoordinateSpace;
  reference?: PointReference;
  dryRun?: boolean;
}

export interface LongPressOptions extends TapPointOptions {
  durationMs?: number;
  steps?: number;
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

export interface AppSwitcherOptions extends CommandOptions {
  start?: Point;
  end?: Point;
  startX?: number;
  startY?: number;
  endX?: number;
  endY?: number;
  steps?: number;
  durationMs?: number;
  dryRun?: boolean;
}

export interface TerminateAppOptions extends CommandOptions {
  signal?: number;
  dryRun?: boolean;
}

export interface UninstallAppOptions extends CommandOptions {
  bundleId?: string;
  name?: string;
  ignoreMissing?: boolean;
  dryRun?: boolean;
}

export interface ObserveOptions extends CommandOptions {
  label?: string;
  out?: string;
  maxLongSide?: number;
  fullSize?: boolean;
  minConfidence?: number;
  noOcr?: boolean;
  noVlm?: boolean;
}

export interface ScreenshotOptions extends CommandOptions {
  label?: string;
  out?: string;
}

export interface StepOptions extends CommandOptions {
  actionFile?: string;
  dryRun?: boolean;
  pageWaitMs?: number;
  noPage?: boolean;
  minConfidence?: number;
  maxLongSide?: number;
  noRefine?: boolean;
  refineCropRatio?: number;
  fullSize?: boolean;
  noOcr?: boolean;
  noVlm?: boolean;
}

export interface DaemonOptions extends CommandOptions {
  socket?: string;
  timeoutMs?: number;
}

export class IosVisualUi {
  screenshot(options?: ScreenshotOptions): Promise<unknown>;
  observe(options?: ObserveOptions): Promise<unknown>;
  step(action?: CoretapAction | string | null, options?: StepOptions): Promise<unknown>;
  tap(target: string, options?: TapOptions): Promise<unknown>;
  tapPoint(point: Point, options?: TapPointOptions): Promise<unknown>;
  longPress(point: Point, options?: LongPressOptions): Promise<unknown>;
  openApp(name: string, options?: StepOptions & { bundleId?: string; strategy?: OpenAppStrategy; killExisting?: boolean; searchTarget?: string; resultTarget?: string }): Promise<unknown>;
  press(button: CoretapButton, options?: PressOptions): Promise<unknown>;
  pressHome(options?: PressOptions): Promise<unknown>;
  typeText(text: string, options?: TypeTextOptions): Promise<unknown>;
  key(key: KeyboardKey, options?: KeyOptions): Promise<unknown>;
  clearText(options?: KeyOptions): Promise<unknown>;
  lock(options?: PressOptions): Promise<unknown>;
  volumeUp(options?: PressOptions): Promise<unknown>;
  volumeDown(options?: PressOptions): Promise<unknown>;
  scroll(direction: ScrollDirection, options?: ScrollOptions): Promise<unknown>;
  appSwitcher(options?: AppSwitcherOptions): Promise<unknown>;
  terminateApp(bundleId: string, options?: TerminateAppOptions): Promise<unknown>;
  uninstallApp(app: string, options?: UninstallAppOptions): Promise<unknown>;
  expectText(expected: string, options?: TextOptions): Promise<unknown>;
  waitForText(expected: string, options?: TextOptions): Promise<unknown>;
  wait(ms: number, options?: CommandOptions): Promise<unknown>;
}

export { Coretap as CoretapClient };
