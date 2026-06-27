export type CoretapBackend = "simulator" | "device";
export type CoretapFormat = "text" | "json" | "ndjson";
export type CoretapDaemonMode = "off" | "auto" | "on";
export type CoretapTunnelMode = "userspace" | "tunneld";
export type CoordinateSpace = "px" | "normalized" | "hid";
export type ButtonState = "press" | "down" | "up" | "canceled";
export type CoretapButton = "home" | "lock" | "power" | "volume-up" | "volume-down" | "mute" | "siri";
export type ScrollDirection = "down" | "up";

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
    stop(options?: CommandOptions): Promise<unknown>;
    cache(options?: CommandOptions): Promise<unknown>;
    run(options: CommandOptions & { image: string; target: string }): Promise<unknown>;
    gc(options?: CommandOptions & { dryRun?: boolean }): Promise<unknown>;
  };
  ocr: {
    status(options?: CommandOptions): Promise<unknown>;
    check(options?: CommandOptions): Promise<unknown>;
  };
  daemon: {
    start(options?: DaemonOptions): Promise<unknown>;
    status(options?: DaemonOptions): Promise<unknown>;
    stop(options?: DaemonOptions): Promise<unknown>;
  };
  setup(options?: CommandOptions): Promise<unknown>;
  checkEnvironment(options?: CommandOptions): Promise<unknown>;
  status(options?: CommandOptions): Promise<unknown>;
  discover(options?: CommandOptions): Promise<unknown>;
  doctor(options?: CommandOptions): Promise<unknown>;
  runFlow(flow: string, options?: CommandOptions & FlowOptions): Promise<unknown>;
  replay(path: string, options?: CommandOptions): Promise<unknown>;
  raw(args: string[], options?: CommandOptions): Promise<unknown>;
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

export interface TapPoint extends Point {
  space: CoordinateSpace;
}

export interface TextOptions extends CommandOptions {
  image?: string;
  timeoutMs?: number;
  pollIntervalMs?: number;
  lang?: string;
  psm?: number;
  caseSensitive?: boolean;
}

export interface TapTextOptions extends CommandOptions {
  dryRun?: boolean;
  lang?: string;
  psm?: number;
  minConfidence?: number;
  caseSensitive?: boolean;
}

export interface TapOptions extends CommandOptions {
  dryRun?: boolean;
}

export interface TapAtOptions extends CommandOptions {
  frame?: string;
  width?: number;
  height?: number;
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

export interface DragOptions extends CommandOptions {
  space?: CoordinateSpace;
  frame?: string;
  width?: number;
  height?: number;
  steps?: number;
  durationMs?: number;
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

export interface ScreenshotOptions extends CommandOptions {
  label?: string;
  out?: string;
  maxLongSide?: number;
  fullSize?: boolean;
}

export interface FlowOptions {
  dryRun?: boolean;
  timeoutMs?: number;
  pollIntervalMs?: number;
  caseSensitive?: boolean;
}

export interface DaemonOptions extends CommandOptions {
  socket?: string;
  timeoutMs?: number;
}

export class IosVisualUi {
  screenshot(options?: ScreenshotOptions): Promise<unknown>;
  locate(target: string, options?: CommandOptions): Promise<unknown>;
  tap(target: string, options?: TapOptions): Promise<unknown>;
  tapTarget(target: string, options?: TapOptions): Promise<unknown>;
  tapText(text: string, options?: TapTextOptions): Promise<unknown>;
  tapAt(point: TapPoint, options?: TapAtOptions): Promise<unknown>;
  press(button: CoretapButton, options?: PressOptions): Promise<unknown>;
  pressHome(options?: PressOptions): Promise<unknown>;
  typeText(text: string, options?: TypeTextOptions): Promise<unknown>;
  lock(options?: PressOptions): Promise<unknown>;
  volumeUp(options?: PressOptions): Promise<unknown>;
  volumeDown(options?: PressOptions): Promise<unknown>;
  drag(from: Point, to: Point, options?: DragOptions): Promise<unknown>;
  scroll(direction: ScrollDirection, options?: ScrollOptions): Promise<unknown>;
  expectText(expected: string, options?: TextOptions): Promise<unknown>;
  waitForText(expected: string, options?: TextOptions): Promise<unknown>;
  wait(ms: number, options?: CommandOptions): Promise<unknown>;
  waitForStable(): Promise<unknown>;
  snapshot(name: string): Promise<unknown>;
}

export { Coretap as CoretapClient };
