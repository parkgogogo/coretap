export interface CoretapOptions {
  binary?: string;
  command?: string[];
  backend?: "simulator" | "device";
  device?: string;
  artifactRoot?: string;
}

export class CoretapError extends Error {
  code: string;
  category: string;
  stage: string;
  response: unknown;
}

export class Coretap {
  static connect(options?: CoretapOptions): Promise<Coretap>;
  static attachFromEnvironment(): Promise<CoretapRun>;
  model: {
    install(options?: { force?: boolean }): Promise<unknown>;
    check(options?: { deep?: boolean }): Promise<unknown>;
    warm(): Promise<unknown>;
    status(): Promise<unknown>;
    stop(): Promise<unknown>;
    cache(): Promise<unknown>;
    gc(options?: { dryRun?: boolean }): Promise<unknown>;
  };
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
  space: "px" | "normalized" | "hid";
  x: number;
  y: number;
}

export class IosVisualUi {
  screenshot(options?: { label?: string }): Promise<unknown>;
  locate(target: string): Promise<unknown>;
  tap(target: string, options?: { dryRun?: boolean }): Promise<unknown>;
  tapAt(point: Point, options?: { frame?: string; dryRun?: boolean }): Promise<unknown>;
  expectText(
    expected: string,
    options?: {
      image?: string;
      timeoutMs?: number;
      pollIntervalMs?: number;
      caseSensitive?: boolean;
    },
  ): Promise<unknown>;
  waitForText(
    expected: string,
    options?: {
      image?: string;
      timeoutMs?: number;
      pollIntervalMs?: number;
      caseSensitive?: boolean;
    },
  ): Promise<unknown>;
  wait(ms: number): Promise<unknown>;
  waitForStable(): Promise<unknown>;
  snapshot(name: string): Promise<unknown>;
}

export { Coretap as CoretapClient };
