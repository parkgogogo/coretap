# Coretap

Coretap is a local VLM-first mobile-use runtime for iOS real devices and
Simulators. The public surface is intentionally small: agents observe the
screen, execute one typed mobile-use action with `step`, and use text
assertions for test checks.

The runtime keeps the device backend, OCR, and model pack as built-in
capabilities instead of exposing multiple competing command paths.

## Capabilities

- Real-device CoreDevice automation through `pymobiledevice3`, without WDA.
- Userspace CoreDevice tunnel by default, so normal agent runs do not need
  administrator privileges.
- Built-in MAI-UI 2B MLX 6-bit grounding profile:
  `builtin:mai-ui-2b-mlx-6bit@1`.
- Screenshot normalization for CoreDevice rotation/display-size mismatches.
- Built-in macOS Vision OCR for Chinese and English text assertions.
- Compact JSON stdout by default for agent and test-kit use. Runtime screenshots
  use temporary files and are deleted unless artifacts or tracing are explicitly
  enabled.
- A Node test kit that wraps the same `observe`, `step`, `assert text`, and
  `wait text` CLI surface.

The public model pack is fixed to `mlx-community/MAI-UI-2B-6bit-v2` at revision
`cb57cf2fc99f28cb7691459f712d2a276342f804`. Coretap installs it into
`~/Library/Application Support/Coretap/models` and records its own manifest.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/parkgogogo/coretap/main/install.sh | bash
```

Useful installer options:

```bash
./install.sh --skip-model
./install.sh --skip-simulator
./install.sh --skip-device
./install.sh --skip-node-smoke
./install.sh --no-brew-install
./install.sh --no-warm
```

For development from a checkout:

```bash
uv tool install --force --editable .
coretap setup
coretap doctor
```

## Public CLI

Infrastructure:

```bash
coretap status
coretap config check
coretap discover
coretap doctor
coretap daemon start
coretap daemon status
coretap daemon stop
coretap model install
coretap model check
coretap model warm
coretap model status
```

Mobile-use and test surface:

```bash
coretap --backend device --device "$UDID" observe

coretap --backend device --device "$UDID" \
  step --action '{"type":"tap","target":"the App Store search field"}'

coretap --backend device --device "$UDID" \
  step --action '{"type":"typeText","text":"测试文本"}'

coretap --backend device --device "$UDID" assert text --text "搜索"
coretap --backend device --device "$UDID" wait text --text "搜索"
```

Supported `step` action types:

| Type | Purpose |
| --- | --- |
| `tap` | VLM grounding from a semantic target to a device tap. |
| `tapPoint` | Explicit point tap for already-known coordinates. |
| `longPress` | Explicit point long press, implemented through CoreDevice HID drag/hold. |
| `openApp` | High-level app launch. Known or explicit bundle ids use CoreDevice/DVT launch first; Spotlight is fallback. |
| `typeText` | CoreDevice pasteboard + edit-menu text input. |
| `key` | Virtual keyboard key events such as `enter` or `backspace`. |
| `clear` | Repeated Backspace to clear focused text. |
| `press` | Device buttons such as `home`, `lock`, `volume-up`, `volume-down`. |
| `scroll` | Touchscreen scroll gesture. |
| `appSwitcher` | Named home-indicator gesture for entering the iOS app switcher. |
| `terminateApp` | Idempotently terminate a foreground/background app by bundle id. |
| `uninstallApp` | Idempotently uninstall an app by bundle id, with known-name aliases such as `小红书`. |
| `wait` | Timed wait inside a mobile-use step. |

Terminology is intentional here: Coretap exposes `terminateApp`, not
`closeApp`. Closing an app usually means sending it to the background or
returning to Home, which is covered by `press home`. `terminateApp` means ending
the app process by bundle id and verifying the PID is no longer running.

`tap` is the primary semantic click path. It uses the built-in VLM grounding
model, maps the model point back to the source screenshot, and dispatches
through CoreDevice HID. The VLM coordinate is the final tap coordinate when the
target is found. `tapPoint` and `longPress` are explicit point primitives for
cases where the caller already has a trusted coordinate.

`step` captures the before frame when the action needs screen context, executes
the action, waits `--page-wait-ms` milliseconds, then returns a compact page
observation from a fresh screenshot. The default page wait is 6000 ms. Use
explicit `assert text` or `wait text` commands when a test needs OCR assertions.

For non-ASCII `typeText`, Coretap uses pasteboard input. If `pasteAt` is not
provided, Coretap requires a recent tap anchor from the common `tap text field`
then `typeText` flow so calls do not paste into an unknown focus target.
Coretap does not fall back to pinyin candidate input by default; if no recent
tap anchor is available, non-ASCII `typeText` fails fast with
`TEXT_INPUT_TARGET_UNKNOWN`.

## Chain Logs

Use `--trace-id` to tie multiple commands into one replayable chain log:

```bash
TRACE=mobile-flow-$(date +%Y%m%d%H%M%S)

coretap --backend device --device "$UDID" --trace-id "$TRACE" --trace-title "通用移动链路" \
  step --action '{"type":"openApp","name":"App Store"}'

coretap --backend device --device "$UDID" --trace-id "$TRACE" \
  step --action '{"type":"tap","target":"the App Store search field"}'
```

The compact response includes debug paths when tracing is enabled. Without an
explicit `--artifact-root`, Coretap writes trace data under
`~/Library/Caches/Coretap/artifacts`:

- `~/Library/Caches/Coretap/artifacts/traces/<trace-id>/trace.json`
- `~/Library/Caches/Coretap/artifacts/traces/<trace-id>/events.jsonl`
- `~/Library/Caches/Coretap/artifacts/traces/<trace-id>/event-000001.response.json`

Each event records command argv, cwd, status, duration, request id, action
summary, key screenshots, model input, grounding point, tap/type result, page
observation, and the full JSON response path. The per-step `artifactDir`
continues to hold raw screenshots and model/OCR artifacts only when artifacts
are persistent.

By default, commands do not leave persistent screenshots or model/OCR raw files
behind. Use one of these when you need evidence for debugging:

```bash
coretap --keep-artifacts observe
coretap --artifact-root ./artifacts/coretap observe
coretap --trace-id "$TRACE" step --action '{"type":"press","button":"home"}'
```

## Observe

`observe` captures a screenshot and returns a structured page view:

```bash
coretap --backend device --device "$UDID" observe --label home
```

By default the returned frame is compressed to the same 1368 px long-side size
used for VLM input. The source frame is kept only for the duration of the
command unless artifacts are explicitly persistent. Pass `--max-long-side` to
change the preview size or `--full-size` when the returned frame must be
original resolution.

The OCR layer uses macOS Vision with built-in `zh-Hans` and `en-US`
recognition. It includes text, confidence, pixel bounding boxes, center points,
normalized coordinates, and engine metadata. There is no separate OCR engine or
language selector.

`observe` also runs the built-in VLM visual inventory by default. The returned
`visual.elements` list supplements OCR with icon-only and image-like UI targets
such as app icons, tab icons, cloud/download buttons, and unlabeled controls.
Each visual element has a semantic `label`, `role`, normalized `center`, optional
normalized `bbox`, and model self-rated `confidence`. Use `--no-vlm` to skip
this slower page-understanding pass, or `--no-ocr` to keep visual inventory while
skipping OCR.

`step` uses the same visual inventory for the post-action page observation by
default. The internal `step-before` screenshot does not run VLM; it only captures
the screen context needed by the action.

## Real Devices

Coretap's real-device path uses CoreDevice and does not use WDA:

```bash
coretap --coredevice-tunnel-mode userspace --backend device --device "$UDID" observe
```

`userspace` is the default because it avoids root/admin privileges. If you
prefer a long-running tunneld service, opt in explicitly:

```bash
sudo pymobiledevice3 remote tunneld --daemonize
coretap --coredevice-tunnel-mode tunneld --backend device --device "$UDID" observe
```

Prerequisites:

- Pair/trust the iPhone and enable Developer Mode.
- Mount DDI or perform any device-specific developer setup required by
  `pymobiledevice3` for that iOS version.

By default, non-daemon commands auto-start `coretapd`. The daemon keeps the MLX
model and CoreDevice userspace worker resident, which avoids repeated process
startup and teardown in agent loops.

## Node Test Kit

```js
const { Coretap } = require("./packages/node");

const client = await Coretap.connect({
  backend: "device",
  device: process.env.UDID,
});

await client.checkEnvironment();

const run = await client.openRun({ name: "app-store-search" });
await run.test("search app", async (ui) => {
  await ui.pressHome();
  await ui.tap("the App Store app icon");
  await ui.tap("the App Store search field");
  await ui.typeText("测试文本");
  await ui.expectText("测试文本");
  await ui.key("enter");
  await ui.terminateApp("com.apple.AppStore");
  await ui.uninstallApp("小红书");
});
await run.close();
```

The Node package is a thin test kit over the local Coretap CLI. If `coretap` is
not installed or not on `PATH`, it raises `CORETAP_CLI_NOT_INSTALLED` with the
install command and `CORETAP_BIN` override hint.

`openRun({ name })` automatically assigns a trace id and passes it to every
`ui.*` command in that run, so the CLI artifacts and the JavaScript test steps
share one chain log under Coretap's cache artifact root unless `artifactRoot` is
provided.

## Intentional Cuts

Coretap no longer exposes standalone coordinate tap, OCR tap, locate,
screenshot, act loop, replay, JSON flow runner, direct press/type/key/clear/
drag/scroll, or OCR status subcommands. Use `observe` plus `step` actions
instead.
