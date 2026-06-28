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
- macOS Vision OCR first, with Tesseract fallback for Chinese and English text
  assertions.
- JSON response envelopes by default for agent and test-kit use.
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
./install.sh --skip-ocr
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
  step --action '{"schema":"coretap.action.v2","type":"tap","target":"the App Store search field"}' \
  --expect-change

coretap --backend device --device "$UDID" \
  step --action '{"schema":"coretap.action.v2","type":"typeText","text":"小红书"}' \
  --expect-text "小红书"

coretap --backend device --device "$UDID" assert text --text "搜索"
coretap --backend device --device "$UDID" wait text --text "搜索"
```

Supported `step` action types:

| Type | Purpose |
| --- | --- |
| `tap` | VLM grounding from a semantic target to a device tap. |
| `typeText` | CoreDevice pasteboard + edit-menu text input. |
| `key` | Virtual keyboard key events such as `enter` or `backspace`. |
| `clear` | Repeated Backspace to clear focused text. |
| `press` | Device buttons such as `home`, `lock`, `volume-up`, `volume-down`. |
| `scroll` | Touchscreen scroll gesture. |
| `wait` | Timed wait inside a mobile-use step. |

`tap` is the only public click path. It uses the built-in VLM grounding model,
maps the model point back to the source screenshot, runs safety checks, and then
dispatches through CoreDevice HID. OCR is for `observe`, `assert text`,
`wait text`, and text postconditions, not for clicking.

## Observe

`observe` captures a screenshot and returns a structured page view:

```bash
coretap --backend device --device "$UDID" observe --label home
```

By default the returned frame is compressed to the same 1368 px long-side size
used for VLM input. The source frame is preserved in artifacts when resizing is
needed. Pass `--max-long-side` to change the preview size or `--full-size` when
the returned frame must be original resolution.

The OCR layer includes text, confidence, pixel bounding boxes, center points,
normalized coordinates, and engine metadata. `--lang` defaults to
`chi_sim+eng`.

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
  await ui.tap("the App Store app icon", { expectChange: true });
  await ui.tap("the App Store search field", { expectChange: true });
  await ui.typeText("小红书", {
    expectText: "小红书",
  });
  await ui.key("enter", { expectChange: true });
  await ui.expectText("小红书");
});
await run.close();
```

The Node package is a thin test kit over the local Coretap CLI. If `coretap` is
not installed or not on `PATH`, it raises `CORETAP_CLI_NOT_INSTALLED` with the
install command and `CORETAP_BIN` override hint.

## Intentional Cuts

Coretap no longer exposes public coordinate tap, OCR tap, locate, screenshot,
act loop, replay, JSON flow runner, direct press/type/key/clear/drag/scroll, or
OCR status subcommands. Use `observe` plus `step` actions instead.
