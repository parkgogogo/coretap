# Coretap

Coretap is an early local visual UI automation runtime for iOS real-device and
Simulator workflows. It exposes a small CLI plus a Node client surface while
keeping model/OCR capabilities as built-in product capabilities.

This MVP intentionally implements the product skeleton first:

- `pymobiledevice3` real-device backend for CoreDevice HID taps.
- `simctl` Simulator backend for discovery and screenshots.
- Built-in MAI-UI 2B MLX 6-bit grounding profile:
  `builtin:mai-ui-2b-mlx-6bit@1`.
- Fixed model pack install/check/warm/run/status/stop/cache/gc commands.
- OCR capability through the local `tesseract` CLI for text assertions only.
- JSON/NDJSON-friendly command envelopes and artifact capture.
- A thin async Node test client that calls the CLI and consumes JSON.

The public grounding model is fixed to
`mlx-community/MAI-UI-2B-6bit-v2` at revision
`cb57cf2fc99f28cb7691459f712d2a276342f804`. Coretap installs it into
`~/Library/Application Support/Coretap/models` and records its own manifest.

There is an explicit `internal:test-fixture-grounder` profile for local
simulator regression. It is OCR-backed and should not be treated as a product
grounding profile.

## Simulator Tap Support

`simctl` supports screenshots but does not expose a touch/tap primitive. Coretap
uses `fb-idb`/`idb_companion` for real Simulator taps when a companion is
available, while keeping `simctl` for discovery and screenshots.

Dry-run remains available for coordinate regression:

```bash
coretap tap target --backend simulator --device booted --target "Settings" --dry-run
```

For real taps, install or provide `idb_companion`. In this workspace it can be
provided through:

```bash
export CORETAP_IDB_COMPANION_PATH="$PWD/.tools/idb-companion.universal/bin/idb_companion"
```

Coretap records frame pixel, normalized, and executor coordinates. The shared
coordinate is normalized `[0,1]`; real devices convert normalized points to HID
U16, while Simulator converts normalized points to IDB logical points.

## Real Device CoreDevice Support

Real-device automation uses `pymobiledevice3 developer core-device` and does
not use WDA. On current `pymobiledevice3`, Coretap defaults to the CoreDevice
userspace tunnel path:

```bash
PYMOBILEDEVICE3_UDID="$UDID" pymobiledevice3 developer core-device ... --userspace
```

This is the default because it does not require root/admin privileges, which is
important for agent-driven CLI runs.

CoreDevice screenshots are normalized to the primary display size reported by
`get-display-info` before grounding/OCR/tap coordinates are used. This keeps the
frame pixel coordinates aligned with the HID coordinate space even when the raw
CoreDevice screenshot service returns a rotated PNG.

On userspace CoreDevice HID, `pymobiledevice3` can dispatch the touch but hang
while closing its media stream. Coretap treats that as an attempted tap with
`completionStatus: "timeout"` and `deliveryStatus: "unknown"` so test flows can
continue to the next screenshot or assertion for confirmation.

Prerequisites:

- Pair/trust the iPhone and enable Developer Mode.
- Mount DDI or perform any device-specific developer setup required by
  `pymobiledevice3` for that iOS version.

If you prefer a long-running tunneld service, opt in explicitly:

```bash
sudo pymobiledevice3 remote tunneld --daemonize
coretap --coredevice-tunnel-mode tunneld --backend device --device "$UDID" screenshot
```

You can also set `CORETAP_COREDEVICE_TUNNEL_MODE=tunneld`. If tunneld is missing
in that mode, Coretap reports `COREDEVICE_TUNNELD_UNAVAILABLE` with the
suggested daemon command instead of trying to parse a broken or empty screenshot.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/parkgogogo/coretap/main/install.sh | bash
```

The installer sets up the wrapper CLI, `pymobiledevice3`, Simulator tap support,
OCR when Homebrew is available, the built-in MAI-UI model pack, and then runs
`coretap doctor`.

Useful installer options:

```bash
./install.sh --skip-model
./install.sh --skip-ocr
./install.sh --skip-simulator
./install.sh --skip-device
./install.sh --no-brew-install
./install.sh --no-warm
```

For development from a checkout:

```bash
uv tool install --force --editable .
coretap setup --format json
```

`coretap assert text` and `coretap wait text` currently use the local
`tesseract` CLI. Install it separately, for example:

```bash
brew install tesseract
```

If Xcode is installed but `xcode-select` points at CommandLineTools, set:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
```

## Example

```bash
coretap screenshot --backend simulator --device booted --format json
coretap locate --backend simulator --device booted --target "Settings app icon" --format json
coretap tap point --backend simulator --device booted --space normalized --x 0.5 --y 0.5 --format json
coretap assert text --backend simulator --device booted --text "Settings" --format json
coretap wait text --backend simulator --device booted --text "General" --format json
```

For deterministic simulator fixture regression:

```bash
coretap --profile internal:test-fixture-grounder \
  --backend simulator \
  --device booted \
  run examples/settings-flow.json \
  --format json
```
