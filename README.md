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

## Quick Start

```bash
uv tool install --force --editable .
coretap setup
coretap status --format json
coretap discover --backend simulator --format json
coretap model install --format json
coretap model check --deep --format json
coretap model warm --format json
coretap ocr check --format json
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
