---
name: coretap-mobile-use-lab
description: Scenario-driven hardening workflow for Coretap mobile-use on iOS real devices. Use when Codex needs to operate a phone through Coretap, run realistic mobile-use scenarios, debug VLM/OCR/type/tap behavior, collect trace evidence, fix Coretap issues, validate repairs, and update the project skill and scenario catalog.
---

# Coretap Mobile Use Lab

## Overview

Use Coretap as the only phone-control surface, then iterate like a mobile-use test engineer and developer in one loop: run realistic tasks, collect trace evidence, diagnose failures from artifacts, patch Coretap, validate the fix, and add a harder scenario.

Keep the loop evidence-driven. Do not rely on memory of a prior screen, manual coordinate guesses, or the model's own visual ability as a fallback for operating the phone.

## Operating Rules

- Use only `coretap` commands for iOS device actions. Shell, file reads, JSON parsing, and image viewing are allowed for analysis.
- Make VLM semantic `step tap` the primary path for touch targets. The target text should describe the user-visible goal directly, for example `download the official App Store result by Xingin`.
- Do not add app-specific hard-coded logic for a single scenario. Fix generic failure classes only.
- Do not use OCR to rewrite tap coordinates. OCR is for page observation and explicit `assert text` / `wait text` checks.
- Do not use inline step validation. A `step` action executes, waits for the default page interval, and returns the resulting page state. Use a separate assertion command when a test needs a textual proof.
- Always set `--trace-id`; set `--artifact-root` only when the user explicitly wants artifacts in a specific directory. Otherwise let Coretap keep trace artifacts under `~/Library/Caches/Coretap/artifacts`.
- Record command `durationMs`, artifact directories, and the exact failing evidence before fixing code.
- After a fix, rerun the failing scenario and one adjacent scenario that could regress.
- Update this skill or `references/scenario-catalog.md` whenever a new stable pattern, failure mode, or escalation rule is learned.

## Scenario Loop

1. Pick one scenario from [scenario-catalog.md](references/scenario-catalog.md), starting from the lowest level that is not reliably green.
2. Define success in user-visible terms, for example "App Store shows 打开 for 小红书" rather than "tap coordinate X".
3. Prepare state with Coretap actions such as `uninstallApp`, `terminateApp`, `press home`, or `openApp`.
4. Run each action through `step`, `observe`, `assert text`, or `wait text` with a shared trace id.
5. Stop at the first unexpected result. Inspect the response JSON, screenshots, model input, OCR tokens, grounding point, page observation, and trace event.
6. Classify the failure:
   - `grounding`: VLM found the wrong element or failed to find the target.
   - `execution`: CoreDevice tap/type/key/scroll did not deliver reliably.
   - `state`: prior app/device state invalidated the next step.
   - `assertion`: OCR text check failed despite correct device state, or passed despite the wrong state.
   - `performance`: the step works but is too slow for an agent loop.
   - `daemon`: a stale resident worker or daemon ran old code.
7. Patch Coretap or the skill pattern, then run focused tests and the scenario again.
8. Increase scenario complexity only after the current tier is stable and measured.

Before trusting a surprising scenario result after code changes, run `coretap --daemon off daemon status` and check the daemon code fingerprint under the returned daemon status payload. The default daemon mode should auto-restart stale daemons, but older running daemons may execute old logic until the new client refreshes them. If behavior contradicts current source/tests, treat stale daemon code as a first-class suspect.

## Command Template

```bash
export UDID=<connected-device-udid>
export RUN_ROOT="$HOME/Library/Caches/Coretap/artifacts/mobile-use-lab/$(date -u +%Y%m%dT%H%M%SZ)-<scenario>"
export TRACE_ID="$(basename "$RUN_ROOT")"
mkdir -p "$RUN_ROOT"
export CT="uv run python -m coretap --backend device --device $UDID --artifact-root $RUN_ROOT --trace-id $TRACE_ID"
```

Use `--trace-title "<human readable scenario>"` on the first command.

Common action forms:

```bash
$CT step --action '{"type":"press","button":"home"}'
$CT step --action '{"type":"openApp","name":"App Store"}'
$CT step --action '{"type":"tap","target":"the Search tab in App Store"}'
$CT step --action '{"type":"tap","target":"the App Store search field"}'
$CT step --action '{"type":"typeText","text":"小红书","replace":true}'
$CT assert text --text "小红书" --timeout-ms 5000
$CT step --action '{"type":"tap","target":"the first search suggestion for 小红书"}'
$CT step --action '{"type":"tap","target":"download the official Xiaohongshu App Store result by Xingin"}'
$CT wait text --text "打开" --timeout-ms 120000 --poll-interval-ms 2000
```

## Stable Patterns

For `typeText replace=true`, inspect `textEntryContext.replaceDecision` in the response. If Coretap classifies the nearby text as `placeholder`, it should skip destructive clearing and type directly.

For Chinese or shifted/symbol-heavy text entry, prefer `typeText` with the actual target text after a VLM tap focuses the field. The mobile-use path uses pasteboard input, visual edit-menu detection, and OCR verification. This avoids unstable Chinese IME candidate commits and prevents blind paste-menu taps.

For ASCII search phrases with spaces, do not force CoreDevice HID keyboard typing. On devices left in a Chinese IME, HID `space` can commit candidates instead of inserting a literal space. `step typeText` should treat spaced ASCII as paste-backed text and verify it visually.

For short ASCII app names or search terms that only need letter case, such as `Safari` or `OpenAI`, CoreDevice keyboard input is acceptable because uppercase letters are safe while the visual paste menu may be fragile in bottom or edge-aligned search fields.

For top search fields, a semantic target such as `App Store search field at the top` records the post-focus text-entry anchor near the focused field. This matters because iOS search bars can move after focus; reusing the initial tap point for paste or long-press can hit a recommendation row.

Do not let wording like `top` override actual geometry blindly. Some iOS search fields are visible near the bottom after a scroll, then relocate above the keyboard when focused. If VLM grounds a search/address field at the bottom of the screen, Coretap should record the active text-entry anchor near the focused field, not at the top of the screen.

When OCR merges iOS edit-menu items into a single token such as `粘贴...自动填充`, treat it as a valid paste menu if the token contains `粘贴` or `Paste`. Click the substring center for the paste segment. Do not use an unverified geometric paste-menu offset.

For Safari bottom address/search fields, the complete focused text may appear in the upper suggestions overlay while the bottom field shows only the tail of a long URL. Text-entry verification may accept an exact or compact URL match in that upper overlay when the original focus anchor is a bottom address bar.

For iOS search fields, OCR may render the focused field as `Q蓝牙` or `Q 小红书` while the same exact text also appears in result rows. Text-input verification must prefer a UI-prefix-stripped match near the text-entry anchor over an exact match elsewhere on the page.

Targets for result rows, suggestion rows, cards, or list items are not text-entry targets, even if their wording mentions context like `below the search field` or `not the search text field`. Do not write a new text-entry anchor after tapping those rows; it can pollute the next `typeText` action.

For App Store install flows, do not treat the query text in the search field as proof that results loaded. After tapping a search suggestion, wait for target-specific result evidence such as the developer name, bundle-specific title, ranking/category, or another unique result token. Then VLM-tap the visible download control with a direct semantic target.

If a result tap opens the App Store detail page, continue semantically from the new page, for example `the blue iCloud download button below the app title`. This is normal state recovery, not a coordinate fallback.

Do not use broad global checks such as `进行中` as App Store install proof. Search pages can show repeated result cards, ads, or lower-page progress labels. Prefer a fresh `observe`, then decide the next semantic action from the page state.

On real devices, `openApp` bundle launch is only confirmed when CoreDevice/DVT returns a positive process id. A `pid:null` launch result means the app did not become launchable; with `strategy:"bundle"` treat this as blocked, and with `strategy:"auto"` let Coretap fall back to Spotlight instead of reporting bundle-launch success.

Do not assert the search field focus by waiting for `取消`; some App Store states keep the keyboard visible without exposing that label. A short default page wait followed by `typeText` and a query-text assertion is the stable focus check.

For App Store developer evidence, Vision OCR may include the small developer icon as a prefix, for example `g Xingin`. Coretap exact text matching strips Vision single-letter UI prefixes, so `assert text --text "Xingin"` should match that token.

For web-search/read scenarios, include negative checks for browser error text such as `打不开` and `丢失网络连接`. A query visible in Safari's bottom address bar is not proof that a results page loaded.

When a system permission sheet overlays an app-owned legal modal, handle only the system sheet and stop before app-owned consent buttons such as `同意`, unless the user explicitly authorizes consent.

## Evidence Checklist

For every failed scenario, preserve:

- Trace file: `<RUN_ROOT>/traces/<trace-id>/trace.json`
- Event log: `<RUN_ROOT>/traces/<trace-id>/events.jsonl`
- Step response JSON.
- Before/after screenshot paths.
- Model input image and grounding JSON for VLM actions.
- OCR plain text and token boxes for text assertions.
- A concise diagnosis: expected state, actual state, root cause, fix, validation result.

## Escalation

Use Oracle Pro when a failure requires model/runtime research, CoreDevice behavior is unclear, or a design choice affects the whole mobile-use architecture. Provide raw traces, screenshots, current code paths, and the exact question. Wait for the result; do not use Deep Research mode unless the user explicitly asks.

When Oracle Pro suggests a pattern that proves useful in a real scenario, update this skill so future runs inherit it.
