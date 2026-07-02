# Coretap Mobile-Use Scenario Catalog

Use this catalog to create increasingly difficult, realistic tasks. A tier is stable only when it succeeds twice in a row on a real device, has trace evidence, and has no unexplained workaround.

## Tier 0: Environment And Primitive Checks

Goal: confirm the lab can trust the device, daemon, screenshot, OCR, model, and simple CoreDevice actions.

- `env-health`: run `status`, `config check`, `discover`, `doctor`, `model check`, and `daemon status`.
- `daemon-fingerprint`: after code edits, run `coretap --daemon off daemon status` and confirm the daemon code fingerprint matches the current client; default daemon mode should refresh stale daemons automatically.
- `observe-home`: press Home, observe the home screen, verify screenshot orientation and OCR language.
- `open-known-app`: open App Store by bundle/name, terminate it, reopen it.
- `semantic-icon-tap`: from Home, VLM-tap a visible app icon, then press Home.

## Tier 1: Single-Screen Agent Actions

Goal: make individual actions reliable enough for an agent loop.

- `appstore-search-focus`: open App Store, navigate to Search, VLM-tap the search field, verify keyboard/focus through page observation or a separate text check.
- `text-entry-ascii`: type `xiaohongshu` into App Store search and assert it is visible.
- `text-entry-cjk`: VLM-tap the App Store top search field, type `小红书`, and assert it is visible. Expected strategy is `visual_paste_verified`, not pinyin keyboard.
- `search-suggestion`: type a query and VLM-tap the first matching App Store suggestion, avoiding keyboard candidate bars.
- `scroll-results`: scroll App Store results down and back up with visible page-state changes.

## Tier 2: Short User Goals

Goal: complete practical goals with 3-8 actions and no manual coordinate fallback.

- `install-appstore-app`: uninstall target app, open App Store, search, wait for unique official-result evidence such as developer/title/category, download that result, and wait for `打开`.
- `open-installed-app`: open a known installed app by semantic request, verify app-specific landing text or state.
- `first-launch-permission-modal`: from a newly opened app, handle a system permission modal with a VLM tap and verify the next app-owned modal or screen. Stop before accepting app legal/privacy terms unless the user explicitly asks.
- `settings-toggle-readonly`: open Settings and navigate to a named settings page without changing destructive state.
- `web-search-read`: open Safari, search a short term, and verify results page text.

## Tier 3: Robust Mobile Use

Goal: tolerate app state, ads, keyboard candidates, lists, and delayed network state.

- `install-with-ad-result`: search an app whose App Store result page contains an ad before the target result; verify the official developer/title before tapping download, and download the target, not the ad.
- `resume-from-random-state`: begin with the target app already open, backgrounded, or on a details page, then complete the goal.
- `first-launch-legal-consent-boundary`: detect app-owned legal or privacy prompts, report the required user consent, and do not auto-accept unless explicitly authorized.
- `multi-app-assertion`: complete a task in one app, open another app, and verify the first task's user-visible result.
- `recover-from-wrong-page`: intentionally start on a wrong details page and recover using Back/Search without manual coordinates.

## Required Metrics

For each scenario run, record:

- `success`: true/false.
- `totalMs`: sum of command `durationMs` on the intended path.
- `stepCount`: number of Coretap commands.
- `vlmStepCount`: number of VLM semantic actions.
- `failedStep`: first failed command, if any.
- `dominantCost`: screenshot, model, CoreDevice delivery, OCR/assertion, app/network wait, or state recovery.
- `trace`: trace path.
- `artifactRoot`: scenario artifact root.

## Known Patterns

- A `step` action should return the resulting page state after the default wait; use that state to choose the next action.
- Text assertions are separate commands. Use `assert text` for immediate checks and `wait text` for delayed state.
- App Store result pages may OCR developer names with a leading icon prefix, for example `g Xingin`. Treat this as an assertion concern, not a VLM grounding failure; Coretap exact matching should strip Vision single-letter UI prefixes.
- App Store top search fields move after focus. A semantic target such as `App Store search field at the top` should record the active text-entry anchor near the focused field.
- Settings and other iOS search fields may be visible near the bottom after scrolling, then relocate above the keyboard when focused. Anchor text entry near the focused field instead of trusting target words like `top`.
- For CJK and shifted/symbol-heavy text, the stable agent path is pasteboard plus visual edit-menu detection plus OCR verification. Pinyin keyboard input is not the default mobile-use path.
- Short ASCII app/search names with uppercase letters, such as `Safari` or `OpenAI`, can use CoreDevice keyboard input; uppercase letters are safe, but spaces and shifted punctuation are not.
- ASCII text containing spaces is paste-backed in agent `step typeText`; HID space can be swallowed by a non-English IME.
- Safari bottom address fields can show the complete edited text in a top suggestions overlay while the bottom bar shows only a suffix; text-entry verification should tolerate that layout.
- Vision may merge iOS edit-menu items into one token such as `粘贴...自动填充`. If the token contains the exact paste label, click the paste substring center.
- Text-input verification must consider all exact and UI-prefix-stripped candidates before applying the near-anchor filter. A far exact result such as `蓝牙` should not hide a nearby focused-field token such as `Q蓝牙`.
- Result rows, suggestion rows, list items, and cards should not create text-entry anchors, even when the target text mentions context such as `below the search field`.
- If a row/suggestion/list/card tap still reports `textEntryAnchor`, first check the daemon code fingerprint with `coretap --daemon off daemon status`. A stale development daemon can execute old anchor-writing code even when current tests pass.
- Do not use the App Store search field query alone as result-page proof. Require official app evidence such as title, developer, category/rank, or the final `打开` state.
- If a result-row download tap opens the App Store detail page, continue semantically by tapping `the blue iCloud download button below the app title`; this is a valid recovery path and often more stable than retrying the same target.
- Do not use global `进行中` as App Store install proof. It can belong to repeated lower result cards or ads. After official result evidence appears, VLM-tap the official result's visible download control, then verify the final `打开` state.
- Web-search/read scenarios must assert that browser network-error text such as `打不开` and `丢失网络连接` is absent; a query visible in the address bar alone is not a loaded result page.
- System permission sheets can overlay app-owned consent modals. Handling the system sheet is allowed, but the scenario must stop at app-owned legal/privacy buttons such as `同意` unless the user explicitly authorizes consent.
- `terminateApp` should be judged by actual running checks. A nonzero `pidAfter` can be stale or non-running; use `runningAfter`/`pidAfterRunning` for state decisions.
- Real-device `openApp` bundle launch is confirmed by a positive CoreDevice/DVT pid. A `pid:null` result is not launch success; strict bundle strategy should block, while auto strategy may fall back to Spotlight.

## Validated Scenario Notes

- `settings-bluetooth-readonly`: `artifacts/coretap/mobile-use-lab/20260629T-settings-bluetooth-textverify-fix/traces/20260629T-settings-bluetooth-textverify-fix/trace.json`
- Result: Settings readonly navigation succeeded. Coretap opened Settings, VLM tapped a bottom-visible search field, used the active text-entry anchor, pasted `蓝牙`, verified `Q蓝牙` near the anchor, then VLM tapped the first `蓝牙` result row and landed on the Bluetooth privacy page without changing toggles.
- Timing: `45146ms` total traced command duration; `typeText 蓝牙` was `12008ms` with one attempt.
- `install-appstore-app` repeat: `artifacts/coretap/mobile-use-lab/20260629T035000CST-install-xiaohongshu-rerun/traces/20260629T035000CST-install-xiaohongshu-rerun/trace.json`
- Result: 小红书 install succeeded from uninstall + terminated App Store in eight steps. Final official `Xingin` result showed `打开`.
- Timing: `78967ms` total traced command duration; App Store-open-to-final was `63319ms`; search-field-to-final was `54525ms`.

## Promotion Rule

Do not create a harder scenario until the current scenario has:

- one clean success from a prepared state;
- one success from a slightly different state;
- no hidden manual coordinate fallback;
- a documented timing baseline;
- any new stable usage pattern folded back into `SKILL.md`.
