# Coretap 功能测试清单

最后更新：2026-06-27

## 测试范围

这份清单用于连接真实 iPhone 后做手工功能回归测试。当前角色只做 QA：

- 测试过程中不修改代码。
- 测试过程中不修复问题。
- 记录每一项的通过/失败、耗时、artifact 路径和真实设备表现。
- 跑完后先输出问题清单，等用户确认后，再切换到开发工程师角色修复。

默认设备：

```bash
export UDID=00008150-001270190A44401C
export CT='coretap --format json --backend device --device 00008150-001270190A44401C'
```

推荐执行方式：

- 真实交互性能测试默认使用 daemon 模式，除非用例明确写了 `--daemon off`。
- 输出统一使用 JSON。
- 测试截图、OCR、VLM、tap、drag、scroll 前，尽量让手机处于竖屏主屏幕。
- `lock`、`power`、`siri`、`mute` 这些有明显副作用的操作，除非用户明确同意，否则先不测真实执行，只测 dry-run。

## 通过标准

一条用例通过，需要同时满足：

- JSON 返回符合预期的成功或预期失败状态。
- 如果不是 `--dry-run`，真实设备行为也必须符合命令意图。

对耗时敏感的命令，需要记录 `durationMs`，并注明是冷启动还是 daemon/model 预热后的热启动。

## 功能测试清单

| ID | 模块 | 命令 | 预期结果 | 备注 |
| --- | --- | --- | --- | --- |
| ENV-01 | CLI 状态 | `coretap --daemon off --format json status` | `ok: true`，返回版本信息 | 不操作设备。 |
| ENV-02 | 配置检查 | `coretap --daemon off --format json config check` | `ok: true`，或返回清晰可执行的配置错误 | 只记录缺失项，不修复。 |
| ENV-03 | 设备发现 | `coretap --daemon off --format json --backend device discover` | 能发现已连接 iPhone，UDID 匹配 `$UDID` | 验证配对、信任、可见性。 |
| ENV-04 | Doctor | `coretap --daemon off --format json --backend device --device "$UDID" doctor` | `result.ready` 为 true，或失败项足够明确 | QA 阶段不安装、不修复。 |
| ENV-05 | OCR 状态 | `coretap --daemon off --format json ocr status` | 能找到 Tesseract | 中英文语言包在 ENV-06 检查。 |
| ENV-06 | OCR 检查 | `coretap --daemon off --format json ocr check` | `chi_sim` 和 `eng` 都可用 | 默认 OCR 语言应为 `chi_sim+eng`。 |
| ENV-07 | 模型状态 | `coretap --daemon off --format json model status` | 返回内置模型 profile | 预期 profile：`builtin:mai-ui-2b-mlx-6bit@1`。 |
| ENV-08 | 模型检查 | `coretap --daemon off --format json model check` | 模型包已安装且可用 | 只记录缺失模型或异常。 |
| DAEMON-01 | Daemon 状态 | `coretap --format json daemon status` | 返回正在运行或已停止状态 | 交互测试前基线。 |
| DAEMON-02 | Daemon 启动 | `coretap --format json daemon start` | daemon 启动成功，或提示已运行 | 记录启动耗时。 |
| DAEMON-03 | Daemon 复用 | `coretap --format json daemon status` | `running: true`，包含 worker 信息 | 验证长驻路径可用。 |
| DAEMON-04 | Daemon 停止 | `coretap --format json daemon stop` | daemon 能干净停止 | 放到所有性能测试之后执行。 |
| SS-01 | 默认截图 | `$CT screenshot --label qa-preview` | 返回的 `frame` 长边约为 1368 px | `sourceFrame` 应保留原始尺寸。 |
| SS-02 | 原图截图 | `$CT screenshot --label qa-full --full-size` | 返回的 `frame` 是设备原始分辨率 | 对照真实手机确认方向正确。 |
| SS-03 | 自定义截图尺寸 | `$CT screenshot --label qa-1024 --max-long-side 1024` | 返回的 `frame` 长边约为 1024 px | 验证压缩参数生效。 |
| SS-04 | 指定截图路径 | `$CT screenshot --out artifacts/coretap/qa-out.png` | 文件存在，JSON 指向该文件 | 验证路径处理。 |
| PRESS-01 | Home 键 | `$CT press home` | 设备回到主屏幕 | 高频安全命令。 |
| PRESS-02 | 音量加 dry-run | `$CT press volume-up --dry-run` | `attempted: false`，返回解析后的 HID button | 基线测试先不改变设备音量。 |
| PRESS-03 | 音量减 dry-run | `$CT press volume-down --dry-run` | `attempted: false`，返回解析后的 HID button | 基线测试先不改变设备音量。 |
| TAP-01 | 坐标点击 dry-run | `$CT tap point --space normalized --x 0.5 --y 0.5 --dry-run` | 返回 normalized、px、HID 坐标换算结果 | 不操作设备。 |
| TAP-02 | 坐标点击 | `$CT tap point --space normalized --x 0.5 --y 0.92` | 点击能送达到设备 | 只在当前屏幕该坐标安全时执行。 |
| OCR-01 | 中文文字断言 | `$CT assert text --text "搜索" --timeout-ms 3000` | 能找到主屏幕上的“搜索” | 不需要额外传 `--lang`。 |
| OCR-02 | 英文文字断言 | `$CT assert text --text "ChatGPT" --timeout-ms 3000` | 如果图标文字可见，应能找到 ChatGPT | 如果当前页面不可见，标记为环境相关。 |
| OCR-03 | 文字点击 dry-run | `$CT tap text "搜索" --dry-run` | OCR 找到文字并返回中心点，但不点击 | 验证精确文字路径。 |
| OCR-04 | 文字点击 | `$CT tap text "搜索"` | 打开并聚焦 iOS 搜索 | 验证 OCR 到坐标再到点击的闭环。 |
| WAIT-01 | 固定等待 | `$CT wait --ms 500` | 等待后返回 `waitedMs: 500` | 不要求设备有变化。 |
| WAIT-02 | 等待文字 | `$CT wait text --text "搜索" --timeout-ms 3000` | 语义等同文字断言 | 如果搜索不可见，先执行 home。 |
| TYPE-01 | ASCII 输入 | `$CT type "hello@example.com" --paste-at 0.2,0.54` | 当前聚焦输入框收到完全一致的 ASCII 文本，并通过 OCR/Vision 验证 | 建议在 OCR-04 打开空搜索框后执行；如需清空旧内容再输入，追加 `--replace`。 |
| TYPE-02 | Unicode 输入 dry-run | `$CT type "搜索" --paste-at 0.2,0.54 --dry-run` | `ok: true`，`text.asciiOnly: false`，不实际输入 | 当前输入路径使用 CoreDevice pasteboard + iOS 编辑菜单粘贴。 |
| VLM-01 | 定位图标 | `$CT locate --target "the ChatGPT app icon"` | grounding 返回 `status: found`，坐标看起来合理 | 记录冷启动/热启动耗时。 |
| VLM-02 | 图标点击 dry-run | `$CT tap target --target "the ChatGPT app icon" --dry-run` | grounding 成功，但不真正点击 | 验证语义目标路径，无副作用。 |
| VLM-03 | 图标点击 | `$CT tap target --target "the ChatGPT app icon"` | 从主屏幕打开 ChatGPT App | 主要 VLM 闭环测试。 |
| GESTURE-01 | 拖拽 dry-run | `$CT drag --from 0.5,0.75 --to 0.5,0.25 --dry-run` | 拖拽路径能转换成 HID 坐标 | 不操作设备。 |
| GESTURE-02 | 向下滚动 | `$CT scroll down` | 当前可滚动页面向下移动 | 在安全的可滚动页面执行。 |
| GESTURE-03 | 向上滚动 | `$CT scroll up` | 当前可滚动页面向上移动 | 放在 GESTURE-02 后执行。 |
| FLOW-01 | 最小 flow | `coretap --format json --backend device --device "$UDID" run artifacts/coretap/qa-flow.json` | 逐步执行 flow，并返回每一步结果 | 执行前先创建最小 QA flow artifact。 |
| REPLAY-01 | 复放 tap-target 结果 | `coretap --format json replay <tap-target-artifact-dir>` | 使用保存的 tap-target 截图重新跑 grounding | 当前只支持 tap-target replay。 |
| TEST-01 | 子命令包装器 | `coretap --format json --backend device --device "$UDID" test -- coretap --daemon off --format json status` | 子命令退出码为 0，并捕获 stdout/stderr 日志 | 只验证 `test` 包装能力，不验证 UI 行为。 |
| NODE-01 | Node 包 smoke | 在 `packages/node` 下执行 `npm test` | Node test kit smoke 通过 | 不替代真机 CLI 功能测试。 |
| NODE-02 | CLI 未安装提示 | 临时用无效 `CORETAP_BIN` 执行 Node test kit | 返回 `CORETAP_CLI_NOT_INSTALLED`，并包含安装提示 | 只测环境提示。 |

## 最小 Flow Artifact

只用于 FLOW-01。这是手工 QA artifact，不是长期自动化测试脚本。

```json
{
  "version": 1,
  "name": "qa-home-search-smoke",
  "steps": [
    {
      "press": {
        "button": "home"
      }
    },
    {
      "screenshot": {
        "label": "qa-flow-home"
      }
    },
    {
      "assertText": {
        "text": "搜索",
        "timeoutMs": 3000
      }
    },
    {
      "tapText": {
        "text": "搜索"
      }
    },
    {
      "type": {
        "text": "hello@example.com",
        "pasteAt": "0.2,0.54"
      }
    }
  ]
}
```

建议位置：

```bash
mkdir -p artifacts/coretap
$EDITOR artifacts/coretap/qa-flow.json
```

## 耗时记录

| 用例 ID | 冷/热启动 | durationMs | Artifact 目录 | 结果 |
| --- | --- | ---: | --- | --- |
| VLM-01 | 冷启动 |  |  |  |
| VLM-01 | 热启动 |  |  |  |
| VLM-03 | 热启动 |  |  |  |
| OCR-01 | 热启动 |  |  |  |
| OCR-04 | 热启动 |  |  |  |

## 问题清单模板

| ID | 严重程度 | 模块 | 命令 | 预期 | 实际 | Artifact / 证据 | 复现备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BUG-001 | P0/P1/P2/P3 |  |  |  |  |  |  |

严重程度说明：

- P0：阻塞大部分真机自动化能力。
- P1：阻塞主流程，例如截图、OCR 点击、VLM 点击、输入、daemon 复用。
- P2：重要问题，但有 workaround 或影响范围有限。
- P3：文案、提示、文档或非阻塞一致性问题。
