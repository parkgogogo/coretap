# Coretap 功能测试清单

最后更新：2026-06-28

这份清单只覆盖当前保留的公开能力。Coretap 的主路径已经收敛为：

- `observe`：截图 + OCR 页面观察。
- `step`：单步 mobile-use action，VLM 语义点击和基础操作都从这里进入。
- `assert text` / `wait text`：文本断言和等待。
- `setup/status/config/discover/doctor/model/daemon`：环境和常驻服务能力。

默认设备：

```bash
export UDID=<connected-device-udid>
export CT="coretap --backend device --device $UDID"
```

## 通过标准

- JSON 返回 `ok: true`，或返回预期的结构化失败。
- 真实设备行为符合命令意图。
- 对模型相关命令记录 `durationMs`、冷/热启动状态和 artifact 目录。
- 功能测试只记录问题，不在测试过程中修代码。

## 功能测试

| ID | 模块 | 命令 | 预期结果 | 备注 |
| --- | --- | --- | --- | --- |
| ENV-01 | CLI 状态 | `coretap --daemon off status` | 返回版本、能力和默认 JSON envelope | 不操作设备。 |
| ENV-02 | 配置检查 | `coretap --daemon off config check` | `ok: true`，或错误可执行 | 检查默认 OCR 和 grounding 配置。 |
| ENV-03 | 设备发现 | `coretap --daemon off --backend device discover` | 能发现已连接 iPhone，UDID 匹配 `$UDID` | 验证配对、信任、可见性。 |
| ENV-04 | Doctor | `coretap --daemon off --backend device --device "$UDID" doctor` | `result.ready` 为 true，或失败项明确 | OCR 也通过 doctor 体现，不再单独测 `ocr` 命令。 |
| MODEL-01 | 模型状态 | `coretap --daemon off model status` | 返回内置模型 profile | 预期 `builtin:mai-ui-2b-mlx-6bit@1`。 |
| MODEL-02 | 模型检查 | `coretap --daemon off model check` | 模型包已安装且可用 | 记录缺失或异常。 |
| MODEL-03 | 模型预热 | `coretap model warm` | 模型可加载，后续 VLM step 更快 | 记录耗时。 |
| DAEMON-01 | Daemon 状态 | `coretap daemon status` | 返回正在运行或已停止状态 | 交互测试前基线。 |
| DAEMON-02 | Daemon 启动 | `coretap daemon start` | daemon 启动成功，或提示已运行 | 记录启动耗时。 |
| DAEMON-03 | Daemon 停止 | `coretap daemon stop` | daemon 能干净停止 | 放到所有测试之后执行。 |
| OBSERVE-01 | 默认观察 | `$CT observe --label qa-observe` | 返回压缩截图、`sourceFrame` 和 OCR token JSON | 默认长边 1368。 |
| OBSERVE-02 | 原图观察 | `$CT observe --label qa-full --full-size` | 返回原始分辨率 frame | 用于排查方向/坐标问题。 |
| OBSERVE-03 | 自定义尺寸 | `$CT observe --label qa-1024 --max-long-side 1024` | 返回长边约 1024 的 frame | 验证压缩参数。 |
| STEP-01 | Home 键 | `$CT step --action '{"schema":"coretap.action.v2","type":"press","button":"home"}' --expect-change` | 设备回到主屏幕 | 高频安全命令。 |
| STEP-02 | VLM 语义点击 dry-run | `$CT step --action '{"schema":"coretap.action.v2","type":"tap","target":"the Watch app icon"}' --dry-run` | grounding 成功但不真正点击 | 目标必须是当前主屏真实可见图标；如 Watch 不可见，替换为可见图标。 |
| STEP-03 | VLM 语义点击 | `$CT step --action '{"schema":"coretap.action.v2","type":"tap","target":"the Watch app icon"}' --expect-change` | 从主屏幕打开目标 App | 主闭环用例；目标必须真实可见。 |
| STEP-04 | 文本输入 | `$CT step --action '{"schema":"coretap.action.v2","type":"typeText","text":"小红书"}' --expect-text "小红书"` | 当前聚焦输入框收到文本并通过 OCR 断言 | 先用 VLM 点击聚焦搜索框；step 会从 OCR 上下文解析粘贴锚点。 |
| STEP-05 | 清空文本 | `$CT step --action '{"schema":"coretap.action.v2","type":"clear","count":20}'` | 当前输入框文本被删除 | 用于搜索框复用。 |
| STEP-06 | 键盘 Enter | `$CT step --action '{"schema":"coretap.action.v2","type":"key","key":"enter"}'` | 提交当前搜索或表单，命令返回成功 | 当前必须已有聚焦输入框；页面可能已提前加载结果，不用固定要求 screenChanged。 |
| STEP-07 | 滚动向下 | `$CT step --action '{"schema":"coretap.action.v2","type":"scroll","direction":"down"}' --expect-change` | 当前可滚动页面向下移动 | 必须在内容明显超过一屏的列表页执行。 |
| STEP-08 | 滚动向上 | `$CT step --action '{"schema":"coretap.action.v2","type":"scroll","direction":"up"}' --expect-change` | 当前可滚动页面向上移动 | 放在 STEP-07 后，且页面不能已经在顶部。 |
| STEP-09 | 等待 | `$CT step --action '{"schema":"coretap.action.v2","type":"wait","ms":500}'` | 返回单步执行成功 | 不要求设备变化。 |
| ASSERT-01 | 中文断言 | `$CT assert text --text "搜索" --timeout-ms 3000` | 能找到“搜索” | 不需要传 `--lang`。 |
| ASSERT-02 | 英文断言 | `$CT assert text --text "ChatGPT" --timeout-ms 3000` | 如果当前页可见，应找到 ChatGPT | 环境相关。 |
| WAIT-01 | 等待文字 | `$CT wait text --text "搜索" --timeout-ms 3000` | 等价于带轮询的文本断言 | 当前页不可见时先回主屏。 |
| NODE-01 | Node smoke | `node packages/node/smoke.js` | Node test kit smoke 通过 | 不替代真机功能测试。 |
| NODE-02 | CLI 未安装提示 | 使用无效 `CORETAP_BIN` 执行 Node test kit | 返回 `CORETAP_CLI_NOT_INSTALLED` 和安装提示 | 只测环境提示。 |

## App Store 小红书链路

用于验证真实 mobile-use 效率和稳定性：

1. `$CT step --action '{"schema":"coretap.action.v2","type":"press","button":"home"}' --expect-change`
2. `$CT step --action '{"schema":"coretap.action.v2","type":"openApp","name":"App Store"}' --expect-change`
3. 如当前不在搜索页：`$CT step --action '{"schema":"coretap.action.v2","type":"tap","target":"the App Store search tab"}'`
4. `$CT step --action '{"schema":"coretap.action.v2","type":"tap","target":"the App Store search field"}'`
5. `$CT step --action '{"schema":"coretap.action.v2","type":"typeText","text":"小红书"}' --expect-text "小红书"`
6. `$CT step --action '{"schema":"coretap.action.v2","type":"key","key":"enter"}'`
7. `$CT wait text --text "小红书" --timeout-ms 10000`
8. 如按钮可见且 App 未安装：`$CT step --action '{"schema":"coretap.action.v2","type":"tap","target":"the GET or cloud download button for Xiaohongshu"}' --expect-change`
9. 如触发下载：`$CT wait text --text "打开" --timeout-ms 120000`

每一步记录 `durationMs`、是否冷启动、artifact 目录和真实设备表现。

## 耗时记录

| 用例 ID | 冷/热启动 | durationMs | Artifact 目录 | 结果 |
| --- | --- | ---: | --- | --- |
| MODEL-03 | 冷启动 |  |  |  |
| STEP-02 | 热启动 |  |  |  |
| STEP-03 | 热启动 |  |  |  |
| OBSERVE-01 | 热启动 |  |  |  |
| App Store 链路 | 热启动 |  |  |  |

## 问题清单模板

| ID | 严重程度 | 模块 | 命令 | 预期 | 实际 | Artifact / 证据 | 复现备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BUG-001 | P0/P1/P2/P3 |  |  |  |  |  |  |

严重程度说明：

- P0：阻塞大部分真机 mobile-use 能力。
- P1：阻塞主流程，例如 observe、VLM tap、输入、daemon 复用。
- P2：重要问题，但有 workaround 或影响范围有限。
- P3：文案、提示、文档或非阻塞一致性问题。
