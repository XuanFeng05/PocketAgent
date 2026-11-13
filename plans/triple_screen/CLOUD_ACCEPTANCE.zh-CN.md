# 云端研究增量验收与合并说明

日期：2026-11-13。环境：Linux、Python 3.12.14、独立 `.venv`、CPU；安装 `requirements-research.txt`（已引用 `requirements-dev.txt` 和核心依赖），没有安装 PyTorch/CUDA。

本增量以本地页面修复提交 `21cd849` 为共同基线，包含云端 `ed99ec1` 的 US100 研究、`bc8d0b1` 的 US500 研究以及本次验收修复。代码可以合并；通过范围是研究模块、数据审计和自动化工程验证，不是新增美股 Dashboard 或实盘盈利验收。

## 本次修复

1. **US100 跨阶段身份保护**：旧流程只检查同阶段已有 checkpoint，首次 validation 可能接受训练后修改的规则。现在训练恢复、validate、costs、analyze 均核对完整候选族、冻结训练签名、参数、日期、数据/配置及源码身份。新 checkpoint 记录引擎、指标统计、CLI 与技术标准化模块的 SHA；旧格式只校验原来记录的字段，不追写当前签名。已有一致输入快照不重复写入。
2. **跨平台 UTF-8**：研究 JSON、名单和 JSONL 文本读写显式使用 UTF-8，避免非 ASCII 供应商名在部分 Windows 默认编码下落盘失败。回归使用真正关闭 UTF-8 mode 的子进程，检查落盘、回读与禁止联网的缓存复用；这不是原生 Windows 浏览器测试。
3. **状态文档**：根目录状态明确分开当前 US500、原页面修复和首次快照历史；`CURRENT_SCOPE.zh-CN.md` 标明为初始本地 100 股范围，避免把后续云端授权误判为越界。

## 本轮验收证据

| 检查 | 范围及结果 |
|---|---|
| 最终 Python 全量回归 | `python -B -m pytest -q -p no:cacheprovider`：617 passed、2 skipped、17 warnings，73.53 秒；跳过为可选 PyTorch 训练相关项 |
| 前端行为 | `node tests/frontend/test_triple_ui.js`，10 项通过 |
| 独立页面/API 与取消 | `test_triple_dashboard.py`、`test_triple_cancellation.py`，31 项通过；Linux 合成 fixture，无真实行情依赖 |
| 依赖与入口 | `python -m pip check` 通过；US100/US500 下载、研究及质量 CLI 的 `--help` 可执行 |
| 真实数据重新审计 | 固定 500 股全部通过原始响应重建、标准化字段、日历与文件哈希核对；保留零量行，不补造或删除行情 |
| 已完成研究工件 | 两段各 1540 个组合的矩阵、逐股指标、账户汇总和冻结时序复核通过；独立重算净值、现金/交易账及同步区块统计 |
| 真实账户重放 | 8 股重新生成特征，与缓存一致；2 段 × 3 策略 × 8 股共 48 个账户重算匹配，信号形成早于入场 |
| 修改边界 | 原 `app_layer`、原三重滤网执行器及原页面测试未被研究增量覆盖；5 个编码修复模块 AST 除编码参数外一致 |

本轮重新执行自动化测试、500 股数据审计和工件复核，**没有声称以修复后源码完整重跑了全部 154 万个股票×策略任务**。完整批次来源于已完成的 `bc8d0b1` 研究检查点；本次新增保护和编码修复不改变交易计算。运行日志、原始行情、统计输出及回测结果仅保留云端 `runtime_layer/reports/`，不提交 Git。

数据审计首次在 180 秒外层上限停止，未记录数据错误；排查单股 CPU 审计耗时后，以 600 秒上限重试并完成。超时日志保留，没有将超时伪装为通过或重新下载行情。

## 配置与复现

- US100 下载：`config/research/us100_download.json`；US500 下载：`config/research/us500_download.json`。
- 固定股票：`config/universe/us_research100.txt`、`us_research500.txt`；500 清单前 100 项保留原顺序，不重新按实时指数选股。
- US500 主要规则：`config/research/multiscale_us500_v1.json`；数据质量敏感性：`config/research/us500_quality_sensitivity.json`。
- Yahoo Finance 日线，请求 2020-01-01 至 2026-07-12，标准化价格为供应商 OHLC 乘以 adjclose/vendor_close，成交量单位为股。该调整定义与 BaoStock 参数不同。
- 所有路径相对仓库根目录；Windows 可用 `py -3.12 -m venv .venv`，Linux 可用 `python3.12 -m venv .venv`，激活环境后执行下列命令。

```bash
python -m pip install -r requirements-research.txt
python -B -m pytest -q -p no:cacheprovider
node tests/frontend/test_triple_ui.js
python -m research_layer.multiscale_data --audit-only
```

只有在用户授权下载的环境中，缺行情时才运行：

```bash
python -m research_layer.multiscale_data
```

首次研究运行可用：

```bash
python -m research_layer.cli.multiscale_search --stage all --workers 4
python -m research_layer.cli.multiscale_quality
```

已有旧源码产生的研究目录时，保留原工件；源码 SHA 改变后不得修改旧签名来绕过保护。重新运行应复制研究配置、使用新的 `output` 目录，并将质量配置的 `strategy_config` 指向该副本。下载目录与固定股票清单可以继续使用已审计版本。恢复同一次研究应使用与其签名一致的代码、配置与行情。旧 US100 checkpoint 未记录 CLI/标准化模块 SHA，不能追认这些依赖曾被完整冻结。

## 不在本次通过范围

- 新 US100/US500 CLI 与 Dashboard 整合、CLI 历史自动导入、GUI 跨进程续算。
- 新的浏览器操作或原生 Windows 全流程验收。
- 共享组合资金、跨股票仓位分配、做空、加仓、盘中真实撮合或实盘。
- 无偏收益证明：当前成分和后验流动性筛选、SW/FERG/TLN 的历史衔接风险、不同数据快照、交易成本近似均仍披露；数据未满足完整周线预热时保留现金。
- 新的未见测试：2022–2025 已用于探索，只能称回顾性验证。工程通过和高胜率都不等于稳定超额收益。
