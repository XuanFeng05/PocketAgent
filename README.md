# PocketAgent

CPU-first strategy research and backtesting. 保留原A股100股研究；在 `codex/local100-ui-repair` 页面修复基础上增加独立的 **100只非仙股美股**验证路径。两个市场的清单、数据、日历和执行版本分开，研究结果不等同实盘收益。

**当前状态与云端接手入口：[PROJECT_STATUS.zh-CN.md](PROJECT_STATUS.zh-CN.md)。** 早期 docs/plans 保留历史决策与实验摘要，其中的完成声明应结合最新状态说明及实际代码核对。

2026-11-14：较早 US100 bundle 已离线集成，本地继续完成周线入场／退出四组对照。见 [bundle 审查](plans/triple_screen/BUNDLE_REVIEW.zh-CN.md) 和 [本地研究结果](plans/triple_screen/WEEKLY_COMPONENT_RESULTS.zh-CN.md)。原策略默认保持，新增组件案例由 CLI/API 读取配置运行。

后续 US500 云端下载与研究摘要见 [CLOUD_ACCEPTANCE](plans/triple_screen/CLOUD_ACCEPTANCE.zh-CN.md) 和 [DOWNLOAD_SOURCES](plans/triple_screen/DOWNLOAD_SOURCES.zh-CN.md)；这两份摘要不代表后续 US500 代码或原始数据已合入当前分支。美股数据口径与本地 A 股前复权口径隔离；Git 不包含云端原始行情。

## 仓库范围

包含源码、策略配置、股票代码清单、合成测试和研究说明。**不包含下载的行情、数据库、模型权重、运行结果、虚拟环境、缓存、论文或个人本机配置。** `runtime_layer/` 只有说明文件进入 Git。

因此，clone 本仓库不会得到本地100股行情，也不能仅凭历史实验摘要复现市场收益。不会因为读取页面或运行合成测试而自动补齐行情。

## 代码结构

- `strategy_layer/`：规则策略与三重滤网配置。
- `feature_layer/triple_screen.py`：已完成周线 MACD、日 EFI 和时点控制。
- `research_layer/triple_screen.py`：前复权标准化净值、触发和退出状态。
- `research_layer/triple_runner.py`：多股票、成本情景与工件保存。
- `research_layer/triple_validation.py`：固定规则的独立时间窗口验证。
- `research_layer/triple_study.py`：参数邻域与滤网移除对照。
- `research_layer/triple_shards.py`：本地分片及校验恢复。
- `app_layer/`：本地 Dashboard、API 与页面。
- `data_layer/`、`download_layer/`：存储与显式下载工具。本轮已获云端下载授权；页面不会自动下载，本机取数仍由用户另行安排。
- `agent_layer/`：保留的旧训练兼容代码；CPU研究入口不需要 PyTorch。

## 本地使用

建议 Python 3.12和独立虚拟环境。测试安装 `requirements-dev.txt`（已引用基础依赖），美股另安装 `requirements-us.txt`。本轮云端已准备依赖；旧训练依赖独立放在 `requirements-training.txt`，当前CPU研究不需要PyTorch或CUDA。

在已经准备好环境的项目根目录运行：

```text
python -m app_layer.cli.run_dashboard --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/triple.html` 使用A股，或 `http://127.0.0.1:8000/triple.html?market=US` 使用美股。沿用本地修复分支的真实进度、任务边界取消、可编辑年度窗口、10案例与子报告。没有行情时不会自动下载。

美股下载与合并配置见 [US_MERGE](plans/triple_screen/US_MERGE.zh-CN.md)，市场口径见 [US_SCOPE](plans/triple_screen/US_SCOPE.zh-CN.md)。100股清单为 `config/universe/us_research100.txt`，日线独立存入 `runtime_layer/data_us`；数据和运行结果不进Git。

CLI 固定规则配置：

```text
python -m research_layer.cli.triple_screen --config config/research/triple_screen_v1.json
```

此命令要求本机已有 `runtime_layer/data`。带 `--demo` 的旧 CPU 回测入口可用于合成行情检查：

```text
python -m backtest_layer.cli.run --demo --strategy sma_trend
```

## 测试与证据边界

无市场数据的环境可先运行：

```text
python -B -m pytest -q -p no:cacheprovider
node tests/frontend/test_triple_ui.js
node --check app_layer/frontend/triple.js
```

Dashboard集成测试使用临时合成分片，不依赖原机器行情；Node测试覆盖payload、日期、案例、格式化与状态逻辑。`node --check`只检查语法，不证明界面行为正确。

实验规格见 [SPEC](plans/triple_screen/SPEC.zh-CN.md)，历史本地结果解读见 [M6_RESULTS](plans/triple_screen/M6_RESULTS.zh-CN.md)。这些 Markdown 中出现的运行目录和 Windows 绝对路径是原机器上的证据位置，运行工件没有随仓库上传。
