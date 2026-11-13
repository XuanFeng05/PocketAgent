# PocketAgent

CPU-first A-share strategy research and backtesting. 当前重点是在已有本地 **100 股**上研究前复权三重滤网，比较规则、成本和不同时期的表现。研究结果不等同实盘收益。

**当前状态与云端接手入口：[PROJECT_STATUS.zh-CN.md](PROJECT_STATUS.zh-CN.md)。** 早期 docs/plans 保留历史决策与实验摘要，其中的完成声明应结合最新状态说明及实际代码核对。

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
- `data_layer/`、`download_layer/`：存储与显式下载工具。使用下载工具前应由用户安排，当前任务禁止自动下载。
- `agent_layer/`：保留的旧训练兼容代码；CPU研究入口不需要 PyTorch。

## 本地使用

建议 Python 3.12。依赖清单为 `requirements.txt`、`requirements-dev.txt`；旧训练依赖独立放在 `requirements-training.txt`。环境准备由使用者安排，本次源码同步没有下载或安装依赖。

在已经准备好环境的项目根目录运行：

```text
python -m app_layer.cli.run_dashboard --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/triple.html`。这个页面目前可读取本地股票清单、提交任务及查看单股/批量报告；**年度显示、完整案例选择、进度/取消等仍有待修项**，详见当前状态。没有行情时不会自动下载。

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
python -B -m pytest -q -p no:cacheprovider --ignore=tests/integration/test_triple_dashboard.py
node --check app_layer/frontend/triple.js
```

被排除的 Dashboard 测试文件目前直接依赖原机器的100股行情；后续应改成临时合成 fixture。`node --check` 只检查语法，不证明界面行为正确。

实验规格见 [SPEC](plans/triple_screen/SPEC.zh-CN.md)，历史本地结果解读见 [M6_RESULTS](plans/triple_screen/M6_RESULTS.zh-CN.md)。这些 Markdown 中出现的运行目录和 Windows 绝对路径是原机器上的证据位置，运行工件没有随仓库上传。
