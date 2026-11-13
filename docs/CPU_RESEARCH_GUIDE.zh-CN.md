# PocketAgent CPU 策略研究使用说明

更新日期：2026-11-07。P0／P1 CPU 闭环及 P2 批量参数、滚动验证、FIFO 报告已完成。默认工作台、命令行、账户和回测都可在没有 PyTorch 的环境中运行。新功能详见 [批量研究与滚动验证](PARAMETER_RESEARCH_GUIDE.zh-CN.md)。

## 启动工作台

在项目目录运行：

```powershell
Set-Location -LiteralPath 'D:\project\on-going\PocketAgent'
.\.venv\Scripts\python.exe -m app_layer.cli.run_dashboard --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/`。若该端口已被占用，指定其他端口，不结束已有应用进程。

先选择“演示数据”运行一次，检查策略、净值、成交和对照报告；演示行情为合成数据。之后可切换到本地 CSV 或 `runtime_layer/data` 行情目录。原有数据和训练界面仍可通过“数据与原有工作台”进入，训练／模型回放需要单独的可选训练环境。

本机已创建项目内 `.venv` 并安装 CPU 依赖。其他机器首次设置：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt` 为 CPU 运行依赖，`requirements-dev.txt` 额外包含 pytest；`requirements-training.txt` 单独保存原有训练依赖。当前 CPU 虚拟环境没有安装 torch，未修改全局 Python 环境。

## 当前支持的策略与输入

- `sma_trend`：收盘快均线高于慢均线且空仓时发买入信号，反之持仓时发卖出信号。默认快窗口 5、慢窗口 20。
- `breakout`：收盘突破此前窗口最高收盘价时入场，跌破退出窗口最低收盘价时离场。
- `mean_reversion`：收盘向下偏离均线达到阈值时入场，回到均线时离场。
- `buy_hold`：尝试买入，未成交且仍空仓则继续尝试；首次实际成交后持有至期末估值。首次成交量受账户与流动性限制，不承诺一次满仓。
- `cash`：始终保持现金。

候选与 cash／buy-hold 使用同一个账户、执行器和执行配置。净值每天在收盘估值，信号在下一可用交易日开盘尝试执行。交易记录保留信号、委托、执行时间及拒单原因。

CSV 至少包含：

```text
symbol,date,open,high,low,close,volume,is_st
000001.SZ,2025-01-02,10.0,10.2,9.9,10.1,1000000,0
000001.SZ,2025-01-03,10.1,10.3,10.0,10.2,1200000,0
```

这里两行仅演示字段，均线策略需要足够历史。日期可用 `date` 或 `datetime`；若无 symbol 列，需显式指定股票。可选 `amount`、`limit_pct`、`limit_reference_close`、`market_can_buy`、`market_can_sell`、`is_tradeable`。布尔字段必须有明确的 0／1 或 true／false，不将缺失 ST 当正常股票。日期缺失、重复日期、非法 OHLC、非有限价格与多股票混用会被拒绝。

本地行情读取不复权日线并合并按日期记录的 ST 状态；状态覆盖不完整时明确报错。读取不会下载、重建缓存或改写原数据库。

## 命令行运行

演示回测：

```powershell
.\.venv\Scripts\python.exe -m backtest_layer.cli.run --demo --strategy sma_trend
```

读取现有本地行情：

```powershell
.\.venv\Scripts\python.exe -m backtest_layer.cli.run --db runtime_layer/data --symbol 000001.SZ --start 2025-01-02 --end 2025-02-28 --strategy sma_trend --output runtime_layer/reports/research
```

使用 `--csv <路径>` 可读取 CSV。`--parameters` 接受 JSON，例如 `'{"fast_window":5,"slow_window":20}'`；`--execution` 接受显式执行参数，`--no-baselines` 仅用于不需要对照的调试。Windows shell 的 JSON 引号处理与终端版本有关，Dashboard 可直接填写参数。

CLI 默认输出到 `runtime_layer/reports/research`，Dashboard 输出到 `runtime_layer/research/runs` 并在 `requests` 记录任务状态。目前网页最近实验列表展示网页创建的任务；CLI 结果直接从其输出目录读取。

每次运行创建唯一目录，保存 `config.json`、`input.csv`、`result.json`、`equity.csv`、`fills.csv` 和 `status.json`。配置含数据和实现源码 SHA-256，可定位具体输入与实现；运行失败保留失败状态，不覆盖已有实验。

## 已修复的正确性问题

1. 旧环境新增推进前账户快照；评估与回放用真正时点的 NAV，RL reward 的 next-state 语义保持不变。
2. 月线快慢索引均排除当前月尚未完成的历史 K 线；缺少月线时可从 daily 聚合。
3. 日内聚合按实际交易 session 和时钟分桶，缺 bar 不移动后续窗口，Python 与 SQL 口径一致。
4. 数据包先完整校验和 staging，再替换并保留回滚路径；损坏、取消或切换失败不先清空原数据。
5. 旧回放的 Buy-and-Hold 改为真实执行对照；简单价格收益单独标识。新 CPU 对照也共用执行器。
6. 旧模型回放默认继承 checkpoint 的执行配置，空白控件不再强制覆盖成 20% 仓位；显式零费用也能保留。

派生行情与 feature-parts 的计算签名已更新，下一次显式构建会识别旧规则。已有特征与 Agent Cache 没有被就地重建或删除；旧 monthly 数据集在继续用于模型研究前需显式重建。

## 新核心与兼容性

`backtest_layer` 保存唯一的账户、动作、执行器、市场快照、日线引擎与数据入口。原 `agent_layer` 路径兼容导出相同类对象，避免两套费用和持仓语义。

`strategy_layer` 生成 `OrderIntent`。当前数量语义是可执行容量比例，不是目标持仓权重；策略不能直接修改账户。`research_layer` 已支持单次、有限参数批次和滚动验证，FIFO 随报告保存。离线机器批次与共享资金组合仍属后续阶段。

## 结果适用边界

当前是独立单股账户的日线研究，不是共享资金组合。开盘容量使用前一个观察日成交额作为代理，不使用当日最终成交额改善成交，无法精确重建真实集合竞价。

公司分红、拆股等公司行为尚未建模；显式标记 corporate_action 的输入会拒绝。实际研究应选择不受这些事件影响的区间或补齐公司行为处理。IPO、特殊历史涨跌幅规则需要明确的逐日限制字段，不能将默认规则视为覆盖全部历史情况。期末持仓按收盘市值估值，不强制卖出；现金不计利息。

## 验证与资产保护

完整 CPU 测试命令：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

P0／P1 验收时为 84 passed；P2 集成后完整测试为 **166 passed、2 skipped、7 warnings**。两项跳过分别是模型测试模块与单个张量转换测试，需要可选 PyTorch；其余执行约束测试仍运行。7 条警告来自 pandas／NumPy 的 timedelta 弃用提示，没有测试失败。真实神经网络 checkpoint 回放未做端到端验收。

另完成 Dashboard 实际点击、净值曲线、成交价格、已保存记录展示，以及本地 `000001.SZ` 在 2025-01-02 至 2025-02-28 的只读日线回放。该回放验证工程流程，不作为策略有效性结论。

源码改造前快照位于 `D:\project\backups\PocketAgent_refactor-20261107\source-before.zip`，包含 176 个原文件，并保留原 Git diff 与 staged diff。未删除原文件，未重置工作区、提交或推送。账户与执行器原实现迁入新核心，旧路径保留兼容。

保护清单包含 runtime 与 references 下 10,181 个原文件，合计 31,564,285,528 字节；验收比较了路径、大小和修改时间，未发现变化。这里是资产元数据核验，不声称对 31 GB 资产逐字节重新哈希。测试和演示报告写在外部备份目录或临时目录中。

界面截图、只读数据回放产物、资产核验与环境记录保存在 `D:\project\backups\PocketAgent_refactor-20261107\evidence`。
