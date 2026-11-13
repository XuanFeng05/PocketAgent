# PocketAgent 策略研究与回测改造方案

日期：2026-11-07。产品方向：面向个人研究的策略挖掘与回测工作台，日常研究以笔记本 CPU 为基础，可将完整独立实验分发到不互相通信的学校电脑。

这份方案最初用于旧项目归档与改造规划。2026-11-07 已按用户授权完成 P0、P1 与 P2 的有限参数挖掘、滚动验证及 FIFO 报告。使用方法见 [CPU 使用说明](CPU_RESEARCH_GUIDE.zh-CN.md) 和 [批量研究与滚动验证](PARAMETER_RESEARCH_GUIDE.zh-CN.md)。后续训练源码清理与离线机器分发仍按阶段推进，现有训练资产未删除。

## 产品目标与首版范围

PocketAgent 应帮助研究者把一个交易想法变成可复现的证据：明确适用条件、进出场规则、仓位和成本，比较不同历史时期的表现，检查失败原因。优先保证数据时点、执行和评估正确，再扩大策略数量。

首版采用日线和有限股票样本，建立独立单股账户的研究闭环。选择 2 至 3 个可解释的策略家族，例如趋势、突破、均值回归；每类只开放少量参数，总实验先控制在几十组。这些是研究候选，不预设其有效性。

首版交付数据选择、策略配置、批量实验、交易回放、对照报告和实验导入导出。收益、回撤、费用、换手、拒单原因和不同时间段表现应能追溯到成交事件。

跨股票训练或逐股回测平均值均不等于共享资金组合。首版界面明确标为逐股研究；真实组合需要新增统一现金、同时持仓、调仓与组合级约束，列入后续阶段。

深度学习与强化学习退出默认产品流程。轻量监督学习可在回测可信后作为可选策略类型接入，遵守同样的数据边界和评估规则。

## 当前基础与实施边界

当前代码在 GitHub 基线 `d21694c` 上有 32 个已修改的跟踪文件，另有新 CLI、分布式实现和测试等未跟踪文件。实施时应保留这些本地工作，建立明确文件清单，避免以远端版本覆盖本机。

当前训练主线是跨股票共享参数的单股多周期 PPO。`SingleSymbolMultiFrequencyPolicy` 继承 `MultiFrequencyMAPPO` 以复用网络和兼容格式，每个 episode 的股票轴为一，不代表多股票联合持仓策略。

实际配置来自 JSON、CLI、controller 和代码默认值。`config/data.yaml`、`train.yaml`、`eval.yaml`、`backtest.yaml` 均为空文件；README 的三折与 252 日测试说明也不适用于所有入口。新方案建立唯一可校验的实验配置，UI 和 CLI 不再各自设置隐含默认值。

## 现有模块处理清单

“保留”表示保留能力并修复缺陷，不表示无需修改。“可删除”均附迁移门槛；当前源码删除与旧目录删除是两项独立工作。

| 现有位置 | 处理决定 | 具体改造与保留边界 |
| --- | --- | --- |
| `download_layer/` | 保留 | 复用下载、重试和数据目录；补可重现的数据版本及来源记录，不恢复 old 下载链路 |
| `data_layer/storage/`、`inventory/`、`validators/` | 保留并修复 | 保留存储、覆盖率、股票池、质量检查；数据包改为先校验、暂存、再切换；补历史样本选择和缺失时隙记录 |
| `feature_layer/builders/`、`indicator_registry.py`、`rules.py` | 保留并调整 | 技术指标与时点约束用于规则策略；修周期聚合与月线可见性；将特征计算与神经网络张量组装分开 |
| `feature_layer/datasets/`、`model_input_registry.py`、`config/features/` | 拆分 | 保留数据契约、列顺序、schema 与增量结果；研究侧使用轻量表格／分片，不要求每个策略生成完整 LSTM 窗口和 Agent Tensor Cache |
| `agent_layer/portfolio/account.py` | 保留并迁移 | 迁入通用回测核心，保留现金、持仓、可卖数量、费用、估值和平均成本语义；FIFO 成交报告独立实现 |
| `agent_layer/environment/execution.py`、`actions/` | 保留并重构接口 | 复用成交与拒单规则，去除对 Agent 特有市场结构的硬依赖；现有 size 为方向对应的执行比例，不能直接当目标仓位权重 |
| `agent_layer/data/timeline.py`、`timeline_types.py`、`feature_parts.py` | 提取通用部分 | 保留行情时间流、字段可用时间和市场约束；将轻量行情快照与模型序列张量拆开 |
| `agent_layer/environment/trading_env.py`、`single_symbol_env.py` | 重写回测调度 | 建立无 reward、episode、policy tensor 的回测循环；先留兼容适配，修净值时间错位后再退出 RL 环境 |
| `agent_layer/experiments/splits.py` | 保留并迁移 | 统一时间切分与重叠样本隔离；以交易日期划分多股票数据，不直接对拼接后的数据行做随机划分 |
| `agent_layer/experiments/run_store.py`、`run_config.py` | 借鉴后重写 | 保留不可变配置、原子保存、运行状态和日志；新实验不以 checkpoint、训练 step 为中心 |
| `evaluation_layer/metrics/` | 保留并验证 | 指标输入统一为真实日终净值；补现金基准、成交级统计、稳健性结果；DSR／PBO 输入需能追溯试验集合 |
| `evaluation_layer/backtest/runner.py`、`single_symbol_runner.py`、相关 CLI | 重写 | 拆开策略执行、指标、结果持久化；规则策略不经过 torch 模型加载；真实执行 baseline 替代价格比对照 |
| `app_layer/backend/server.py`、`agent_controller.py`、`evaluation_controller.py` | 重构 | 入口改为数据、策略、实验、结果；规则回测不强制导入 torch；独立策略实验取代训练配置与 checkpoint 导出流程 |
| `app_layer/backend/jobs.py` | 有条件复用 | 可用于短任务调度；实验状态应落盘，重启后识别中断任务，不把内存 job 状态当唯一事实 |
| `app_layer/frontend/`、`visualization_layer/` | 保留基础并改版 | 保留行情、净值、成交回放；增加策略参数、对照结果、批次和失败原因；训练曲线和 GPU 面板后续移除 |
| `tests/` | 保留并迁移 | 当前动作、执行约束和缓存测试不能随目录一并删；保留仍适用的行为测试，补研究与回测关键路径 |
| `requirements.txt`、`pyproject.toml` | 重写依赖组织 | 基础安装只包含研究与回测依赖；过渡期训练依赖单列可选文件，基础启动和测试不需要 torch／CUDA |
| `agent_layer/models/`、`training/`、`rewards/` | 后续可归档删除 | 新研究闭环通过且所有活跃 imports 退出后删除；如仍保留可选模型插件，应明确维护范围 |
| worker、validator、DDP launcher、训练专用 Dashboard 与 cache CLI | 后续可删除 | 对应 UI、路由、包导入和调用点全部迁移，通用数据读取能力已抽出 |
| 两个根目录 `_tmp_*training*`／`_tmp_run_local_subprocess_smoke.py` 脚本 | 审查后可删除 | 有效验收逻辑转入测试后删除；不能仅根据临时命名判断无价值 |
| 空 YAML 和纯占位包 | 后续可删除 | 新配置入口建立，确认无实际 imports、说明或工具依赖后清理 |
| `.pytest_cache/`、`.ruff_cache/`、`__pycache__/` | 可再生成 | 属于可清理缓存，不与模型、数据和研究结果混在同一删除动作中 |
| `runtime_layer/`、`references/` | 按资产保留 | 行情、特征、数据包、论文继续保留；旧 checkpoint／训练日志单独列清单后处理，不因代码转型整目录清空 |

### 必须拆开的启动依赖

`server.py` 当前无条件导入 `agent_controller`，后者导入 torch 与训练组件；另一条路径是 `server.py → evaluation_controller → evaluation_layer.backtest.runner → torch`。`evaluation_layer/backtest/__init__.py` 和 `agent_layer/experiments/__init__.py` 的提前导入也会间接加载模型。

因此，新增无 torch 的函数还不够；包入口、controller 注册、CLI 和依赖安装都必须一起分离。基础环境中不安装 torch 也能打开 Dashboard、运行规则策略和读取结果，才算完成解耦。

## 目标模块与接口

目标结构保留现有分层。`strategy_layer`、`backtest_layer` 和 `research_layer` 已覆盖 P1／P2：规则策略、共享执行、有限参数批次、滚动时间验证、输入快照与结果持久化。业务逻辑不放在 controller 中。

```text
download_layer       下载与来源
data_layer           存储 数据质量 历史股票样本
feature_layer        时点正确的指标与因子
strategy_layer       策略注册 参数 状态 信号
backtest_layer       行情快照 统一账户 执行 净值与事件
research_layer       实验配置 参数批次 时间切分 试验记录
evaluation_layer     指标 baseline 比较 FIFO 与稳健性报告
visualization_layer  行情 净值 成交展示
app_layer            CLI 与 Dashboard 编排
runtime_layer        可追溯的运行产物
```

建议接口如下，名称可在实施时按现有风格调整，职责必须保持：

```text
Strategy.initialize(config)
Strategy.on_bar(visible_market, account_view) -> OrderIntent[]
BacktestEngine.run(strategy, dataset, execution_config) -> BacktestResult
ExperimentRunner.run(experiment_manifest) -> ExperimentResult
```

策略只读取当时可见的数据和账户视图，不能直接修改账户。执行器统一处理委托、部分成交、拒单、费用和持仓变化。规则策略、baseline 和后续模型共用这一执行器。

`OrderIntent` 显式声明数量语义，例如股数或目标权重；目标权重须通过统一的订单构建器换算，不能填进现有买卖比例字段。信号时间、委托时间、成交时间、估值时间分别记录。

首版日线约定为收盘后生成信号、下一可交易时点执行。若模拟下一日开盘成交，不能把该日最终成交量、高低价作为开盘前已知信息。日线无法确定的盘中路径采用明确的保守规则或要求更细粒度数据。

数据区分特征价格、成交价格和公司行为处理口径；费用、市场规则与数据修订均带版本。策略不能根据回测结束后才知道的信息选择股票或构造特征。

## 必须先修复的正确性问题

| 优先级 | 问题与当前定位 | 验收场景 |
| --- | --- | --- |
| P1 | 净值时间错位：`evaluation_layer/backtest/runner.py:100`、`:249` 与 `agent_layer/environment/trading_env.py:107` | 前日收盘 10000、次日开盘 11000、次日收盘 10500；前日日终仍为 10000；CLI、验证、回放口径一致 |
| P1 | monthly 历史 cutoff 未排除当前月：`feature_layer/datasets/compact.py:1671` | 已有当月最终月 K 时，月末 09:35 不得看到其最终 close；月中不完整月 K 同样按可用时点处理；当前周期 snapshot 不能重复纳入 |
| P1 | 导入先删除旧数据后校验：`data_layer/storage/portable_bundle.py:450` | 损坏包、取消、磁盘失败均保留原数据；先完整校验和 staging，再切换；切换失败可恢复 |
| P2 | 日内聚合按行号分桶：`feature_layer/builders/aggregation.py:167` 与 `materialize.py:1049` | 缺 09:40 时 15 分钟窗口仍按交易时钟划分；缺口可见，不跨午休错误合并；Python 与 SQL 路径一致 |
| P2 | Buy-and-Hold 仅为首尾 close 比：`evaluation_layer/backtest/runner.py:581` | 与候选策略共用费用、资金、可成交时间及约束；价格收益另列参考，不冒充可执行 baseline |
| P2 | GUI 20% 仓位默认与单股执行 100% 默认不一致：`app_layer/backend/evaluation_controller.py:325` | UI、CLI 使用同一显式 execution 配置；任何覆盖值写入实验清单；同配置得到同结果 |

这些验收用小型合成行情先建立可解释的预期结果，再用实际数据检查。语法检查不能代替这些行为验证。

## old 的吸收清单

旧主线提交为 `0ad1f60d71840cac2778886472fce4d5bf3d2b84`。历史源码存于外部完整归档，不恢复到当前运行路径。原验收说明副本与本轮修订见 [旧项目吸收与归档记录](legacy/PocketAgent_old_review.zh-CN.md)。

| 顺序 | 来源 | 吸收内容 | 重写时必须修正 |
| --- | --- | --- | --- |
| 1 | `scripts/rl/evaluate.py` | 同一个环境执行 model、cash、buy-hold、random，输出逐股票和汇总对照 | old buy-hold 只发一次 BUY_MEDIUM；新版本明确仓位、不可成交时的处理与期末估值；random 固定 seed 并限定可行动作 |
| 2 | `src/backtest/metrics.py` | FIFO 买卖匹配、费用分摊、已实现盈亏、持仓时间和成交质量 | 区分匹配片段、卖单、完整开平仓三种计数；处理期初持仓、同时间戳顺序、超量卖出与期末未平仓 |
| 3 | `src/features/market_features.py` | 个股相对指数／行业收益、市场成交与趋势上下文 | 建立历史时点行业归属与数据可用时间；旧同时间戳 left join 不足以保证正确；20 根 K 线不能标成 20 日 |
| 4 | `src/agent/risk_manager.py` | 原始信号、约束原因、最终执行的可解释记录 | 只借鉴结构，旧止损和仓位阈值不直接继承；执行约束与策略逻辑分别记录 |

FIFO 用于研究报表，当前账户平均成本语义不随之静默改变。净值指标继续以真实日终 NAV 为输入，不迁回旧日内 bar 年化公式。旧 SB3 模型、独立账户、下载存储和 Dashboard 均不回迁。

## 研究流程与计算预算

每个实验固定数据快照、历史股票样本、策略版本、参数、执行配置、时间切分、种子及全部候选清单。配置生成稳定 ID；成功、失败、被淘汰和重复运行都保留，不能只记录最佳结果。

按时间划分研究、验证和最终测试区间。滚动验证中的参数选择仅使用该窗口过去的数据；最终保留区间不参与反复调参。未来收益标签跨边界时，按其实际时间跨度剔除重叠样本并设置间隔，而不是对所有实验使用任意固定行数。多股票同日数据须在同一时间侧。

研究报告同时展示样本数量、成交数量、不同年份和股票的分布、成本敏感性、相邻参数稳定性及对少数极端交易的依赖。逐股统计与组合统计分开。DSR／PBO 只有在试验数量、候选相关性和输入收益矩阵定义清楚后才解释，不能当作通过即有效的证书。

先用缓存后的指标表和向量化计算做粗筛，再用统一执行器详细重放入选候选。粗筛结果明确标为近似结果；不能绕过可成交性与成本直接进入最终排名。

笔记本先测小数据集的总耗时、峰值内存、读取量和每秒处理 bar 数，再确定批次大小。学校电脑按完整策略参数／验证窗口分工，不把同一账户的一条时间序列拆给不同机器后简单拼接收益。

离线任务包包含代码版本与补丁哈希、数据 manifest、任务 ID、完整参数和预期输出 schema。每台电脑独立运行，结果包带校验和；汇总时拒绝数据版本不一致的结果并去重。所有机器的候选都计入同一试验登记册，避免各机器只提交赢家而低估搜索次数。

## 分阶段交付与删除门槛

| 阶段 | 交付 | 完成条件 |
| --- | --- | --- |
| P0 数据与评估基础 | 修复上述正确性问题，确认依赖和保留资产 | 合成边界测试通过；导入失败不破坏旧数据；真实日终净值正确 |
| P1 CPU 单股闭环 | 通用账户／执行器、策略接口、一个规则策略、cash 和 buy-hold、基础报告 | 无 torch 环境下 CLI 和 Dashboard 可启动并完成回测；重复运行的成交与指标一致 |
| P2 有限策略挖掘 | 2 至 3 个策略家族、少量参数、多窗口、完整试验登记、FIFO、逐股对照 | 全部试验可重现；费用／拒单／未平仓可追踪；最终测试与调参隔离 |
| P3 离线批次与清理 | 任务包／结果包、失败重试、去重汇总；退出训练 UI 与依赖 | 断网机器结果可验证导入；活跃路径不再依赖训练组件；迁移测试通过后删除训练专属源码 |
| 后续扩展 | 市场／行业特征、真实组合、可选轻量 ML | 各自增加明确的数据和执行契约；不把逐股平均替代组合回测 |

训练源码清理必须同时检查 imports、包 `__init__`、CLI、路由、前端控件、依赖文件和文档。对当前未提交工作先记录和保存；运行数据、模型、论文与实验结果另立清单，不能夹带在源码清理中。

## 本轮归档与验证

`PocketAgent_old` 归档位置为 `D:\project\backups\PocketAgent_old-20261107\PocketAgent_old.zip`。它完整保存删除前剩余的 `.git` 与未跟踪的 `LEGACY_VALUE.md`，包含未被分支／标签引用的对象；不是只打包可达提交的 Git bundle。

逐文件 SHA-256、ZIP 完整性、完整解压、Git HEAD／全部 refs 和对象检查均已核对。删除状态与校验值以 [归档记录](legacy/PocketAgent_old_archive.json) 为准。原目录此前已无旧源码工作树、虚拟环境和独立数据文件；本轮归档不声称恢复此前已删除且未进入 Git 的资产。

后续 P0／P1/P2 实施使用独立 `.venv`，未安装 PyTorch；已完成 212 项 CPU／回归测试，跳过 2 个需要模型／张量依赖的测试项，另有 7 条 pandas／NumPy 弃用警告。新增的 P3 与后续扩展继续使用同一 CPU 环境；组合、上下文、轻量模型、离线任务与稳健性诊断各自有针对性测试。Dashboard 真实点击、API 持久化与恢复、本地日线只读回放均已验证。既有数据与未提交源码的保护记录保存在 `D:\project\backups\PocketAgent_complete-20261107`，具体使用和限制见完整研究工作流说明。
