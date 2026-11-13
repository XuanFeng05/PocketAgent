# PocketAgent_old 遗留项目验收记录

## 1. 验收结论

`PocketAgent_old` 是 PocketAgent 在当前分层架构之前的早期实现。它完成过数据下载、特征构建、规则风控、单股交易环境、Stable-Baselines3 PPO 训练、回测评估和独立 Dashboard，但这些实现已经不再适合作为当前项目的运行代码。

本次验收后的处理结论如下：

- 不保留旧工作树源码、旧虚拟环境、旧行情数据、模型、训练结果和缓存。
- 保留 `.git`，因为其中保存着早期提交历史和 Phase 0、Phase 1 标签。
- 保留本文件，记录旧项目中仍值得借鉴的设计，以及后续从 Git 历史提取代码时必须注意的问题。
- 旧代码仅作为历史参考，不应重新并入当前训练主线。

当前正式项目是相邻目录 `PocketAgent`。它使用 `download_layer`、`data_layer`、`feature_layer`、`agent_layer`、`evaluation_layer`、`visualization_layer` 和 `app_layer` 的分层结构，并已经替代本旧项目的主要职责。

## 2. 旧项目技术路线

旧项目采用传统 `src/` 结构，主要模块包括：

- `src/data`：BaoStock 数据下载、CSV 落盘、DuckDB 存储、覆盖率与一致性检查。
- `src/features`：K 线、日线、大盘、行业、持仓和规则状态特征。
- `src/agent`：规则化市场状态、个股状态、做 T 状态机、持仓管理和风险管理。
- `src/rl`：离散动作交易环境、Gymnasium 适配器、Stable-Baselines3 PPO 和训练日志。
- `src/backtest`：账户、订单、成交、费用、滑点、回测引擎和绩效指标。
- `src/app/rl_dashboard`：旧训练与评估 Dashboard。
- `scripts/rl`：单股训练、universe 轮训、PPO 与 baseline 对照评估。
- `scripts/data`：旧数据下载、覆盖检查与健康检查。

旧训练方式并不是当前 MAPPO 方案。它让同一个 Stable-Baselines3 PPO 模型依次在多个股票上训练，但每次环境仍只加载一只股票，动作是有限个离散买卖档位。当前项目使用方向头和连续仓位比例头、多频率序列编码、独立 Agent Cache、统一执行环境以及可扩展的分布式训练，因此两套训练代码的模型结构、输入合同、动作空间和 checkpoint 格式均不兼容。

## 3. 仍有参考价值的设计

### 3.1 大盘与行业相对强弱特征

历史文件：

- `src/features/market_features.py`
- `src/features/feature_builder.py`

旧实现定义过以下市场上下文：

- `index_ret_30m`：指数短周期收益。
- `index_ret_1d`：指数日收益。
- `index_ma20_distance`：指数相对 20 日均线的位置。
- `market_amount_ratio`：市场成交额相对历史水平。
- `industry_ret_1d`、`industry_ret_5d`：行业短期收益。
- `stock_minus_index_ret`：个股相对指数的超额收益。
- `stock_minus_industry_ret`：个股相对行业的超额收益。

这些字段目前没有出现在正式项目的 `feature_v1` 中。它们与“单个 episode 只运行一只股票”并不冲突，因为指数和行业数据可以作为该股票的外部市场上下文，而不是联合持仓或多股票联合动作。

后续真正采用前必须重新实现，不能直接复制旧代码，原因包括：

1. 必须使用决策时刻已经可见的数据，严格避免未来函数。
2. 行业归属必须按历史时点处理，不能用今天的行业映射覆盖历史。
3. 指数和行业 K 线必须加入正式数据源、覆盖检查和 preflight。
4. 必须定义停牌、缺失 K 线和不同频率的对齐规则。
5. 特征必须进入当前 Feature Dataset、Agent Cache 和模型输入注册表，而不是在训练环境中临时计算。

这部分是旧项目最值得保留的特征层思路，但优先级应低于当前 Agent 训练链路稳定性。

### 3.2 成交级评估指标

历史文件：

- `src/backtest/metrics.py`

旧回测层包含当前评估层仍可参考的成交统计：

- 按 FIFO 方式把买卖成交重建为已平仓交易。
- 成交数、拒单数、买入数和卖出数。
- 总买入额、总卖出额、总换手和总费用。
- 已平仓交易数量、胜率、平均单笔盈亏和平均单笔收益。
- 最佳与最差单笔收益。
- 平均盈利与平均亏损的盈亏比。
- 最大回撤的峰值时间和谷值时间。

这些指标适合补充当前 `evaluation_layer` 的交易质量报告，但不能直接搬运旧权益指标代码。旧代码使用日内 bar 数量进行年化，例如按 `244 * 8` 估算 30 分钟周期；当前项目已经明确要求从日终 NAV 计算 Sharpe、Sortino、波动率、Calmar 和最大回撤，避免把 5 分钟 bar 当成独立交易日导致指标膨胀。

建议未来只迁移“成交重建和成交级统计”的思想，组合收益指标继续使用当前评估层口径，并补充以下测试：

- 多次分批买入后分批卖出的 FIFO 成本。
- 手续费、印花税和过户费对已实现盈亏的影响。
- episode 结束时尚未平仓仓位的处理。
- 无成交、只有买入、部分卖出和跨日卖出的边界情况。

### 3.3 PPO、Hold、Random 的统一对照流程

历史文件：

- `scripts/rl/evaluate.py`
- `scripts/rl/evaluate_rl_ppo_vs_baselines.py`
- `scripts/rl/run_rl_env_baseline.py`

旧项目曾把 PPO、始终持有和随机动作放在同一评估周期中，并输出 comparison、equity curve、trades 和 sanity report。这种“任何模型都必须与简单 baseline 在同一数据、同一费用和同一执行规则下比较”的原则仍然有效。

当前评估层后续应保留以下思想：

- 模型、Buy-and-Hold、空仓和受约束随机策略共用同一个执行环境。
- 所有策略使用相同的初始资金、手续费、滑点、涨跌停和 T+1 规则。
- 输出逐股票结果和跨股票汇总，不能只看总体平均值。
- 对收益、回撤、交易数和 NAV 有限性进行 sanity check。
- 明确区分训练期、验证期、测试期，禁止评估数据进入训练。

旧脚本依赖 Stable-Baselines3 模型、旧离散动作空间和旧 DataFrame 环境，不能直接读取当前 MAPPO checkpoint，因此只保留评估流程思路。

### 3.4 可解释风控覆盖层

历史文件：

- `src/agent/risk_manager.py`
- `src/agent/t_state_machine.py`
- `src/agent/stock_state.py`
- `src/agent/market_state.py`

旧规则层定义过市场状态、个股状态、风险等级和做 T 状态机，例如：

- 市场状态：禁止交易、防守、震荡、可交易、强进攻。
- 个股状态：强趋势、趋势回调、突破尝试、突破确认、假突破风险、高位加速、冲高回落和破位。
- 风险等级：低、中、高、极端。
- 做 T 状态：正常持仓、卖出候选、已卖等待买回、买回候选、当日完成和止损模式。

这些规则使用人工阈值，不能作为当前模型 reward，也不应强行覆盖训练动作，否则模型会学习到人为策略而不是从数据中学习。但未来部署推理服务时，可以参考其结构建立独立于模型的安全层：

- 对法规或账户硬约束直接屏蔽动作。
- 对高风险状态输出警告和解释，不随意增加 reward punishment。
- 记录模型原始动作、过滤原因和最终执行建议，便于人工审核。
- 风控参数必须配置化并通过历史回测验证。

其中“可解释状态输出”有价值，旧阈值本身没有经过充分统计验证，不应继承。

### 3.5 持仓与交易规则建模

旧项目已经意识到以下 A 股执行问题：

- 100 股一手的整数约束。
- T+1 下总持仓与可卖持仓的区别。
- 手续费、最低佣金、印花税和滑点。
- 涨跌停、无成交量和无有效价格时的拒单。
- invalid action 只造成账户不变，不应默认附加惩罚。

这些原则已经由当前 `agent_layer.environment` 和 `agent_layer.portfolio` 更完整地实现。旧代码在这一部分没有需要回迁的实现，只具有项目演进记录价值。

## 4. 不再有迁移价值的部分

### 4.1 Stable-Baselines3 PPO 主线

旧模型使用 MLP PPO 和离散动作，不能利用当前从 5 分钟到月线的多频率输入，也不支持当前方向头与仓位比例头。旧 checkpoint 与当前模型完全不兼容，继续维护只会形成第二套训练口径。

### 4.2 旧交易环境与回测账户

当前训练和评估必须共用同一套 A 股执行逻辑。旧 `TradingEnv`、旧 backtest account 和旧 order builder 若重新启用，会造成训练、验证和回测成交规则不一致，因此不再保留工作树代码。

### 4.3 BaoStock 专用数据路径

旧数据层主要支持 daily 和 30min，依赖 CSV 加单一 DuckDB。当前项目已有更完整的数据下载层、分区存储、portable bundle、全频率覆盖检查和 Feature Dataset 构建流程。旧下载与存储脚本没有继续维护价值。

### 4.4 旧 Dashboard

旧 Dashboard 把训练控制、数据下载、评估、日志解析和静态页面集中在一个大型 Python 文件中。当前 `app_layer` 已按 controller、jobs、server 和 frontend 分离，并覆盖 Cache、preflight、分布式训练和评估入口。旧 Dashboard 不应恢复。

### 4.5 旧模型与实验结果

现存旧训练摘要显示，其代表性 universe 实验大致为：

- 候选 universe：99 只股票。
- 实际训练：25 只股票。
- 每只股票约 2,000 steps。
- 总目标约 50,000 steps。
- 主要使用 2021-2023 年 daily 数据。
- 使用 Stable-Baselines3 PPO。

这只能证明旧训练流程曾经运行，不能证明策略有效。缺少可靠的 walk-forward、独立测试期、多 seed、过拟合概率和稳定盈利证据，因此旧模型和 checkpoint 不具有研究结论价值。

## 5. 旧代码质量与直接复用风险

验收时发现：

- 旧项目没有正式测试用例。
- Python 文件可以静态解析，但缺少行为回归保障。
- 部分旧指标的年化口径不符合当前约定。
- 旧动作空间、环境接口、数据结构和 checkpoint 格式均已变化。
- 旧 README 和代码注释虽记录了设计意图，但不能替代测试和数据契约。
- 部分未提交的旧 Dashboard 与下载控制器调整没有保留；这些内容已被当前分层实现替代。

因此，从 Git 历史提取任何代码时，都应先把它当作伪代码或需求说明，再按当前模块边界重写并添加测试，而不是复制文件。

## 6. Git 历史保留说明

`.git` 中保留 11 个早期提交，旧主线 HEAD 为：

```text
0ad1f60 Fix RL universe training and evaluation pipeline
```

保留的阶段标签包括：

- `phase-0`
- `phase-1-data-core`
- `phase-1-watchlist-collector`
- `phase-1-inventory`
- `phase-1-data-availability`
- `phase-1-data-consistency`
- `phase-1-data-health-check`
- `phase-1-readme`

在不恢复整个工作树的情况下，可以使用以下命令查看历史文件：

```powershell
git -C D:\project\on-going\PocketAgent_old show 0ad1f60:src/features/market_features.py
git -C D:\project\on-going\PocketAgent_old show 0ad1f60:src/backtest/metrics.py
git -C D:\project\on-going\PocketAgent_old log --oneline --decorate --all
git -C D:\project\on-going\PocketAgent_old tag --list
```

如需临时检出整个旧版本，应检出到新的临时目录，不要重新把它混入当前 `PocketAgent`：

```powershell
git -C D:\project\on-going\PocketAgent_old worktree add D:\project\on-going\PocketAgent_legacy_review 0ad1f60
```

审阅完成后再移除临时 worktree。

## 7. 本次清理记录

清理日期：2026-07-17。

清理前目录约 1.186 GB，共约 30,139 个文件。主要空间占用为：

- `.venv`：约 971 MB。
- `data`：约 135 MB。
- `.git`：约 56 MB。
- `runs`：约 18.8 MB。
- `models`：约 3.1 MB。
- 其余日志、universe 归档和 Python 缓存。

已删除：

- `.venv`
- `.pytest_cache`
- `data`
- `logs`
- `models`
- `runs`
- `universe`
- `config`、`scripts`、`src`、`tests` 工作树目录
- 根目录旧入口、依赖和项目配置文件
- `__pycache__`、`.pyc` 和 `.pyo`

清理后仅保留：

- `.git`：旧提交、分支和标签历史。
- `LEGACY_VALUE.md`：本验收说明。

预计释放约 1.13 GB。本目录不再具备直接启动能力，这是有意的归档状态。

## 8. 后续参考优先级

未来需要补充当前项目时，建议按以下顺序使用旧历史：

1. **评估层成交统计**：参考 FIFO 平仓交易和成交级指标，按当前数据结构重写。
2. **评估 baseline**：让 MAPPO、Buy-and-Hold、空仓和受约束随机策略共用当前执行环境。
3. **大盘与行业上下文**：先建立历史时点正确的数据合同和 preflight，再新增特征。
4. **推理安全层**：参考旧风险状态输出结构，但重新校准所有阈值。
5. **不恢复旧训练主线**：Stable-Baselines3 PPO、旧环境和旧 Dashboard 不再投入维护。

## 9. 最终判定

`PocketAgent_old` 的运行资产和工作树代码可以删除。其剩余价值已经由本文件提炼，具体实现仍可从 `.git` 历史按需查看。

旧项目最值得继承的是三个原则，而不是旧代码本身：

1. 单股动作也应观察大盘和行业环境。
2. 模型必须与简单 baseline 在完全一致的执行规则下比较。
3. 训练策略与实盘安全护栏应解耦，硬约束负责屏蔽，不能随意通过 reward punishment 代替。
