# 本地100股阶段的集成验收

日期：2026-11-12。本记录是本地已实现范围验收，不是历史全A股目标完成声明。用户明确只有这100股，缺失数据以后再补，并禁止自主下载。

| 要求 | 实际行为与证据 | 结果 |
|---|---|---|
| 固定规则与价格口径 | SPEC、triple_screen_v1.json、源输入adjust=pre；完整v1保持冻结 | 本地完成 |
| 周/日时点与预热 | 原始周MACD、EFI2、100完成周，未来扰动/节假日/ns-us精度测试 | 完成 |
| 触发与退出状态 | 前日高点严格突破、候选过期、固定stop、跳空、T+1歧义锁存、停牌延迟、无当日重入 | 完成 |
| 逐股四成本回放 | triple_20261112_101445_a783cd4e，400/400任务，1704工件哈希 | 本地完成 |
| 固定规则独立年度窗 | validation_20261112_103518_e8b55a31，四窗400/400，顶层+4子run共2023哈希 | 本地完成，不是参数选优walk-forward |
| 参数邻域/滤网对照 | study_20261112_105916_14b30718，1000/1000任务，全部配对，5063哈希 | 本地完成，不选择赢家 |
| 原版行为不变 | primary_equivalence.json：100股输入、指标、净值和fills与前批完全一致 | 完成 |
| 分片及恢复 | 新triple_shards.py；205股mock拆100/100/5；真实3股×2成本分两片，恢复6任务全复用，43哈希 | 能力完成，未证明真实全市场容量 |
| 使用入口和报告 | CLI、独立离线HTML、股票/成本筛选与明细展开实测；页面选择100股、启动/轮询/历史/报告 | 完成本地页面链路；原通用研究页保持不变 |
| 本地数据保护 | 100股200个pre/none文件前后SHA/大小/mtime相同，原有结果不覆盖 | 完成 |
| 历史全市场及退市覆盖 | 当前仅100股；用户以后补数据 | 待数据，未完成 |
| 复权版本权威核验 | 2020–2025未触发所定义异常；2026年16股反跳原因未知 | 待材料，未完成 |
| 真正未见新日期测试 | 2026-07-13起数据本地缺失；未运行 | 待数据，未完成 |

## 验证

最终核心测试：466 passed、2 skipped、7 warnings。页面及既有 Dashboard 集成测试12项通过；`node --check app_layer/frontend/triple.js`通过。跳过项属于可选torch；警告来自原有NumPy/pandas timedelta弃用。日志：runtime_layer/reports/triple_screen/m6_final_tests_20261112.log。

核心/编排独立review没有发现阻断本地阶段的新问题；修复了日期精度、平价候选误失败、未使用amount误拒、恢复身份、成本ID碰撞、累积checkpoint磁盘增长、报告布局与消融标签问题。没有修改用户原始行情、提交或推送Git。

## 本地入口（不会下载）

在项目根运行：

```powershell
.\.venv\Scripts\python.exe -B -m research_layer.cli.triple_screen --config config/research/triple_screen_v1.json
.\.venv\Scripts\python.exe -B -m research_layer.triple_validation --config plans/triple_screen/local100_annual_config.json --windows plans/triple_screen/local100_annual_windows.json
.\.venv\Scripts\python.exe -B -m research_layer.triple_study --config plans/triple_screen/m6_study_config.json --cases plans/triple_screen/m6_study_cases.json
.\.venv\Scripts\python.exe -B -m research_layer.triple_shards --config runtime_layer/reports/triple_screen/local100_request_20261112.json --shard-size 100
```

启动本地 Dashboard 后访问 `/triple.html`。页面只读取本地 `runtime_layer/data/market_parts/*/daily_pre.parquet`，显示当前100股；启动的任务记录在 `runtime_layer/reports/triple_dashboard`，生成的离线报告由本地 `/api/triple/runs/<job_id>/report` 提供。

这些命令创建新的本地实验，不覆盖旧结果。无需为了查看结果重新运行；直接打开已有report.html或summary.json。分片主路径为按股票存储的market；CSV入口64MiB限制属于资源保护，不等同所有格式的全市场内存验证。

## 结论边界

同期限98股、2022–2024、单边10bps下，原版平均累计收益约+0.42%，中位数-8.82%；这证明研究链路能比较规则，不证明稳健盈利。邻域/消融全部结果见M6_RESULTS.zh-CN.md。全市场和新日期数据到位后再继续原目标，不能自行下载或把当前100股替代全市场。
