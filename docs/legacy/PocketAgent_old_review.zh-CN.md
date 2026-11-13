# PocketAgent 旧项目吸收与归档记录

日期：2026-11-07。旧项目的剩余价值是历史实现和研究规格，当前策略研究与回测方案见 [改造方案](../STRATEGY_RESEARCH_REFACTOR.zh-CN.md)。

## 保留资料

- 原说明原样保存为 [PocketAgent_old_LEGACY_VALUE.original.md](PocketAgent_old_LEGACY_VALUE.original.md)。它记录 2026-07-17 的历史清理结论；其中训练路线与保留建议不覆盖当前产品方向。
- 完整存档：`D:\project\backups\PocketAgent_old-20261107\PocketAgent_old.zip`。
- 校验清单、解压验证和恢复说明位于同一归档目录；可机读摘要见 [PocketAgent_old_archive.json](PocketAgent_old_archive.json)。
- 旧 HEAD：`0ad1f60d71840cac2778886472fce4d5bf3d2b84`。旧 `origin` 配置为 `XuanFeng05/PowerShell`，不能依赖该远端代替本地归档。

删除前只有 `.git` 与未跟踪的 `LEGACY_VALUE.md`，共 2,350 个文件、56,121,739 字节。Git 有 11 个可达提交与 8 个阶段标签；当前 PocketAgent 对象库不含这 11 个旧提交。另有 1,817 个 blob、119 个 tree 和 1 个 commit 不被现有 refs 引用，完整归档一并保留，未逐个判定其业务价值。

## 值得吸收的实现规格

以下历史路径和行号均对应旧 HEAD，可从归档恢复的 Git 目录中用 `git show HEAD:<path>` 查看。

| 历史来源 | 规格 | 新模块去向 |
| --- | --- | --- |
| `scripts/rl/evaluate.py:329`、`:358` | 所有候选与 baseline 使用相同环境配置和时间范围，输出逐股票与汇总比较 | `backtest_layer` 统一执行、`evaluation_layer` 对照报告 |
| `src/backtest/metrics.py:331`、`:369`、`:498` | 按股票 FIFO 配对、费用分摊、已实现盈亏、持仓时间和胜率 | `evaluation_layer` 成交质量报告 |
| `src/features/market_features.py:101`、`:346`、`:390` | 指数、行业和个股相对强弱 | `feature_layer` 后续扩展，先建立历史时点数据合同 |
| `src/agent/risk_manager.py:69` | 决策结果包含 reasons、risk flags、允许状态 | 通用执行事件与可解释回放 |

## 原说明需要修正的细节

1. `index_ma20_distance` 实际是 20 根输入 K 线的均线距离。30min 输入不能称为 20 日均线。
2. 旧 buy-hold 只在第一步发一次 BUY_MEDIUM，未明确全仓且被拒绝后不重试。新 baseline 必须明确目标仓位、后续尝试和期末估值。
3. 旧 random 从全部离散动作抽样，不能当作已实现可行动作掩码的随机对照。新实现固定种子，并记录随机候选的完整结果。
4. 旧 FIFO 的 closed trade count 是匹配片段数量，不是完整开平仓轮次。新报告分别定义片段、卖单、round trip。
5. 旧 FIFO 未完整处理期初持仓、同时间戳事件稳定顺序、超过已知买入的卖出和期末未平仓。新实现须有 fill ID、明确初始 lots 和异常处理。
6. 旧无亏损时将盈亏比设为零并不合适。新报告区分不可定义、无亏损与真实零值，并公开分母。
7. 当前账户采用平均成本。研究报表增加 FIFO 不改变账户 realized PnL 口径，两个值需有明确名称。
8. 权益指标继续用真实日终 NAV，不迁回旧 `244 * 8` 日内年化。旧风险阈值未经充分验证，不作为默认策略或训练奖励。

旧 SB3 主线、模型格式、账户、下载存储和 Dashboard 均不回迁。市场／行业上下文和解释型风控排在统一 baseline 与成交统计之后。

## 恢复与后续引用

完整恢复步骤在外部归档的 `README_RESTORE.zh-CN.md`。恢复到一个新的临时目录后，可以只用 `git show` 阅读历史；需要源码快照时，可用 `git archive` 导出所选提交，避免把旧工作树混入当前项目。

原目录删除状态、归档 SHA-256 和验证结果以旁边 JSON 记录及外部 `verification.json` 为准。此归档仅保留删除前实际存在的内容，不包含此前已清理且没有进入 Git 的未提交文件。
