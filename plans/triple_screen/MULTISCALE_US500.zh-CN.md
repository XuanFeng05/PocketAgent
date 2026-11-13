# 500股多周期信号与组合研究

本轮在`ed99ec1`研究增量上扩展，保留原100只美股，增加400只；独立数据、配置和运行目录，不覆盖原始Elder实验。用户已明确授权扩大股票范围。新增股票来自公开iShares IWB持仓快照，按基金持仓规模及价格/流动性规则选取，不能按策略收益替换成员。此样本存在当前成分、存续和后验流动性筛选偏差，不能声称历史时点可投资股票池。

## 先登记、后观察

`config/research/multiscale_us500_v1.json`在查看本轮收益前固定指标、阈值、持有期、选择方式、费用和统计方法。10个日线信号加全部45个两两AND，共55种入场条件；与7种周线过滤、4种退出交叉，合计1540组。包括无周过滤对照，不能预设周线过滤必然改善结果。EFI、RSI、Stochastic、Williams %R、Bollinger、EMA、Donchian和MACD代表回调、均值回归、趋势、动量和突破等公开技术方法；并不意味着这些交叉规则都是市场公认策略。

## 单信号与条件研究

信号在收盘形成，下一交易日开盘计入，分别在第1、3、5、10、20、40个交易日收盘计出，买入当日算第1天。事件表包括净胜率、平均及中位净收益、盈亏比、尾部损失、最大有利/不利波动，以及与同日期股票池、相同条件股票池的差值。跨评价期末缺少退出日期的事件公开排除，不借用未来评价区间。

同一指标另在已完成周线上计算，再比较相同日数的后续表现；周信号只在新完整周发布时触发。日内/跨股票/连续日期事件会重叠，事件数量不是独立样本量，事件收益也不是可交易账户的收益。先判断信号是否超过同日期基准，再看周趋势、波动、成交量等条件是否改变表现。提前卖出提高胜率不能独自证明信号衰减，也可能截断大盈利单。

## 交易回测

各段都从新账户开始：2022–2023训练，训练后冻结候选，2024–2025回顾性验证。两段都使用2020以来历史预热；新上市股票不补造历史，100个完整周尚未达到时保留现金。2026行情只用于质量检查与已声明的股票池筛选。所有年份已经观察过，不称为全新样本外。

每只股票独立账户初始资金相等，单次计划风险为该账户权益1%，初始止损为买入价下方2×前一日ATR14，无杠杆。次日开盘入场，费用单边10bps。退出为持有5/10/20日（另有保护止损），或3ATR追踪并在周过滤失效后次日开盘退出。新追踪止损次日生效，跳空按开盘成交，不能用事后低点修复亏损。同日退出后不重入。

主比较将官方French日RF作为前收盘闲置现金的计息代理，并单列利息贡献；这不是券商承诺收益。训练冻结的候选再做0/5/10/20bps成本及零现金利息敏感性。基准为同股票池等初始资金买入持有与RF现金；必须披露策略暴露较低、资金没有跨股票共享的区别。

## 选择与多重尝试

先满足至少1000笔已平仓、100只股票有已平仓交易的门槛。按RF超额均值/回撤选5个排名候选，另选收益最高、正超额且利润因子大于1的最高胜率候选，并固定保留4个事先指定的策略对照。排名靠前不自动等于盈利，未达到盈利门槛时明确记录没有候选，不降低门槛凑赢家。

验证期对全部1540个候选同步日期区块重采样，主区块20个交易日，40/60作稳健性；39999次，给出全候选族的最大均值检验及Holm调整。选择后的个体区间只作已知选择规则下的诊断。DSR仅在IID和假设有效尝试次数下计算，不当作过拟合概率。累计尝试至少为前轮72加本轮1540，先前未枚举探索仍需另计。

所有行情、持仓源快照、运行结果、缓存和图表保留`runtime_layer`；可交付的Git增量只含代码、配置、测试与方法文档。此研究尚不自动接入Dashboard，也不构成实盘验收。

## 复现入口

从项目根目录，在Python3.12独立虚拟环境中安装`requirements-research.txt`。本轮增加Numba的CPU执行加速，不需要PyTorch或CUDA。

```bash
python -m pip install -r requirements-research.txt
python -m research_layer.multiscale_data
python -m research_layer.multiscale_data --audit-only
python -m research_layer.cli.multiscale_search --stage all --workers 4
python -m research_layer.cli.multiscale_quality
```

交付清单已经固定500个代码，下载器默认按该清单下载，不再次根据最新持仓选股；`--only SYMBOL ...`仅重试清单内成员。首次扩展时尝试过但未入选的代码和原因留在云端`us500_expansion`，它们不属于最终500股回测范围。不得用遍历整个daily目录代替读取固定清单。

源数据质量另有重要边界：SW、FERG、TLN的Yahoo历史包含上市、跨市场或合并衔接问题。供应商零成交量只被标为不可交易，不能认证为真实停牌。`--diagnostics-only`可重建异常变化与一手上市史诊断；`multiscale_quality`按`us500_quality_sensitivity.json`把这三个账户整段改为RF现金，保留500分母，对全部1540组重算两段结果和全族统计，仍使用原训练冻结的选择。这是保守数据敏感性，不是修复后的真实价格，不在主/质量两组中事后挑更好的一组。该CLI还提供与100周/价格资格起点一致的买入持有对照；它仍没有匹配策略仓位暴露。

## 技术定义来源

- 指标计算定义：[Fidelity指标指南](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide)、[StockCharts Force Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/force-index)、[Price Channels](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/price-channels)。具体参数及触发规则在`multiscale_signals.SIGNAL_DEFINITIONS`中冻结。
- 策略背景：[Original Turtle Rules](https://www.tradingblox.com/originalturtles/originalturtlerules.pdf)讨论突破、波动风险定仓等组成部分；本轮通道研究没有实现完整海龟加仓/相关性/全部退出规则。
- 统计解释：[Bailey与López de Prado的DSR原文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。完整的区块检验与多重比较来源也保存在统计模块中。
- 股票目录来源：[iShares IWB官方页](https://www.ishares.com/us/products/239707/ishares-russell-1000-etf)。它是基金持仓快照，不是回测各历史时点的成员名单。
