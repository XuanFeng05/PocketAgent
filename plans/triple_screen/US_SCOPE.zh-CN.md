# 美股100股替代实验

登记日期：2026-11-13。用户授权以100只非仙股美股替代受连接限制的A股数据，继续三重滤网工程验证。固定候选见 `config/universe/us_research100.txt`，不修改原A股100股清单。

- 候选为100家大型、活跃公司的美股主板股票；每股下载时核对USD、股票类型、NYSE/Nasdaq挂牌、元数据最新价至少10美元，以及请求末段60交易日的成交额中位数至少2500万美元。若失败，记录原因，不静默替换。筛选基于当前可见事实，属于有存活样本偏差的探索性工程样本，不称历史全市场或未见测试。
- 来源为Yahoo Finance公开chart接口，免费访问不等于数据供应SLA或商业再分发授权。原始JSON、拆股/分红事件、供应商OHLC、AdjClose/Close调整因子、参数、获取时间与哈希均保留。不开启自动价格修补，不补行情缺口或成交量，不混用其他数据源。
- 时间为2020-01-01至2026-07-12，日线实际目标末日2026-07-10；100周预热；主要评价2022–2025，2026先检查质量。输出 `runtime_layer/data_us`，与A股及合成fixture分开。
- 价格字段为Yahoo拆股/分红调整OHLC，存储层通用adjust=pre不表示BaoStock前复权版本。成交量保留供应商股数，不乘分红价格因子；不声称与真实券商资金账户等价。
- 日历来自固定版本exchange_calendars的XNYS，使用纽约本地日期和真实常规/提前收盘时刻，含2025-01-09等特殊休市；按日历核对每股日期，保留所有自然日和关闭日事实。
- 独立规则版本 `triple_screen_us_pre_v1`：指标、100周预热、做多/空仓、初始止损、成本0/5/10/20bps保持一致；取消A股当天买入不能卖出的锁定。入场当天触及止损，允许当日退出；盘中先后不明时保守视为入场在前、止损在后，并标记歧义。开盘入场后低点触及止损则无需伪造顺序歧义。
- 此版本是标准化技术净值，不模拟券商保证金、现金结算、账户日内交易限制或真实股数；不得称实盘执行复现。原 `triple_screen_pre_v1` 的A股T+1语义保留。
- 下载成功、CLI成功、HTTP成功、前端交互通过分别记账。原Dashboard八项缺口不能因换成美股而自动关闭。

来源依据（读取于2026-11-13）：

- [Yahoo调整收盘价说明](https://help.yahoo.com/kb/SLN28256.html)：拆股和分红调整。供应商Close已含拆股调整，不能把它误称完整原始成交价。
- [yfinance项目说明](https://github.com/ranaroussi/yfinance)：同类研究工具与Yahoo无隶属关系，数据使用遵循Yahoo条款。本实现直接保存chart响应，不依赖yfinance自动修复。
- [NYSE交易日历](https://www.nyse.com/markets/hours-calendars)与[Nasdaq交易日历](https://beta.nasdaqtrader.com/Trader.aspx?id=Calendar)：交易日、节假日与提前收盘的参照。
- [exchange_calendars XNYS实现](https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar_xnys.py)：云端使用4.13.2，另检查2025-01-09特殊休市。

“非仙股”以筛选时实际报价及主板、流动性门槛核对，不把多年拆股后较低的历史调整价误当当年的仙股报价。固定清单形成于查看回测收益之前；它仍有当前存活样本选择偏差。
