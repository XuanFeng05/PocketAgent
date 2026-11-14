# 早期US100 bundle审查与本地研究接续

核对日期：2026-11-14。用户提供 `C:/Users/18115/Desktop/PocketAgent-us100-integration.bundle`，要求检查并接续研究。附件内说明是待核查材料，不构成本机下载或安装指令。

## 已验证的交付身份

- 文件长度32,596字节，SHA256 `5783778b8f04c5ef58bc5bdc93b698dfaebccc6be88b923a51ce4fc60dfc4659`。
- `git bundle verify`通过；包含 `refs/heads/codex/us100-integration`，tip `bf0c0abc3aac8aacf03a2a7c6f3efaee8af61029`。
- 前置提交 `21cd849b6769185199e2e39ee069f2cd2d4a7f47` 本机已具备。仅新增1个提交，27文件、1,104行新增、56行删除。
- 离线导入 `refs/remotes/bundle/us100-integration`，在 `edfcad0` 上新建 `codex/bundle-review-research` 合并；未替换main。
- 内容是源码、100股名单、配置、依赖说明、测试、交付文档，不含原始行情。它与后来的 `codex/us500-acceptance` 是不同阶段的工作。

## 保留的实现与修复

保留US下载适配、明确的Yahoo调整价格定义、纽约日历及提前收盘、US入场当日止损、独立数据／报告目录、`?market=US` 与 `/api/triple-us`。原CN默认规则、A股100股、前复权口径和T+1路径保留。

审查未发现该增量直接引入完成周线前视；US同日止损的净值代数及双边成本正确。发现并修复两项收盘时间输入问题：

1. US原校验只检查小时，误接受13:59、16:01等时间。现在要求精确13:00或16:00，检查分、秒及小数秒。
2. CN行情若额外带有 `session_close` 会被误信任。现在此列统一清除，并按所选市场的日历重新计算；附加13:00或非法字符串不会改变原CN结果，US仍以校验日历为准。

另修复一个测试隔离问题：mock runner时懒加载年度／案例模块，会让模块保留mock别名，导致不同测试顺序出现假失败。测试在mock前显式加载依赖，不修改产品运行逻辑。

本地独立研究新增 `drop_weekly_entry` / `drop_weekly_exit`，贯通引擎、runner、study和报告标签。原四种variant及默认配置保持。组件实验通过CLI/API配置使用，页面默认十案例菜单未扩充。

## 证据边界

原bundle全量本地回归511 passed、3 skipped；新增研究和修复后的最终回归 **538 passed、3 skipped、11 warnings，65.39秒**，日志 `runtime_layer/reports/bundle_research/final_repair_pytest.log`。Node12项行为测试通过。新增测试包含可手算净值、CN/US止损差异、停牌后T+1退出、未来行情扰动，以及收盘时间字段隔离。

3项skip包括本机未安装的 `exchange_calendars` 实际日历测试和2项旧训练依赖测试。本机未安装依赖，未执行行情下载；US合成测试与HTTP合同通过，不代表真实美股行情或浏览器点击复验。云端原文中的下载、US100收益、US500结果属于交付方记录。

本地100股四组×两区间共800任务，在收盘字段修复前后各运行一次。原版与旧开发基线逐股核对：实际日期、行情与日历哈希、指标、净值、成交、交易完全一致；信号与决策保留全部原有字段，仅允许此前已加入的variant注释字段。详见 [独立研究结果](WEEKLY_COMPONENT_RESULTS.zh-CN.md)。

## 研究限制与后续交接

US downloader用下载时 `regularMarketPrice` 和区间末段60日成交额筛选，对更早回测是后验筛选。因此固定US100是存活且活跃大公司样本，不能当作历史可知的无偏股票池。严格要求全区间行情覆盖也不适合直接套到包含新上市公司的US500。

获得真实美股快照后还需核对原始响应、调整因子与manifest；当前不能仅靠 `adjust=pre` 或交付说明认证数据。后续US500下载器／多周期研究源码仍需单独拿到并做差异审查，不能用此早期bundle冒充完全同步。当前没有证据需要覆盖本地前复权数据。

原数据、运行报告、模型与缓存均不进Git。下载示例仅用于可下载的环境；本机继续遵守不自行下载的要求。
