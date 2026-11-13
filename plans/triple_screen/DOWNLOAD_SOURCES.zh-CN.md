# BaoStock 与 Yahoo 并列下载入口

2026-11-13。入口同时提供 A 股 BaoStock 与美股 Yahoo Finance，市场配置、目录、日历与调整含义保持独立。安装 `requirements-research.txt` 即包含 dev/core 依赖及美股交易日历；无需 PyTorch/CUDA。

## 页面

在原 **Download Data** 页面选择 **Data Source**：

- **BaoStock / A 股** 为默认，保留原股票、日期、频率、复权与分片设置。
- **Yahoo Finance / 美股** 使用登记配置，只能选择固定 US500 或原 US100；默认 US500。页面展示实际股票数量、请求日期与独立目录，隐藏 A 股参数，不接受任意路径或临时换股。
- 两者使用同一下载任务进度、停止与报告入口。Yahoo 的取消在已启动股票完成后生效；成功、失败和未启动取消股票均有记录。`partial/failed/cancelled` 不改写成成功，刷新后能恢复活动任务的来源。

美股下载接入页面，不等于 US500 策略研究已接入原 A 股 Triple Screen 页面。后者仍按 A 股分片、市场与执行约束工作。

## CLI

所有命令从仓库根目录执行。旧 BaoStock 命令不变；可显式写来源：

```bash
python -m download_layer.cli.download --source baostock \
  --storage shard --storage-root runtime_layer/data \
  --symbols config/universe/available_universe.txt \
  --start 2020-01-01 --end 2026-07-12 \
  --freqs d --adjustflags 2 --workers 1 --sleep 0.5 \
  --report runtime_layer/runs/cloud_download_report.csv
```

注意 BaoStock 不指定 `--adjustflags` 时仍默认 `3,2`；当前只研究前复权需显式指定 `2`。

Yahoo 使用独立 JSON 配置，不接受把 BaoStock 的复权、分片或交易参数直接套过来：

```bash
python -m download_layer.cli.download --source yahoo \
  --config config/research/us500_download.json

python -m download_layer.cli.download --source yahoo \
  --config config/research/us500_download.json --audit-only

# 只重试固定名单内指定股票；保留原成功快照及此前失败记录。
python -m download_layer.cli.download --source yahoo \
  --config config/research/us500_download.json --only AAPL MSFT
```

原100股用 `config/research/us100_download.json`。`--audit-only` 是不联网的全名单审计，不能同时传 `--only`。失败、部分失败和取消均返回非零退出码；配置错误在联网前拒绝。

| 项目 | BaoStock A 股 | Yahoo 美股研究 |
|---|---|---|
| 股票 | 原 A 股名单 | 固定 US100 / US500，500前100项保持原顺序 |
| 默认目录 | `runtime_layer/data` | `runtime_layer/data_us_research` / `runtime_layer/data_us500` |
| 存储 | 原分片和目录数据库 | 每股原始JSON、标准化日线CSV、元数据及日历 |
| 日历 | 项目原 BaoStock 日历流程 | `exchange_calendars` XNYS，保留完整自然日期 |
| 价格调整 | BaoStock 1/2/3 原定义 | vendor OHLC × adjclose/vendor close，供应商拆股/分红调整 |
| 成交量 | 沿用原适配定义 | 股，保留原始供应商成交量 |
| 周线 | 沿用各研究配置 | 本阶段只下载日线，研究时生成已完成周线 |

Yahoo 标准化列中的 `adjust=pre` 是项目兼容标签，不能理解为与 BaoStock `adjustflag=2` 价格语义或数值相同。供应商 OHLC 本身可能已拆股调整，不能凭调整因子反推出原交易所名义价格。

成功缓存必须通过源响应重建、证券身份、日期范围与哈希检查才复用；错误缓存不能静默覆盖或被只读审计认证成功。相同数据根目录有独占任务锁，避免页面和CLI同时写同一股票。正常结束释放自身锁；意外进程终止留下的锁需先确认任务已停止，再处理，程序不擅自判断锁已过期。

配置保留单次请求超时、重试次数、并发数和总调度时限。总调度时限停止新任务，活动请求仍在股票边界结束；不是进程级硬截止。Linux 批量任务可在外层用 `timeout 2400 ...`，Windows 可由启动进程设置总超时。不要因超时删除有效数据或修改签名。

## 历史长度与数据版本

研究配置当前请求 **2020-01-01 至 2026-07-12**；结束日为周日，最后实际行情为 **2026-07-10**。历史足够长的股票覆盖约六年半；新上市股票较短，按逐股数据与100个已完成周线确定评价起点。

这个窗口不是接口长度上限。2026-11-13 用同一 Yahoo 日线接口独立探测，从1970-01-01请求至2026-07-12，实际得到：

| 股票 | 最早实际日线 | 最后实际日线 | 行数 |
|---|---|---|---:|
| AAPL | 1980-12-12 | 2026-07-10 | 11,485 |
| MSFT | 1986-03-13 | 2026-07-10 | 10,159 |

这些探测没有扩展研究股票池，也没有混入正式2020起点的行情目录。它们只证明两只股票在本次请求的可用范围，不能保证所有股票均有相同长度或完整上市历史。Yahoo 帮助说明历史范围取决于交易历史及其可用数据，历史通常不早于1970；见[历史数据说明](https://help.yahoo.com/kb/finance/historical-prices-adjusted-close-sln2311.html)及[图表范围说明](https://help.yahoo.com/kb/SLN28287.html)。项目用公开 chart 响应，不依赖网页付费 CSV 下载按钮；本次可访问不等于官方承诺的稳定免费API。

修改日期请复制配置并使用新的数据根目录和报告目录，不覆盖旧成功快照。前复权历史会随分红/拆股与供应商修订变化，新下载数据不保证与旧版本的价格、哈希或研究结果逐项相同。

## 交付与验证口径

代码、固定名单、配置、测试与说明提交 GitHub。用户已进一步授权保存行情；行情作为独立个人研究快照包交付，包含源响应、CSV、逐股下载时间/请求/文件哈希及审计清单，不进入 Git 历史。Yahoo 的[数据供应说明](https://help.yahoo.com/kb/SLN2352.html)限制行情再分发，因此不把行情公开作为 GitHub Release 数据集发布。

2026-11-13后续检查时，当前工作目录只保留了较早的US100代码/行情，US500源码通过已保存验收bundle恢复。目录不一致的原因未查明；不将其断言为平台重置或用户删除。本轮按原固定500名单复用有效100股缓存并重新补下载其余400股。此前US500完整研究工件不在当前目录，历史验收记录保留其来源；本轮重新审计下载数据和工程代码，未冒称重跑此前1540组全策略批次。新行情不能冒充旧批次的精确快照。
