# 云端研究同步记录

同步日期：2026-11-14。来源分支：`codex/us500-acceptance`，最新提交：`aea32a540c626835162b91ebbe6b295214d5ffa4`。该分支的前序下载入口提交为 `5c157756becaa18f59fdb2216894e68446ba1528`，US100/US500研究提交为 `ed99ec1`、`bc8d0b1`。

## 云端已确认内容

- 云端是全新 Linux/Python 3.12.14 环境，可以安装 CPU 研究依赖并下载行情；没有安装 PyTorch/CUDA。
- 云端完成了 Yahoo Finance US500 数据快照：固定 500 股、500/500 成功、798,427 根日线，实际行情最后日期为 2026-07-10；714 条零成交量记录原样保留。
- 云端请求范围为 2020-01-01 至 2026-07-12；较新股票可能没有完整的100根已完成周线，报告中按实际预热资格后移评价起点。
- 云端还记录了 US100／US500 多周期研究工程和 Yahoo/BaoStock 并列下载入口，最新完整回归记录为 647 passed、2 skipped、19 warnings。

## 数据口径隔离

- 云端美股使用 Yahoo 日线，标准化价格是供应商 OHLC × `adjclose/vendor_close`，成交量单位为股，日历为 XNYS。
- 本地 A 股使用 BaoStock 前复权 `adjustflag=2` 和项目交易日历。两种价格调整定义、市场规则、交易日历和成本含义不同，不能把云端美股结果直接并入 A 股结果，也不能要求两边收益逐值相同。
- 云端原始行情、JSON 源响应、CSV、数据库、缓存和研究运行结果没有进入 GitHub；本地没有自动下载或复制这些数据。要真正同步数据，需要云端提供可访问的快照包或明确的本地路径。

## 云端代码待合并项

云端前序提交 `5c15775` 新增/修改了：

- `download_layer/services/yahoo_downloader.py`
- `app_layer/backend/yahoo_download_controller.py`
- `app_layer/frontend/download.js`
- 下载 CLI 的 `--source yahoo --config ...` 入口
- `config/research/us100_download.json`、`us500_download.json`
- `config/universe/us_research100.txt`、`us_research500.txt`
- Yahoo 下载相关测试和 `DOWNLOAD_SOURCES.zh-CN.md`

这些变更尚未直接合并到本地 `codex/local100-ui-repair`，原因是本地 Git fetch 受网络 TLS EOF 阻断，并且本地修复分支与云端代码需要逐文件合并。当前本地 A 股三重滤网和页面状态不受影响。

## 推荐合并方式

云端继续使用自己的环境下载和验证；提交代码时保留清晰 commit/PR。后续在本地拿到可访问的云端代码提交或快照路径后，先放入临时 worktree，逐文件审阅，再选择性合并：

1. 先合并独立下载层、US 配置和下载测试；
2. 再核对 `app_layer/backend/server.py`、`app.js` 与本地三重滤网页面的冲突；
3. 保持 Yahoo 数据目录与 A 股 `runtime_layer/data` 隔离；
4. 重新跑本地页面和研究核心测试；
5. 不把美股数据或研究结果提交到 GitHub。

云端当前的研究验收也明确：美股工程验证不等于 A 股100股结果验收，不等于稳定盈利证明，不等于 Windows 浏览器验收。

来源：

- [云端最新验收说明](https://github.com/XuanFeng05/PocketAgent/blob/codex/us500-acceptance/plans/triple_screen/CLOUD_ACCEPTANCE.zh-CN.md)
- [Yahoo/BaoStock 下载口径说明](https://github.com/XuanFeng05/PocketAgent/blob/codex/us500-acceptance/plans/triple_screen/DOWNLOAD_SOURCES.zh-CN.md)
- [下载入口提交 5c15775](https://github.com/XuanFeng05/PocketAgent/commit/5c157756becaa18f59fdb2216894e68446ba1528)
- [最新验收提交 aea32a5](https://github.com/XuanFeng05/PocketAgent/commit/aea32a540c626835162b91ebbe6b295214d5ffa4)
