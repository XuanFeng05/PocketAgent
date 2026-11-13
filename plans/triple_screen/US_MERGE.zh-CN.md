# 美股适配交付与下载配置

基线：`codex/local100-ui-repair` 的 `21cd849b6769185199e2e39ee069f2cd2d4a7f47`。整合分支：`codex/us100-integration`。先读取并沿用 `PROJECT_STATUS.zh-CN.md` 与 `LOCAL_REPAIR.zh-CN.md` 的页面、进度、取消、重启核对和报告逻辑；没有用较早云端页面实现覆盖本地修复。

## 合并内容

- 新增Yahoo日线下载模块 `download_layer/cli/download_us.py`，固定清单 `config/universe/us_research100.txt`，可选CPU依赖 `requirements-us.txt`。
- 新增 `us100_primary.json`、`us100_annual.json`、`us100_study.json`，都是同一100股、独立US数据路径及美东日历。年度/研究配置由对应Python runner使用；现有 `research_layer.cli.triple_screen` 只执行基础实验，不冒充年度/研究CLI。
- 独立规则版本 `triple_screen_us_pre_v1`，显式纽约收盘时刻与当日止损语义；A股版本及原100股清单保留。
- 在已修好的页面上增加 `?market=US` 和 `/api/triple-us`，独立请求历史和报告目录，继承原进度、取消、重启、子报告功能。
- 新增美股规则、数据拒绝、市场隔离及四模式HTTP合同测试，扩展原Node测试检查美股路由和安全子报告链接。
- 交付仅含源码、股票代码、配置、测试和说明；原始行情、供应商JSON、数据库、环境和运行工件均不提交。

## 下载配置

以下是复现配置，不会随页面启动自动执行。本轮只在云端取数；合并源码不授权在本机自动安装或下载。

| 参数 | 固定值 |
| --- | --- |
| 行情源 | Yahoo Finance公开chart接口，本次实测可达；不保证持续可用或商业授权 |
| 股票 | `config/universe/us_research100.txt`，100个唯一代码，不自动换股 |
| 资格 | NYSE/Nasdaq、USD、EQUITY；获取时最新价≥10美元，请求末段60交易日Close×Volume中位数≥2500万美元 |
| 日期 | 2020-01-01至2026-07-12；日线实际截至2026-07-10 |
| 频率/调整 | 仅日线；Yahoo AdjClose/Close乘OHLC，成交量保持供应商股数；模块固定此口径，无BaoStock `--adjustflags` 参数 |
| 日历 | exchange_calendars XNYS，纽约本地日期、常规/提前收盘、完整自然日 |
| 并发/节奏 | 串行1 worker，间隔0.5秒，单次网络timeout 20秒 |
| 数据目录 | `runtime_layer/data_us`，不覆盖原 `runtime_layer/data` |
| 失败 | 原始响应/已成功分片保留，逐股明确原因，整体退出码2；不自动替换、补量或修价 |
| 快照 | 原始JSON、供应商CSV、调整因子、文件SHA-256和请求时间写manifest；已有manifest时拒绝覆盖 |

在已获授权、激活Python 3.12虚拟环境的项目根目录安装：

```text
python -m pip install -r requirements-dev.txt -r requirements-us.txt
```

本次实测Python 3.12.14、exchange-calendars 4.13.2、tzdata 2026.6、pandas 2.3.3、numpy 2.5.4、duckdb 1.5.6、pyarrow 22.0.0。显式tzdata依赖兼顾Windows没有系统IANA时区库的情况；本次未执行Windows浏览器验收。

连接探测先创建独立AAPL清单并请求一个月；以下Python外层对探测设置60秒上限：

```text
python -c "from pathlib import Path; p=Path('runtime_layer/us_probe'); p.mkdir(parents=True,exist_ok=True); (p/'symbols.txt').write_text('AAPL\n',encoding='utf-8')"
python -c "import subprocess,sys; subprocess.run([sys.executable,'-m','download_layer.cli.download_us','--symbols','runtime_layer/us_probe/symbols.txt','--start','2026-06-01','--end','2026-06-30','--storage-root','runtime_layer/us_probe/data','--timeout','20','--sleep','0.5'],timeout=60,check=True)"
```

只有探测成功才执行100股。云端本次外层限制1800秒，完整配置命令为：

```text
python -m download_layer.cli.download_us --symbols config/universe/us_research100.txt --start 2020-01-01 --end 2026-07-12 --storage-root runtime_layer/data_us --timeout 20 --sleep 0.5
```

若某股失败，用只包含失败代码的新清单和新目录排查；保留原manifest，不删除成功分片。当前模块不提供自动续传或合并不同快照；未完成数据对账前不开始市场评价。新建下载目录时，研究配置的source_path和calendar_path也要同步改为该快照，不能混源。

## 运行与验证

```text
python -m research_layer.cli.triple_screen --config plans/triple_screen/us100_primary.json --output runtime_layer/reports/us100_primary
python -m app_layer.cli.run_dashboard --host 127.0.0.1 --port 8000
python -B -m pytest -q -p no:cacheprovider
node tests/frontend/test_triple_ui.js
```

访问 `http://127.0.0.1:8000/triple.html?market=US`。默认评价2022-01-03至2025-12-31；2026已下载行情仅检查质量，不标成未见测试。年度/案例可以在页面选择，并打开对应子报告。取消在当前股票/成本步骤结束后生效。

## 本地取得提交

远端发布成功时，获取 `codex/us100-integration` 后合并到本地修复分支；提交以本次交付回复中的完整hash为准。如提供源码Git bundle，可使用：

```text
git switch codex/local100-ui-repair
git fetch /path/to/PocketAgent-us100-integration.bundle codex/us100-integration
git merge --ff-only FETCH_HEAD
```

bundle以21cd849为前置提交，仅含本次源码增量。若本地修复分支已有额外提交，快进合并会安全停止，应先比较差异再正常合并，不强制重置。`LOCAL_REPAIR.zh-CN.md` 的原本地验收记录保持原样；云端新证据列在根目录状态文档，二者不互相冒充。
