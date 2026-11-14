# 云端下载口径摘要

本文是从云端 `codex/us500-acceptance` 分支同步的口径摘要，不包含原始行情或下载缓存。

## A 股 BaoStock

```bash
python -m download_layer.cli.download --source baostock \
  --storage shard --storage-root runtime_layer/data \
  --symbols config/universe/available_universe.txt \
  --start 2020-01-01 --end 2026-07-12 \
  --freqs d --adjustflags 2 --workers 1 --sleep 0.5 \
  --report runtime_layer/runs/cloud_download_report.csv
```

`adjustflag=2` 是 BaoStock 前复权。结束日使用周日 2026-07-12，让 2026-07-10 所在周拥有完整自然日历。当前本地100股工作流就是这一口径；行情数据不进 Git。

## 美股 Yahoo

云端代码增加了独立配置：

```bash
python -m download_layer.cli.download --source yahoo \
  --config config/research/us500_download.json

python -m download_layer.cli.download --source yahoo \
  --config config/research/us500_download.json --audit-only

python -m download_layer.cli.download --source yahoo \
  --config config/research/us500_download.json --only AAPL MSFT
```

Yahoo 数据独立保存在美股目录，使用供应商 OHLC × `adjclose/vendor_close`，XNYS 日历，成交量单位为股。它不继承 BaoStock 的复权参数、A 股交易规则或本地三重滤网市场约束。

## 同步边界

- 下载入口代码可以在后续逐文件合并。
- 云端数据快照不能从 Git commit 恢复，因为原始数据、运行结果和缓存没有提交 Git。
- 如果要把云端数据带回本地，需要云端提供快照包、文件共享路径或其他明确的可访问交付物；本文件不声称数据已经本地同步。
