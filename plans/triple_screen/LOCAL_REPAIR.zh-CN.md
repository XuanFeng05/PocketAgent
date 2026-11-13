# 本地100股页面修复与云端合并边界

2026-11-13。基线a91b4ad；分支codex/local100-ui-repair。用户要求本地与云端并行推进，修好后上传。当前范围仍是已有100股页面闭环，不要求全A股覆盖。

## 本地负责范围

- app_layer/backend/triple_controller.py、server.py：输入预检、进度、请求状态、取消、重启对账、子报告链接。
- app_layer/backend/jobs.py：保留partial/interrupted终态，避免通用job API与专用请求状态不一致。
- app_layer/frontend/triple.html/js/css：批次默认、数字参数、可编辑窗口、十案例选择、结果和状态显示、历史查看。
- research_layer/triple_runner.py/triple_validation.py/triple_study.py：可选cancel_check与父子任务级进度，原数值规则不变；CLI callback适配run_started事件。
- tests/integration/test_triple_dashboard.py：临时合成分片和四模式端到端测试；另新增test_triple_cancellation.py、tests/frontend/test_triple_ui.js。

## 实际页面验证

本地127.0.0.1临时服务，仅使用已有数据：

| 模式/操作 | job ID | 实测结果 |
|---|---|---|
| 100股取消 | triple_screen_20261113_115840_d93b5e | 15/100时终态cancelled，已完成结果保留 |
| 单股 | triple_screen_20261113_115852_be108f | 1/1完成，报告可查看 |
| 年度 | triple_screen_20261113_115859_6a35d6 | 页面日期生成2022/2023两窗口，2/2完成，百分比显示及子报告链接正常 |
| 参数/滤网对照 | triple_screen_20261113_115908_e51cc0 | 十案例10/10完成，移除周线报告明确显示对应变体 |
| 100股批次 | triple_screen_20261113_115949_d9e4bb | 100/100完成，刷新后历史与结果保留 |

本地证据runtime_layer/reports/local100_repair/page_acceptance.json与page_verified.png，运行数据和图片不提交。1618个原始数据文件大小/mtime与上传前记录一致；没有下载或安装依赖。

最终全量pytest：497 passed、2 skipped、7条原有NumPy/pandas弃用警告，100.99秒。日志runtime_layer/reports/local100_repair/final_pytest_20261113.log。前端Node测试验证payload、窗口、10案例、格式化与报告错误显示；独立review未发现四种页面工作流的阻断问题。

## 仍然明确保留的边界

1. 取消在股票/成本任务边界生效，当前步骤不抢占；不会删除已完成工件。
2. 页面历史仅列页面请求，尚无自动导入任意既有CLI历史；基础CLI提供校验恢复，年度/研究中断保存部分结果但没有GUI续算按钮。
3. 股票和不同预热起点、样本内研究及比例成本等口径沿用已登记规则；修复不证明策略盈利。
4. 本地版本已修复这些页面文件。云端如果同时修改，合并时核对diff和测试，不能直接覆盖。

## 云端美股配置的后续合并

用户说明云端可能因网络问题先用美股做验证。等云端提供下载脚本/配置、依赖版本、数据源字段映射、股票与日期、commit/PR后再合并。工程验证和市场效果分别标注；保留市场ID、交易日历、复权含义、成交量单位、T+1与触发约束，不伪装成.SH/.SZ或直接覆盖A股默认配置。当前本地没有这些美股工件，不能声明已合并或已验证。
