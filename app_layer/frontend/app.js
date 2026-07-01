const DEFAULT_DB_PATH = 'runtime_layer/data';
const DEFAULT_DOWNLOAD_SYMBOLS_FILE = 'config/universe/download_symbols.txt';
const VISUAL_DEFAULT_VISIBLE_BARS = 120;
const VISUAL_WINDOW_BARS = 1200;
const VISUAL_MAX_FETCH_BARS = 9600;
const VISUAL_MAX_BUFFER_BARS = 102400;
const VISUAL_MAX_VISIBLE_BARS = 102400;
const FREQUENCY_ORDER = ['5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'];
const VISUAL_DEFAULT_FREQS = [...FREQUENCY_ORDER];
const VISUAL_DEFAULT_ADJUSTS = ['pre', 'none', 'post'];

const state = {
  inventory: [],
  activeJobId: null,
  downloadJobActive: false,
  pollTimer: null,
  materializeJobId: null,
  materializePollTimer: null,
  bundleJobId: null,
  bundlePollTimer: null,
  featureJobId: null,
  featurePollTimer: null,
  agentCacheJobId: null,
  agentCachePollTimer: null,
  agentJobId: null,
  agentPollTimer: null,
  agentSpec: null,
  agentRunId: null,
  agentRuns: [],
  agentRunMetrics: [],
  agentRunLogs: [],
  agentMetricSeq: 0,
  agentLogSeq: 0,
  agentLoadedRunId: null,
  agentPollBusy: false,
  agentPollGeneration: 0,
  agentPollFailures: 0,
  agentActiveTab: 'setup',
  agentChartHidden: {},
  agentChartHover: {},
  agentChartGroupsCollapsed: {},
  agentSelectedManageRun: null,
  evaluationModels: [],
  evaluationRuns: [],
  evaluationRunId: null,
  evaluationReplay: {
    activeTab: 'market',
    runId: null,
    payload: null,
    hoverIndex: null,
    pollTimer: null,
    appliedRunId: null,
    appliedEventCount: 0,
  },
  featureSpec: null,
  featureIndicatorConfig: null,
  modelInputBlueprint: null,
  modelInputDefault: null,
  modelInputCatalog: [],
  modelInputCompiled: null,
  modelInputValidation: null,
  modelInputDirty: false,
  modelInputDragIndex: null,
  modelInputInsertIndex: null,
  inventoryViewMode: 'group',
  expandedInventorySymbols: new Set(),
  selectedInventoryKeys: new Set(),
  configSymbolPath: DEFAULT_DOWNLOAD_SYMBOLS_FILE,
  visualization: {
    payload: null,
    selectedFreq: 'daily',
    loading: false,
    windowOffset: 0,
    totalRows: 0,
    prefetching: null,
    beforeLoadSize: VISUAL_WINDOW_BARS,
    afterLoadSize: VISUAL_WINDOW_BARS,
    beforeTriggerAbsolute: null,
    afterTriggerAbsolute: null,
    pendingZoom: null,
    mainOverlays: new Set(),
    mainOverlayPreferenceExplicit: false,
    panelSelections: ['volume', 'macd', 'kd'],
    panelSelectionsByFreq: {},
    displayInitialized: false,
    inspectorVisible: true,
    loadGeneration: 0,
    abortController: null,
    viewStart: 0,
    viewEnd: 0,
    hoverIndex: null,
    hoverPanel: null,
    referenceGuides: {
      active: false,
      panel: null,
      price: null,
      volume: null,
    },
    modeSnapshots: {
      market: null,
      evaluation: null,
    },
    drag: {
      active: false,
      startX: 0,
      startViewStart: 0,
      startViewEnd: 0,
    },
  },
  detail: {
    symbol: null,
    slices: [],
    selectedFreq: null,
    selectedAdjust: null,
    offset: 0,
    limit: 100,
    rows: [],
    hasMore: false,
    loading: false,
  },
};

let visualReferenceClickTimer = null;
let modelInputValidationTimer = null;
let modelInputValidationGeneration = 0;
let agentExportConflictResolver = null;

const LANGUAGE_STORAGE_KEY = 'pocketagent.language';
const I18N_EN_TO_ZH = {
  'Layer Console': '分层控制台',
  'Visualization': '可视化',
  'Download': '下载',
  'Data': '数据',
  'Feature': '特征',
  'Agent': '智能体',
  'Evaluation': '评估',
  'Config': '配置',
  'Refresh': '刷新',
  'Table': '表格',
  'Values': '数值',
  'Database': '数据库',
  'Symbol': '股票代码',
  'Adjust': '复权',
  'Daily': '日线',
  'Weekly': '周线',
  'Monthly': '月线',
  'Sub Panels': '副图数量',
  'Main + 1': '主图 + 1',
  'Main + 2': '主图 + 2',
  'Main + 3': '主图 + 3',
  'Indicators': '指标',
  'Loading...': '加载中...',
  'K-line': 'K线',
  'Technical Values': '技术指标数值',
  'Move across the chart to inspect indicator values.': '在图表上移动鼠标以查看指标数值。',
  'Download Layer': '下载层',
  'Download Data': '下载数据',
  'Market Data Root': '市场数据根目录',
  'Shard Storage Root': '分片存储根目录',
  'Symbols File': '股票列表文件',
  'Use Config Selection': '使用配置选择',
  'Edit Symbols In Config': '在配置中编辑股票',
  'Symbol file is managed in Config.': '股票文件在配置页中管理。',
  'Start Date': '开始日期',
  'End Date': '结束日期',
  'Frequencies': '频率',
  'Adjust Modes': '复权模式',
  'None': '不复权',
  'Pre': '前复权',
  'Post': '后复权',
  'none': '不复权',
  'pre': '前复权',
  'post': '后复权',
  'Request Sleep': '请求间隔',
  'Workers': '并发数',
  'Replace existing symbol rows': '替换已有股票数据',
  'Skip verified completed slices': '跳过已验证完成分片',
  'Start Download': '开始下载',
  'Stop': '停止',
  'Download Job': '下载任务',
  'No active job': '无活动任务',
  'Progress': '进度',
  'Current': '当前',
  'Completed': '完成',
  'Succeeded': '成功',
  'Failed': '失败',
  'Saved Rows': '保存行数',
  'No download CSV loaded.': '未加载下载 CSV。',
  'Download Logs': '下载日志',
  'No logs.': '无日志。',
  'Failed / Warning Download Rows': '失败 / 警告下载行',
  'Freq': '频率',
  'Status': '状态',
  'Rows': '行数',
  'Error': '错误',
  'No failed or warning rows.': '没有失败或警告行。',
  'Data Layer': '数据层',
  'Local Database': '本地数据库',
  'Symbols': '股票数',
  'Total Rows': '总行数',
  'Date Range': '日期范围',
  'Derived Bars': '派生周期',
  'Base Frequency': '基础频率',
  'CSV Output': 'CSV 输出路径',
  'Targets': '目标周期',
  'Build Derived Bars': '生成派生周期',
  'No materialize job': '无派生周期任务',
  'Derived Bar Logs': '派生周期日志',
  'Portable Bundle': '迁移包',
  'Export data/feature shards into one portable DuckDB file for migration. Import restores the original shard layout.': '将数据/特征分片导出为一个可迁移 DuckDB 文件；导入时恢复原分片结构。',
  'Bundle Path': '迁移包路径',
  'Feature Dataset Dir': '特征数据集目录',
  'Include': '包含',
  'Data Layer shards': '数据层分片',
  'Feature Layer parts': '特征层分片',
  'Replace existing shards on import': '导入时替换已有分片',
  'Export Bundle': '导出迁移包',
  'Import Bundle': '导入迁移包',
  'Inspect Bundle': '检查迁移包',
  'No bundle job': '无迁移包任务',
  'Bundle Logs': '迁移包日志',
  'Inventory': '数据列表',
  'Delete Selected': '删除选中',
  'All frequencies': '全部频率',
  'All adjustments': '全部复权',
  'Grouped': '分组',
  'Slices': '分片',
  'Freq / Adjust': '频率 / 复权',
  'Start': '开始',
  'End': '结束',
  'Actions': '操作',
  'No data.': '无数据。',
  'Coverage Check': '覆盖检查',
  'Minimum Rows': '最小行数',
  'Required Start': '要求开始',
  'Required End': '要求结束',
  'Check Coverage': '检查覆盖',
  'Available': '可用',
  'No results.': '无结果。',
  'Available Universe': '可用股票池',
  'Candidate File': '候选文件',
  'Output File': '输出文件',
  'CSV Report': 'CSV 报告',
  'Build Universe': '生成股票池',
  'Feature Layer': '特征层',
  'Dataset Builder': '数据集构建器',
  'Trade Frequency': '交易频率',
  'Max Decisions': '最大决策数',
  'Build Chunk Size': '构建分块大小',
  'Build Workers': '构建并发数',
  'Low memory mode': '低内存模式',
  'Market parquet cache': '行情 parquet 缓存',
  'Rebuild market cache': '重建行情缓存',
  'Incremental build': '增量构建',
  'Force rebuild feature parts': '强制重建特征分片',
  'Open auction decision': '开盘集合竞价决策',
  'Market Cache Directory': '行情缓存目录',
  'Output Directory': '输出目录',
  'Available Frequencies / Enabled for This Dataset': '可用频率 / 本数据集启用',
  'Sequence Windows': '序列窗口',
  'Preflight': '预检查',
  'Preview': '预览',
  'Build Feature Dataset': '生成特征数据集',
  'Preflight Status: idle': '预检查状态：空闲',
  'Run preflight before building.': '构建前请先运行预检查。',
  'Dataset Preview': '数据集预览',
  'No preview.': '无预览。',
  'Decision Time': '决策时间',
  'Stage': '阶段',
  'Execution': '执行',
  'Market Rule': '市场规则',
  'Visible Bar': '可见K线',
  'Market Rows': '行情行数',
  'Build Status: idle': '构建状态：空闲',
  'Feature Logs': '特征日志',
  'Indicator Configuration': '指标配置',
  'Add Indicator': '添加指标',
  'Save Indicators': '保存指标',
  'Feature Contract': '特征契约',
  'Indicator Details': '指标详情',
  'Name': '名称',
  'Included': '包含',
  'Outputs': '输出',
  'Formula': '公式',
  'Trading Rule Inputs': '交易规则输入',
  'Rule': '规则',
  'Value': '值',
  'Note': '备注',
  'Model Input Blueprint': '模型输入蓝图',
  'Add Feature': '添加特征',
  'Add Group': '添加分组',
  'Add Comment': '添加注释',
  'Reset Default': '重置默认',
  'Save Blueprint': '保存蓝图',
  'Loading blueprint...': '正在加载蓝图...',
  'Search': '搜索',
  'Close': '关闭',
  'Quality & Debug Outputs': '质量与调试输出',
  'File': '文件',
  'Grain': '数据粒度',
  'Columns': '列',
  'Agent Layer': '智能体层',
  'Configure, run, and monitor persistent single-stock training.': '配置、运行并监控持久化单股训练。',
  'Open Monitor Window': '打开监控窗口',
  'Setup': '设置',
  'Live Monitor': '实时监控',
  'Runs & Checkpoints': '运行与检查点',
  'Run Setup': '运行设置',
  'Profile': '配置档',
  'Smoke': '快速测试',
  'Formal': '正式',
  'Custom': '自定义',
  'Run Name': '运行名称',
  'Feature Dataset': '特征数据集',
  'Training Symbols': '训练股票',
  'Validation Symbols': '验证股票',
  'Total Timesteps': '总步数',
  'Seed': '随机种子',
  'Device': '设备',
  'Start Training': '开始训练',
  'Stop Training': '停止训练',
  'Evaluation Layer': '评估层',
  'Config Layer': '配置层',
  'Save': '保存',
  'Cancel': '取消',
  'Apply': '应用',
  'Edit': '编辑',
  'Delete': '删除',
  'View': '查看',
  'Manage': '管理',
  'Select All': '全选',
  'Clear': '清空',

  'idle': '空闲',
  'running': '运行中',
  'completed': '完成',
  'failed': '失败',
  'queued': '排队中',
  'cancelled': '已取消',
  'Preflight Status': '预检查状态',
  'Build Status': '构建状态',
  'Training Status': '训练状态',
  'Validation Status': '验证状态',
  'Search symbol': '搜索股票代码',

  'PocketAgent Console': 'PocketAgent 控制台',
  '中文': '中文',
  'CSV:': 'CSV：',
  'Check': '检查',
  'Message': '消息',
  'daily': '日线',
  'weekly': '周线',
  'monthly': '月线',
  'Leave empty for all Store symbols.': '留空则使用特征存储中的全部股票。',
  'Symbol Limit': '股票数量限制',
  'Symbol Sample Seed': '股票抽样种子',
  'Walk-forward Fold': '滚动验证折',
  'Fold 1': '第 1 折',
  'Fold 2': '第 2 折',
  'Fold 3': '第 3 折',
  'Parallel Envs': '并行环境数',
  'Total Steps': '总训练步数',
  'Episode Days': '单回合天数',
  'Validation Days': '验证天数',
  'Use Agent Cache': '使用智能体缓存',
  'Training Frequencies': '训练频率',
  'Uncheck a frequency to exclude it. If none are checked, the Agent uses every frequency embedded in the Feature Dataset.': '取消勾选即可排除该频率；如果一个都不选，智能体会使用特征数据集中包含的全部频率。',
  'Model Output Directory': '模型输出目录',
  'Frozen Agent v1 Contract': '固定版 Agent v1 契约',
  'Section': '分区',
  'Setting': '设置',
  'Training Parameters': '训练参数',
  'PPO Optimization': 'PPO 优化参数',
  'Learning Rate': '学习率',
  'Final Learning Rate': '最终学习率',
  'Gamma': '折扣因子 Gamma',
  'GAE Lambda': 'GAE Lambda',
  'Rollout Steps': '采样步数',
  'Minibatch Size': '小批量大小',
  'Update Epochs': '更新轮数',
  'Clip Ratio': '裁剪比例',
  'Value Clip': '价值裁剪',
  'Value Coefficient': '价值损失系数',
  'Entropy Coefficient': '熵系数',
  'Max Gradient Norm': '最大梯度范数',
  'Target KL': '目标 KL',
  'Model Capacity': '模型容量',
  'Input Projection': '输入投影维度',
  'LSTM Hidden': 'LSTM 隐层维度',
  'LSTM Layers': 'LSTM 层数',
  'Fused Market': '融合行情维度',
  'Context Embedding': '上下文嵌入维度',
  'Runtime Embedding': '运行状态嵌入维度',
  'Local State': '本地状态维度',
  'Global State': '全局状态维度',
  'Dropout': 'Dropout',
  'Fixed at zero for PPO likelihood consistency.': '为保持 PPO 似然一致性固定为 0。',
  'Execution & Costs': '交易执行与成本',
  'Initial Cash': '初始资金',
  'Lot Size': '每手股数',
  'Max Position Ratio': '最大持仓比例',
  'Bar Participation': 'K线成交参与率',
  'Auction Participation': '集合竞价参与率',
  'Commission Rate': '佣金率',
  'Minimum Commission': '最低佣金',
  'Stamp Duty': '印花税',
  'Transfer Fee': '过户费',
  'Base Slippage': '基础滑点',
  'Impact Coefficient': '冲击成本系数',
  'Maximum Slippage': '最大滑点',
  'Holding Bar Cap': '持仓 K 线数上限',
  'Reward': '奖励',
  'scale × log(NAV after / NAV before) − drawdown penalty − turnover penalty − invalid-action penalty': '缩放 × log(操作后净值 / 操作前净值) − 回撤惩罚 − 换手惩罚 − 非法动作惩罚',
  'Reward Kind': '奖励类型',
  'Scale': '缩放',
  'Drawdown Penalty': '回撤惩罚',
  'Turnover Penalty': '换手惩罚',
  'Invalid Action Penalty': '非法动作惩罚',
  'Checkpoint & Validation Schedule': '检查点与验证计划',
  'Checkpoint Every Updates': '每多少次更新保存检查点',
  'Validate Every Updates': '每多少次更新验证',
  'Keep Last': '保留最近数量',
  'Best Metric': '最佳指标',
  'Sharpe': '夏普',
  'Calmar': '卡玛',
  'Total Return': '总收益',
  'Validation Sample Seed': '验证抽样种子',
  'Quick Validation Days': '快速验证天数',
  'Periodic Validation Device': '周期验证设备',
  'Auto': '自动',
  'Final Validation Device': '最终验证设备',
  'Run preflight before training.': '训练前请先运行预检查。',
  'Live Training': '实时训练',
  'Training Status: idle': '训练状态：空闲',
  'Pause': '暂停',
  'Resume': '继续',
  'Save Now': '立即保存',
  'Validation Status: idle': '验证状态：空闲',
  'Reward & NAV': '奖励与净值',
  'Loss': '损失',
  'Policy Health': '策略健康度',
  'Throughput': '吞吐量',
  'Latest Telemetry': '最新遥测',
  'Validation': '验证',
  'Agent Logs': '智能体日志',
  'Training Runs': '训练运行',
  'Run': '运行',
  'Training': '训练',
  'Steps': '步数',
  'Updated': '更新时间',
  'Action': '操作',
  'No runs loaded.': '未加载训练运行。',
  'Selected Run Checkpoints': '已选运行的检查点',
  'Size': '大小',
  'Modified': '修改时间',
  'Path': '路径',
  'Select a run.': '请选择一个运行。',
  'Coming soon.': '即将加入。',
  'Symbol Universe': '股票池',
  'Universe File': '股票池文件',
  'File Path': '文件路径',
  'Load': '加载',
  'Use For Download': '用于下载',
  'No symbol file loaded.': '未加载股票文件。',
  'Data Defaults': '数据默认设置',
  'Default Frequencies': '默认频率',
  'Default Adjust Modes': '默认复权模式',
  'Apply To Download': '应用到下载页',
  'Persistent YAML editing can be added after the UI flow is stable.': '界面流程稳定后可再加入持久化 YAML 编辑。',
  'K-line Data': 'K线数据',
  'Frequency': '频率',
  'Datetime': '时间',
  'Open': '开盘',
  'High': '最高',
  'Low': '最低',
  'Volume': '成交量',
  'Amount': '成交额',
  'PctChg': '涨跌幅',
  'No rows.': '无数据行。',
  'Scroll to load more.': '滚动加载更多。',

  'Training Progress': '训练进度',
  'Cumulative Training Steps': '训练累计步数',
  'Training Speed': '训练速度',
  'Hardware Performance': '硬件性能',
  'Hardware Usage': '硬件使用率',
  'Time Breakdown': '耗时拆分',
  'Trading Performance': '交易表现',
  'Reward & Asset': '收益与资产',
  'Actions & Costs': '动作与成本',
  'Algorithm Learning': '算法学习',
  'Validation Evaluation': '验证评估',
  'Validation Return': '验证收益',
  'Validation Risk': '验证风险',
  'Training Status: idle': '训练状态：空闲',
  'Validation Status: idle': '验证状态：空闲',
  'Training status': '训练状态',
  'Validation status': '验证状态',
  'Idle': '空闲',
  'Queued': '排队中',
  'Running': '运行中',
  'Paused': '已暂停',
  'Stopped': '已停止',
  'Completed': '已完成',
  'Failed': '失败',
  'Interrupted': '已中断',
  'Starting': '启动中',
  'Pending': '等待中',
  'Task created': '已创建任务',
  'Launching training process': '启动训练进程',
  'Validating training config': '校验训练配置',
  'Loading Feature Parts': '读取 Feature Parts',
  'Preparing data cache': '准备缓存数据',
  'Building train/validation splits': '构建训练/验证切分',
  'Building model and optimizer': '构建模型与优化器',
  'Starting rollout': '开始采样',
  'Collecting samples': '采样中',
  'Policy update': '策略更新',
  'Saving checkpoint': '保存检查点',
  'Training completed': '训练完成',
  'Heartbeat is stale; worker is still alive': 'heartbeat 暂未更新，worker 仍存活',
  'Validation task pending': '有待验证任务',
  'Phase': '阶段',
  'Updates / Episodes': '更新 / Episode',
  'Parallel Envs': '并行环境数',
  'Speed': '速度',
  'ETA': '预计剩余',
  'Elapsed': '运行耗时',
  'Data Cache': '数据缓存',
  'Latest Checkpoint': '最近检查点',
  'Best Checkpoint': '最佳检查点',
  'No logs yet.': '暂无日志。',
  'Reward': '奖励',
  'Asset / Cash': '资产 / 现金',
  'Total Loss': '总损失',
  'Policy / Value Loss': '策略 / 价值损失',
  'Entropy / Gradient': '熵 / 梯度',
  'Buy / Hold / Sell': '买 / 持有 / 卖',
  'Executed / Blocked': '成交 / 拒单',
  'Blocked Reasons': '拒单原因',
  'Turnover / Fees': '成交额 / 费用',
  'State': '状态',
  'Sample': '样本',
  'Return': '收益率',
  'Max Drawdown': '最大回撤',
  'Observations': '观测数',
  'No data': '暂无数据',
  'No visible metric data': '当前可见指标暂无数据',
  'Record': '记录',
  'Count': '数量',
  'Step': '步数',
  'Percent': '百分比',
  'Milliseconds': '毫秒',
  'Return / Reward': '收益 / 奖励',
  'Count / Cost': '数量 / 成本',
  'Loss': '损失',
  'PPO Health': 'PPO 状态',
  'Validation': '验证',
  'Score': '分数',
  'Training status: polling temporarily failed': '训练状态：轮询暂时失败',
  'Training status: queued / validating request': '训练状态：排队中 / 校验请求',
  'First sub panel': '第一个副图',
  'Second sub panel': '第二个副图',
  'Third sub panel': '第三个副图',
};
const I18N_ZH_TO_EN = Object.fromEntries(Object.entries(I18N_EN_TO_ZH).map(([en, zh]) => [zh, en]));
let languageObserver = null;
let languageApplying = false;

function currentLanguage() {
  return localStorage.getItem(LANGUAGE_STORAGE_KEY) || 'en';
}

function translateExactText(text, language) {
  const original = String(text ?? '');
  const trimmed = original.trim();
  if (!trimmed) return text;
  const dictionary = language === 'zh' ? I18N_EN_TO_ZH : I18N_ZH_TO_EN;
  const translated = dictionary[trimmed];
  if (translated) return original.replace(trimmed, translated);
  if (language === 'zh') {
    const statusMatch = trimmed.match(/^([A-Za-z ]+ Status):\s*(.+)$/);
    if (statusMatch && I18N_EN_TO_ZH[statusMatch[1]]) {
      const statusValue = I18N_EN_TO_ZH[statusMatch[2]] || statusMatch[2];
      return original.replace(trimmed, `${I18N_EN_TO_ZH[statusMatch[1]]}：${statusValue}`);
    }
  }
  return text;
}

function translateElementAttribute(element, attribute, language) {
  if (!element.hasAttribute(attribute)) return;
  const value = element.getAttribute(attribute);
  const translated = translateExactText(value, language);
  if (translated !== value) element.setAttribute(attribute, translated);
}

function applyLanguage(language = currentLanguage()) {
  languageApplying = true;
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (['SCRIPT', 'STYLE', 'CANVAS'].includes(parent.tagName)) return NodeFilter.FILTER_REJECT;
      if (parent.closest('.log-panel, pre, code')) return NodeFilter.FILTER_REJECT;
      return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  textNodes.forEach(node => {
    const translated = translateExactText(node.nodeValue, language);
    if (translated !== node.nodeValue) node.nodeValue = translated;
  });
  document.querySelectorAll('[placeholder], [title], [aria-label]').forEach(element => {
    translateElementAttribute(element, 'placeholder', language);
    translateElementAttribute(element, 'title', language);
    translateElementAttribute(element, 'aria-label', language);
  });
  document.querySelectorAll('input[type="button"], input[type="submit"]').forEach(element => {
    const translated = translateExactText(element.value, language);
    if (translated !== element.value) element.value = translated;
  });
  const button = $('languageToggleBtn');
  if (button) {
    button.textContent = language === 'zh' ? 'EN' : 'ZH';
    button.title = language === 'zh' ? 'Switch to English' : 'Switch to Chinese';
  }
  window.setTimeout(() => { languageApplying = false; }, 0);
}

function setupLanguageToggle() {
  const button = $('languageToggleBtn');
  if (button) {
    button.addEventListener('click', () => {
      const next = currentLanguage() === 'zh' ? 'en' : 'zh';
      localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
      applyLanguage(next);
    });
  }
  applyLanguage(currentLanguage());
  if (!languageObserver) {
    languageObserver = new MutationObserver(() => {
      if (languageApplying || currentLanguage() !== 'zh') return;
      window.setTimeout(() => {
        if (!languageApplying && currentLanguage() === 'zh') applyLanguage('zh');
      }, 30);
    });
    languageObserver.observe(document.body, { childList: true, subtree: true });
  }
}

function $(id) { return document.getElementById(id); }

function escapeHtml(value) {
  return String(value ?? '-')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : String(value);
}

function formatBytes(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let current = number;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }
  const digits = index === 0 ? 0 : current >= 10 ? 1 : 2;
  return `${current.toFixed(digits)} ${units[index]}`;
}

function formatPercent(value) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(4)}%` : String(value);
}

function formatSignedPercent(value) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  const sign = number > 0 ? '+' : '';
  return `${sign}${(number * 100).toFixed(4)}%`;
}

function formatSignedNumber(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  const sign = number > 0 ? '+' : '';
  return `${sign}${number.toFixed(digits)}`;
}

function barMoveValue(bar) {
  const pct = Number(bar?.pctChg);
  if (Number.isFinite(pct) && Math.abs(pct) > 1e-12) return pct;
  const close = Number(bar?.close);
  const open = Number(bar?.open);
  if (Number.isFinite(close) && Number.isFinite(open)) return close - open;
  return 0;
}

function barMoveColor(bar) {
  const move = barMoveValue(bar);
  if (move > 0) return '#ff335f';
  if (move < 0) return '#00b878';
  return '#64748b';
}

function priceCompareColor(value, previousClose) {
  const number = Number(value);
  const baseline = Number(previousClose);
  if (!Number.isFinite(number) || !Number.isFinite(baseline)) return '#64748b';
  if (number > baseline) return '#ff335f';
  if (number < baseline) return '#00b878';
  return '#64748b';
}

function previousCloseForIndex(index) {
  const bars = state.visualization.payload?.bars || [];
  const previous = bars[Math.max(0, Number(index) - 1)];
  const previousClose = Number(previous?.close);
  return Number.isFinite(previousClose) && Number(index) > 0 ? previousClose : null;
}

function barColorByPreviousClose(index, bar = null) {
  const currentBar = bar || state.visualization.payload?.bars?.[index];
  const previousClose = previousCloseForIndex(index);
  return priceCompareColor(currentBar?.close, previousClose);
}

function signedColor(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '#64748b';
  if (number > 0) return '#ff335f';
  if (number < 0) return '#00b878';
  return '#64748b';
}

function formatLegendDateTime(value) {
  const text = String(value || '-');
  return text === '-' ? '-' : text.slice(0, 16);
}

function legendItem(label, value, color = '#e5f0ff', extraClass = '') {
  const className = `legend-value ${extraClass}`.trim();
  return `
    <span class="legend-item">
      <span class="legend-key">${escapeHtml(label)}</span>
      <span class="${className}" style="color:${escapeHtml(color)}">${escapeHtml(value)}</span>
    </span>
  `;
}

function priceScaleForBars(bars, pad, height, extraValues = []) {
  const plotH = height - pad.top - pad.bottom;
  let minLow = Number.POSITIVE_INFINITY;
  let maxHigh = Number.NEGATIVE_INFINITY;
  bars.forEach(item => {
    const low = Number(item.low);
    const high = Number(item.high);
    if (Number.isFinite(low)) minLow = Math.min(minLow, low);
    if (Number.isFinite(high)) maxHigh = Math.max(maxHigh, high);
  });
  extraValues.forEach(value => {
    if (value === null || value === undefined || value === '') return;
    const number = Number(value);
    if (!Number.isFinite(number)) return;
    minLow = Math.min(minLow, number);
    maxHigh = Math.max(maxHigh, number);
  });
  if (!Number.isFinite(minLow) || !Number.isFinite(maxHigh)) {
    minLow = 0;
    maxHigh = 1;
  }
  const range = Math.max(0.000001, maxHigh - minLow);
  const minPrice = minLow - range * 0.08;
  const maxPrice = maxHigh + range * 0.08;
  const yFor = value => pad.top + ((maxPrice - value) / (maxPrice - minPrice)) * plotH;
  const valueForY = y => maxPrice - ((Math.max(pad.top, Math.min(height - pad.bottom, y)) - pad.top) / plotH) * (maxPrice - minPrice);
  return { minPrice, maxPrice, yFor, valueForY };
}

function volumeScaleForBars(bars, pad, height) {
  const plotH = height - pad.top - pad.bottom;
  const volumes = bars.map(item => Number(item.volume)).filter(Number.isFinite);
  const maxVolume = Math.max(1, ...volumes);
  const yFor = value => pad.top + plotH - (value / maxVolume) * plotH;
  const valueForY = y => ((height - pad.bottom - Math.max(pad.top, Math.min(height - pad.bottom, y))) / plotH) * maxVolume;
  return { maxVolume, yFor, valueForY };
}

function getVisualDbPath() {
  return ($('visualDbPath')?.value || DEFAULT_DB_PATH).trim();
}

function normalizeVisualSymbol(value) {
  const text = String(value || '').trim().toUpperCase();
  if (/^\d{6}$/.test(text)) {
    return /^[69]/.test(text) ? `${text}.SH` : `${text}.SZ`;
  }
  return text;
}

function getVisualSymbolValue() {
  return normalizeVisualSymbol($('visualSymbolInput')?.value || '');
}

function setVisualSymbolValue(symbol) {
  const input = $('visualSymbolInput');
  if (input) input.value = normalizeVisualSymbol(symbol);
}

async function apiGet(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok || data.ok === false) throw new Error(data.error || `GET ${url} failed`);
  return data;
}

async function apiPost(url, payload = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) throw new Error(data.error || `POST ${url} failed`);
  return data;
}

function showSection(sectionId) {
  document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
  $(sectionId).classList.add('active');
  document.querySelector(`[data-section="${sectionId}"]`).classList.add('active');
  document.body.dataset.section = sectionId;
  if (sectionId === 'visualization') {
    drawVisualizationPayload(state.visualization.payload);
  }
}

function parseSymbols(text) {
  return text.split(/\n|,/).map(s => s.trim().toUpperCase()).filter(Boolean);
}

function getCheckedValues(selector) {
  return [...document.querySelectorAll(selector)]
    .filter(item => item.checked)
    .map(item => item.value);
}

function setCheckedValues(selector, values) {
  const selected = new Set(values);
  document.querySelectorAll(selector).forEach(item => {
    item.checked = selected.has(item.value);
  });
}

function getDataDbPath() {
  return ($('dataDbPath')?.value || $('downloadDbPath')?.value || DEFAULT_DB_PATH).trim();
}

function setSharedDbPath(path) {
  const value = (path || DEFAULT_DB_PATH).trim();
  ['visualDbPath', 'dataDbPath', 'downloadDbPath', 'featureDbPath', 'configDefaultDbPath'].forEach(id => {
    if ($(id)) $(id).value = value;
  });
}

function normalizeOptionalValue(value) {
  if (!value || value === '-') return null;
  return value;
}

function frequencyRank(freq) {
  const index = FREQUENCY_ORDER.indexOf(String(freq || '').toLowerCase());
  return index >= 0 ? index : FREQUENCY_ORDER.length;
}

function compareFrequency(a, b) {
  return frequencyRank(a) - frequencyRank(b) || String(a || '').localeCompare(String(b || ''));
}

function sortFrequencies(values) {
  return [...values].sort(compareFrequency);
}

function compareInventorySlice(a, b) {
  return compareFrequency(a.freq, b.freq) || String(a.adjust || '').localeCompare(String(b.adjust || ''));
}

function inventoryKey(row) {
  return [
    String(row.symbol || '').toUpperCase(),
    String(row.freq || '-'),
    String(row.adjust || '-'),
  ].join('|');
}

function inventoryItemFromKey(key) {
  const [symbol, freq, adjust] = String(key || '').split('|');
  return {
    symbol: String(symbol || '').toUpperCase(),
    freq: normalizeOptionalValue(freq),
    adjust: normalizeOptionalValue(adjust),
  };
}

function getInventorySlices() {
  return (state.inventory || []).map(row => ({
    symbol: String(row.symbol || '').toUpperCase(),
    freq: String(row.freq || '').trim() || '-',
    adjust: row.adjust || '-',
    rows: Number(row.rows ?? 0) || 0,
    start_datetime: row.start_datetime || null,
    end_datetime: row.end_datetime || null,
  })).filter(row => row.symbol);
}

function compareDateText(a, b, pickMin = true) {
  if (!a) return b || null;
  if (!b) return a || null;
  return pickMin ? (String(a) <= String(b) ? a : b) : (String(a) >= String(b) ? a : b);
}

function getInventoryGroups() {
  const groups = new Map();
  for (const row of state.inventory) {
    const symbol = String(row.symbol || '').toUpperCase();
    if (!symbol) continue;
    if (!groups.has(symbol)) {
      groups.set(symbol, {
        symbol,
        total_rows: 0,
        daily_rows: 0,
        start_datetime: null,
        end_datetime: null,
        freqs: [],
        slices: [],
      });
    }
    const group = groups.get(symbol);
    const freq = String(row.freq || '').trim() || '-';
    const sliceRows = Number(row.rows ?? 0) || 0;
    const slice = {
      symbol,
      freq,
      adjust: row.adjust || '-',
      rows: sliceRows,
      start_datetime: row.start_datetime || null,
      end_datetime: row.end_datetime || null,
    };
    group.slices.push(slice);
    group.total_rows += sliceRows;
    if (freq === 'daily') group.daily_rows += sliceRows;
    group.start_datetime = compareDateText(group.start_datetime, slice.start_datetime, true);
    group.end_datetime = compareDateText(group.end_datetime, slice.end_datetime, false);
    if (slice.freq && !group.freqs.includes(slice.freq)) group.freqs.push(slice.freq);
  }
  for (const group of groups.values()) {
    group.freqs = sortFrequencies(group.freqs);
    group.slices.sort(compareInventorySlice);
  }
  return [...groups.values()].sort((a, b) => a.symbol.localeCompare(b.symbol));
}

function renderSummaryCards(elementId, items) {
  $(elementId).innerHTML = items.map(item => `
    <div><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>
  `).join('');
}

function renderInventorySummary(data) {
  const range = data.start_date && data.end_date ? `${data.start_date} -> ${data.end_date}` : '-';
  const storageMode = data.storage_mode || 'duckdb';
  const storagePath = storageMode === 'shard' ? (data.storage_root || '-') : (data.db_path || '-');
  renderSummaryCards('inventorySummary', [
    { label: 'Storage', value: storageMode },
    { label: storageMode === 'shard' ? 'Storage Root' : 'Database', value: storagePath },
    { label: storageMode === 'shard' ? 'Catalog Exists' : 'DB Exists', value: data.db_exists ? 'yes' : 'no' },
    { label: 'Symbols', value: data.total_symbols ?? 0 },
    { label: 'Total Rows', value: data.total_rows ?? 0 },
    { label: 'Date Range', value: range },
  ]);

  const entries = Object.entries(data.freq_summary || {}).sort(([a], [b]) => compareFrequency(a, b));
  $('freqSummary').innerHTML = entries.length
    ? entries.map(([freq, item]) => `<span class="freq-chip">${escapeHtml(freq)} / ${escapeHtml(item.symbols)} symbols / ${escapeHtml(item.rows)} rows</span>`).join('')
    : '<span class="muted">No frequency slices.</span>';
}

function getInventoryAdjustFilterValue() {
  const control = $('inventoryAdjustFilter');
  return control ? control.value.trim() : '';
}

function inventorySliceMatchesFilters(row, { keyword = '', freqFilter = '', adjustFilter = '' } = {}) {
  const symbolOk = !keyword || String(row.symbol || '').toUpperCase().includes(keyword);
  const freqOk = !freqFilter || String(row.freq || '') === freqFilter;
  const adjustOk = !adjustFilter || String(row.adjust || '') === adjustFilter;
  return symbolOk && freqOk && adjustOk;
}

function getFilteredInventoryRows() {
  const keyword = $('inventorySearch').value.trim().toUpperCase();
  const freqFilter = $('inventoryFreqFilter').value.trim();
  const adjustFilter = getInventoryAdjustFilterValue();
  return getInventorySlices().filter(row => (
    inventorySliceMatchesFilters(row, { keyword, freqFilter, adjustFilter })
  )).sort((a, b) => (
    a.symbol.localeCompare(b.symbol)
    || compareFrequency(a.freq, b.freq)
    || String(a.adjust).localeCompare(String(b.adjust))
  ));
}

function getFilteredInventoryGroups() {
  const keyword = $('inventorySearch').value.trim().toUpperCase();
  const freqFilter = $('inventoryFreqFilter').value.trim();
  const adjustFilter = getInventoryAdjustFilterValue();
  return getInventoryGroups().filter(group => {
    const symbolOk = !keyword || String(group.symbol || '').toUpperCase().includes(keyword);
    const sliceOk = (group.slices || []).some(slice => (
      inventorySliceMatchesFilters(slice, { freqFilter, adjustFilter })
    ));
    return symbolOk && sliceOk;
  });
}

function getDisplaySlicesForGroup(group) {
  const freqFilter = $('inventoryFreqFilter').value.trim();
  const adjustFilter = getInventoryAdjustFilterValue();
  return (group.slices || [])
    .filter(slice => inventorySliceMatchesFilters(slice, { freqFilter, adjustFilter }))
    .sort(compareInventorySlice);
}

function getVisibleInventorySliceRows() {
  if (state.inventoryViewMode === 'slice') return getFilteredInventoryRows();
  const rows = [];
  for (const group of getFilteredInventoryGroups()) {
    if (!state.expandedInventorySymbols.has(group.symbol)) continue;
    rows.push(...getDisplaySlicesForGroup(group));
  }
  return rows;
}

function renderFreqTags(group) {
  const displaySlices = getDisplaySlicesForGroup(group);
  const slices = displaySlices.slice(0, 8);
  const extra = displaySlices.length - slices.length;
  return [
    ...slices.map(slice => `<span class="freq-badge">${escapeHtml(slice.freq || '-')} / ${escapeHtml(slice.adjust || '-')}</span>`),
    extra > 0 ? `<span class="freq-badge muted-badge">+${extra}</span>` : '',
  ].join('');
}

function renderInventoryTable() {
  renderInventoryModeControls();
  if (state.inventoryViewMode === 'slice') {
    renderInventorySliceTable();
    return;
  }
  renderInventoryGroupTable();
}

function renderInventoryModeControls() {
  document.querySelectorAll('[data-inventory-mode]').forEach(button => {
    button.classList.toggle('active', button.dataset.inventoryMode === state.inventoryViewMode);
  });
}

function renderInventorySliceTable() {
  const rows = getFilteredInventoryRows();
  if (!rows.length) {
    $('inventoryTableHead').innerHTML = inventorySliceHeadHtml();
    bindSelectAllInventory();
    $('inventoryTableBody').innerHTML = '<tr><td colspan="7" class="muted">No data.</td></tr>';
    $('selectAllInventory').checked = false;
    return;
  }

  $('inventoryTableHead').innerHTML = inventorySliceHeadHtml();
  bindSelectAllInventory();
  $('inventoryTableBody').innerHTML = rows.map(row => {
    const key = inventoryKey(row);
    const checked = state.selectedInventoryKeys.has(key) ? 'checked' : '';
    return inventorySliceRowHtml(row, checked);
  }).join('');

  const visibleKeys = rows.map(inventoryKey);
  $('selectAllInventory').checked = visibleKeys.length > 0 && visibleKeys.every(key => state.selectedInventoryKeys.has(key));
}

function renderInventoryGroupTable() {
  const groups = getFilteredInventoryGroups();
  $('inventoryTableHead').innerHTML = inventoryGroupHeadHtml();
  bindSelectAllInventory();
  if (!groups.length) {
    $('inventoryTableBody').innerHTML = '<tr><td colspan="7" class="muted">No data.</td></tr>';
    $('selectAllInventory').checked = false;
    return;
  }

  $('inventoryTableBody').innerHTML = groups.map(group => {
    const displaySlices = getDisplaySlicesForGroup(group);
    const expanded = state.expandedInventorySymbols.has(group.symbol);
    const expandedHtml = expanded ? inventoryExpandedSliceRowsHtml(group, displaySlices) : '';
    return `
      <tr class="inventory-group-row">
        <td></td>
        <td><strong>${escapeHtml(group.symbol)}</strong></td>
        <td><div class="freq-tag-list">${renderFreqTags(group) || '<span class="muted">No matching slices</span>'}</div></td>
        <td>${escapeHtml(group.total_rows ?? 0)}</td>
        <td>${escapeHtml(String(group.start_datetime ?? '-').slice(0, 10))}</td>
        <td>${escapeHtml(String(group.end_datetime ?? '-').slice(0, 10))}</td>
        <td class="row-actions">
          <button class="secondary small" data-action="detail" data-symbol="${escapeHtml(group.symbol)}">View</button>
          <button class="secondary small" data-action="manage" data-symbol="${escapeHtml(group.symbol)}">${expanded ? 'Hide' : 'Manage'}</button>
        </td>
      </tr>
      ${expandedHtml}
    `;
  }).join('');

  const visibleKeys = getVisibleInventorySliceRows().map(inventoryKey);
  $('selectAllInventory').checked = visibleKeys.length > 0 && visibleKeys.every(key => state.selectedInventoryKeys.has(key));
}

function inventorySliceHeadHtml() {
  return `
    <tr>
      <th><input id="selectAllInventory" type="checkbox" /></th>
      <th>Symbol</th>
      <th>Freq / Adjust</th>
      <th>Rows</th>
      <th>Start</th>
      <th>End</th>
      <th>Actions</th>
    </tr>
  `;
}

function inventoryGroupHeadHtml() {
  return `
    <tr>
      <th><input id="selectAllInventory" type="checkbox" /></th>
      <th>Symbol</th>
      <th>Available Slices</th>
      <th>Total Rows</th>
      <th>Start</th>
      <th>End</th>
      <th>Actions</th>
    </tr>
  `;
}

function inventorySliceRowHtml(row, checked = '') {
  const key = inventoryKey(row);
  return `
    <tr class="inventory-slice-row">
      <td><input class="inventory-select" type="checkbox" data-key="${escapeHtml(key)}" ${checked} /></td>
      <td><strong>${escapeHtml(row.symbol)}</strong></td>
      <td><span class="freq-badge">${escapeHtml(row.freq || '-')} / ${escapeHtml(row.adjust || '-')}</span></td>
      <td>${escapeHtml(row.rows ?? 0)}</td>
      <td>${escapeHtml(String(row.start_datetime ?? '-').slice(0, 10))}</td>
      <td>${escapeHtml(String(row.end_datetime ?? '-').slice(0, 10))}</td>
      <td class="row-actions">
        <button class="secondary small" data-action="detail" data-symbol="${escapeHtml(row.symbol)}" data-freq="${escapeHtml(row.freq)}" data-adjust="${escapeHtml(row.adjust)}">View</button>
        <button class="secondary small danger-text" data-action="delete" data-symbol="${escapeHtml(row.symbol)}" data-freq="${escapeHtml(row.freq)}" data-adjust="${escapeHtml(row.adjust)}">Delete</button>
      </td>
    </tr>
  `;
}

function inventoryExpandedSliceRowsHtml(group, slices) {
  if (!slices.length) {
    return `
      <tr class="inventory-expanded-row">
        <td></td>
        <td colspan="6" class="muted">No matching slices for ${escapeHtml(group.symbol)}.</td>
      </tr>
    `;
  }
  return slices.map(slice => {
    const key = inventoryKey(slice);
    const checked = state.selectedInventoryKeys.has(key) ? 'checked' : '';
    return inventorySliceRowHtml(slice, checked).replace(
      '<tr class="inventory-slice-row">',
      '<tr class="inventory-slice-row inventory-expanded-row">'
    );
  }).join('');
}

function bindSelectAllInventory() {
  const checkbox = $('selectAllInventory');
  if (checkbox) checkbox.addEventListener('change', toggleSelectAllInventory);
}

function getVisualizationGroup() {
  const symbol = getVisualSymbolValue();
  return getInventoryGroups().find(group => group.symbol === symbol) || null;
}

function getCurrentVisualizationFreq() {
  return state.visualization.selectedFreq || 'daily';
}

function getVisualizationFreqs(group) {
  const values = [];
  for (const slice of group?.slices || []) {
    const freq = slice.freq || '-';
    if (!values.includes(freq)) values.push(freq);
  }
  return sortFrequencies(values);
}

function getVisualizationAdjusts(group, freq) {
  const values = [];
  for (const slice of group?.slices || []) {
    if ((slice.freq || '-') !== freq) continue;
    const adjust = slice.adjust || '-';
    if (!values.includes(adjust)) values.push(adjust);
  }
  return values;
}

function getVisualizationSlice() {
  const group = getVisualizationGroup();
  const freq = getCurrentVisualizationFreq();
  const adjust = $('visualAdjustSelect')?.value || null;
  return (group?.slices || []).find(item => (item.freq || '-') === freq && (item.adjust || '-') === adjust) || null;
}

function setVisualizationQuote(payload = null) {
  const summary = payload?.summary || {};
  const symbol = payload?.symbol || getVisualSymbolValue() || '-';
  const price = summary.latest_close;
  const change = summary.change;
  const pct = summary.pctChg;
  const changeNumber = Number(change || 0);
  const quoteClass = changeNumber > 0 ? 'up' : (changeNumber < 0 ? 'down' : 'flat');

  $('visualQuoteSymbol').textContent = symbol;
  $('visualQuotePrice').textContent = formatNumber(price, 3);
  $('visualQuotePrice').className = `quote-price ${quoteClass}`;
  $('visualQuoteChange').textContent = `${formatSignedNumber(change, 3)} ${formatSignedPercent(pct)}`;
  $('visualQuoteChange').className = `quote-change ${quoteClass}`;
  $('visualPriceLegend').innerHTML = payload?.rows ? priceLegendHtml(payload) : '-';
}

function renderVisualizationControls({ keepSymbol = true, keepFreq = true } = {}) {
  if (!$('visualSymbolInput')) return;

  const groups = getInventoryGroups();
  const symbolInput = $('visualSymbolInput');
  const symbolOptions = $('visualSymbolOptions');
  const adjustSelect = $('visualAdjustSelect');
  const previousSymbol = normalizeVisualSymbol(symbolInput.value);
  const previousFreq = getCurrentVisualizationFreq();
  const previousAdjust = adjustSelect.value;

  if (symbolOptions) {
    symbolOptions.innerHTML = groups
      .map(group => `<option value="${escapeHtml(group.symbol)}">${escapeHtml(group.total_rows)} rows</option>`)
      .join('');
  }

  const selectedSymbol = keepSymbol && previousSymbol
    ? previousSymbol
    : (groups[0]?.symbol || '');
  setVisualSymbolValue(selectedSymbol);

  const group = getVisualizationGroup();
  const freqs = group ? getVisualizationFreqs(group) : VISUAL_DEFAULT_FREQS;
  const selectedFreq = keepFreq && freqs.includes(previousFreq)
    ? previousFreq
    : preferDetailFreq(freqs);
  state.visualization.selectedFreq = selectedFreq || freqs[0] || 'daily';
  renderVisualizationFreqButtons(freqs);

  const adjusts = group ? getVisualizationAdjusts(group, getCurrentVisualizationFreq()) : VISUAL_DEFAULT_ADJUSTS;
  const selectedAdjust = adjusts.includes(previousAdjust)
    ? previousAdjust
    : preferDetailAdjust(adjusts);
  adjustSelect.innerHTML = adjusts.map(adjust => `<option value="${escapeHtml(adjust)}">${escapeHtml(adjust)}</option>`).join('');
  adjustSelect.value = selectedAdjust || '';

  setVisualizationQuote(state.visualization.payload);
  if (!selectedSymbol) setVisualStatus('No local data.');
}

function renderVisualizationFreqButtons(availableFreqs) {
  document.querySelectorAll('#visualFreqButtons button[data-freq]').forEach(button => {
    const freq = button.dataset.freq;
    button.disabled = !availableFreqs.includes(freq);
    button.classList.toggle('active', freq === getCurrentVisualizationFreq());
  });
}

function setVisualStatus(message) {
  $('visualChartStatus').textContent = message || '';
  $('visualChartStatus').classList.toggle('hidden', !message);
}

function resetVisualizationView(bars, anchor = 'latest', preferredWindow = null) {
  const count = bars?.length || 0;
  const preferred = Math.min(count, preferredWindow || 120);
  if (anchor === 'start') {
    state.visualization.viewStart = 0;
    state.visualization.viewEnd = preferred;
  } else {
    state.visualization.viewEnd = count;
    state.visualization.viewStart = Math.max(0, count - preferred);
  }
  state.visualization.hoverIndex = null;
  state.visualization.hoverPanel = null;
  state.visualization.referenceGuides = emptyVisualizationReferenceGuides();
  clearVisualizationHover();
}

function clampVisualizationView(start, end) {
  const bars = state.visualization.payload?.bars || [];
  const count = bars.length;
  if (!count) {
    state.visualization.viewStart = 0;
    state.visualization.viewEnd = 0;
    return;
  }
  const minWindow = Math.min(count, 20);
  const windowSize = Math.max(minWindow, Math.min(count, Math.round(end - start)));
  let nextStart = Math.round(start);
  if (nextStart < 0) nextStart = 0;
  if (nextStart + windowSize > count) nextStart = Math.max(0, count - windowSize);
  state.visualization.viewStart = nextStart;
  state.visualization.viewEnd = nextStart + windowSize;
}

function getVisibleBars() {
  const bars = state.visualization.payload?.bars || [];
  const start = Math.max(0, state.visualization.viewStart || 0);
  const end = Math.min(bars.length, state.visualization.viewEnd || bars.length);
  return bars.slice(start, end);
}

function getChartGeometry(canvas) {
  const rect = canvas.getBoundingClientRect();
  return {
    rect,
    pad: { left: 46, right: 66, top: 18, bottom: 26 },
    width: rect.width,
    height: rect.height,
  };
}

function getHoverIndexFromClientX(clientX, canvas = $('visualPriceCanvas')) {
  const bars = getVisibleBars();
  if (!canvas || !bars.length) return null;
  const { rect, pad, width } = getChartGeometry(canvas);
  const plotW = Math.max(1, width - pad.left - pad.right);
  const x = Math.max(pad.left, Math.min(width - pad.right, clientX - rect.left));
  const step = plotW / Math.max(1, bars.length);
  const visibleIndex = Math.max(0, Math.min(bars.length - 1, Math.floor((x - pad.left) / step)));
  return state.visualization.viewStart + visibleIndex;
}

function getVisibleHoverIndex(bars) {
  const hoverIndex = state.visualization.hoverIndex;
  if (hoverIndex === null || hoverIndex === undefined || !bars.length) return -1;
  const visibleIndex = hoverIndex - state.visualization.viewStart;
  return visibleIndex >= 0 && visibleIndex < bars.length ? visibleIndex : -1;
}

function getHoverPanelFromTarget(target) {
  if (target === $('visualPriceCanvas')) return 'price';
  const slot = target?.dataset?.panelSlot;
  if (slot !== undefined) return state.visualization.panelSelections[Number(slot)] || 'none';
  return null;
}

function getVisualizationCanvasFromTarget(target) {
  if (!target) return null;
  if (target.tagName === 'CANVAS') return target;
  return target.closest?.('#visualChartGrid canvas') || null;
}

function emptyVisualizationReferenceGuides() {
  return {
    active: false,
    panel: null,
    price: null,
    volume: null,
  };
}

function hasVisualizationReferenceGuide() {
  return Boolean(state.visualization.referenceGuides?.active);
}

function clearPendingVisualizationReferenceClick() {
  if (!visualReferenceClickTimer) return;
  window.clearTimeout(visualReferenceClickTimer);
  visualReferenceClickTimer = null;
}

function clearVisualizationReferenceGuide(redraw = true) {
  clearPendingVisualizationReferenceClick();
  state.visualization.referenceGuides = emptyVisualizationReferenceGuides();
  if (redraw) drawVisualizationPayload(state.visualization.payload);
}

function resetVisualizationLoadingStrategy() {
  const visualization = state.visualization;
  visualization.prefetching = null;
  visualization.beforeLoadSize = VISUAL_WINDOW_BARS;
  visualization.afterLoadSize = VISUAL_WINDOW_BARS;
  visualization.beforeTriggerAbsolute = null;
  visualization.afterTriggerAbsolute = null;
  visualization.pendingZoom = null;
}

function waitForVisualizationPaint() {
  return new Promise(resolve => window.requestAnimationFrame(() => resolve()));
}

function getVisualizationIndicators(payload = state.visualization.payload) {
  return payload?.features?.indicators || [];
}

function readVisualizationPreferences() {
  try {
    return JSON.parse(window.localStorage.getItem('pocketagent.visualization.features') || 'null');
  } catch (_err) {
    return null;
  }
}

function persistVisualizationPreferences() {
  const visualization = state.visualization;
  visualization.panelSelectionsByFreq[getCurrentVisualizationFreq()] = [...visualization.panelSelections];
  try {
    window.localStorage.setItem('pocketagent.visualization.features', JSON.stringify({
      mainOverlays: [...visualization.mainOverlays],
      panelsByFreq: visualization.panelSelectionsByFreq,
      inspectorVisible: visualization.inspectorVisible,
    }));
  } catch (_err) {
    // Local preferences are optional; chart behavior must not depend on storage.
  }
}

function defaultVisualizationPanels(indicators) {
  const preferred = ['volume'];
  indicators
    .filter(item => item.render_target === 'sub_panel' && item.default_visible)
    .forEach(item => preferred.push(item.id));
  indicators
    .filter(item => item.render_target === 'sub_panel')
    .forEach(item => preferred.push(item.id));
  return [...new Set(preferred)].slice(0, 3).concat(['none', 'none', 'none']).slice(0, 3);
}

function initializeVisualizationFeatureControls(payload) {
  const visualization = state.visualization;
  const indicators = getVisualizationIndicators(payload);
  if (!visualization.displayInitialized) {
    const saved = readVisualizationPreferences();
    const defaults = indicators
      .filter(item => item.render_target === 'main_overlay' && item.default_visible)
      .map(item => item.id);
    visualization.mainOverlayPreferenceExplicit = Array.isArray(saved?.mainOverlays);
    visualization.mainOverlays = new Set(saved?.mainOverlays || defaults);
    visualization.panelSelectionsByFreq = saved?.panelsByFreq || {};
    visualization.inspectorVisible = saved?.inspectorVisible !== false;
    visualization.displayInitialized = true;
  }
  if (!visualization.mainOverlayPreferenceExplicit) {
    indicators
      .filter(item => item.render_target === 'main_overlay' && item.default_visible)
      .forEach(item => visualization.mainOverlays.add(item.id));
  }

  const freq = getCurrentVisualizationFreq();
  const availablePanels = new Set([
    'none',
    'volume',
    ...indicators.filter(item => item.render_target === 'sub_panel').map(item => item.id),
  ]);
  const savedPanels = visualization.panelSelectionsByFreq[freq];
  const candidate = Array.isArray(savedPanels) ? savedPanels : defaultVisualizationPanels(indicators);
  const used = new Set();
  visualization.panelSelections = candidate.map(value => {
    const selected = availablePanels.has(value) && value !== 'none' && !used.has(value) ? value : 'none';
    if (selected !== 'none') used.add(selected);
    return selected;
  }).concat(['none', 'none', 'none']).slice(0, 3);
  visualization.panelSelectionsByFreq[freq] = [...visualization.panelSelections];
  renderVisualizationIndicatorMenu(indicators);
  renderVisualizationPanelSelectors(indicators);
  $('visualFeatureInspector')?.classList.toggle('hidden', !visualization.inspectorVisible);
}

function renderVisualizationIndicatorMenu(indicators = getVisualizationIndicators()) {
  const menu = $('visualIndicatorMenu');
  if (!menu) return;
  const overlays = indicators.filter(item => item.render_target === 'main_overlay');
  menu.innerHTML = overlays.length ? `
    <strong>Main overlays</strong>
    ${overlays.map(item => `
      <label class="visual-indicator-option">
        <input type="checkbox" data-overlay-id="${escapeHtml(item.id)}" ${state.visualization.mainOverlays.has(item.id) ? 'checked' : ''} />
        <span>${escapeHtml(item.label)}</span>
      </label>
    `).join('')}
  ` : '<span class="muted">No overlays for this frequency.</span>';
}

function renderVisualizationPanelSelectors(indicators = getVisualizationIndicators()) {
  const panelIndicators = indicators.filter(item => item.render_target === 'sub_panel');
  const options = [
    { id: 'none', label: 'None' },
    { id: 'volume', label: 'VOL' },
    ...panelIndicators.map(item => ({ id: item.id, label: item.label })),
  ];
  document.querySelectorAll('.visualPanelSelect').forEach(select => {
    const slot = Number(select.dataset.panelSlot || 0);
    select.innerHTML = options.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join('');
    select.value = state.visualization.panelSelections[slot] || 'none';
  });
}

async function loadVisualizationChart(options = {}) {
  const symbol = getVisualSymbolValue();
  setVisualSymbolValue(symbol);
  const adjust = $('visualAdjustSelect')?.value;
  const freq = getCurrentVisualizationFreq();
  if (!symbol || !adjust || !freq) {
    state.visualization.abortController?.abort();
    state.visualization.loadGeneration += 1;
    state.visualization.loading = false;
    resetVisualizationLoadingStrategy();
    state.visualization.windowOffset = 0;
    state.visualization.totalRows = 0;
    state.visualization.payload = null;
    setVisualizationQuote();
    drawVisualizationPayload(null);
    return;
  }

  state.visualization.abortController?.abort();
  const controller = new AbortController();
  const generation = state.visualization.loadGeneration + 1;
  state.visualization.abortController = controller;
  state.visualization.loadGeneration = generation;
  resetVisualizationLoadingStrategy();
  state.visualization.loading = true;
  setVisualStatus('Loading...');
  try {
    const db = encodeURIComponent(getVisualDbPath());
    const limit = encodeURIComponent(String(VISUAL_WINDOW_BARS));
    const url = `/api/visualization/kline?db=${db}&symbol=${encodeURIComponent(symbol)}&freq=${encodeURIComponent(freq)}&adjust=${encodeURIComponent(adjust)}&limit=${limit}`;
    const res = await apiGet(url, { signal: controller.signal });
    if (generation !== state.visualization.loadGeneration) return;
    state.visualization.payload = res.data;
    state.visualization.windowOffset = Number(res.data.offset || 0);
    state.visualization.totalRows = Number(res.data.total_rows || 0);
    initializeVisualizationFeatureControls(res.data);
    resetVisualizationView(res.data.bars || [], options.anchor || 'latest', options.preferredWindow || null);
    setVisualizationQuote(res.data);
    drawVisualizationPayload(res.data);
    setVisualStatus(res.data.rows ? '' : 'No rows for this slice.');
  } catch (err) {
    if (err.name === 'AbortError') return;
    setVisualStatus(`Load failed: ${err.message}`);
    state.visualization.windowOffset = 0;
    state.visualization.totalRows = 0;
    state.visualization.payload = null;
    drawVisualizationPayload(null);
  } finally {
    if (generation === state.visualization.loadGeneration) {
      state.visualization.loading = false;
      if (state.visualization.payload?.bars?.length) {
        void primeVisualizationHistory(generation);
      }
    }
  }
}

function visualizationWindowUrl(offset, limit) {
  const db = encodeURIComponent(getVisualDbPath());
  const symbol = encodeURIComponent(getVisualSymbolValue());
  const freq = encodeURIComponent(getCurrentVisualizationFreq());
  const adjust = encodeURIComponent($('visualAdjustSelect')?.value || 'none');
  return `/api/visualization/kline?db=${db}&symbol=${symbol}&freq=${freq}&adjust=${adjust}&offset=${encodeURIComponent(String(offset))}&limit=${encodeURIComponent(String(limit))}`;
}

function mergeAlignedFeatureValues(currentValues, incomingValues, context) {
  const merged = new Array(context.mergedEnd - context.mergedOffset).fill(null);
  (currentValues || []).forEach((value, index) => {
    merged[context.currentOffset + index - context.mergedOffset] = value;
  });
  (incomingValues || []).forEach((value, index) => {
    merged[context.incomingOffset + index - context.mergedOffset] = value;
  });
  const trimStart = context.nextOffset - context.mergedOffset;
  return merged.slice(trimStart, trimStart + context.finalLength);
}

function mergeFeatureSeries(currentSeries, incomingSeries, context) {
  const currentById = new Map((currentSeries || []).map(item => [item.id, item]));
  const incomingById = new Map((incomingSeries || []).map(item => [item.id, item]));
  const ids = [...new Set([...currentById.keys(), ...incomingById.keys()])];
  return ids.map(id => {
    const current = currentById.get(id);
    const incoming = incomingById.get(id);
    const descriptor = { ...(current || {}), ...(incoming || {}) };
    if (current?.values || incoming?.values) {
      descriptor.values = mergeAlignedFeatureValues(current?.values, incoming?.values, context);
    }
    return descriptor;
  });
}

function mergeVisualizationFeatures(currentFeatures, incomingFeatures, context) {
  if (!currentFeatures?.indicators?.length) return incomingFeatures || currentFeatures;
  if (!incomingFeatures?.indicators?.length) return currentFeatures;
  const currentById = new Map(currentFeatures.indicators.map(item => [item.id, item]));
  const incomingById = new Map(incomingFeatures.indicators.map(item => [item.id, item]));
  const ids = [...new Set([...currentById.keys(), ...incomingById.keys()])];
  const indicators = ids.map(id => {
    const current = currentById.get(id);
    const incoming = incomingById.get(id);
    const descriptor = { ...(current || {}), ...(incoming || {}) };
    descriptor.model_series = mergeFeatureSeries(current?.model_series, incoming?.model_series, context);
    descriptor.display_series = mergeFeatureSeries(current?.display_series, incoming?.display_series, context);
    descriptor.coverage = mergeAlignedFeatureValues(current?.coverage, incoming?.coverage, context);
    return descriptor;
  });
  return {
    ...currentFeatures,
    ...incomingFeatures,
    status: 'ready',
    offset: context.nextOffset,
    rows: context.finalLength,
    warmup_rows: Math.max(Number(currentFeatures.warmup_rows || 0), Number(incomingFeatures.warmup_rows || 0)),
    indicators,
  };
}

function mergeVisualizationWindow(incoming, direction) {
  const payload = state.visualization.payload;
  const currentBars = payload?.bars || [];
  const incomingBars = incoming?.bars || [];
  if (!payload || !incomingBars.length) return false;

  const currentOffset = state.visualization.windowOffset;
  const incomingOffset = Number(incoming.offset || 0);
  const mergedOffset = Math.min(currentOffset, incomingOffset);
  const mergedEnd = Math.max(currentOffset + currentBars.length, incomingOffset + incomingBars.length);
  const merged = new Array(mergedEnd - mergedOffset);
  currentBars.forEach((bar, index) => { merged[currentOffset + index - mergedOffset] = bar; });
  incomingBars.forEach((bar, index) => { merged[incomingOffset + index - mergedOffset] = bar; });
  if (merged.some(bar => !bar)) return false;

  const oldViewStartAbsolute = currentOffset + state.visualization.viewStart;
  const oldViewEndAbsolute = currentOffset + state.visualization.viewEnd;
  const oldHoverAbsolute = state.visualization.hoverIndex === null
    ? null
    : currentOffset + state.visualization.hoverIndex;
  const oldDragStartAbsolute = state.visualization.drag.active
    ? currentOffset + state.visualization.drag.startViewStart
    : null;
  const oldDragEndAbsolute = state.visualization.drag.active
    ? currentOffset + state.visualization.drag.startViewEnd
    : null;

  let bars = merged;
  let nextOffset = mergedOffset;
  if (bars.length > VISUAL_MAX_BUFFER_BARS) {
    const trim = bars.length - VISUAL_MAX_BUFFER_BARS;
    if (direction === 'after') {
      bars = bars.slice(trim);
      nextOffset += trim;
      state.visualization.beforeLoadSize = VISUAL_WINDOW_BARS;
      state.visualization.beforeTriggerAbsolute = nextOffset + VISUAL_WINDOW_BARS;
    } else {
      bars = bars.slice(0, VISUAL_MAX_BUFFER_BARS);
      state.visualization.afterLoadSize = VISUAL_WINDOW_BARS;
      state.visualization.afterTriggerAbsolute = nextOffset + bars.length - VISUAL_WINDOW_BARS;
    }
  }

  state.visualization.windowOffset = nextOffset;
  state.visualization.totalRows = Number(incoming.total_rows || payload.total_rows || bars.length);
  state.visualization.viewStart = Math.max(0, oldViewStartAbsolute - nextOffset);
  state.visualization.viewEnd = Math.min(bars.length, oldViewEndAbsolute - nextOffset);
  const nextHoverIndex = oldHoverAbsolute === null ? null : oldHoverAbsolute - nextOffset;
  state.visualization.hoverIndex = nextHoverIndex !== null && nextHoverIndex >= 0 && nextHoverIndex < bars.length
    ? nextHoverIndex
    : null;
  if (state.visualization.drag.active) {
    state.visualization.drag.startViewStart = oldDragStartAbsolute - nextOffset;
    state.visualization.drag.startViewEnd = oldDragEndAbsolute - nextOffset;
  }

  const mergedFeatures = mergeVisualizationFeatures(payload.features, incoming.features, {
    currentOffset,
    incomingOffset,
    mergedOffset,
    mergedEnd,
    nextOffset,
    finalLength: bars.length,
  });

  state.visualization.payload = {
    ...payload,
    offset: nextOffset,
    rows: bars.length,
    total_rows: state.visualization.totalRows,
    has_more_before: nextOffset > 0,
    has_more_after: nextOffset + bars.length < state.visualization.totalRows,
    start_datetime: bars[0]?.datetime || null,
    end_datetime: bars[bars.length - 1]?.datetime || null,
    bars,
    features: mergedFeatures,
  };
  return true;
}

function applyPendingVisualizationZoom() {
  const pending = state.visualization.pendingZoom;
  const bars = state.visualization.payload?.bars || [];
  if (!pending || !bars.length) return;
  const nextWindow = Math.min(VISUAL_MAX_VISIBLE_BARS, bars.length, pending.windowSize);
  const anchorLocal = pending.anchorAbsolute - state.visualization.windowOffset;
  const nextStart = anchorLocal - nextWindow * pending.anchorRatio;
  clampVisualizationView(nextStart, nextStart + nextWindow);
  if (nextWindow >= pending.windowSize || bars.length >= VISUAL_MAX_VISIBLE_BARS) {
    state.visualization.pendingZoom = null;
  }
}

async function primeVisualizationHistory(generation) {
  await waitForVisualizationPaint();
  if (generation !== state.visualization.loadGeneration) return;
  await prefetchVisualizationDirection('before', true);
}

async function prefetchVisualizationDirection(direction, force = false) {
  const visualization = state.visualization;
  const payload = visualization.payload;
  const bars = payload?.bars || [];
  if (!bars.length || visualization.prefetching || visualization.loading) return false;
  if (direction === 'before' && !payload.has_more_before) return false;
  if (direction === 'after' && !payload.has_more_after) return false;

  const currentOffset = visualization.windowOffset;
  const totalRows = visualization.totalRows;
  const requestedSize = direction === 'before'
    ? visualization.beforeLoadSize
    : visualization.afterLoadSize;
  const offset = direction === 'before'
    ? Math.max(0, currentOffset - requestedSize)
    : currentOffset + bars.length;
  const limit = direction === 'before'
    ? currentOffset - offset
    : Math.min(requestedSize, totalRows - offset);
  if (limit <= 0) return false;

  const generation = visualization.loadGeneration;
  const oldBoundary = direction === 'before' ? currentOffset : currentOffset + bars.length;
  visualization.prefetching = direction;
  let loaded = false;
  try {
    if (!force) await waitForVisualizationPaint();
    const res = await apiGet(visualizationWindowUrl(offset, limit), {
      signal: visualization.abortController?.signal,
    });
    if (generation !== visualization.loadGeneration) return false;
    loaded = mergeVisualizationWindow(res.data, direction);
    if (!loaded) return false;
    if (direction === 'before') {
      visualization.beforeTriggerAbsolute = oldBoundary;
      visualization.beforeLoadSize = Math.min(VISUAL_MAX_FETCH_BARS, requestedSize * 2);
    } else {
      visualization.afterTriggerAbsolute = oldBoundary;
      visualization.afterLoadSize = Math.min(VISUAL_MAX_FETCH_BARS, requestedSize * 2);
    }
    applyPendingVisualizationZoom();
    drawVisualizationPayload(state.visualization.payload);
    return true;
  } catch (err) {
    if (err.name !== 'AbortError') console.warn('Visualization prefetch failed:', err);
    return false;
  } finally {
    if (generation === visualization.loadGeneration) {
      visualization.prefetching = null;
      if (loaded) void maybePrefetchVisualization();
    }
  }
}

async function maybePrefetchVisualization() {
  const visualization = state.visualization;
  const payload = visualization.payload;
  const bars = payload?.bars || [];
  if (!bars.length || visualization.prefetching || visualization.loading) return;

  const absoluteStart = visualization.windowOffset + visualization.viewStart;
  const absoluteEnd = visualization.windowOffset + visualization.viewEnd;
  const needsBefore = payload.has_more_before
    && Number.isFinite(visualization.beforeTriggerAbsolute)
    && absoluteStart <= visualization.beforeTriggerAbsolute;
  const needsAfter = payload.has_more_after
    && Number.isFinite(visualization.afterTriggerAbsolute)
    && absoluteEnd >= visualization.afterTriggerAbsolute;
  if (!needsBefore && !needsAfter) return;

  const direction = needsBefore && needsAfter
    ? (absoluteStart - visualization.windowOffset <= visualization.windowOffset + bars.length - absoluteEnd ? 'before' : 'after')
    : (needsBefore ? 'before' : 'after');
  await prefetchVisualizationDirection(direction);
}

function drawVisualizationPayload(payload) {
  const bars = payload ? getVisibleBars() : [];
  updateVisualizationPanelVisibility();
  drawPriceCanvas($('visualPriceCanvas'), bars, payload);
  for (let slot = 0; slot < 3; slot += 1) {
    const canvas = $(`visualPanelCanvas${slot}`);
    const legend = $(`visualPanelLegend${slot}`);
    const selection = state.visualization.panelSelections[slot] || 'none';
    if (selection === 'volume') {
      drawVolumeCanvas(canvas, bars, legend);
      continue;
    }
    const indicator = getVisualizationIndicators(payload).find(item => item.id === selection);
    if (selection !== 'none' && indicator) drawIndicatorCanvas(canvas, bars, indicator, legend);
    else {
      drawCanvasMessage(canvas, selection === 'none' ? 'No indicator selected' : 'Indicator unavailable');
      if (legend) legend.innerHTML = '-';
    }
  }
  if (payload?.mode === 'evaluation' && state.evaluationReplay.activeTab === 'evaluation') {
    renderEvaluationStats(evaluationEventForVisualizationIndex());
  } else {
    renderTechnicalInspector(payload);
  }
}

function updateVisualizationPanelVisibility() {
  const count = Math.max(1, Math.min(3, Number($('visualSubPanelCount')?.value || 3)));
  const grid = document.querySelector('.visual-chart-grid');
  grid?.classList.remove('panel-count-1', 'panel-count-2', 'panel-count-3');
  grid?.classList.add(`panel-count-${count}`);
  document.querySelectorAll('.visual-sub-panel').forEach(panel => {
    const slot = Number(panel.dataset.panelSlot || 0);
    panel.classList.toggle('hidden', slot >= count);
  });
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function drawPanelBackground(ctx, width, height) {
  ctx.fillStyle = '#10151d';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#202938';
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i += 1) {
    const y = (height / 5) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 1; i < 8; i += 1) {
    const x = (width / 8) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function drawCanvasMessage(canvas, text) {
  const { ctx, width, height } = setupCanvas(canvas);
  drawPanelBackground(ctx, width, height);
  ctx.fillStyle = '#8ba3c7';
  ctx.font = '13px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width / 2, height / 2);
}

function drawAxisLabelPair(ctx, width, pad, y, text) {
  ctx.fillStyle = '#a9c8f5';
  ctx.font = '12px Inter, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  ctx.fillText(text, pad.left - 8, y);
  ctx.textAlign = 'right';
  ctx.fillText(text, width - 8, y);
}

function compactBarsForCanvas(bars, plotWidth) {
  const maxBuckets = Math.max(240, Math.floor(plotWidth));
  if (bars.length <= maxBuckets) return bars;
  const bucketSize = Math.ceil(bars.length / maxBuckets);
  const compacted = [];
  for (let start = 0; start < bars.length; start += bucketSize) {
    const end = Math.min(bars.length, start + bucketSize);
    const first = bars[start];
    const last = bars[end - 1];
    let high = Number.NEGATIVE_INFINITY;
    let low = Number.POSITIVE_INFINITY;
    let volume = 0;
    for (let index = start; index < end; index += 1) {
      const bar = bars[index];
      const barHigh = Number(bar.high);
      const barLow = Number(bar.low);
      const barVolume = Number(bar.volume);
      if (Number.isFinite(barHigh)) high = Math.max(high, barHigh);
      if (Number.isFinite(barLow)) low = Math.min(low, barLow);
      if (Number.isFinite(barVolume)) volume += barVolume;
    }
    compacted.push({
      ...last,
      open: first.open,
      high: Number.isFinite(high) ? high : last.high,
      low: Number.isFinite(low) ? low : last.low,
      volume,
      _sourceStart: start,
      _sourceEnd: end - 1,
    });
  }
  return compacted;
}

function featureModelSeriesMap(indicator) {
  return new Map((indicator?.model_series || []).map(series => [series.id, series]));
}

function resolvedFeatureSeries(indicator) {
  const modelById = featureModelSeriesMap(indicator);
  return (indicator?.display_series || []).map(series => ({
    ...series,
    values: series.values || modelById.get(series.model_field)?.values || [],
  }));
}

function visibleFeatureValues(values) {
  return (values || []).slice(state.visualization.viewStart, state.visualization.viewEnd);
}

function compactNumericValues(values, plotWidth, mode = 'last') {
  const maxPoints = Math.max(240, Math.floor(plotWidth));
  if (values.length <= maxPoints) return values;
  const bucketSize = Math.ceil(values.length / maxPoints);
  const result = [];
  for (let start = 0; start < values.length; start += bucketSize) {
    const bucket = values.slice(start, Math.min(values.length, start + bucketSize))
      .filter(value => value !== null && value !== undefined)
      .map(Number)
      .filter(Number.isFinite);
    if (!bucket.length) result.push(null);
    else if (mode === 'max_abs') result.push(bucket.reduce((best, value) => Math.abs(value) > Math.abs(best) ? value : best, bucket[0]));
    else result.push(bucket[bucket.length - 1]);
  }
  return result;
}

function drawFeatureLine(ctx, values, pad, plotW, yFor, color) {
  const compact = compactNumericValues(values, plotW);
  const step = plotW / Math.max(1, compact.length);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.35;
  ctx.beginPath();
  let drawing = false;
  compact.forEach((value, index) => {
    if (value === null || value === undefined) {
      drawing = false;
      return;
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      drawing = false;
      return;
    }
    const x = pad.left + step * index + step / 2;
    const y = yFor(number);
    if (!drawing) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    drawing = true;
  });
  ctx.stroke();
  ctx.restore();
}

function drawMainPriceOverlays(ctx, payload, pad, plotW, yFor) {
  getVisualizationIndicators(payload)
    .filter(indicator => indicator.render_target === 'main_overlay' && state.visualization.mainOverlays.has(indicator.id))
    .forEach(indicator => {
      resolvedFeatureSeries(indicator).forEach(series => {
        drawFeatureLine(ctx, visibleFeatureValues(series.values), pad, plotW, yFor, series.color);
      });
    });
}

function visibleMainOverlayRange(payload) {
  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;
  const start = state.visualization.viewStart;
  const end = state.visualization.viewEnd;
  getVisualizationIndicators(payload)
    .filter(indicator => indicator.render_target === 'main_overlay' && state.visualization.mainOverlays.has(indicator.id))
    .forEach(indicator => {
      resolvedFeatureSeries(indicator).forEach(series => {
        const values = series.values || [];
        for (let index = start; index < Math.min(end, values.length); index += 1) {
          if (values[index] === null || values[index] === undefined || values[index] === '') continue;
          const number = Number(values[index]);
          if (!Number.isFinite(number)) continue;
          minValue = Math.min(minValue, number);
          maxValue = Math.max(maxValue, number);
        }
      });
    });
  return Number.isFinite(minValue) && Number.isFinite(maxValue) ? [minValue, maxValue] : [];
}

function mainOverlayLegendItems(payload, index) {
  if (index === null || index === undefined) return [];
  const parts = [];
  getVisualizationIndicators(payload)
    .filter(indicator => indicator.render_target === 'main_overlay' && state.visualization.mainOverlays.has(indicator.id))
    .forEach(indicator => {
      resolvedFeatureSeries(indicator).forEach(series => {
        parts.push(legendItem(series.label, formatNumber(series.values?.[index], 3), series.color));
      });
    });
  return parts;
}


function evaluationEventForVisualizationIndex(index = null) {
  const payload = state.visualization.payload;
  if (payload?.mode !== 'evaluation') return null;
  const bars = payload.bars || [];
  if (!bars.length) return null;
  let selectedIndex = index;
  if (selectedIndex === null || selectedIndex === undefined) {
    selectedIndex = state.visualization.hoverIndex;
  }
  if (selectedIndex === null || selectedIndex === undefined || !bars[selectedIndex]) {
    selectedIndex = Math.max(0, Math.min(bars.length - 1, (state.visualization.viewEnd || bars.length) - 1));
  }
  return bars[selectedIndex]?._evaluationEvent || null;
}

function evaluationFillInfo(event) {
  const fills = event?.execution?.fills || [];
  const filled = fills.find(item => item.status === 'filled') || null;
  const blocked = fills.find(item => item.status === 'blocked') || null;
  const action = String(event?.decision?.action || 'hold').toLowerCase();
  const syntheticBlocked = !filled && !blocked && action !== 'hold'
    ? { status: 'blocked', side: action, reason: 'No fill was created for this action.' }
    : null;
  const primary = filled || blocked || syntheticBlocked || null;
  return {
    filled,
    blocked: blocked || syntheticBlocked,
    primary,
    executed: Boolean(filled || event?.execution?.executed),
    rejected: Boolean(blocked || syntheticBlocked || event?.execution?.blocked),
  };
}

function evaluationActionLabel(event) {
  const execution = event?.execution || {};
  const raw = String(event?.decision?.raw_action || event?.decision?.action || 'hold').toLowerCase();
  if (execution.executed) return `Executed ${String(execution.side || raw).toUpperCase()}`;
  if (execution.blocked) return `Blocked ${String(execution.side || raw).toUpperCase()}`;
  return raw.toUpperCase();
}

function evaluationLegendItems(event) {
  if (!event) return [];
  const execution = event.execution || {};
  const color = execution.executed
    ? (String(execution.side).toLowerCase() === 'sell' ? '#fbbf24' : '#60a5fa')
    : execution.blocked ? '#f97316' : '#94a3b8';
  const items = [
    legendItem('Decision', evaluationActionLabel(event), color),
  ];
  if (execution.blocked_reason) items.push(legendItem('Reason', execution.blocked_reason, '#f97316'));
  if (execution.executed) {
    items.push(legendItem('Trade', `${formatNumber(execution.shares, 0)} @ ${formatNumber(execution.price, 3)}`, color));
  }
  return items;
}

function drawEvaluationActionOverlays(ctx, visibleBars, renderBars, pad, plotW, yFor) {
  const payload = state.visualization.payload;
  if (payload?.mode !== 'evaluation' || !visibleBars.length || !renderBars.length) return;
  const step = plotW / Math.max(1, renderBars.length);
  renderBars.forEach((bar, index) => {
    const start = Number.isInteger(bar._sourceStart) ? bar._sourceStart : index;
    const end = Number.isInteger(bar._sourceEnd) ? bar._sourceEnd : start;
    const events = visibleBars.slice(start, end + 1).map(item => item._evaluationEvent).filter(Boolean);
    if (!events.length) return;
    let marker = null;
    for (const event of events) {
      const info = evaluationFillInfo(event);
      if (info.filled) marker = { event, fill: info.filled, type: info.filled.side === 'sell' ? 'sell' : 'buy', blocked: false };
      else if (info.blocked && !marker) marker = { event, fill: info.blocked, type: info.blocked.side === 'sell' ? 'blocked_sell' : 'blocked_buy', blocked: true };
    }
    if (!marker) return;
    const x = pad.left + step * index + step / 2;
    const price = Number(marker.fill.price ?? marker.fill.execution_price ?? marker.fill.reference_price ?? bar.close ?? bar.open);
    const y = yFor(Number.isFinite(price) ? price : Number(bar.close || bar.open));
    ctx.save();
    ctx.lineWidth = 2;
    if (marker.blocked) {
      const isBlockedSell = marker.type === 'blocked_sell';
      ctx.strokeStyle = isBlockedSell ? '#a78bfa' : '#f97316';
      ctx.fillStyle = isBlockedSell ? 'rgba(167, 139, 250, 0.22)' : 'rgba(249, 115, 22, 0.22)';
      ctx.beginPath();
      if (isBlockedSell) {
        ctx.moveTo(x, y + 11);
        ctx.lineTo(x + 8, y - 3);
        ctx.lineTo(x - 8, y - 3);
      } else {
        ctx.moveTo(x, y - 11);
        ctx.lineTo(x - 8, y + 3);
        ctx.lineTo(x + 8, y + 3);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 4, y - 4);
      ctx.lineTo(x + 4, y + 4);
      ctx.moveTo(x + 4, y - 4);
      ctx.lineTo(x - 4, y + 4);
      ctx.stroke();
    } else {
      const isBuy = marker.type === 'buy';
      ctx.fillStyle = isBuy ? '#60a5fa' : '#fbbf24';
      ctx.strokeStyle = '#0f172a';
      ctx.beginPath();
      if (isBuy) {
        ctx.moveTo(x, y - 10); ctx.lineTo(x - 7, y + 5); ctx.lineTo(x + 7, y + 5);
      } else {
        ctx.moveTo(x, y + 10); ctx.lineTo(x - 7, y - 5); ctx.lineTo(x + 7, y - 5);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  });
}

function drawPriceCanvas(canvas, bars, payload) {
  if (!canvas) return;
  if (!bars.length) {
    drawCanvasMessage(canvas, 'No K-line data');
    $('visualPriceLegend').innerHTML = '-';
    return;
  }

  const { ctx, width, height } = setupCanvas(canvas);
  drawPanelBackground(ctx, width, height);

  const pad = { left: 46, right: 66, top: 18, bottom: 26 };
  const plotW = width - pad.left - pad.right;
  const renderBars = compactBarsForCanvas(bars, plotW);
  const { minPrice, maxPrice, yFor } = priceScaleForBars(renderBars, pad, height, visibleMainOverlayRange(payload));
  const step = plotW / Math.max(1, renderBars.length);
  const candleW = Math.max(1, Math.min(10, step * 0.58));

  ctx.strokeStyle = '#2a3546';
  ctx.fillStyle = '#a9c8f5';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < 5; i += 1) {
    const value = minPrice + ((maxPrice - minPrice) / 4) * i;
    const y = yFor(value);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    drawAxisLabelPair(ctx, width, pad, y, value.toFixed(3));
  }

  renderBars.forEach((bar, index) => {
    const open = Number(bar.open);
    const close = Number(bar.close);
    const high = Number(bar.high);
    const low = Number(bar.low);
    if (![open, close, high, low].every(Number.isFinite)) return;
    const x = pad.left + step * index + step / 2;
    const up = close >= open;
    const color = up ? '#ff335f' : '#00b878';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, yFor(high));
    ctx.lineTo(x, yFor(low));
    ctx.stroke();

    const top = yFor(Math.max(open, close));
    const bottom = yFor(Math.min(open, close));
    const bodyH = Math.max(1, bottom - top);
    ctx.fillRect(x - candleW / 2, top, candleW, bodyH);
  });

  drawMainPriceOverlays(ctx, payload, pad, plotW, yFor);
  drawEvaluationActionOverlays(ctx, bars, renderBars, pad, plotW, yFor);

  drawTimeLabels(ctx, bars, pad, width, height, plotW);
  const visibleHoverIndex = getVisibleHoverIndex(bars);
  const hoverBar = visibleHoverIndex >= 0 ? bars[visibleHoverIndex] : null;
  const referenceGuide = state.visualization.referenceGuides;
  const hasPriceGuide = referenceGuide.active
    && referenceGuide.panel === 'price'
    && Number.isFinite(Number(referenceGuide.price));
  const hoverPriceY = !referenceGuide.active && hoverBar && state.visualization.hoverPanel === 'price' && Number.isFinite(Number(hoverBar.close))
    ? yFor(Number(hoverBar.close))
    : null;
  drawHoverCrosshair(ctx, bars, pad, width, height, plotW, hoverPriceY);
  if (hoverBar && state.visualization.hoverPanel === 'price' && hoverPriceY !== null) {
    drawPriceHoverLabels(ctx, width, pad, hoverPriceY, hoverBar);
  }
  if (hasPriceGuide) {
    drawPriceReferenceGuide(ctx, width, height, pad, yFor(Number(referenceGuide.price)), Number(referenceGuide.price), hoverBar);
  }
  $('visualPriceLegend').innerHTML = priceLegendHtml(payload, hoverBar, state.visualization.hoverIndex);
}

function priceLegendHtml(payload, hoverBar = null, hoverIndex = null) {
  const base = `<span class="legend-base">${escapeHtml(payload?.freq || '-')} / ${escapeHtml(payload?.adjust || '-')}</span>`;
  if (!hoverBar) return `<span class="legend-line">${base}</span>`;
  const previousClose = previousCloseForIndex(hoverIndex);
  const moveColor = barColorByPreviousClose(hoverIndex, hoverBar);
  const parts = [
    legendItem('Time', formatLegendDateTime(hoverBar.datetime), '#e5f0ff'),
    legendItem('O', formatNumber(hoverBar.open, 3), priceCompareColor(hoverBar.open, previousClose)),
    legendItem('H', formatNumber(hoverBar.high, 3), priceCompareColor(hoverBar.high, previousClose)),
    legendItem('L', formatNumber(hoverBar.low, 3), priceCompareColor(hoverBar.low, previousClose)),
    legendItem('C', formatNumber(hoverBar.close, 3), priceCompareColor(hoverBar.close, previousClose)),
    legendItem('Pct', formatSignedPercent(hoverBar.pctChg), moveColor),
  ];
  if (String(payload?.freq || '').toLowerCase() === 'daily') {
    parts.push(legendItem('Turn', formatNumber(hoverBar.turn, 4), '#f8fafc', 'white'));
    if (hoverBar.is_st === true) parts.push(legendItem('Status', 'ST', '#ff335f'));
  }
  if (payload?.mode === 'evaluation') {
    parts.push(...evaluationLegendItems(hoverBar._evaluationEvent));
  }
  parts.push(...mainOverlayLegendItems(payload, hoverIndex));
  return `<span class="legend-line">${base}${parts.join('')}</span>`;
}

function drawPriceHoverLabels(ctx, width, pad, y, bar) {
  const color = barColorByPreviousClose(state.visualization.hoverIndex, bar);
  ctx.save();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.fillRect(0, y - 12, pad.left + 8, 24);
  ctx.fillRect(width - pad.right + 8, y - 12, pad.right - 8, 24);
  ctx.fillStyle = '#ffffff';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(formatNumber(bar.close, 3), (pad.left + 8) / 2, y);
  ctx.fillText(formatSignedPercent(bar.pctChg), width - pad.right / 2 + 4, y);
  ctx.restore();
}

function drawHorizontalReferenceLine(ctx, width, height, pad, y, color = '#f8fafc') {
  const clampedY = Math.max(pad.top || 0, Math.min(height - (pad.bottom || 0), Number(y)));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(pad.left || 0, clampedY);
  ctx.lineTo(width - (pad.right || 0), clampedY);
  ctx.stroke();
  ctx.restore();
  return clampedY;
}

function drawPriceReferenceGuide(ctx, width, height, pad, y, value, hoverBar = null) {
  if (y < pad.top || y > height - pad.bottom) return;
  const baseline = Number(hoverBar?.close);
  const pct = Number.isFinite(baseline) && baseline !== 0 ? value / baseline - 1 : null;
  const color = signedColor(pct);
  const clampedY = drawHorizontalReferenceLine(ctx, width, height, pad, y, color);
  const priceText = formatNumber(value, 3);
  const rightText = Number.isFinite(Number(pct)) ? formatSignedPercent(pct) : '-';
  drawSideLabel(ctx, 0, pad.left + 8, clampedY, priceText, color);
  drawSideLabel(ctx, width - pad.right + 8, pad.right - 8, clampedY, rightText, color);
}

function drawVolumeCanvas(canvas, bars, legendElement = null) {
  if (!canvas) return;
  if (!bars.length) {
    drawCanvasMessage(canvas, 'No volume data');
    if (legendElement) legendElement.innerHTML = '-';
    return;
  }

  const { ctx, width, height } = setupCanvas(canvas);
  drawPanelBackground(ctx, width, height);
  const pad = { left: 46, right: 66, top: 12, bottom: 24 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const renderBars = compactBarsForCanvas(bars, plotW);
  const { maxVolume, yFor: yForVolume } = volumeScaleForBars(renderBars, pad, height);
  const step = plotW / Math.max(1, renderBars.length);
  const barW = Math.max(1, Math.min(10, step * 0.62));

  renderBars.forEach((bar, index) => {
    const volume = Number(bar.volume);
    if (!Number.isFinite(volume)) return;
    const x = pad.left + step * index + step / 2 - barW / 2;
    const h = (volume / maxVolume) * plotH;
    const sourceIndex = bar._sourceStart ?? index;
    ctx.fillStyle = barColorByPreviousClose(state.visualization.viewStart + sourceIndex, bar);
    ctx.fillRect(x, pad.top + plotH - h, barW, h);
  });

  ctx.fillStyle = '#a9c8f5';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < 3; i += 1) {
    const value = (maxVolume / 2) * i;
    const y = pad.top + plotH - (value / maxVolume) * plotH;
    drawAxisLabelPair(ctx, width, pad, y, formatCompactNumber(value));
  }
  drawTimeLabels(ctx, bars, pad, width, height, plotW);
  const visibleHoverIndex = getVisibleHoverIndex(bars);
  const hoverBar = visibleHoverIndex >= 0 ? bars[visibleHoverIndex] : null;
  const hoverVolume = hoverBar ? Number(hoverBar.volume) : null;
  const referenceGuide = state.visualization.referenceGuides;
  const hasVolumeGuide = referenceGuide.active
    && referenceGuide.panel === 'volume'
    && Number.isFinite(Number(referenceGuide.volume));
  const hoverVolumeY = !referenceGuide.active && hoverBar && state.visualization.hoverPanel === 'volume' && Number.isFinite(hoverVolume)
    ? yForVolume(hoverVolume)
    : null;
  drawHoverCrosshair(ctx, bars, pad, width, height, plotW, hoverVolumeY);
  if (hoverBar && state.visualization.hoverPanel === 'volume' && hoverVolumeY !== null) {
    drawValueSideLabel(ctx, width, pad, hoverVolumeY, formatCompactNumber(hoverBar.volume), barColorByPreviousClose(state.visualization.hoverIndex, hoverBar));
  }
  if (hasVolumeGuide) {
    drawVolumeReferenceGuide(ctx, width, height, pad, yForVolume(Number(referenceGuide.volume)), Number(referenceGuide.volume), hoverBar);
  }
  if (legendElement) {
    legendElement.innerHTML = volumeLegendHtml(hoverBar || bars[bars.length - 1], hoverBar ? state.visualization.hoverIndex : state.visualization.viewEnd - 1);
  }
}

function volumeLegendHtml(bar = null, index = null) {
  if (!bar) return 'VOL: -';
  return `<span class="legend-line">${legendItem('VOL', formatCompactNumber(bar.volume), barColorByPreviousClose(index, bar))}</span>`;
}

function drawVolumeReferenceGuide(ctx, width, height, pad, y, value, hoverBar = null) {
  if (y < pad.top || y > height - pad.bottom) return;
  const color = hoverBar ? barColorByPreviousClose(state.visualization.hoverIndex, hoverBar) : '#64748b';
  const clampedY = drawHorizontalReferenceLine(ctx, width, height, pad, y, color);
  const valueText = formatCompactNumber(value);
  drawSideLabel(ctx, 0, pad.left + 8, clampedY, valueText, color);
  drawSideLabel(ctx, width - pad.right + 8, pad.right - 8, clampedY, valueText, color);
}

function drawIndicatorCanvas(canvas, bars, indicator, legendElement = null) {
  if (!canvas || !bars.length) {
    if (canvas) drawCanvasMessage(canvas, 'No indicator data');
    if (legendElement) legendElement.innerHTML = '-';
    return;
  }
  const series = resolvedFeatureSeries(indicator).map(item => ({
    ...item,
    visibleValues: visibleFeatureValues(item.values),
  }));
  let maxAbsValue = 1e-9;
  let hasNumericValue = false;
  series.forEach(item => {
    item.visibleValues.forEach(value => {
      if (value === null || value === undefined || value === '') return;
      const number = Number(value);
      if (!Number.isFinite(number)) return;
      hasNumericValue = true;
      maxAbsValue = Math.max(maxAbsValue, Math.abs(number));
    });
  });
  if (!hasNumericValue) {
    drawCanvasMessage(canvas, 'Indicator is warming up');
    if (legendElement) legendElement.innerHTML = `${escapeHtml(indicator.label)} / warming up`;
    return;
  }

  const { ctx, width, height } = setupCanvas(canvas);
  drawPanelBackground(ctx, width, height);
  const pad = { left: 46, right: 66, top: 12, bottom: 24 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  let minValue;
  let maxValue;
  if (indicator.axis?.mode === 'bounded') {
    minValue = Number(indicator.axis.min);
    maxValue = Number(indicator.axis.max);
  } else {
    minValue = -maxAbsValue;
    maxValue = maxAbsValue;
  }
  const range = Math.max(1e-12, maxValue - minValue);
  const yFor = value => pad.top + ((maxValue - value) / range) * plotH;

  ctx.font = '12px Inter, sans-serif';
  (indicator.axis?.reference_lines || []).forEach(value => {
    const y = yFor(Number(value));
    ctx.save();
    ctx.strokeStyle = '#344155';
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.restore();
  });
  for (let index = 0; index < 3; index += 1) {
    const value = minValue + (range / 2) * index;
    drawAxisLabelPair(ctx, width, pad, yFor(value), formatNumber(value, 4));
  }

  series.filter(item => item.style === 'histogram').forEach(item => {
    const values = compactNumericValues(item.visibleValues, plotW, 'max_abs');
    const step = plotW / Math.max(1, values.length);
    const barWidth = Math.max(1, Math.min(8, step * 0.62));
    const zeroY = yFor(0);
    values.forEach((value, index) => {
      if (value === null || value === undefined) return;
      const number = Number(value);
      if (!Number.isFinite(number)) return;
      const y = yFor(number);
      ctx.fillStyle = number > 0 ? '#ff335f' : number < 0 ? '#00b878' : '#64748b';
      ctx.fillRect(pad.left + step * index + (step - barWidth) / 2, Math.min(y, zeroY), barWidth, Math.max(1, Math.abs(zeroY - y)));
    });
  });
  series.filter(item => item.style !== 'histogram').forEach(item => {
    drawFeatureLine(ctx, item.visibleValues, pad, plotW, yFor, item.color);
  });

  drawTimeLabels(ctx, bars, pad, width, height, plotW);
  drawHoverCrosshair(ctx, bars, pad, width, height, plotW);
  const hoverIndex = getVisibleHoverIndex(bars) >= 0
    ? state.visualization.hoverIndex
    : Math.max(0, state.visualization.viewEnd - 1);
  if (legendElement) legendElement.innerHTML = indicatorLegendHtml(indicator, hoverIndex);
}

function indicatorLegendHtml(indicator, index) {
  const parts = resolvedFeatureSeries(indicator).map(series => (
    legendItem(series.label, formatFeatureValue(series.values?.[index]), series.color)
  ));
  return `<span class="legend-line">${parts.join('')}</span>`;
}

function formatFeatureValue(value) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return Math.abs(number) < 0.01 && number !== 0 ? number.toFixed(6) : number.toFixed(4);
}

function renderTechnicalInspector(payload) {
  const content = $('visualInspectorContent');
  const time = $('visualInspectorTime');
  if (!content || !time) return;
  const bars = payload?.bars || [];
  if (!bars.length) {
    time.textContent = '-';
    content.innerHTML = '<div class="muted">No technical indicator data.</div>';
    return;
  }
  const index = state.visualization.hoverIndex !== null
    ? state.visualization.hoverIndex
    : Math.max(0, state.visualization.viewEnd - 1);
  const renderKey = `${state.visualization.loadGeneration}:${state.visualization.windowOffset + index}`;
  if (content.dataset.renderKey === renderKey) return;
  content.dataset.renderKey = renderKey;
  time.textContent = formatLegendDateTime(bars[index]?.datetime);
  const indicators = getVisualizationIndicators(payload);
  content.innerHTML = indicators.length ? indicators.map(indicator => {
    const coverage = Number(indicator.coverage?.[index]);
    const rows = (indicator.model_series || []).map(series => {
      const value = series.values?.[index];
      const number = Number(value);
      const clip = series.clip;
      const atLimit = value !== null && value !== undefined && Number.isFinite(number) && Array.isArray(clip)
        && (number <= Number(clip[0]) || number >= Number(clip[1]));
      return `<div class="visual-inspector-row"><span>${escapeHtml(series.label)}</span><strong>${escapeHtml(formatFeatureValue(value))}${atLimit ? ' *' : ''}</strong></div>`;
    }).join('');
    return `
      <section class="visual-inspector-group">
        <h3>${escapeHtml(indicator.label)} <span class="visual-inspector-maturity">${Number.isFinite(coverage) ? `${Math.round(coverage * 100)}% mature` : ''}</span></h3>
        ${rows}
      </section>
    `;
  }).join('') : '<div class="muted">No technical indicators for this frequency.</div>';
}

function drawFeaturePlaceholderCanvas(canvas, label, featureContract) {
  if (!canvas) return;
  const { ctx, width, height } = setupCanvas(canvas);
  drawPanelBackground(ctx, width, height);
  ctx.strokeStyle = '#263448';
  ctx.setLineDash([6, 5]);
  [0.25, 0.5, 0.75].forEach(pos => {
    const y = height * pos;
    ctx.beginPath();
    ctx.moveTo(48, y);
    ctx.lineTo(width - 66, y);
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.fillStyle = '#8ba3c7';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const status = featureContract?.status || 'pending_feature_layer';
  ctx.fillText(`${label} / ${status}`, width / 2, height / 2);
  drawHoverCrosshair(ctx, getVisibleBars(), { left: 46, right: 66, top: 12, bottom: 16 }, width, height, width - 112);
}

function drawTimeLabels(ctx, bars, pad, width, height, plotW) {
  if (!bars.length) return;
  ctx.fillStyle = '#a9c8f5';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  const labelCount = Math.min(6, bars.length);
  for (let i = 0; i < labelCount; i += 1) {
    const index = Math.floor((bars.length - 1) * (i / Math.max(1, labelCount - 1)));
    const x = pad.left + plotW * (index / Math.max(1, bars.length - 1));
    const text = String(bars[index].datetime || '').slice(0, 10);
    ctx.fillText(text, x, height - 5);
  }
}

function drawHoverCrosshair(ctx, bars, pad, width, height, plotW, horizontalY = null) {
  const hoverIndex = state.visualization.hoverIndex;
  if (hoverIndex === null || hoverIndex === undefined || !bars.length) return;
  const visibleIndex = hoverIndex - state.visualization.viewStart;
  if (visibleIndex < 0 || visibleIndex >= bars.length) return;
  const step = plotW / Math.max(1, bars.length);
  const x = pad.left + step * visibleIndex + step / 2;
  ctx.save();
  ctx.strokeStyle = 'rgba(190, 215, 255, 0.55)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, pad.top || 0);
  ctx.lineTo(x, height - (pad.bottom || 0));
  ctx.stroke();
  if (Number.isFinite(Number(horizontalY))) {
    const y = Math.max(pad.top || 0, Math.min(height - (pad.bottom || 0), Number(horizontalY)));
    ctx.beginPath();
    ctx.moveTo(pad.left || 0, y);
    ctx.lineTo(width - (pad.right || 0), y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawValueSideLabel(ctx, width, pad, y, text, color = '#64748b') {
  drawSideLabel(ctx, width - pad.right + 8, pad.right - 8, y, text, color);
}

function drawSideLabel(ctx, x, boxWidth, y, text, color = '#64748b') {
  ctx.save();
  ctx.fillStyle = color;
  ctx.fillRect(x, y - 12, boxWidth, 24);
  ctx.fillStyle = '#ffffff';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + boxWidth / 2, y);
  ctx.restore();
}

function updateVisualizationReferenceFromPointer(target, clientY) {
  const canvas = getVisualizationCanvasFromTarget(target);
  const panel = getHoverPanelFromTarget(canvas);
  if (panel !== 'price' && panel !== 'volume') return false;
  const bars = getVisibleBars();
  if (!canvas || !bars.length || !Number.isFinite(Number(clientY))) return false;

  const rect = canvas.getBoundingClientRect();
  const y = Number(clientY) - rect.top;
  const guide = state.visualization.referenceGuides;
  guide.active = true;
  guide.panel = panel;
  if (panel === 'price') {
    const pad = { left: 46, right: 66, top: 18, bottom: 26 };
    const scaleBars = compactBarsForCanvas(bars, rect.width - pad.left - pad.right);
    const { valueForY } = priceScaleForBars(scaleBars, pad, rect.height, visibleMainOverlayRange(state.visualization.payload));
    guide.price = valueForY(y);
  } else {
    const pad = { left: 46, right: 66, top: 12, bottom: 24 };
    const scaleBars = compactBarsForCanvas(bars, rect.width - pad.left - pad.right);
    const { valueForY } = volumeScaleForBars(scaleBars, pad, rect.height);
    guide.volume = Math.max(0, valueForY(y));
  }
  return true;
}

function activateVisualizationReferenceGuide(target, clientX, clientY) {
  if (!updateVisualizationReferenceFromPointer(target, clientY)) return;
  updateVisualizationHover(target, clientX, clientY);
}

function updateVisualizationHover(target, clientX, clientY = null) {
  const bars = state.visualization.payload?.bars || [];
  const canvas = getVisualizationCanvasFromTarget(target) || $('visualPriceCanvas');
  const panel = getHoverPanelFromTarget(canvas);
  const index = getHoverIndexFromClientX(clientX, canvas);
  if (index === null || !bars[index]) {
    clearVisualizationHover();
    return;
  }

  state.visualization.hoverIndex = index;
  state.visualization.hoverPanel = panel;
  if (hasVisualizationReferenceGuide() && clientY !== null) {
    updateVisualizationReferenceFromPointer(canvas, clientY);
  }
  drawVisualizationPayload(state.visualization.payload);
}

function clearVisualizationHover() {
  state.visualization.hoverIndex = null;
  state.visualization.hoverPanel = null;
}

function zoomVisualization(deltaY, clientX) {
  const bars = state.visualization.payload?.bars || [];
  if (bars.length <= 1) return;
  const canvas = $('visualPriceCanvas');
  const { rect, pad, width } = getChartGeometry(canvas);
  const plotW = Math.max(1, width - pad.left - pad.right);
  const currentStart = state.visualization.viewStart;
  const currentEnd = state.visualization.viewEnd || bars.length;
  const currentWindow = Math.max(1, currentEnd - currentStart);
  const factor = deltaY > 0 ? 1.18 : 0.82;
  const pendingWindow = Number(state.visualization.pendingZoom?.windowSize || 0);
  const baseWindow = deltaY > 0 ? Math.max(currentWindow, pendingWindow) : currentWindow;
  const desiredWindow = Math.max(20, Math.min(VISUAL_MAX_VISIBLE_BARS, Math.round(baseWindow * factor)));
  const nextWindow = Math.min(bars.length, desiredWindow);
  const localX = Math.max(pad.left, Math.min(width - pad.right, clientX - rect.left));
  const anchorRatio = (localX - pad.left) / plotW;
  const anchorIndex = currentStart + currentWindow * anchorRatio;
  const nextStart = anchorIndex - nextWindow * anchorRatio;
  if (desiredWindow > bars.length && (state.visualization.payload?.has_more_before || state.visualization.payload?.has_more_after)) {
    state.visualization.pendingZoom = {
      windowSize: desiredWindow,
      anchorAbsolute: state.visualization.windowOffset + anchorIndex,
      anchorRatio,
    };
  } else {
    state.visualization.pendingZoom = null;
  }
  clampVisualizationView(nextStart, nextStart + nextWindow);
  drawVisualizationPayload(state.visualization.payload);
  void maybePrefetchVisualization();
}

function startVisualizationDrag(clientX) {
  state.visualization.drag = {
    active: true,
    startX: clientX,
    startViewStart: state.visualization.viewStart,
    startViewEnd: state.visualization.viewEnd,
  };
  document.body.classList.add('visual-dragging');
}

function moveVisualizationDrag(clientX) {
  const drag = state.visualization.drag;
  if (!drag.active) return;
  const bars = state.visualization.payload?.bars || [];
  if (!bars.length) return;
  const canvas = $('visualPriceCanvas');
  const { pad, width } = getChartGeometry(canvas);
  const plotW = Math.max(1, width - pad.left - pad.right);
  const windowSize = Math.max(1, drag.startViewEnd - drag.startViewStart);
  const deltaBars = -Math.round((clientX - drag.startX) * (windowSize / plotW));
  clampVisualizationView(drag.startViewStart + deltaBars, drag.startViewEnd + deltaBars);
  clearVisualizationHover();
  drawVisualizationPayload(state.visualization.payload);
  void maybePrefetchVisualization();
}

function endVisualizationDrag() {
  if (!state.visualization.drag.active) return;
  state.visualization.drag.active = false;
  document.body.classList.remove('visual-dragging');
  void maybePrefetchVisualization();
}

function formatCompactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  if (Math.abs(number) >= 100000000) return `${(number / 100000000).toFixed(2)}B`;
  if (Math.abs(number) >= 10000) return `${(number / 10000).toFixed(2)}W`;
  return number.toFixed(0);
}
async function refreshInventory() {
  try {
    const db = encodeURIComponent(getDataDbPath());
    const res = await apiGet(`/api/data/inventory?db=${db}`);
    const data = res.data;
    state.inventory = data.symbols || [];
    const validKeys = new Set(getInventorySlices().map(row => inventoryKey(row)));
    state.selectedInventoryKeys = new Set([...state.selectedInventoryKeys].filter(key => validKeys.has(key)));
    const validSymbols = new Set(getInventoryGroups().map(group => group.symbol));
    state.expandedInventorySymbols = new Set([...state.expandedInventorySymbols].filter(symbol => validSymbols.has(symbol)));
    renderInventorySummary(data);
    renderInventoryTable();
    renderVisualizationControls();
    await loadVisualizationChart();
  } catch (err) {
    alert(`Refresh inventory failed: ${err.message}`);
  }
}

function setDownloadRunning(active) {
  state.downloadJobActive = Boolean(active);
  if ($('startDownloadBtn')) $('startDownloadBtn').disabled = state.downloadJobActive;
  if ($('stopDownloadBtn')) $('stopDownloadBtn').disabled = !state.downloadJobActive;
}

function renderJob(job) {
  const progress = Math.round((Number(job.progress || 0)) * 100);
  $('jobStatus').textContent = `${job.status || '-'} / ${job.message || ''}`;
  $('jobStatus').className = `job-status ${job.status || ''}`;
  $('jobProgressBar').style.width = `${progress}%`;
  renderSummaryCards('jobSummary', [
    { label: 'Progress', value: `${progress}%` },
    { label: 'Current', value: job.current || '-' },
    { label: 'Completed', value: `${job.completed ?? 0} / ${job.total ?? '-'}` },
    { label: 'Succeeded', value: job.succeeded ?? 0 },
    { label: 'Failed', value: job.failed ?? 0 },
    { label: 'Skipped', value: job.skipped ?? 0 },
    { label: 'Saved Rows', value: job.saved_rows ?? 0 },
  ]);
  $('downloadReportPath').textContent = job.result?.report_path || '-';
  const logs = job.logs || [];
  $('downloadJobLogs').textContent = logs.length
    ? logs.slice(-200).map(item => `[${item.time || '-'}] ${String(item.level || 'info').toUpperCase()} ${item.message || ''}`).join('\n')
    : 'No logs yet.';
  setDownloadRunning(['queued', 'running'].includes(job.status));
}

async function loadDownloadReport(reportPath) {
  if (!reportPath) return;
  try {
    const res = await apiGet(`/api/download/report?path=${encodeURIComponent(reportPath)}`);
    const report = res.data;
    $('downloadReportSummary').textContent = `Report: ${report.path} / Total ${report.total} / Issues ${report.issue_rows ?? report.failed} / Failed ${report.failed_count ?? report.failed} / Warnings ${report.warnings ?? 0}`;
    renderFailedRows(report.failed_rows || []);
  } catch (err) {
    $('downloadReportSummary').textContent = `Read CSV failed: ${err.message}`;
    renderFailedRows([]);
  }
}

function renderFailedRows(rows) {
  if (!rows.length) {
    $('failedDownloadTableBody').innerHTML = '<tr><td colspan="6" class="muted">No failed or warning rows.</td></tr>';
    return;
  }
  $('failedDownloadTableBody').innerHTML = rows.map(row => `
    <tr>
      <td>${escapeHtml(row.symbol)}</td>
      <td>${escapeHtml(row.freq || '-')}</td>
      <td>${escapeHtml(row.adjust || '-')}</td>
      <td>${escapeHtml(row.status ?? row.ok ?? row.success ?? '-')}</td>
      <td>${escapeHtml(row.rows ?? row.saved_rows ?? 0)}</td>
      <td class="error-cell">${escapeHtml(row.error || row.message || row.issues || '-')}</td>
    </tr>
  `).join('');
}

function shouldShowTurnColumn() {
  return String(state.detail.selectedFreq || '').toLowerCase() === 'daily';
}

function getModalKlineColumnCount() {
  return shouldShowTurnColumn() ? 10 : 8;
}

function renderModalTableHead() {
  const dailyHeaders = shouldShowTurnColumn() ? '<th>Turn</th><th>Status</th>' : '';
  $('modalKlineTableHead').innerHTML = `
    <tr>
      <th>Datetime</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Amount</th>
      ${dailyHeaders}
      <th>PctChg</th>
    </tr>
  `;
}

function setModalTableMessage(message) {
  renderModalTableHead();
  $('modalKlineTableBody').innerHTML = `<tr><td colspan="${getModalKlineColumnCount()}" class="muted">${escapeHtml(message)}</td></tr>`;
}

async function resumeActiveDownloadJob() {
  try {
    const res = await apiGet('/api/jobs');
    const active = (res.jobs || [])
      .filter(job => job.type === 'download' && ['queued', 'running'].includes(job.status))
      .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0];
    if (active?.job_id) {
      startPollingJob(active.job_id);
    } else {
      setDownloadRunning(false);
    }
  } catch (err) {
    setDownloadRunning(false);
    $('downloadReportSummary').textContent = `Job sync failed: ${err.message}`;
  }
}

function startPollingJob(jobId) {
  state.activeJobId = jobId;
  if (state.pollTimer) clearInterval(state.pollTimer);
  $('downloadReportSummary').textContent = 'Waiting for download CSV.';
  renderFailedRows([]);

  async function poll() {
    try {
      const res = await apiGet(`/api/jobs/${jobId}`);
      renderJob(res.job);
      if (['completed', 'failed', 'cancelled'].includes(res.job.status)) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        setDownloadRunning(false);
        await refreshInventory();
        if (res.job.result?.report_path) await loadDownloadReport(res.job.result.report_path);
      }
    } catch (err) {
      $('downloadReportSummary').textContent = `Job polling failed: ${err.message}`;
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      setDownloadRunning(false);
    }
  }

  poll();
  state.pollTimer = setInterval(poll, 1000);
}

function getSelectedDownloadFreqs() {
  return getCheckedValues('.downloadFreq');
}

function getSelectedDownloadAdjustflags() {
  return getCheckedValues('.downloadAdjustflag');
}

async function startDownload() {
  if (state.downloadJobActive) return;
  const freqs = getSelectedDownloadFreqs();
  if (!freqs.length) return alert('Select at least one frequency.');
  const adjustflags = getSelectedDownloadAdjustflags();
  if (!adjustflags.length) return alert('Select at least one adjust mode.');

  const payload = {
    db_path: $('downloadDbPath').value.trim(),
    storage_mode: 'shard',
    storage_root: $('downloadStorageRoot')?.value.trim() || 'runtime_layer/data',
    workers: Number($('downloadWorkers')?.value || 1),
    symbols_file: $('downloadSymbolsFile').value.trim(),
    start: $('downloadStart').value,
    end: $('downloadEnd').value,
    freqs,
    adjustflags,
    sleep: Number($('downloadSleep').value || 0),
    replace_symbol: $('downloadReplace').checked,
    skip_existing: $('downloadSkipExisting').checked,
  };

  try {
    setDownloadRunning(true);
    const res = await apiPost('/api/download/start', payload);
    startPollingJob(res.data.job_id);
  } catch (err) {
    setDownloadRunning(false);
    alert(`Start download failed: ${err.message}`);
  }
}

async function stopDownload() {
  if (!state.activeJobId || !state.downloadJobActive) return;
  try {
    await apiPost(`/api/jobs/${state.activeJobId}/cancel`, {});
    $('jobStatus').textContent = 'cancel requested';
  } catch (err) {
    alert(`Stop download failed: ${err.message}`);
  }
}

function renderResultTable(targetBodyId, rows) {
  if (!rows || !rows.length) {
    $(targetBodyId).innerHTML = '<tr><td colspan="5" class="muted">No results.</td></tr>';
    return;
  }
  $(targetBodyId).innerHTML = rows.map(row => `
    <tr>
      <td>${escapeHtml(row.symbol)}</td>
      <td>${escapeHtml(row.available)}</td>
      <td>${escapeHtml(row.rows ?? 0)}</td>
      <td>${escapeHtml(String(row.start_datetime ?? '-').slice(0, 10))}</td>
      <td>${escapeHtml(String(row.end_datetime ?? '-').slice(0, 10))}</td>
    </tr>
  `).join('');
}

async function checkCoverage() {
  const payload = {
    db_path: getDataDbPath(),
    symbols_file: $('coverageSymbolsFile').value.trim(),
    min_rows: Number($('coverageMinRows').value || 0),
    required_start: $('coverageRequiredStart').value || null,
    required_end: $('coverageRequiredEnd').value || null,
    output: $('coverageOutput').value.trim(),
  };
  try {
    const res = await apiPost('/api/data/check-coverage', payload);
    const data = res.data;
    renderSummaryCards('coverageSummary', [
      { label: 'Total', value: data.total },
      { label: 'Available', value: data.available },
      { label: 'Unavailable', value: data.unavailable },
      { label: 'CSV', value: data.report_path || '-' },
    ]);
    renderResultTable('coverageTableBody', data.rows || []);
  } catch (err) {
    $('coverageSummary').innerHTML = `<div class="error-box">Check failed: ${escapeHtml(err.message)}</div>`;
    renderResultTable('coverageTableBody', []);
  }
}

async function buildUniverse() {
  const payload = {
    db_path: getDataDbPath(),
    candidates: $('universeCandidates').value.trim(),
    output: $('universeOutput').value.trim(),
    report: $('universeReport').value.trim(),
    min_rows: Number($('universeMinRows').value || 0),
    required_start: $('universeRequiredStart').value || null,
    required_end: $('universeRequiredEnd').value || null,
  };
  try {
    const res = await apiPost('/api/data/build-universe', payload);
    const data = res.data;
    renderSummaryCards('universeSummary', [
      { label: 'Total', value: data.total },
      { label: 'Available', value: data.available },
      { label: 'Unavailable', value: data.unavailable },
      { label: 'Output', value: data.output_path || '-' },
      { label: 'CSV', value: data.report_path || '-' },
    ]);
    renderResultTable('universeTableBody', data.rows || []);
  } catch (err) {
    $('universeSummary').innerHTML = `<div class="error-box">Build failed: ${escapeHtml(err.message)}</div>`;
    renderResultTable('universeTableBody', []);
  }
}

function getSelectedMaterializeTargets() {
  return getCheckedValues('.materializeTarget');
}

function syncMaterializeTargetsForBase() {
  const base = $('materializeBaseFreq').value;
  const targetsByBase = {
    '5min': ['15min'],
    '30min': ['60min'],
    daily: ['weekly', 'monthly'],
  };
  const selected = new Set(targetsByBase[base] || []);
  document.querySelectorAll('.materializeTarget').forEach(input => {
    input.disabled = !selected.has(input.value);
    input.checked = selected.has(input.value);
  });
}

function renderMaterializeJob(job) {
  const progress = Math.round((Number(job.progress || 0)) * 100);
  const active = ['queued', 'running'].includes(job.status);
  $('materializeBarsBtn').disabled = active;
  $('stopMaterializeBtn').disabled = !active || Boolean(job.cancel_requested);
  $('materializeJobStatus').textContent = `${job.status || '-'} / ${job.message || ''}`;
  $('materializeJobStatus').className = `job-status ${job.status || ''}`;
  $('materializeProgressBar').style.width = `${progress}%`;
  const result = job.result || {};
  renderSummaryCards('materializeSummary', [
    { label: 'Progress', value: `${progress}%` },
    { label: 'Units', value: `${job.completed ?? 0} / ${job.total ?? '-'}` },
    { label: 'Current', value: job.current || '-' },
    { label: 'Succeeded', value: job.succeeded ?? result.ok ?? '-' },
    { label: 'Up to Date', value: result.skipped ?? '-' },
    { label: 'Failed', value: job.failed ?? result.failed ?? '-' },
    { label: 'Saved Rows', value: job.saved_rows ?? result.saved_rows ?? 0 },
    { label: 'CSV', value: result.report_path || '-' },
  ]);
  const logs = job.logs || [];
  $('materializeJobLogs').textContent = logs.length
    ? logs.slice(-200).map(item => `[${item.time || '-'}] ${String(item.level || 'info').toUpperCase()} ${item.message || ''}`).join('\n')
    : 'No logs yet.';
}

function startPollingMaterializeJob(jobId) {
  state.materializeJobId = jobId;
  if (state.materializePollTimer) clearInterval(state.materializePollTimer);

  async function poll() {
    try {
      const res = await apiGet(`/api/jobs/${jobId}`);
      renderMaterializeJob(res.job);
      if (['completed', 'failed', 'cancelled'].includes(res.job.status)) {
        clearInterval(state.materializePollTimer);
        state.materializePollTimer = null;
        state.materializeJobId = null;
        $('materializeBarsBtn').disabled = false;
        $('stopMaterializeBtn').disabled = true;
        await refreshInventory();
      }
    } catch (err) {
      $('materializeJobStatus').textContent = `Job polling failed: ${err.message}`;
      clearInterval(state.materializePollTimer);
      state.materializePollTimer = null;
      $('materializeBarsBtn').disabled = false;
      $('stopMaterializeBtn').disabled = !state.materializeJobId;
    }
  }

  poll();
  state.materializePollTimer = setInterval(poll, 1000);
}

async function materializeBars() {
  if (state.materializeJobId) return alert('A derived-bar job is already active.');
  const targets = getSelectedMaterializeTargets();
  if (!targets.length) return alert('Select at least one target frequency.');
  const payload = {
    db_path: getDataDbPath(),
    symbols_file: $('materializeSymbolsFile').value.trim(),
    base_freq: $('materializeBaseFreq').value,
    adjust: $('materializeAdjust').value,
    targets,
    start: $('materializeStart').value || null,
    end: $('materializeEnd').value || null,
    output: $('materializeOutput').value.trim(),
  };

  try {
    $('materializeBarsBtn').disabled = true;
    $('stopMaterializeBtn').disabled = false;
    const res = await apiPost('/api/data/materialize-bars', payload);
    startPollingMaterializeJob(res.data.job_id);
  } catch (err) {
    $('materializeBarsBtn').disabled = false;
    $('stopMaterializeBtn').disabled = true;
    alert(`Build derived bars failed: ${err.message}`);
  }
}

async function stopMaterializeBars() {
  if (!state.materializeJobId) return;
  $('stopMaterializeBtn').disabled = true;
  $('materializeJobStatus').textContent = 'running / Cancel requested; waiting for the current target transaction';
  try {
    await apiPost(`/api/jobs/${state.materializeJobId}/cancel`, {});
  } catch (err) {
    $('stopMaterializeBtn').disabled = false;
    alert(`Stop derived bars failed: ${err.message}`);
  }
}


function getBundlePayload() {
  return {
    bundle_path: $('bundlePath')?.value.trim() || 'runtime_layer/bundles/pocketagent_bundle.duckdb',
    db_path: getDataDbPath(),
    feature_output_dir: $('bundleFeatureOutputDir')?.value.trim() || 'runtime_layer/reports/feature_dataset',
    include_data: Boolean($('bundleIncludeData')?.checked),
    include_feature: Boolean($('bundleIncludeFeature')?.checked),
    replace_existing: Boolean($('bundleReplaceExisting')?.checked),
    overwrite: true,
  };
}

function renderBundleSummary(result = {}, progress = null) {
  const cards = [];
  if (progress !== null) cards.push({ label: 'Progress', value: `${progress}%` });
  if ('bundle_path' in result) cards.push({ label: 'Bundle', value: result.bundle_path || '-' });
  if ('file_count' in result) cards.push({ label: 'Files', value: result.file_count });
  if ('restored_files' in result) cards.push({ label: 'Restored', value: result.restored_files });
  if ('skipped_files' in result) cards.push({ label: 'Skipped', value: result.skipped_files });
  if ('verified_files' in result) cards.push({ label: 'Verified', value: result.verified_files });
  if ('total_bytes' in result) cards.push({ label: 'Raw Size', value: formatBytes(result.total_bytes) });
  if ('bundle_size' in result) cards.push({ label: 'Bundle Size', value: formatBytes(result.bundle_size) });
  if ('compressed_payload_bytes' in result) cards.push({ label: 'Payload', value: formatBytes(result.compressed_payload_bytes) });
  if ('restored_bytes' in result) cards.push({ label: 'Restored Size', value: formatBytes(result.restored_bytes) });
  renderSummaryCards('bundleSummary', cards.length ? cards : [{ label: 'Status', value: '-' }]);
}

function renderBundleJob(job) {
  const progress = Math.round((Number(job.progress || 0)) * 100);
  const active = ['queued', 'running'].includes(job.status);
  $('exportBundleBtn').disabled = active;
  $('importBundleBtn').disabled = active;
  $('inspectBundleBtn').disabled = active;
  $('stopBundleBtn').disabled = !active || Boolean(job.cancel_requested);
  $('bundleJobStatus').textContent = `${job.status || '-'} / ${job.message || ''}`;
  $('bundleJobStatus').className = `job-status ${job.status || ''}`;
  $('bundleProgressBar').style.width = `${progress}%`;
  renderBundleSummary(job.result || {}, progress);
  const logs = job.logs || [];
  $('bundleJobLogs').textContent = logs.length
    ? logs.slice(-200).map(item => `[${item.time || '-'}] ${String(item.level || 'info').toUpperCase()} ${item.message || ''}`).join('\n')
    : 'No logs yet.';
}

function startPollingBundleJob(jobId) {
  state.bundleJobId = jobId;
  if (state.bundlePollTimer) clearInterval(state.bundlePollTimer);

  async function poll() {
    try {
      const res = await apiGet(`/api/jobs/${jobId}`);
      renderBundleJob(res.job);
      if (['completed', 'failed', 'cancelled'].includes(res.job.status)) {
        clearInterval(state.bundlePollTimer);
        state.bundlePollTimer = null;
        state.bundleJobId = null;
        $('exportBundleBtn').disabled = false;
        $('importBundleBtn').disabled = false;
        $('inspectBundleBtn').disabled = false;
        $('stopBundleBtn').disabled = true;
        if (res.job.type === 'portable_bundle_import' && res.job.status === 'completed') {
          await refreshInventory();
        }
      }
    } catch (err) {
      $('bundleJobStatus').textContent = `Bundle polling failed: ${err.message}`;
      clearInterval(state.bundlePollTimer);
      state.bundlePollTimer = null;
      state.bundleJobId = null;
      $('exportBundleBtn').disabled = false;
      $('importBundleBtn').disabled = false;
      $('inspectBundleBtn').disabled = false;
      $('stopBundleBtn').disabled = true;
    }
  }

  poll();
  state.bundlePollTimer = setInterval(poll, 1000);
}

async function exportBundle() {
  if (state.bundleJobId) return alert('A bundle job is already active.');
  const payload = getBundlePayload();
  if (!payload.include_data && !payload.include_feature) return alert('Select at least one layer to export.');
  try {
    $('exportBundleBtn').disabled = true;
    $('importBundleBtn').disabled = true;
    $('inspectBundleBtn').disabled = true;
    $('stopBundleBtn').disabled = false;
    const res = await apiPost('/api/data/bundle/export', payload);
    startPollingBundleJob(res.data.job_id);
  } catch (err) {
    $('exportBundleBtn').disabled = false;
    $('importBundleBtn').disabled = false;
    $('inspectBundleBtn').disabled = false;
    $('stopBundleBtn').disabled = true;
    alert(`Export bundle failed: ${err.message}`);
  }
}

async function importBundle() {
  if (state.bundleJobId) return alert('A bundle job is already active.');
  const payload = getBundlePayload();
  if (!payload.include_data && !payload.include_feature) return alert('Select at least one layer to import.');
  const replaceText = payload.replace_existing ? ' Existing data/feature shards will be replaced.' : '';
  if (!confirm(`Import portable bundle from ${payload.bundle_path}?${replaceText}`)) return;
  try {
    $('exportBundleBtn').disabled = true;
    $('importBundleBtn').disabled = true;
    $('inspectBundleBtn').disabled = true;
    $('stopBundleBtn').disabled = false;
    const res = await apiPost('/api/data/bundle/import', payload);
    startPollingBundleJob(res.data.job_id);
  } catch (err) {
    $('exportBundleBtn').disabled = false;
    $('importBundleBtn').disabled = false;
    $('inspectBundleBtn').disabled = false;
    $('stopBundleBtn').disabled = true;
    alert(`Import bundle failed: ${err.message}`);
  }
}

async function inspectBundle() {
  const payload = getBundlePayload();
  try {
    const res = await apiGet(`/api/data/bundle/inspect?path=${encodeURIComponent(payload.bundle_path)}`);
    const data = res.data || {};
    renderBundleSummary(data);
    const summary = (data.summary || []).map(row => (
      `${row.layer || '-'} / ${row.item_type || '-'}: ${row.files || 0} files, ${formatBytes(row.bytes || 0)}`
    ));
    $('bundleJobStatus').textContent = `inspected / ${data.bundle_path || payload.bundle_path}`;
    $('bundleJobStatus').className = 'job-status completed';
    $('bundleProgressBar').style.width = '100%';
    $('bundleJobLogs').textContent = summary.length ? summary.join('\n') : 'Bundle has no items.';
  } catch (err) {
    $('bundleJobStatus').textContent = `inspect failed / ${err.message}`;
    $('bundleJobStatus').className = 'job-status failed';
    renderBundleSummary({});
  }
}

async function stopBundleJob() {
  if (!state.bundleJobId) return;
  $('stopBundleBtn').disabled = true;
  $('bundleJobStatus').textContent = 'running / Cancel requested';
  try {
    await apiPost(`/api/jobs/${state.bundleJobId}/cancel`, {});
  } catch (err) {
    $('stopBundleBtn').disabled = false;
    alert(`Stop bundle job failed: ${err.message}`);
  }
}

function getSelectedFeatureFreqs() {
  return getCheckedValues('.featureFreq');
}

function getFeatureSequenceWindows() {
  const result = {};
  document.querySelectorAll('.featureWindow').forEach(input => {
    const freq = input.dataset.freq;
    const value = Number(input.value || 0);
    if (freq && value > 0) result[freq] = value;
  });
  return result;
}

function getFeatureDatasetPayload() {
  const frequencies = getSelectedFeatureFreqs();
  return {
    db_path: ($('featureDbPath').value || DEFAULT_DB_PATH).trim(),
    symbols_file: $('featureSymbolsFile').value.trim(),
    adjust: $('featureAdjust').value,
    trade_freq: $('featureTradeFreq').value,
    frequencies,
    start: $('featureStart').value || null,
    end: $('featureEnd').value || null,
    include_open_auction: $('featureOpenAuction').checked,
    max_decisions: Number($('featureMaxDecisions').value || 0),
    feature_build_chunk_size: Number($('featureBuildChunkSize').value || 16),
    feature_build_workers: Number($('featureBuildWorkers').value || 1),
    feature_low_memory: $('featureLowMemory') ? $('featureLowMemory').checked : true,
    market_parquet_cache_enabled: $('featureMarketParquetCache') ? $('featureMarketParquetCache').checked : true,
    market_parquet_cache_force: $('featureForceMarketParquetCache') ? $('featureForceMarketParquetCache').checked : false,
    market_parquet_cache_root: $('featureMarketParquetCacheRoot') ? $('featureMarketParquetCacheRoot').value.trim() : 'runtime_layer/data/market_parquet_cache',
    feature_incremental_enabled: $('featureIncrementalBuild') ? $('featureIncrementalBuild').checked : true,
    feature_force_rebuild_parts: $('featureForceRebuildParts') ? $('featureForceRebuildParts').checked : false,
    feature_intermediate_format: 'parquet',
    sequence_windows: getFeatureSequenceWindows(),
    output_dir: $('featureOutputDir').value.trim(),
  };
}

function renderFeatureSpec(spec) {
  if (!spec) return;
  state.featureSpec = spec;
  renderSummaryCards('featureContractSummary', [
    { label: 'Spec', value: `${spec.name || '-'} / ${spec.version || '-'}` },
    { label: 'Base', value: spec.base_frequency || '-' },
    { label: 'Trade', value: spec.trade_frequency || '-' },
    { label: 'Default Freqs', value: (spec.default_frequencies || []).join(', ') || '-' },
    { label: 'Stages', value: (spec.decision_stages || []).join(', ') || '-' },
    { label: 'Fields', value: `${(spec.market_fields || []).length} market / ${(spec.context_fields || []).length} context / ${(spec.constraint_fields || []).length} market rules / ${(spec.portfolio_fields || []).length + (spec.environment_fields || []).length} env contract` },
  ]);

  $('featureFrequencyPolicy').innerHTML = (spec.frequency_policy || []).map(item => `
    <div class="feature-note">
      <strong>${escapeHtml(item.freq)}</strong>
      <span>${escapeHtml(item.source)}</span>
      <small>${escapeHtml(item.visible_rule)}</small>
    </div>
  `).join('');

  $('featureIndicatorTableBody').innerHTML = (spec.indicator_details || []).map(item => `
    <tr>
      <td>${escapeHtml(item.name)}</td>
      <td>${item.included ? '<span class="status-pill ok">yes</span>' : '<span class="status-pill muted-pill">no</span>'}</td>
      <td>${escapeHtml((item.outputs || []).join(', ') || '-')}</td>
      <td>${escapeHtml(item.formula || '-')}</td>
    </tr>
  `).join('') || '<tr><td colspan="4" class="muted">No indicator details.</td></tr>';

  $('featureRuleTableBody').innerHTML = (spec.trading_rules || []).map(item => `
    <tr>
      <td>${escapeHtml(item.rule)}</td>
      <td>${escapeHtml(item.value)}</td>
      <td>${escapeHtml(item.note)}</td>
    </tr>
  `).join('') || '<tr><td colspan="3" class="muted">No trading rules.</td></tr>';

  $('featureFormatTableBody').innerHTML = (spec.table_formats || []).map(item => `
    <tr>
      <td>${escapeHtml(item.file)}</td>
      <td>${escapeHtml(item.grain)}</td>
      <td>${escapeHtml((item.columns || []).join(', '))}</td>
    </tr>
  `).join('') || '<tr><td colspan="3" class="muted">No output format.</td></tr>';
}

const INDICATOR_PARAM_LABELS = {
  macd: [['fast', 'Fast'], ['slow', 'Slow'], ['signal', 'Signal']],
  kd: [['lookback', 'Lookback'], ['smooth_k', 'K Smooth'], ['smooth_d', 'D Smooth']],
  efi: [['fast', 'Fast'], ['slow', 'Slow'], ['baseline', 'Baseline']],
  ema_channel: [['fast', 'Fast'], ['slow', 'Slow']],
};

function renderFeatureIndicatorEditor() {
  const config = state.featureIndicatorConfig;
  if (!config) return;
  $('featureIndicatorConfigPath').textContent = config.path || '';
  const frequencies = config.available_frequencies || [];
  $('featureIndicatorEditor').innerHTML = (config.indicators || []).map((item, index) => {
    const params = INDICATOR_PARAM_LABELS[item.kind] || [];
    return `
      <div class="indicator-editor" data-indicator-index="${index}">
        <div class="indicator-editor-head">
          <div class="indicator-editor-flags">
            <label class="checkbox-line no-margin"><input class="indicatorEnabled" type="checkbox" ${item.enabled ? 'checked' : ''} /> Enabled</label>
            <label class="checkbox-line no-margin"><input class="indicatorDefaultVisible" type="checkbox" ${item.default_visible ? 'checked' : ''} /> Default visible</label>
          </div>
          <button class="icon-danger indicatorDelete" type="button" title="Remove indicator" aria-label="Remove indicator">×</button>
        </div>
        <div class="form-grid two-cols">
          <label>ID<input class="indicatorId" value="${escapeHtml(item.id)}" /></label>
          <label>Type<select class="indicatorKind">${(config.supported_kinds || []).map(kind => `<option value="${escapeHtml(kind)}" ${kind === item.kind ? 'selected' : ''}>${escapeHtml(kind)}</option>`).join('')}</select></label>
          <label>Display<select class="indicatorRenderTarget" disabled>${(config.supported_render_targets || []).map(target => `<option value="${escapeHtml(target)}" ${target === item.render_target ? 'selected' : ''}>${escapeHtml(target)}</option>`).join('')}</select></label>
        </div>
        <div class="field-label">Frequencies</div>
        <div class="checkbox-grid indicatorFreqs">
          ${frequencies.map(freq => `<label class="checkbox-line no-margin"><input type="checkbox" value="${escapeHtml(freq)}" ${(item.frequencies || []).includes(freq) ? 'checked' : ''} /> ${escapeHtml(freq)}</label>`).join('')}
        </div>
        <div class="form-grid indicatorParams">
          ${params.map(([name, label]) => `<label>${escapeHtml(label)}<input type="number" min="1" max="5000" data-param="${escapeHtml(name)}" value="${Number(item.params?.[name] || 1)}" /></label>`).join('')}
        </div>
        <div class="muted compact-text">${escapeHtml((item.outputs || []).join(', ') || 'Outputs appear after save.')}</div>
      </div>
    `;
  }).join('') || '<div class="muted">No indicators configured.</div>';
}

function collectFeatureIndicators() {
  return [...document.querySelectorAll('.indicator-editor')].map(editor => {
    const params = {};
    editor.querySelectorAll('[data-param]').forEach(input => { params[input.dataset.param] = Number(input.value); });
    return {
      id: editor.querySelector('.indicatorId').value.trim(),
      kind: editor.querySelector('.indicatorKind').value,
      enabled: editor.querySelector('.indicatorEnabled').checked,
      default_visible: editor.querySelector('.indicatorDefaultVisible').checked,
      render_target: editor.querySelector('.indicatorRenderTarget').value,
      frequencies: [...editor.querySelectorAll('.indicatorFreqs input:checked')].map(input => input.value),
      params,
    };
  });
}

async function loadFeatureIndicators() {
  try {
    const res = await apiGet('/api/feature/indicators');
    state.featureIndicatorConfig = res.data;
    renderFeatureIndicatorEditor();
  } catch (err) {
    $('featureIndicatorConfigStatus').textContent = `Load failed: ${err.message}`;
  }
}

function addFeatureIndicator() {
  const config = state.featureIndicatorConfig;
  if (!config) return;
  const used = new Set((config.indicators || []).map(item => item.id));
  let suffix = 1;
  while (used.has(`ema_custom_${suffix}`)) suffix += 1;
  config.indicators.push({
    id: `ema_custom_${suffix}`,
    kind: 'ema_channel',
    enabled: true,
    default_visible: true,
    render_target: 'main_overlay',
    frequencies: ['5min', '30min', 'daily', 'weekly', 'monthly'],
    params: { fast: 34, slow: 55 },
    outputs: [],
  });
  renderFeatureIndicatorEditor();
}

async function saveFeatureIndicators() {
  try {
    const res = await apiPost('/api/feature/indicators/save', { indicators: collectFeatureIndicators() });
    state.featureIndicatorConfig = res.data;
    renderFeatureIndicatorEditor();
    await loadFeatureSpec();
    await loadModelInputBlueprint();
    await loadVisualizationChart();
    $('featureIndicatorConfigStatus').textContent = 'Indicator configuration saved and applied.';
  } catch (err) {
    $('featureIndicatorConfigStatus').textContent = `Save failed: ${err.message}`;
  }
}

async function loadFeatureSpec() {
  try {
    const res = await apiGet('/api/feature/spec');
    renderFeatureSpec(res.data);
  } catch (err) {
    $('modelInputStatus').textContent = `Load feature spec failed: ${err.message}`;
  }
}

function cloneModelInputValue(value) {
  return JSON.parse(JSON.stringify(value));
}

function modelInputItemId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function applyModelInputPayload(payload) {
  state.modelInputBlueprint = cloneModelInputValue(payload.blueprint || { items: [] });
  state.modelInputDefault = cloneModelInputValue(payload.default_blueprint || { items: [] });
  state.modelInputCatalog = payload.catalog || [];
  state.modelInputCompiled = payload.compiled || null;
  state.modelInputValidation = payload.validation || null;
  state.modelInputDirty = false;
  $('modelInputBlueprintPath').textContent = payload.path || 'config/features/model_input.json';
  renderModelInputBlueprint();
  renderModelInputCatalog();
}

async function loadModelInputBlueprint() {
  try {
    const res = await apiGet('/api/feature/model-input');
    applyModelInputPayload(res.data || {});
    $('modelInputStatus').textContent = 'Blueprint loaded. Feature rows define the exact model channel order.';
  } catch (err) {
    $('modelInputStatus').textContent = `Load blueprint failed: ${err.message}`;
    $('modelInputList').innerHTML = '<div class="muted">Blueprint unavailable.</div>';
  }
}

function modelInputCatalogItem(name) {
  return state.modelInputCatalog.find(item => item.name === name) || null;
}

function modelInputChannelBadges(item) {
  if (!item.enabled || !state.modelInputCompiled) return [];
  if (item.stream === 'market_sequence') {
    return (item.frequencies || []).map(freq => {
      const index = (state.modelInputCompiled.channels_by_frequency?.[freq] || []).indexOf(item.name);
      return index >= 0 ? `${freq} #${index + 1}` : null;
    }).filter(Boolean);
  }
  const channels = state.modelInputCompiled[item.stream] || [];
  const index = channels.indexOf(item.name);
  return index >= 0 ? [`${item.stream} #${index + 1}`] : [];
}

function renderModelInputSummary() {
  const compiled = state.modelInputCompiled || {};
  const items = state.modelInputBlueprint?.items || [];
  const enabled = items.filter(item => item.type === 'feature' && item.enabled !== false).length;
  const shapeText = Object.entries(compiled.shapes || {})
    .map(([freq, shape]) => `${freq} ${shape?.[0] || 0}x${shape?.[1] || 0}`)
    .join(' / ') || '-';
  renderSummaryCards('modelInputSummary', [
    { label: 'Enabled Features', value: enabled },
    { label: 'Market Shapes', value: shapeText },
    { label: 'Decision Context', value: compiled.decision_context_shape?.[0] ?? 0 },
    { label: 'Runtime State', value: compiled.runtime_state_shape?.[0] ?? 0 },
    { label: 'Schema', value: String(compiled.schema_hash || state.modelInputBlueprint?.schema_hash || '-').slice(0, 12) },
  ]);
}

function renderModelInputValidation() {
  const validation = state.modelInputValidation;
  if (!validation) {
    $('modelInputValidation').innerHTML = '<span class="muted">Waiting for validation.</span>';
    return;
  }
  const issues = [...(validation.errors || []), ...(validation.warnings || [])];
  if (!issues.length) {
    $('modelInputValidation').innerHTML = '<span class="status-pill ok">Valid</span><span>Model input order and shapes are ready.</span>';
    return;
  }
  $('modelInputValidation').innerHTML = `
    <span class="status-pill ${validation.valid ? 'warn' : 'bad'}">${validation.valid ? 'Warning' : 'Invalid'}</span>
    <div>${issues.map(item => `<div>${escapeHtml(item.message)}</div>`).join('')}</div>
  `;
}

function renderModelInputFeatureRow(item, index) {
  const catalog = modelInputCatalogItem(item.name) || {};
  const badges = modelInputChannelBadges(item);
  const frequencies = catalog.available_frequencies || [];
  const frequencyControls = item.stream === 'market_sequence'
    ? `<div class="model-input-frequencies">${frequencies.map(freq => `
        <label><input type="checkbox" data-model-input-frequency="${escapeHtml(freq)}" ${item.frequencies?.includes(freq) ? 'checked' : ''} />${escapeHtml(freq)}</label>
      `).join('')}</div>`
    : '';
  return `
    <div class="model-input-row model-input-feature ${item.enabled === false ? 'disabled' : ''}" data-model-input-index="${index}" draggable="true">
      <button class="model-input-drag" type="button" title="Drag to reorder" aria-label="Drag to reorder">⋮⋮</button>
      <label class="model-input-enable" title="Include in model input"><input type="checkbox" data-model-input-enabled ${item.enabled !== false ? 'checked' : ''} /></label>
      <div class="model-input-body">
        <div class="model-input-title-line">
          <strong>${escapeHtml(item.name)}</strong>
          <span class="model-input-stream">${escapeHtml(item.stream)}</span>
          ${(badges.length ? badges : ['not compiled']).map(value => `<span class="model-input-channel">${escapeHtml(value)}</span>`).join('')}
        </div>
        <div class="model-input-description">${escapeHtml(catalog.description || 'Feature definition is not available in the active specification.')}</div>
        ${frequencyControls}
        <details class="model-input-details">
          <summary>Definition</summary>
          <div><strong>Group:</strong> ${escapeHtml(catalog.group || '-')}</div>
          <div><strong>Type:</strong> ${escapeHtml(catalog.category || '-')}</div>
          <div><strong>Formula:</strong> ${escapeHtml(catalog.formula || '-')}</div>
          <div><strong>Clip:</strong> ${escapeHtml(catalog.clip ? catalog.clip.join(' to ') : 'none')}</div>
          <div><strong>Missing:</strong> zero</div>
        </details>
      </div>
      <div class="model-input-row-actions">
        <button type="button" data-model-input-action="up" title="Move up" aria-label="Move up">↑</button>
        <button type="button" data-model-input-action="down" title="Move down" aria-label="Move down">↓</button>
        <button type="button" data-model-input-action="insert" title="Insert after" aria-label="Insert after">＋</button>
        <button type="button" data-model-input-action="delete" class="danger-text" title="Remove" aria-label="Remove">×</button>
      </div>
    </div>
  `;
}

function renderModelInputBlueprint() {
  const items = state.modelInputBlueprint?.items || [];
  $('saveModelInputBlueprintBtn').textContent = state.modelInputDirty ? 'Save Blueprint *' : 'Save Blueprint';
  $('modelInputList').innerHTML = items.map((item, index) => {
    if (item.type === 'feature') return renderModelInputFeatureRow(item, index);
    if (item.type === 'group') {
      return `
        <div class="model-input-row model-input-group" data-model-input-index="${index}" draggable="true">
          <button class="model-input-drag" type="button" title="Drag to reorder" aria-label="Drag to reorder">⋮⋮</button>
          <input data-model-input-text="label" value="${escapeHtml(item.label || '')}" aria-label="Group title" />
          <div class="model-input-row-actions">
            <button type="button" data-model-input-action="up" title="Move up" aria-label="Move up">↑</button>
            <button type="button" data-model-input-action="down" title="Move down" aria-label="Move down">↓</button>
            <button type="button" data-model-input-action="insert" title="Insert after" aria-label="Insert after">＋</button>
            <button type="button" data-model-input-action="delete" class="danger-text" title="Remove" aria-label="Remove">×</button>
          </div>
        </div>`;
    }
    return `
      <div class="model-input-row model-input-comment" data-model-input-index="${index}" draggable="true">
        <button class="model-input-drag" type="button" title="Drag to reorder" aria-label="Drag to reorder">⋮⋮</button>
        <textarea data-model-input-text="text" rows="2" aria-label="Blueprint comment">${escapeHtml(item.text || '')}</textarea>
        <div class="model-input-row-actions">
          <button type="button" data-model-input-action="up" title="Move up" aria-label="Move up">↑</button>
          <button type="button" data-model-input-action="down" title="Move down" aria-label="Move down">↓</button>
          <button type="button" data-model-input-action="insert" title="Insert after" aria-label="Insert after">＋</button>
          <button type="button" data-model-input-action="delete" class="danger-text" title="Remove" aria-label="Remove">×</button>
        </div>
      </div>`;
  }).join('') || '<div class="muted">No blueprint items. Add a feature to begin.</div>';
  renderModelInputSummary();
  renderModelInputValidation();
}

function markModelInputDirty({ rerender = true } = {}) {
  state.modelInputDirty = true;
  $('saveModelInputBlueprintBtn').textContent = 'Save Blueprint *';
  $('modelInputStatus').textContent = 'Unsaved blueprint changes.';
  if (rerender) renderModelInputBlueprint();
  scheduleModelInputValidation();
}

function scheduleModelInputValidation() {
  if (modelInputValidationTimer) clearTimeout(modelInputValidationTimer);
  const generation = ++modelInputValidationGeneration;
  modelInputValidationTimer = setTimeout(async () => {
    try {
      const res = await apiPost('/api/feature/model-input/validate', { blueprint: state.modelInputBlueprint });
      if (generation !== modelInputValidationGeneration) return;
      state.modelInputValidation = res.data.validation;
      state.modelInputCompiled = res.data.compiled;
      renderModelInputBlueprint();
    } catch (err) {
      if (generation !== modelInputValidationGeneration) return;
      state.modelInputValidation = { valid: false, errors: [{ message: err.message }], warnings: [] };
      state.modelInputCompiled = null;
      renderModelInputBlueprint();
    }
  }, 250);
}

function moveModelInputItem(from, to) {
  const items = state.modelInputBlueprint?.items || [];
  if (from < 0 || from >= items.length || to < 0 || to >= items.length || from === to) return;
  const [item] = items.splice(from, 1);
  items.splice(to, 0, item);
  markModelInputDirty();
}

function insertModelInputItem(item, index = null) {
  const items = state.modelInputBlueprint?.items || [];
  const target = index === null ? items.length : Math.max(0, Math.min(index, items.length));
  items.splice(target, 0, item);
  markModelInputDirty();
}

function openModelInputCatalog(insertIndex = null) {
  state.modelInputInsertIndex = insertIndex;
  $('modelInputCatalog').classList.remove('hidden');
  $('modelInputCatalogSearch').value = '';
  renderModelInputCatalog();
  $('modelInputCatalogSearch').focus();
}

function renderModelInputCatalog() {
  const target = $('modelInputCatalogGroups');
  if (!target) return;
  const keyword = ($('modelInputCatalogSearch')?.value || '').trim().toLowerCase();
  const used = new Set((state.modelInputBlueprint?.items || [])
    .filter(item => item.type === 'feature')
    .map(item => `${item.stream}:${item.name}`));
  const groups = new Map();
  state.modelInputCatalog
    .filter(item => !keyword || `${item.name} ${item.group} ${item.description} ${item.formula}`.toLowerCase().includes(keyword))
    .forEach(item => {
      const key = `${item.stream} / ${item.group}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    });
  target.innerHTML = [...groups.entries()].map(([group, entries]) => `
    <section class="model-input-catalog-group">
      <h3>${escapeHtml(group)}</h3>
      ${entries.map(item => {
        const alreadyUsed = used.has(`${item.stream}:${item.name}`);
        return `<div class="model-input-catalog-item">
          <div><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.description || '')}</span></div>
          <button type="button" data-model-input-catalog-name="${escapeHtml(item.name)}" ${alreadyUsed ? 'disabled' : ''}>${alreadyUsed ? 'Added' : 'Add'}</button>
        </div>`;
      }).join('')}
    </section>
  `).join('') || '<div class="muted">No matching features.</div>';
}

function addModelInputCatalogFeature(name) {
  const catalog = modelInputCatalogItem(name);
  if (!catalog) return;
  insertModelInputItem({
    id: modelInputItemId(`feature_${catalog.stream}_${catalog.name}`),
    type: 'feature',
    name: catalog.name,
    stream: catalog.stream,
    frequencies: catalog.stream === 'market_sequence' ? [...(catalog.available_frequencies || [])] : [],
    missing_policy: 'zero',
    enabled: true,
  }, state.modelInputInsertIndex);
  if (state.modelInputInsertIndex !== null) state.modelInputInsertIndex += 1;
  renderModelInputCatalog();
}

async function saveModelInputBlueprint() {
  try {
    const res = await apiPost('/api/feature/model-input/save', { blueprint: state.modelInputBlueprint });
    applyModelInputPayload(res.data || {});
    $('modelInputStatus').textContent = 'Blueprint saved. New feature datasets will use this exact channel order.';
  } catch (err) {
    $('modelInputStatus').textContent = `Save failed: ${err.message}`;
  }
}

function resetModelInputBlueprint() {
  if (!state.modelInputDefault) return;
  if (!window.confirm('Reset the model input blueprint to the current Feature Spec defaults?')) return;
  state.modelInputBlueprint = cloneModelInputValue(state.modelInputDefault);
  markModelInputDirty();
}

function renderFeaturePreflight(data) {
  const checks = data?.checks || [];
  const stCoverage = data?.st_status_coverage || {};
  renderSummaryCards('featurePreflightSummary', [
    { label: 'Status', value: data?.status || '-' },
    { label: 'Symbols', value: data?.symbol_count ?? 0 },
    { label: 'Decisions', value: data?.decisions_estimate ?? 0 },
    { label: 'Maturity Target', value: data?.warmup_bars ?? '-' },
    { label: 'Est. MB', value: data?.estimated_numeric_mb ?? '-' },
    { label: 'Market Rows', value: Object.entries(data?.market_rows_estimate || {}).map(([freq, rows]) => `${freq}:${rows}`).join(' / ') || '-' },
    { label: 'ST Days', value: stCoverage.st_days ?? 0 },
    { label: 'Missing ST Days', value: stCoverage.missing_days ?? 0 },
  ]);
  $('featurePreflightTableBody').innerHTML = checks.map(item => `
    <tr>
      <td>${escapeHtml(item.name)}</td>
      <td><span class="status-pill ${item.status === 'pass' ? 'ok' : item.status === 'warn' ? 'warn' : 'bad'}">${escapeHtml(item.status)}</span></td>
      <td>${escapeHtml(item.message)}</td>
    </tr>
  `).join('') || '<tr><td colspan="3" class="muted">No checks.</td></tr>';
}

function renderFeaturePreview(data) {
  const decisions = data?.decisions || [];
  const available = data?.available_market_rows || {};
  $('featurePreviewNote').textContent = data?.note || 'No preview.';
  $('featurePreviewTableBody').innerHTML = decisions.map(row => {
    const rows = available[row.decision_id] || {};
    return `
      <tr>
        <td>${escapeHtml(String(row.decision_time || '-').slice(0, 16))}</td>
        <td>${escapeHtml(row.stage || '-')}</td>
        <td>${escapeHtml(row.symbol || '-')}</td>
        <td>${escapeHtml(formatNumber(row.execution_price, 3))}</td>
        <td>${escapeHtml(`${row.is_st ? 'ST' : 'Normal'} / ${formatPercent(row.limit_pct)} / B:${row.market_can_buy ? 'Y' : 'N'} S:${row.market_can_sell ? 'Y' : 'N'}`)}</td>
        <td>${escapeHtml(String(row.visible_bar_end || '-').slice(0, 16))}</td>
        <td>${escapeHtml(Object.entries(rows).map(([freq, count]) => `${freq}:${count}`).join(' / ') || '-')}</td>
      </tr>
    `;
  }).join('') || '<tr><td colspan="7" class="muted">No preview rows.</td></tr>';
}

function renderFeatureJob(job) {
  const progress = Math.round((Number(job.progress || 0)) * 100);
  const active = ['queued', 'running'].includes(job.status);
  const result = job.result || {};
  const summary = result.summary || {};
  const marketRows = result.market_rows || summary.market_rows || {};
  $('buildFeatureDatasetBtn').disabled = active;
  $('stopFeatureDatasetBtn').disabled = !active || Boolean(job.cancel_requested);
  $('featureJobStatus').textContent = `Build Status: ${job.status || '-'} / ${job.message || ''}`;
  $('featureJobStatus').className = `job-status ${job.status || ''}`;
  $('featureProgressBar').style.width = `${progress}%`;
  renderSummaryCards('featureSummary', [
    { label: 'Progress', value: `${progress}%` },
    { label: 'Decisions', value: result.decisions ?? summary.decisions ?? job.saved_rows ?? 0 },
    { label: 'Frequencies', value: (result.frequencies || summary.frequencies || []).join(', ') || '-' },
    { label: 'Output', value: result.output_dir || '-' },
    { label: 'Market Rows', value: Object.entries(marketRows).map(([freq, rows]) => `${freq}:${rows}`).join(' / ') || '-' },
  ]);
  const logs = job.logs || [];
  $('featureJobLogs').textContent = logs.length
    ? logs.slice(-200).map(item => `[${item.time || '-'}] ${String(item.level || 'info').toUpperCase()} ${item.message || ''}`).join('\n')
    : 'No logs yet.';
}

function startPollingFeatureJob(jobId) {
  state.featureJobId = jobId;
  if (state.featurePollTimer) clearInterval(state.featurePollTimer);

  async function poll() {
    try {
      const res = await apiGet(`/api/jobs/${jobId}`);
      renderFeatureJob(res.job);
      if (['completed', 'failed', 'cancelled'].includes(res.job.status)) {
        clearInterval(state.featurePollTimer);
        state.featurePollTimer = null;
        state.featureJobId = null;
        $('buildFeatureDatasetBtn').disabled = false;
        $('stopFeatureDatasetBtn').disabled = true;
      }
    } catch (err) {
      $('featureJobStatus').textContent = `Job polling failed: ${err.message}`;
      clearInterval(state.featurePollTimer);
      state.featurePollTimer = null;
      $('buildFeatureDatasetBtn').disabled = false;
      $('stopFeatureDatasetBtn').disabled = !state.featureJobId;
    }
  }

  poll();
  state.featurePollTimer = setInterval(poll, 1000);
}

async function buildFeatureDataset() {
  let payload;
  try {
    payload = getFeatureDatasetPayload();
  } catch (err) {
    $('featureJobStatus').textContent = `Build Status: failed / invalid form: ${err.message}`;
    $('featureJobStatus').className = 'job-status failed';
    return;
  }
  if (!payload.frequencies.length) return alert('Select at least one input frequency.');
  if (state.featureJobId) return alert('A feature dataset build is already active.');

  try {
    $('buildFeatureDatasetBtn').disabled = true;
    $('stopFeatureDatasetBtn').disabled = false;
    $('featureJobStatus').textContent = 'Build Status: queued / Preparing symbol batches';
    $('featureJobStatus').className = 'job-status running';
    $('featureProgressBar').style.width = '0%';
    const res = await apiPost('/api/feature/build-dataset', payload);
    startPollingFeatureJob(res.data.job_id);
  } catch (err) {
    $('buildFeatureDatasetBtn').disabled = false;
    $('stopFeatureDatasetBtn').disabled = true;
    $('featureJobStatus').textContent = `Build Status: failed / ${err.message}`;
    $('featureJobStatus').className = 'job-status failed';
    alert(`Build feature dataset failed: ${err.message}`);
  }
}

async function stopFeatureDataset() {
  if (!state.featureJobId) return;
  $('stopFeatureDatasetBtn').disabled = true;
  $('featureJobStatus').textContent = 'Build Status: running / Cancel requested; finishing the current symbol safely';
  try {
    await apiPost(`/api/jobs/${state.featureJobId}/cancel`, {});
  } catch (err) {
    $('stopFeatureDatasetBtn').disabled = false;
    alert(`Stop feature dataset build failed: ${err.message}`);
  }
}

async function preflightFeatureDataset() {
  let payload;
  try {
    payload = getFeatureDatasetPayload();
  } catch (err) {
    $('featurePreflightStatus').textContent = `Preflight Status: failed / invalid form: ${err.message}`;
    $('featurePreflightStatus').className = 'job-status failed';
    $('featurePreflightTableBody').innerHTML = `<tr><td colspan="3" class="error-cell">Preflight failed before request: ${escapeHtml(err.message)}</td></tr>`;
    return;
  }
  if (!payload.frequencies.length) return alert('Select at least one input frequency.');
  const button = $('preflightFeatureDatasetBtn');
  if (button) button.disabled = true;
  $('featurePreflightStatus').textContent = 'Preflight Status: checking database coverage...';
  $('featurePreflightStatus').className = 'job-status running';
  try {
    const res = await apiPost('/api/feature/preflight', payload);
    renderFeaturePreflight(res.data);
    $('featurePreflightStatus').textContent = `Preflight Status: ${res.data.status || '-'}`;
    $('featurePreflightStatus').className = `job-status ${res.data.status === 'error' ? 'failed' : res.data.status || ''}`;
  } catch (err) {
    $('featurePreflightStatus').textContent = `Preflight Status: failed / ${err.message}`;
    $('featurePreflightStatus').className = 'job-status failed';
    $('featurePreflightTableBody').innerHTML = `<tr><td colspan="3" class="error-cell">Preflight failed: ${escapeHtml(err.message)}</td></tr>`;
  } finally {
    if (button) button.disabled = false;
  }
}

async function previewFeatureDataset() {
  let payload;
  try {
    payload = getFeatureDatasetPayload();
  } catch (err) {
    $('featurePreviewNote').textContent = `Preview failed before request: ${err.message}`;
    $('featurePreviewTableBody').innerHTML = `<tr><td colspan="7" class="error-cell">Preview failed before request: ${escapeHtml(err.message)}</td></tr>`;
    return;
  }
  if (!payload.frequencies.length) return alert('Select at least one input frequency.');
  const button = $('previewFeatureDatasetBtn');
  if (button) button.disabled = true;
  $('featurePreviewNote').textContent = 'Preview running...';
  try {
    const res = await apiPost('/api/feature/preview', payload);
    renderFeaturePreview(res.data);
  } catch (err) {
    $('featurePreviewNote').textContent = `Preview failed: ${err.message}`;
    $('featurePreviewTableBody').innerHTML = `<tr><td colspan="7" class="error-cell">Preview failed: ${escapeHtml(err.message)}</td></tr>`;
  } finally {
    if (button) button.disabled = false;
  }
}

async function loadAgentSpec() {
  try {
    const res = await apiGet('/api/agent/spec');
    state.agentSpec = res.data || {};
    applyAgentProfile('smoke');
    if ($('agentDevice')) {
      const cudaOption = $('agentDevice').querySelector('option[value="cuda"]');
      if (cudaOption) cudaOption.disabled = !state.agentSpec.cuda_available;
      if (!state.agentSpec.cuda_available) $('agentDevice').value = 'cpu';
    }
    renderAgentContract();
    await loadAgentRuns();
  } catch (err) {
    $('agentContractTableBody').innerHTML = `<tr><td colspan="3" class="error-cell">Agent spec failed: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function agentNumber(id) {
  return Number($(id)?.value || 0);
}

function setAgentValue(id, value) {
  const element = $(id);
  if (!element || value === undefined || value === null) return;
  if (element.type === 'checkbox') element.checked = Boolean(value);
  else element.value = String(value);
}

function setAgentFrequencies(frequencies) {
  const selected = new Set((frequencies || []).map(String));
  const useDefaults = selected.size === 0;
  document.querySelectorAll('.agentFrequency').forEach(input => {
    input.checked = useDefaults
      ? input.value !== 'monthly'
      : selected.has(input.value);
  });
}

function getAgentFrequencies() {
  return Array.from(document.querySelectorAll('.agentFrequency:checked')).map(input => input.value);
}

function applyAgentProfile(profile) {
  const config = state.agentSpec?.profiles?.[profile];
  if (!config) return;
  setAgentValue('agentProfile', profile);
  setAgentValue('agentRunName', '');
  setAgentValue('agentStorePath', config.store_path);
  setAgentValue('agentSymbolsFile', config.symbols_file || '');
  setAgentValue('agentSymbolLimit', config.symbol_limit);
  setAgentValue('agentSymbolSeed', config.symbol_seed);
  setAgentValue('agentFold', config.fold);
  setAgentValue('agentDevice', config.device);
  setAgentValue('agentParallelEnvs', config.parallel_envs);
  setAgentValue('agentUseCache', config.use_agent_cache);
  setAgentFrequencies(config.frequencies || state.agentSpec?.defaults?.frequencies || []);
  setAgentValue('agentTotalSteps', config.total_steps);
  setAgentValue('agentEpisodeDays', config.episode_days);
  setAgentValue('agentValidationDays', config.validation_days);
  setAgentValue('agentSeed', config.seed);
  setAgentValue('agentOutputDir', config.output_dir);
  const ppo = config.ppo || {};
  const model = config.model || {};
  const execution = config.execution || {};
  const reward = config.reward || {};
  const checkpoint = config.checkpoint || {};
  const validation = config.validation || {};
  const values = {
    agentLearningRate: ppo.learning_rate, agentFinalLearningRate: ppo.final_learning_rate,
    agentGamma: ppo.gamma, agentGaeLambda: ppo.gae_lambda,
    agentRolloutSteps: ppo.rollout_steps, agentMinibatchSize: ppo.minibatch_size,
    agentUpdateEpochs: ppo.update_epochs, agentClipRatio: ppo.clip_ratio,
    agentValueClip: ppo.value_clip, agentValueCoefficient: ppo.value_coefficient,
    agentEntropyCoefficient: ppo.entropy_coefficient,
    agentMaxGradientNorm: ppo.maximum_gradient_norm, agentTargetKl: ppo.target_kl,
    agentInputProjection: model.input_projection_size, agentLstmHidden: model.lstm_hidden_size,
    agentLstmLayers: model.lstm_layers, agentFusedMarket: model.fused_market_size,
    agentContextEmbedding: model.context_embedding_size,
    agentRuntimeEmbedding: model.runtime_embedding_size,
    agentLocalState: model.local_state_size, agentGlobalState: model.global_state_size,
    agentInitialCash: execution.initial_cash, agentLotSize: execution.lot_size,
    agentMaxPosition: execution.max_position_ratio,
    agentBarParticipation: execution.bar_participation_rate,
    agentAuctionParticipation: execution.auction_participation_rate,
    agentCommission: execution.commission_rate,
    agentMinimumCommission: execution.minimum_commission,
    agentStampDuty: execution.stamp_duty_rate,
    agentTransferFee: execution.transfer_fee_rate,
    agentBaseSlippage: execution.base_slippage_rate,
    agentImpactCoefficient: execution.impact_coefficient,
    agentMaximumSlippage: execution.maximum_slippage_rate,
    agentHoldingBarCap: execution.holding_bar_cap,
    agentRewardScale: reward.scale, agentHurdleRate: reward.hurdle_rate_annual,
    agentDrawdownPenalty: reward.drawdown_penalty,
    agentTurnoverPenalty: reward.turnover_penalty,
    agentInvalidPenalty: reward.invalid_action_penalty,
    agentCheckpointInterval: checkpoint.checkpoint_interval_updates,
    agentValidationInterval: checkpoint.validation_interval_updates,
    agentKeepLast: checkpoint.keep_last, agentBestMetric: checkpoint.best_metric,
    agentValidationSymbolLimit: validation.symbol_limit,
    agentValidationSymbolSeed: validation.symbol_seed,
    agentQuickValidationDays: validation.quick_days,
    agentPeriodicValidationDevice: validation.periodic_device,
    agentFinalValidationDevice: validation.final_device,
  };
  Object.entries(values).forEach(([id, value]) => setAgentValue(id, value));
}

function renderAgentContract() {
  const spec = state.agentSpec || {};
  const model = spec.model || {};
  const ppo = spec.ppo || {};
  const execution = spec.execution || {};
  const selectedFrequencies = getAgentFrequencies();
  renderSummaryCards('agentContractSummary', [
    { label: 'Algorithm', value: spec.algorithm || '-' },
    { label: 'Encoder', value: `${selectedFrequencies.length || 'All'} independent LSTMs` },
    { label: 'Action', value: 'SELL / HOLD / BUY + size' },
    { label: 'Parallel Envs', value: $('agentParallelEnvs')?.value || '1' },
    { label: 'Device', value: `${spec.cuda_available ? 'CUDA ready' : 'CPU'} / Torch ${spec.torch_version || '-'}` },
    { label: 'Gamma / GAE', value: `${ppo.gamma ?? '-'} / ${ppo.gae_lambda ?? '-'}` },
    { label: 'Reward', value: `Net NAV log return - hurdle ${formatPercent(agentNumber('agentHurdleRate') || 0)}` },
  ]);
  const rows = [
    ['Model', 'Frequencies', selectedFrequencies.join(', ') || 'Feature Dataset all'],
    ['Model', 'LSTM', `${model.lstm_layers ?? '-'} layers x ${model.lstm_hidden_size ?? '-'} hidden`],
    ['Model', 'Pooling', 'Masked attention; one stock per episode'],
    ['PPO', 'Clip / Target KL', `${ppo.clip_ratio ?? '-'} / ${ppo.target_kl ?? '-'}`],
    ['PPO', 'Rollout / Minibatch', `${ppo.rollout_steps ?? '-'} / ${ppo.minibatch_size ?? '-'}`],
    ['Execution', 'Position / Participation', `${formatPercent(execution.max_position_ratio)} / ${formatPercent(execution.bar_participation_rate)}`],
    ['Execution', 'Commission', `${formatPercent(execution.commission_rate)} / min CNY ${execution.minimum_commission ?? '-'}`],
    ['Execution', 'Sell Stamp Duty', formatPercent(execution.stamp_duty_rate)],
    ['Execution', 'Slippage', `${formatPercent(execution.base_slippage_rate)} base / ${formatPercent(execution.maximum_slippage_rate)} cap`],
    ['Validation', 'Protocol', '3 anchored folds / 20-day embargo / 252-day frozen test'],
  ];
  $('agentContractTableBody').innerHTML = rows.map(row => `<tr>${row.map(value => `<td>${escapeHtml(value)}</td>`).join('')}</tr>`).join('');
}

function getAgentPayload() {
  return {
    profile: $('agentProfile').value,
    run_name: $('agentRunName').value.trim(),
    store_path: $('agentStorePath').value.trim(),
    symbols_file: $('agentSymbolsFile').value.trim() || null,
    symbol_limit: agentNumber('agentSymbolLimit'),
    symbol_seed: agentNumber('agentSymbolSeed'),
    fold: Number($('agentFold').value || 3),
    device: $('agentDevice').value,
    parallel_envs: agentNumber('agentParallelEnvs'),
    use_agent_cache: Boolean($('agentUseCache')?.checked),
    frequencies: getAgentFrequencies(),
    total_steps: agentNumber('agentTotalSteps'), episode_days: agentNumber('agentEpisodeDays'),
    validation_days: agentNumber('agentValidationDays'), seed: agentNumber('agentSeed'),
    output_dir: $('agentOutputDir').value.trim(),
    ppo: {
      learning_rate: agentNumber('agentLearningRate'), final_learning_rate: agentNumber('agentFinalLearningRate'),
      gamma: agentNumber('agentGamma'), gae_lambda: agentNumber('agentGaeLambda'),
      rollout_steps: agentNumber('agentRolloutSteps'), minibatch_size: agentNumber('agentMinibatchSize'),
      update_epochs: agentNumber('agentUpdateEpochs'), clip_ratio: agentNumber('agentClipRatio'),
      value_clip: agentNumber('agentValueClip'), value_coefficient: agentNumber('agentValueCoefficient'),
      entropy_coefficient: agentNumber('agentEntropyCoefficient'),
      maximum_gradient_norm: agentNumber('agentMaxGradientNorm'), target_kl: agentNumber('agentTargetKl'),
    },
    model: {
      input_projection_size: agentNumber('agentInputProjection'), lstm_hidden_size: agentNumber('agentLstmHidden'),
      lstm_layers: agentNumber('agentLstmLayers'), fused_market_size: agentNumber('agentFusedMarket'),
      context_embedding_size: agentNumber('agentContextEmbedding'), runtime_embedding_size: agentNumber('agentRuntimeEmbedding'),
      local_state_size: agentNumber('agentLocalState'), global_state_size: agentNumber('agentGlobalState'), dropout: 0,
    },
    execution: {
      initial_cash: agentNumber('agentInitialCash'), lot_size: agentNumber('agentLotSize'),
      max_position_ratio: agentNumber('agentMaxPosition'), bar_participation_rate: agentNumber('agentBarParticipation'),
      auction_participation_rate: agentNumber('agentAuctionParticipation'), commission_rate: agentNumber('agentCommission'),
      minimum_commission: agentNumber('agentMinimumCommission'), stamp_duty_rate: agentNumber('agentStampDuty'),
      transfer_fee_rate: agentNumber('agentTransferFee'), base_slippage_rate: agentNumber('agentBaseSlippage'),
      impact_coefficient: agentNumber('agentImpactCoefficient'), maximum_slippage_rate: agentNumber('agentMaximumSlippage'),
      holding_bar_cap: agentNumber('agentHoldingBarCap'),
    },
    reward: {
      kind: 'net_asset_log_return', scale: agentNumber('agentRewardScale'),
      hurdle_rate_annual: agentNumber('agentHurdleRate'),
      drawdown_penalty: agentNumber('agentDrawdownPenalty'), turnover_penalty: agentNumber('agentTurnoverPenalty'),
      invalid_action_penalty: agentNumber('agentInvalidPenalty'),
    },
    checkpoint: {
      checkpoint_interval_updates: agentNumber('agentCheckpointInterval'),
      validation_interval_updates: agentNumber('agentValidationInterval'),
      keep_last: agentNumber('agentKeepLast'), best_metric: $('agentBestMetric').value,
    },
    validation: {
      symbol_limit: agentNumber('agentValidationSymbolLimit'),
      symbol_seed: agentNumber('agentValidationSymbolSeed'),
      quick_days: agentNumber('agentQuickValidationDays'),
      periodic_device: $('agentPeriodicValidationDevice').value,
      final_device: $('agentFinalValidationDevice').value,
    },
  };
}

function renderAgentPreflight(data) {
  const split = data?.splits;
  const selected = split?.folds?.find(item => Number(item.fold) === Number(data.selected_fold));
  renderSummaryCards('agentPreflightSummary', [
    { label: 'Status', value: data?.status || '-' },
    { label: 'Symbols', value: data?.symbols ?? data?.store?.summary?.symbols ?? 0 },
    { label: 'Validation Symbols', value: data?.validation_symbols ?? '-' },
    { label: 'Schema', value: String(data?.schema_hash || data?.store?.summary?.schema_hash || '-').slice(0, 12) },
    { label: 'Config', value: String(data?.config_hash || '-').slice(0, 12) },
    { label: 'Frequencies', value: (data?.frequencies || []).join(', ') || '-' },
    { label: 'Train', value: selected ? `${selected.train.start} -> ${selected.train.end}` : '-' },
    { label: 'Validation', value: selected ? `${selected.validation.start} -> ${selected.validation.end}` : '-' },
    { label: 'Frozen Test', value: split?.test ? `${split.test.start} -> ${split.test.end}` : '-' },
    { label: 'Device', value: data?.device || '-' },
  ]);
  $('agentPreflightTableBody').innerHTML = (data?.checks || []).map(item => `
    <tr><td>${escapeHtml(item.name)}</td><td><span class="status-pill ${item.status === 'pass' ? 'ok' : item.status === 'warn' ? 'warn' : 'bad'}">${escapeHtml(item.status)}</span></td><td>${escapeHtml(item.message)}</td></tr>
  `).join('') || '<tr><td colspan="3" class="muted">No checks.</td></tr>';
}


function getAgentCachePayload() {
  return {
    feature_dir: $('agentCacheFeatureDir').value.trim(),
    output_dir: $('agentCacheOutputDir').value.trim(),
    symbols_file: $('agentCacheSymbolsFile').value.trim() || null,
    workers: Number($('agentCacheWorkers').value || 1),
    max_decisions_per_symbol: Number($('agentCacheMaxDecisions').value || 0),
    reset: Boolean($('agentCacheReset')?.checked),
    frequencies: getAgentFrequencies(),
  };
}

function renderAgentCacheInspect(data = {}) {
  const manifest = data.manifest || data.inspect?.manifest || {};
  const samples = data.samples || data.inspect?.samples || [];
  const output = data.output_dir || manifest.output_dir || $('agentCacheOutputDir')?.value || '-';
  renderSummaryCards('agentCacheSummary', [
    { label: 'Output', value: output || '-' },
    { label: 'Symbols', value: manifest.symbol_count ?? manifest.symbols?.length ?? data.symbols ?? '-' },
    { label: 'Decisions', value: manifest.decision_count ?? data.decision_count ?? '-' },
    { label: 'Episode Rows', value: data.episode_index_rows ?? data.inspect?.episode_index_rows ?? '-' },
    { label: 'Frequencies', value: (manifest.frequencies || data.frequencies || []).join(', ') || '-' },
    { label: 'Schema', value: String(manifest.schema_hash || data.schema_hash || '-').slice(0, 12) },
    { label: 'Storage', value: manifest.storage || data.storage || '-' },
  ]);
  $('agentCacheTableBody').innerHTML = samples.map(item => {
    const shapes = Object.entries(item.arrays || {}).slice(0, 8).map(([name, shape]) => `${name}:${(shape || []).join('x')}`).join(' / ');
    return `<tr><td>${escapeHtml(item.symbol || '-')}</td><td>${escapeHtml(item.decision_count ?? '-')}</td><td>${escapeHtml(`${String(item.first_decision_time || '-').slice(0,10)} -> ${String(item.last_decision_time || '-').slice(0,10)}`)}</td><td>${escapeHtml(shapes || '-')}</td></tr>`;
  }).join('') || '<tr><td colspan="4" class="muted">No cache sample rows.</td></tr>';
}

function renderAgentCacheJob(job = {}) {
  const progress = Math.max(0, Math.min(100, Math.round(Number(job.progress || 0) * 100)));
  const active = ['queued', 'running'].includes(job.status);
  $('buildAgentCacheBtn').disabled = active;
  if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').disabled = !active;
  $('agentCacheStatus').textContent = `Agent Cache Status: ${job.status || '-'} / ${job.message || ''}`;
  $('agentCacheStatus').className = `job-status ${job.status || ''}`;
  $('agentCacheProgressBar').style.width = `${progress}%`;
  if (job.result) renderAgentCacheInspect(job.result);
}

function startPollingAgentCacheJob(jobId) {
  state.agentCacheJobId = jobId;
  if (state.agentCachePollTimer) clearInterval(state.agentCachePollTimer);
  async function poll() {
    try {
      const res = await apiGet(`/api/jobs/${jobId}`);
      renderAgentCacheJob(res.job);
      if (['completed', 'failed', 'cancelled'].includes(res.job.status)) {
        clearInterval(state.agentCachePollTimer);
        state.agentCachePollTimer = null;
        state.agentCacheJobId = null;
        $('buildAgentCacheBtn').disabled = false;
        if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').disabled = true;
      }
    } catch (err) {
      $('agentCacheStatus').textContent = `Agent Cache Status: polling failed / ${err.message}`;
      $('agentCacheStatus').className = 'job-status failed';
      clearInterval(state.agentCachePollTimer);
      state.agentCachePollTimer = null;
      $('buildAgentCacheBtn').disabled = false;
      if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').disabled = true;
    }
  }
  poll();
  state.agentCachePollTimer = setInterval(poll, 1000);
}

async function buildAgentCache() {
  if (state.agentCacheJobId) return alert('An Agent cache build is already active.');
  try {
    $('buildAgentCacheBtn').disabled = true;
    $('agentCacheStatus').textContent = 'Agent Cache Status: queued / building index-based cache';
    $('agentCacheStatus').className = 'job-status running';
    $('agentCacheProgressBar').style.width = '0%';
    const res = await apiPost('/api/agent/cache/build', getAgentCachePayload());
    startPollingAgentCacheJob(res.data.job_id);
  } catch (err) {
    $('buildAgentCacheBtn').disabled = false;
    if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').disabled = true;
    $('agentCacheStatus').textContent = `Agent Cache Status: failed / ${err.message}`;
    $('agentCacheStatus').className = 'job-status failed';
    alert(`Build Agent cache failed: ${err.message}`);
  }
}

async function stopAgentCacheBuild() {
  const jobId = state.agentCacheJobId;
  if (!jobId) return;
  try {
    if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').disabled = true;
    $('agentCacheStatus').textContent = 'Agent Cache Status: cancelling / stop requested';
    $('agentCacheStatus').className = 'job-status running';
    await apiPost(`/api/jobs/${jobId}/cancel`, {});
  } catch (err) {
    $('agentCacheStatus').textContent = `Agent Cache Status: cancel failed / ${err.message}`;
    $('agentCacheStatus').className = 'job-status failed';
    if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').disabled = false;
  }
}

async function inspectAgentCache() {
  try {
    $('agentCacheStatus').textContent = 'Agent Cache Status: inspecting cache';
    $('agentCacheStatus').className = 'job-status running';
    const res = await apiPost('/api/agent/cache/inspect', { cache_dir: $('agentCacheOutputDir').value.trim() });
    renderAgentCacheInspect(res.data);
    $('agentCacheStatus').textContent = 'Agent Cache Status: inspected';
    $('agentCacheStatus').className = 'job-status completed';
  } catch (err) {
    $('agentCacheStatus').textContent = `Agent Cache Status: failed / ${err.message}`;
    $('agentCacheStatus').className = 'job-status failed';
  }
}

function useAgentCachePathForTraining() {
  const value = $('agentCacheOutputDir').value.trim();
  if (!value) return;
  $('agentStorePath').value = value;
  $('agentPreflightStatus').textContent = `Preflight Status: idle / using Agent cache ${value}`;
}

async function preflightAgent() {
  const button = $('agentPreflightBtn');
  button.disabled = true;
  $('agentPreflightStatus').textContent = 'Preflight Status: validating Store and time splits...';
  $('agentPreflightStatus').className = 'job-status running';
  try {
    const res = await apiPost('/api/agent/runs/preflight', getAgentPayload());
    renderAgentPreflight(res.data);
    $('agentPreflightStatus').textContent = `Preflight Status: ${res.data.status || '-'}`;
    $('agentPreflightStatus').className = `job-status ${res.data.ok ? 'completed' : 'failed'}`;
  } catch (err) {
    $('agentPreflightStatus').textContent = `Preflight Status: failed / ${err.message}`;
    $('agentPreflightStatus').className = 'job-status failed';
  } finally {
    button.disabled = false;
  }
}

function setAgentConfigLocked(locked) {
  document.querySelectorAll('.agent-config-input').forEach(input => { input.disabled = locked; });
  $('agentPreflightBtn').disabled = locked;
  $('startAgentTrainingBtn').disabled = locked;
}

function showAgentTab(name) {
  state.agentActiveTab = name;
  document.querySelectorAll('.agent-tab').forEach(button => button.classList.toggle('active', button.dataset.agentTab === name));
  document.querySelectorAll('.agent-tab-panel').forEach(panel => panel.classList.remove('active'));
  $(`agentTab${name[0].toUpperCase()}${name.slice(1)}`)?.classList.add('active');
  if (name === 'runs') loadAgentRuns();
  if (name === 'monitor') drawAllAgentCharts();
}

const AGENT_STATUS_LABELS = {
  idle: 'Idle', queued: 'Queued', running: 'Running', paused: 'Paused', stopped: 'Stopped',
  completed: 'Completed', failed: 'Failed', interrupted: 'Interrupted', starting: 'Starting', pending: 'Pending',
};

const AGENT_PHASE_LABELS = {
  queued: 'Task created', launching: 'Launching training process', preflight: 'Validating training config',
  loading_feature_parts: 'Loading Feature Parts', preparing_data: 'Preparing data cache',
  building_splits: 'Building train/validation splits', building_model: 'Building model and optimizer',
  starting_rollout: 'Starting rollout', collecting: 'Collecting samples', optimizing: 'Optimizing policy', updated: 'Policy update',
  checkpointing: 'Saving checkpoint', training_completed: 'Training completed', paused: 'Paused',
  stopped: 'Stopped', failed: 'Failed', interrupted: 'Interrupted', completed: 'Completed',
};

function agentStatusLabel(value) {
  return AGENT_STATUS_LABELS[String(value || '').toLowerCase()] || value || '-';
}

function agentPhaseLabel(value) {
  return AGENT_PHASE_LABELS[String(value || '').toLowerCase()] || value || '-';
}

function formatAgentProgressLine(status = {}) {
  const statusLabel = agentStatusLabel(status.status);
  const phaseLabel = agentPhaseLabel(status.phase);
  const steps = status.steps ?? 0;
  const total = status.total_steps ?? '-';
  const heartbeatWarning = status.heartbeat_warning ? ' / Heartbeat is stale; worker is still alive' : '';
  return `Training Status: ${statusLabel} / ${phaseLabel} / ${steps} / ${total}${heartbeatWarning}`;
}

function formatAgentValidationLine(status = {}) {
  const stateText = agentStatusLabel(status.status || 'idle');
  const completed = Number(status.completed || 0);
  const total = Number(status.total || 0);
  const progress = total ? ` / ${completed}/${total}` : '';
  const pending = status.pending_task_id ? ' / Validation task pending' : '';
  return `Validation Status: ${stateText}${progress}${pending}`;
}

function mergeAgentRecords(existing, incoming, maxRecords = 12000) {
  const result = [];
  const seen = new Set();
  [...(existing || []), ...(incoming || [])].forEach(item => {
    const seq = Number(item?.seq || 0);
    const key = seq ? `seq:${seq}` : `${item?.kind || 'record'}:${item?.time || ''}:${result.length}`;
    if (seen.has(key)) return;
    seen.add(key);
    result.push(item);
  });
  result.sort((a, b) => Number(a?.seq || 0) - Number(b?.seq || 0));
  return result.length > maxRecords ? result.slice(-maxRecords) : result;
}

function renderAgentRun(data, { mergeRecords = false } = {}) {
  const job = data.status || {};
  const validationStatus = data.validation_status || {};
  const progress = Math.max(0, Math.min(100, Math.round(Number(job.progress || 0) * 100)));
  const active = ['queued', 'running'].includes(job.status);
  const paused = job.status === 'paused';
  const locked = active || paused;
  setAgentConfigLocked(locked);
  $('pauseAgentRunBtn').disabled = !active;
  $('resumeAgentRunBtn').disabled = !['paused', 'failed', 'interrupted', 'stopped'].includes(job.status) || !job.latest_checkpoint;
  $('checkpointAgentRunBtn').disabled = job.status !== 'running';
  $('stopAgentTrainingBtn').disabled = !active && !paused;
  $('agentJobStatus').textContent = formatAgentProgressLine(job);
  $('agentJobStatus').className = `job-status ${job.status || ''}`;
  $('agentProgressBar').style.width = `${progress}%`;
  const validationProgress = Math.max(0, Math.min(100, Math.round(Number(validationStatus.progress || 0) * 100)));
  $('agentValidationStatus').textContent = formatAgentValidationLine(validationStatus);
  $('agentValidationStatus').className = `job-status ${validationStatus.status || ''} agent-validation-status`;
  $('agentValidationProgressBar').style.width = `${validationProgress}%`;
  renderSummaryCards('agentTrainingSummary', [
    { label: 'Run', value: job.run_name || job.run_id || '-' },
    { label: 'Progress', value: `${progress}%` },
    { label: 'Phase', value: agentPhaseLabel(job.phase) },
    { label: 'Steps', value: `${job.steps ?? 0} / ${job.total_steps ?? '-'}` },
    { label: 'Updates / Episodes', value: `${job.updates ?? 0} / ${job.episodes ?? 0}` },
    { label: 'Parallel Envs', value: job.parallel_envs ?? '-' },
    { label: 'Speed', value: job.steps_per_second ? `${formatNumber(job.steps_per_second, 2)} step/s` : '-' },
    { label: 'Collect Speed', value: job.collect_steps_per_second ? `${formatNumber(job.collect_steps_per_second, 2)} sample/s` : '-' },
    { label: 'Collect / Optimize', value: `${formatAgentDuration(job.collect_seconds)} / ${formatAgentDuration(job.optimize_seconds)}` },
    { label: 'Env Init Time', value: formatAgentDuration(job.initialized_env_seconds) },
    { label: 'Model / Env Step', value: `${formatNumber(Number(job.model_seconds) * 1000, 2)} / ${formatNumber(Number(job.environment_seconds) * 1000, 2)} ms` },
    { label: 'Data / Reset Step', value: `${formatNumber(Number(job.data_load_seconds) * 1000, 2)} / ${formatNumber(Number(job.reset_seconds) * 1000, 2)} ms` },
    { label: 'Samples / Optimizer Steps', value: `${formatNumber(job.samples_per_update, 0)} / ${formatNumber(job.optimizer_steps, 0)}` },
    { label: 'ETA', value: formatAgentDuration(job.eta_seconds) },
    { label: 'Elapsed', value: formatAgentDuration(job.elapsed_seconds) },
    { label: 'Data Cache', value: job.data_cache_total ? `${job.data_cache_completed ?? 0} / ${job.data_cache_total}` : '-' },
    { label: 'Latest Checkpoint', value: job.latest_checkpoint || '-' },
    { label: 'Best Checkpoint', value: job.best_checkpoint || '-' },
  ]);
  state.agentRunMetrics = mergeRecords
    ? mergeAgentRecords(state.agentRunMetrics, data.metrics || [])
    : (data.metrics || []);
  state.agentRunLogs = mergeRecords
    ? mergeAgentRecords(state.agentRunLogs, data.logs || [], 600)
    : (data.logs || []);
  state.agentMetricSeq = state.agentRunMetrics.length ? Number(state.agentRunMetrics[state.agentRunMetrics.length - 1].seq || 0) : 0;
  state.agentLogSeq = state.agentRunLogs.length ? Number(state.agentRunLogs[state.agentRunLogs.length - 1].seq || 0) : 0;
  renderAgentTelemetry(validationStatus);
  renderAgentCheckpoints(data.checkpoints || []);
  $('agentJobLogs').textContent = state.agentRunLogs.length
    ? state.agentRunLogs.map(item => `[${item.time || '-'}] ${String(item.level || 'info').toUpperCase()} ${item.message || ''}`).join('\n')
    : 'No logs yet.';
  drawAllAgentCharts();
}

function formatAgentDuration(seconds) {
  if (!Number.isFinite(Number(seconds))) return '-';
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m ${total % 60}s`;
}

function renderAgentTelemetry(validationStatus = {}) {
  const trainingRows = state.agentRunMetrics.filter(item => item.kind === 'training');
  const latestStep = [...trainingRows].reverse().find(item => Number.isFinite(Number(item.nav))) || {};
  const latestUpdate = [...trainingRows].reverse().find(item => Number.isFinite(Number(item.policy_loss))) || {};
  const latestPerf = Number.isFinite(Number(latestUpdate.collect_seconds)) ? latestUpdate : latestStep;
  const validationRecord = [...state.agentRunMetrics].reverse().find(item => item.kind === 'validation') || {};
  const validation = validationRecord.metrics || {};
  const validationProgress = [...state.agentRunMetrics].reverse().find(item => item.kind === 'validation_progress') || {};
  renderSummaryCards('agentTelemetrySummary', [
    { label: 'Reward', value: formatNumber(latestStep.reward, 6) },
    { label: 'Asset / Cash', value: `${formatNumber(latestStep.nav, 2)} / ${formatNumber(latestStep.cash, 2)}` },
    { label: 'Total Loss', value: formatNumber(latestUpdate.loss, 6) },
    { label: 'Policy / Value Loss', value: `${formatNumber(latestUpdate.policy_loss, 5)} / ${formatNumber(latestUpdate.value_loss, 5)}` },
    { label: 'KL / Clip', value: `${formatNumber(latestUpdate.approximate_kl, 5)} / ${formatNumber(latestUpdate.clip_fraction, 4)}` },
    { label: 'Entropy / Gradient', value: `${formatNumber(latestUpdate.entropy, 5)} / ${formatNumber(latestUpdate.gradient_norm, 4)}` },
    { label: 'Learning Rate', value: formatNumber(latestUpdate.learning_rate, 8) },
    { label: 'Buy / Hold / Sell', value: `${latestStep.buy_actions ?? '-'} / ${latestStep.hold_actions ?? '-'} / ${latestStep.sell_actions ?? '-'}` },
    { label: 'Executed / Blocked', value: `${latestStep.executed_orders ?? '-'} / ${latestStep.blocked_orders ?? '-'}` },
    { label: 'Blocked Reasons', value: Object.entries(latestStep.blocked_reasons || {}).map(([key, value]) => `${key}:${value}`).join(', ') || '-' },
    { label: 'Turnover / Fees', value: `${formatNumber(latestStep.turnover_value, 2)} / ${formatNumber(latestStep.total_fees, 2)}` },
    { label: 'Collect Speed', value: latestPerf.collect_steps_per_second ? `${formatNumber(latestPerf.collect_steps_per_second, 2)} sample/s` : '-' },
    { label: 'Collect / Optimize Time', value: `${formatAgentDuration(latestPerf.collect_seconds)} / ${formatAgentDuration(latestPerf.optimize_seconds)}` },
    { label: 'Env Init Time', value: formatAgentDuration(latestPerf.initialized_env_seconds) },
    { label: 'Model / Env Step', value: `${formatNumber(Number(latestPerf.model_seconds) * 1000, 2)} / ${formatNumber(Number(latestPerf.environment_seconds) * 1000, 2)} ms` },
    { label: 'Data / Reset Step', value: `${formatNumber(Number(latestPerf.data_load_seconds) * 1000, 2)} / ${formatNumber(Number(latestPerf.reset_seconds) * 1000, 2)} ms` },
    { label: 'Data Chunks / Steps', value: `${latestPerf.data_chunks ?? '-'} / ${latestPerf.data_steps ?? '-'}` },
    { label: 'Samples / Optimizer Steps', value: `${formatNumber(latestPerf.samples_per_update, 0)} / ${formatNumber(latestPerf.optimizer_steps, 0)}` },
  ]);
  renderSummaryCards('agentValidationSummary', [
    { label: 'State', value: agentStatusLabel(validationStatus.status || validationRecord.validation_kind || '-') },
    { label: 'Progress', value: validationStatus.total ? `${validationStatus.completed ?? 0} / ${validationStatus.total}` : (validationProgress.total ? `${validationProgress.completed ?? 0} / ${validationProgress.total}` : '-') },
    { label: 'Sample', value: validationRecord.symbols ? `${validationRecord.symbols} symbols / ${validationRecord.days} days` : '-' },
    { label: 'Return', value: validation.total_return === undefined ? '-' : formatPercent(validation.total_return) },
    { label: 'Sharpe', value: formatNumber(validation.sharpe, 3) },
    { label: 'Max Drawdown', value: validation.maximum_drawdown === undefined ? '-' : formatPercent(validation.maximum_drawdown) },
    { label: 'Calmar', value: formatNumber(validation.calmar, 3) },
    { label: 'Observations', value: validation.return_observations ?? '-' },
  ]);
}


const AGENT_CHART_COLORS = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#db2777', '#475569'];
const AGENT_CHART_GROUPS_STORAGE_KEY = 'pocketagent.agent.chartGroups';
const AGENT_CHART_MAX_POINTS = 1400;

function readAgentChartGroupPrefs() {
  try {
    state.agentChartGroupsCollapsed = JSON.parse(window.localStorage.getItem(AGENT_CHART_GROUPS_STORAGE_KEY) || '{}') || {};
  } catch (_err) {
    state.agentChartGroupsCollapsed = {};
  }
}

function persistAgentChartGroupPrefs() {
  try {
    window.localStorage.setItem(AGENT_CHART_GROUPS_STORAGE_KEY, JSON.stringify(state.agentChartGroupsCollapsed || {}));
  } catch (_err) {
    // Chart group persistence is optional.
  }
}

function applyAgentChartGroupState() {
  document.querySelectorAll('[data-agent-chart-group]').forEach(section => {
    const group = section.dataset.agentChartGroup;
    section.classList.toggle('collapsed', Boolean(state.agentChartGroupsCollapsed?.[group]));
  });
}

function agentNumberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function agentNestedValue(row, path) {
  if (!row || !path) return undefined;
  return String(path).split('.').reduce((current, key) => (
    current && Object.prototype.hasOwnProperty.call(current, key) ? current[key] : undefined
  ), row);
}

function agentSeriesValue(series, row, index, rows) {
  if (typeof series.value === 'function') return agentNumberOrNull(series.value(row, index, rows));
  const raw = agentNestedValue(row, series.key);
  const number = agentNumberOrNull(raw);
  if (number === null) return null;
  return number * (series.scale === undefined ? 1 : Number(series.scale));
}

function agentRelativePercent(path) {
  return (_row, index, rows) => {
    const current = agentNumberOrNull(agentNestedValue(rows[index], path));
    if (current === null) return null;
    const baseRow = rows.find(item => agentNumberOrNull(agentNestedValue(item, path)) !== null);
    const base = agentNumberOrNull(agentNestedValue(baseRow, path));
    if (base === null || base === 0) return null;
    return (current / base - 1) * 100;
  };
}

function agentXValue(row, index, mode = 'steps') {
  if (mode === 'seq') return Number(row.seq ?? index + 1);
  if (mode === 'updates') return Number(row.updates ?? row.seq ?? index + 1);
  if (mode === 'validation') return Number(row.updates ?? row.steps ?? row.seq ?? index + 1);
  return Number(row.steps ?? row.seq ?? index + 1);
}

function agentRowTimeMs(row, index = 0) {
  const parsed = Date.parse(row?.time || row?.updated_at || row?.created_at || '');
  if (Number.isFinite(parsed)) return parsed;
  const elapsed = Number(row?.elapsed_seconds);
  if (Number.isFinite(elapsed)) return elapsed * 1000;
  return index * 30000;
}

function agentDisplayBucketMs(rows) {
  const clean = Array.isArray(rows) ? rows.filter(Boolean) : [];
  if (clean.length < 2) return 0;
  const start = agentRowTimeMs(clean[0], 0);
  const end = agentRowTimeMs(clean[clean.length - 1], clean.length - 1);
  const duration = Math.max(0, end - start);
  if (duration >= 3 * 3600 * 1000) return 5 * 60 * 1000;
  if (duration >= 30 * 60 * 1000) return 60 * 1000;
  return 30 * 1000;
}

function aggregateAgentRowsByTime(rows, bucketMs = 0) {
  const clean = Array.isArray(rows) ? rows.filter(Boolean) : [];
  if (!bucketMs || clean.length <= 2) return clean;
  const buckets = new Map();
  clean.forEach((row, index) => {
    const bucket = Math.floor(agentRowTimeMs(row, index) / bucketMs) * bucketMs;
    buckets.set(bucket, { ...(buckets.get(bucket) || {}), ...row, _bucket_time: bucket, _bucket_count: (buckets.get(bucket)?._bucket_count || 0) + 1 });
  });
  const result = [...buckets.entries()].sort((a, b) => a[0] - b[0]).map(([, row]) => row);
  const last = clean[clean.length - 1];
  if (result.length && result[result.length - 1]?.seq !== last.seq) result[result.length - 1] = { ...result[result.length - 1], ...last };
  return result;
}

function decimateAgentRows(rows, maxPoints = AGENT_CHART_MAX_POINTS) {
  const clean = aggregateAgentRowsByTime(rows, agentDisplayBucketMs(rows));
  if (clean.length <= maxPoints) return clean;
  const result = [];
  const stride = clean.length / maxPoints;
  for (let i = 0; i < maxPoints; i += 1) {
    result.push(clean[Math.min(clean.length - 1, Math.floor(i * stride))]);
  }
  const last = clean[clean.length - 1];
  if (result[result.length - 1] !== last) result[result.length - 1] = last;
  return result;
}

function agentRowHasSeriesValue(row, index, rows, series) {
  return series.some(item => Number.isFinite(agentSeriesValue(item, row, index, rows)));
}

function agentRowsForChart(rows, series) {
  const clean = Array.isArray(rows) ? rows.filter(Boolean) : [];
  if (!clean.length || !series.length) return [];
  return clean.filter((row, index) => agentRowHasSeriesValue(row, index, clean, series));
}

function agentNiceTicks(min, max, count = 5) {
  const low = Number(min);
  const high = Number(max);
  if (!Number.isFinite(low) || !Number.isFinite(high)) return [];
  const span = Math.abs(high - low) || 1;
  const rough = span / Math.max(1, count - 1);
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const fraction = rough / pow;
  const step = (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * pow;
  const start = Math.ceil(low / step) * step;
  const result = [];
  for (let value = start; value <= high + step * 0.5 && result.length <= count + 2; value += step) {
    result.push(value);
  }
  return result.length ? result : [low, high];
}

function agentMetricFormat(value, digits = 3, suffix = '') {
  if (value === null || value === undefined || value === '') return '-';
  if (!Number.isFinite(Number(value))) return '-';
  const abs = Math.abs(Number(value));
  let formatted;
  if (abs >= 1000000) formatted = `${formatNumber(Number(value) / 1000000, 2)}M`;
  else if (abs >= 10000) formatted = `${formatNumber(Number(value) / 1000, 1)}K`;
  else formatted = formatNumber(Number(value), abs >= 100 ? 0 : digits);
  return `${formatted}${suffix || ''}`;
}

function agentChartHiddenKey(canvasId, series) {
  return `${canvasId}:${series.key || series.label}`;
}

function agentVisibleSeries(canvasId, series) {
  return series.filter(item => !state.agentChartHidden[agentChartHiddenKey(canvasId, item)]);
}

function renderAgentChartLegend(canvasId, series, rows, hoverIndex) {
  const legend = document.querySelector(`[data-agent-chart-legend="${canvasId}"]`);
  if (!legend) return;
  if (!series.length) {
    legend.innerHTML = '<span class="agent-chart-empty">No data</span>';
    return;
  }
  const index = Number.isInteger(hoverIndex) ? hoverIndex : null;
  legend.innerHTML = series.map((item, seriesIndex) => {
    const hidden = Boolean(state.agentChartHidden[agentChartHiddenKey(canvasId, item)]);
    const value = index === null ? null : agentSeriesValue(item, rows[index] || {}, index, rows);
    const color = item.color || AGENT_CHART_COLORS[seriesIndex % AGENT_CHART_COLORS.length];
    const valueHtml = index === null ? '' : `<strong>${escapeHtml(agentMetricFormat(value, item.digits ?? 3, item.suffix || ''))}</strong>`;
    return `<button class="agent-chart-legend-item ${hidden ? 'hidden-series' : ''}" type="button" data-agent-legend-chart="${escapeHtml(canvasId)}" data-agent-legend-key="${escapeHtml(agentChartHiddenKey(canvasId, item))}"><span class="agent-chart-legend-swatch" style="background:${escapeHtml(color)}"></span><span>${escapeHtml(item.label)}</span>${valueHtml}</button>`;
  }).join('');
}

function drawAgentEmptyChart(ctx, width, height, message) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#9ca3af';
  ctx.font = '13px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message || 'No data', width / 2, height / 2);
}

function drawAgentMetricChart(canvasId, config) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const width = Math.max(360, canvas.clientWidth || 720);
  const height = Math.max(210, canvas.clientHeight || 230);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const allSeries = config.series || [];
  const visibleSeries = agentVisibleSeries(canvasId, allSeries);
  const scopedRows = agentRowsForChart(config.rows || [], visibleSeries.length ? visibleSeries : allSeries);
  const rows = decimateAgentRows(scopedRows, config.maxPoints || AGENT_CHART_MAX_POINTS);
  const pad = { left: 58, right: 18, top: 18, bottom: 42 };
  const plotW = Math.max(1, width - pad.left - pad.right);
  const plotH = Math.max(1, height - pad.top - pad.bottom);
  canvas.__agentChartLayout = { pad, plotW, rowsCount: 0, pointXs: [] };
  const xMode = config.xMode || 'steps';
  const xValues = rows.map((row, index) => agentXValue(row, index, xMode));
  const yValues = [];
  visibleSeries.forEach(series => {
    rows.forEach((row, index) => {
      const value = agentSeriesValue(series, row, index, rows);
      if (Number.isFinite(value)) yValues.push(value);
    });
  });
  const finiteX = xValues.filter(Number.isFinite);
  const hoverState = state.agentChartHover?.[canvasId] || {};
  const hoverIndex = rows.length && Number.isInteger(hoverState.index)
    ? Math.max(0, Math.min(rows.length - 1, hoverState.index))
    : null;
  renderAgentChartLegend(canvasId, allSeries, rows, hoverIndex);
  if (!rows.length || !visibleSeries.length || !yValues.length || !finiteX.length) {
    drawAgentEmptyChart(ctx, width, height, !rows.length ? 'No data' : 'No visible metric data');
    return;
  }
  let minY = Math.min(...yValues);
  let maxY = Math.max(...yValues);
  if (minY === maxY) {
    const delta = Math.max(1, Math.abs(minY) * 0.08);
    minY -= delta;
    maxY += delta;
  } else {
    const padding = (maxY - minY) * 0.08;
    minY -= padding;
    maxY += padding;
  }
  const minX = Math.min(...finiteX);
  const maxX = Math.max(...finiteX);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  const toX = value => pad.left + ((Number(value) - minX) / xSpan) * plotW;
  const toY = value => pad.top + (1 - ((Number(value) - minY) / ySpan)) * plotH;
  canvas.__agentChartLayout = { pad, plotW, rowsCount: rows.length, pointXs: xValues.map(value => toX(value)) };
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#cbd5e1';
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();
  ctx.font = '11px sans-serif';
  ctx.fillStyle = '#64748b';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  agentNiceTicks(minY, maxY, 5).forEach(tick => {
    const y = toY(tick);
    ctx.strokeStyle = '#eef2f7';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.fillText(agentMetricFormat(tick, config.yDigits ?? 3, config.ySuffix || ''), pad.left - 8, y);
  });
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  agentNiceTicks(minX, maxX, 4).forEach(tick => {
    const x = toX(tick);
    ctx.strokeStyle = '#f1f5f9';
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.fillText(agentMetricFormat(tick, 0, ''), x, pad.top + plotH + 7);
  });
  ctx.textAlign = 'center';
  ctx.fillStyle = '#334155';
  ctx.fillText(config.xLabel || (xMode === 'seq' ? 'record' : 'step'), pad.left + plotW / 2, height - 13);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(config.yLabel || 'value', pad.left, 3);

  visibleSeries.forEach((series, seriesIndex) => {
    const color = series.color || AGENT_CHART_COLORS[seriesIndex % AGENT_CHART_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    let open = false;
    rows.forEach((row, index) => {
      const xValue = xValues[index];
      const yValue = agentSeriesValue(series, row, index, rows);
      if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) {
        return;
      }
      const x = toX(xValue);
      const y = toY(yValue);
      if (!open) {
        ctx.moveTo(x, y);
        open = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });

  if (Number.isInteger(hoverIndex) && rows[hoverIndex]) {
    const x = toX(xValues[hoverIndex]);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();
    ctx.restore();
    const tooltipRows = visibleSeries
      .map((series, seriesIndex) => ({
        series,
        color: series.color || AGENT_CHART_COLORS[seriesIndex % AGENT_CHART_COLORS.length],
        value: agentSeriesValue(series, rows[hoverIndex], hoverIndex, rows),
      }))
      .filter(item => Number.isFinite(item.value));
    tooltipRows.forEach(item => {
      const y = toY(item.value);
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
    if (tooltipRows.length) {
      const lines = [`${config.xLabel || (xMode === 'seq' ? 'record' : 'step')}: ${agentMetricFormat(xValues[hoverIndex], 0)}`]
        .concat(tooltipRows.slice(0, 5).map(item => `${item.series.label}: ${agentMetricFormat(item.value, item.series.digits ?? 3, item.series.suffix || '')}`));
      ctx.font = '11px sans-serif';
      const boxW = Math.min(220, Math.max(...lines.map(line => ctx.measureText(line).width)) + 18);
      const boxH = 18 + lines.length * 16;
      const boxX = Math.min(width - boxW - 8, Math.max(pad.left + 4, x + 10));
      const boxY = pad.top + 8;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.fillRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      lines.forEach((line, index) => ctx.fillText(line, boxX + 9, boxY + 8 + index * 16));
    }
  }
}

function drawAllAgentCharts() {
  if (state.agentActiveTab !== 'monitor') return;
  applyAgentChartGroupState();
  const trainingRows = state.agentRunMetrics.filter(item => item.kind === 'training');
  const validationRows = state.agentRunMetrics.filter(item => item.kind === 'validation');
  drawAgentMetricChart('agentStepsChart', {
    rows: trainingRows,
    xMode: 'seq',
    xLabel: 'Record',
    yLabel: 'Count',
    series: [
      { key: 'steps', label: 'steps', digits: 0 },
      { key: 'updates', label: 'updates', digits: 0 },
      { key: 'episodes', label: 'episodes', digits: 0 },
    ],
  });
  drawAgentMetricChart('agentSpeedChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'Speed',
    series: [
      { key: 'steps_per_second', label: 'steps/s', digits: 2 },
      { key: 'collect_steps_per_second', label: 'collect/s', digits: 2 },
      { key: 'optimize_steps_per_second', label: 'optimize/s', digits: 2 },
      { key: 'parallel_envs', label: 'parallel envs', digits: 0 },
    ],
  });
  drawAgentMetricChart('agentHardwareChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'Percent',
    ySuffix: '%',
    series: [
      { key: 'cpu_percent', label: 'CPU', suffix: '%', digits: 1 },
      { key: 'memory_percent', label: 'RAM', suffix: '%', digits: 1 },
      { key: 'gpu_percent', label: 'GPU', suffix: '%', digits: 1 },
      { key: 'gpu_memory_percent', label: 'VRAM', suffix: '%', digits: 1 },
    ],
  });
  drawAgentMetricChart('agentLatencyChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'Milliseconds',
    series: [
      { key: 'model_seconds', label: 'model ms', scale: 1000, digits: 2 },
      { key: 'environment_seconds', label: 'env ms', scale: 1000, digits: 2 },
      { key: 'data_load_seconds', label: 'data ms', scale: 1000, digits: 2 },
      { key: 'reset_seconds', label: 'reset ms', scale: 1000, digits: 2 },
      { key: 'initialized_env_seconds', label: 'init ms', scale: 1000, digits: 2 },
      { key: 'collect_seconds', label: 'collect ms', scale: 1000, digits: 2 },
      { key: 'optimize_seconds', label: 'optimize ms', scale: 1000, digits: 2 },
    ],
  });
  drawAgentMetricChart('agentRewardChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'Return / Reward',
    series: [
      { key: 'reward', label: 'reward', digits: 6 },
      { key: 'nav_pct', value: agentRelativePercent('nav'), label: 'NAV change', suffix: '%', digits: 3 },
    ],
  });
  drawAgentMetricChart('agentActionChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'Count / Cost',
    series: [
      { key: 'buy_actions', label: 'buy', digits: 0 },
      { key: 'hold_actions', label: 'hold', digits: 0 },
      { key: 'sell_actions', label: 'sell', digits: 0 },
      { key: 'blocked_orders', label: 'blocked', digits: 0 },
      { key: 'total_fees', label: 'fees', digits: 2 },
    ],
  });
  drawAgentMetricChart('agentLossChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'Loss',
    series: [
      { key: 'loss', label: 'total', digits: 5 },
      { key: 'policy_loss', label: 'policy', digits: 5 },
      { key: 'value_loss', label: 'value', digits: 5 },
    ],
  });
  drawAgentMetricChart('agentPolicyChart', {
    rows: trainingRows,
    xMode: 'steps',
    xLabel: 'Step',
    yLabel: 'PPO Health',
    series: [
      { key: 'entropy', label: 'entropy', digits: 5 },
      { key: 'approximate_kl', label: 'KL', digits: 5 },
      { key: 'clip_fraction', label: 'clip', digits: 4 },
      { key: 'learning_rate', label: 'LR', digits: 8 },
    ],
  });
  drawAgentMetricChart('agentValidationReturnChart', {
    rows: validationRows,
    xMode: 'validation',
    xLabel: 'Validation',
    yLabel: 'Percent',
    ySuffix: '%',
    series: [
      { key: 'metrics.total_return', label: 'return', scale: 100, suffix: '%', digits: 2 },
      { key: 'metrics.maximum_drawdown', label: 'drawdown', scale: 100, suffix: '%', digits: 2 },
    ],
  });
  drawAgentMetricChart('agentValidationRiskChart', {
    rows: validationRows,
    xMode: 'validation',
    xLabel: 'Validation',
    yLabel: 'Score',
    series: [
      { key: 'metrics.sharpe', label: 'sharpe', digits: 3 },
      { key: 'metrics.calmar', label: 'calmar', digits: 3 },
      { key: 'metrics.return_observations', label: 'observations', digits: 0 },
    ],
  });
}


function agentPollingIntervalMs(status = {}) {
  const steps = Number(status.steps || 0);
  const points = state.agentRunMetrics.length;
  if (steps >= 1000000 || points > 6000) return 30000;
  if (steps >= 300000 || points > 3000) return 15000;
  if (steps >= 50000 || points > 1000) return 5000;
  return 2000;
}

function startPollingAgentRun(runId) {
  state.agentRunId = runId;
  if (state.agentPollTimer) clearTimeout(state.agentPollTimer);
  const generation = ++state.agentPollGeneration;
  state.agentPollBusy = false;
  async function poll() {
    if (state.agentPollBusy || generation !== state.agentPollGeneration) return;
    state.agentPollBusy = true;
    let nextStatus = {};
    let continuePolling = true;
    try {
      const firstLoad = state.agentLoadedRunId !== runId;
      const detailPromise = apiGet(`/api/agent/runs/${runId}${firstLoad ? '' : '?records=0'}`);
      const [res, metrics, logs] = await Promise.all([
        detailPromise,
        firstLoad ? Promise.resolve({ data: [] }) : apiGet(`/api/agent/runs/${runId}/metrics?after=${state.agentMetricSeq}`),
        firstLoad ? Promise.resolve({ data: [] }) : apiGet(`/api/agent/runs/${runId}/logs?after=${state.agentLogSeq}`),
      ]);
      if (!firstLoad) {
        res.data.metrics = metrics.data || [];
        res.data.logs = logs.data || [];
      }
      if (generation !== state.agentPollGeneration) return;
      renderAgentRun(res.data, { mergeRecords: !firstLoad });
      state.agentPollFailures = 0;
      nextStatus = res.data.status || {};
      state.agentLoadedRunId = runId;
      const validationActive = ['starting', 'running'].includes(res.data.validation_status?.status)
        || Boolean(res.data.validation_status?.pending_task_id);
      if (['completed', 'failed', 'stopped', 'interrupted', 'paused'].includes(res.data.status.status) && !validationActive) {
        continuePolling = false;
        state.agentPollTimer = null;
        loadAgentRuns();
      }
    } catch (err) {
      state.agentPollFailures = (state.agentPollFailures || 0) + 1;
      $('agentJobStatus').textContent = `Training Status: polling temporarily failed ${state.agentPollFailures}/5 / ${err.message}`;
      continuePolling = state.agentPollFailures < 5;
      if (!continuePolling) state.agentPollTimer = null;
    } finally {
      if (generation === state.agentPollGeneration) state.agentPollBusy = false;
      if (continuePolling && generation === state.agentPollGeneration) {
        state.agentPollTimer = setTimeout(poll, agentPollingIntervalMs(nextStatus));
      }
    }
  }
  poll();
}

async function startAgentTraining() {
  try {
    setAgentConfigLocked(true);
    $('agentJobStatus').textContent = 'Training Status: queued / validating request';
    const res = await apiPost('/api/agent/runs/start', getAgentPayload());
    state.agentRunId = res.data.run_id;
    showAgentTab('monitor');
    startPollingAgentRun(state.agentRunId);
    await loadAgentRuns();
  } catch (err) {
    setAgentConfigLocked(false);
    alert(`Start Agent training failed: ${err.message}`);
  }
}

async function stopAgentTraining() {
  return controlAgentRun('stop');
}

async function controlAgentRun(action) {
  if (!state.agentRunId) return;
  try {
    await apiPost(`/api/agent/runs/${state.agentRunId}/${action}`, {});
    if (action === 'resume') startPollingAgentRun(state.agentRunId);
    else startPollingAgentRun(state.agentRunId);
  } catch (err) {
    alert(`${action} Agent run failed: ${err.message}`);
  }
}


async function loadAgentRuns() {
  try {
    const res = await apiGet('/api/agent/runs');
    state.agentRuns = res.data || [];
    $('agentRunsTableBody').innerHTML = state.agentRuns.map(run => `
      <tr><td><div class="agent-run-name">${escapeHtml(run.run_name || run.run_id)}</div><div class="agent-run-meta">${escapeHtml(run.run_id || '')}</div></td>
      <td><span class="status-pill ${run.status === 'completed' ? 'ok' : run.status === 'failed' || run.status === 'interrupted' ? 'bad' : 'warn'}">${escapeHtml(run.status || '-')}</span></td>
      <td><span class="status-pill ${run.validation_status === 'completed' || run.validation_status === 'idle' ? 'ok' : run.validation_status === 'failed' ? 'bad' : 'warn'}">${escapeHtml(run.validation_status || 'idle')}</span></td>
      <td>${Math.round(Number(run.progress || 0) * 100)}%</td><td>${run.steps ?? 0} / ${run.total_steps ?? '-'}</td><td>${escapeHtml(run.updated_at || '-')}</td>
      <td><div class="row-actions"><button class="small secondary" data-agent-view-run="${escapeHtml(run.run_id)}">View</button><button class="small secondary" data-agent-manage-run="${escapeHtml(run.run_id)}">Manage</button>${['paused', 'failed', 'interrupted', 'stopped'].includes(run.status) && run.latest_checkpoint ? `<button class="small" data-agent-resume-run="${escapeHtml(run.run_id)}">Resume</button>` : ''}</div></td></tr>
    `).join('') || '<tr><td colspan="7" class="muted">No persistent runs yet.</td></tr>';
  } catch (err) {
    $('agentRunsTableBody').innerHTML = `<tr><td colspan="7" class="error-cell">Runs failed: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function renderAgentCheckpoints(checkpoints) {
  const body = $('agentCheckpointsTableBody');
  if (!body) return;
  body.innerHTML = checkpoints.map(item => `<tr><td>${escapeHtml(item.name)}</td><td>${formatNumber(Number(item.size_bytes) / 1048576, 2)} MB</td><td>${escapeHtml(item.modified_at)}</td><td>${escapeHtml(item.path)}</td></tr>`).join('') || '<tr><td colspan="4" class="muted">No checkpoint saved yet.</td></tr>';
}

function closeAgentRunManageModal() {
  $('agentCheckpointModal')?.classList.add('hidden');
  state.agentSelectedManageRun = null;
}

function renderAgentRunManageModal(data) {
  const job = data.status || {};
  const checkpoints = data.checkpoints || [];
  state.agentSelectedManageRun = job.run_id || null;
  $('agentCheckpointModalTitle').textContent = `Run Checkpoints: ${job.run_name || job.run_id || '-'}`;
  $('agentCheckpointModalSubtitle').textContent = job.run_id || '-';
  renderSummaryCards('agentCheckpointModalSummary', [
    { label: 'Status', value: job.status || '-' },
    { label: 'Steps', value: `${job.steps ?? 0} / ${job.total_steps ?? '-'}` },
    { label: 'Updates / Episodes', value: `${job.updates ?? 0} / ${job.episodes ?? 0}` },
    { label: 'Latest', value: job.latest_checkpoint || '-' },
    { label: 'Best', value: job.best_checkpoint || '-' },
    { label: 'Updated', value: job.updated_at || '-' },
  ]);
  $('agentCheckpointModalTableBody').innerHTML = checkpoints.map(item => {
    const validation = item.validation || {};
    const validationText = validation.status === 'completed'
      ? `return ${formatPercent(validation.total_return)} / dd ${formatPercent(validation.maximum_drawdown)}`
      : (validation.status || '-');
    return `<tr>
      <td><div class="agent-run-name">${escapeHtml(item.name)}</div><div class="agent-run-meta agent-run-path">${escapeHtml(item.path || '')}</div></td>
      <td>${item.step ?? '-'}</td>
      <td>${escapeHtml(item.reason || '-')}</td>
      <td>${escapeHtml(validationText)}</td>
      <td>${formatNumber(Number(item.size_bytes) / 1048576, 2)} MB</td>
      <td>${escapeHtml(item.modified_at || '-')}</td>
      <td><button class="small secondary" type="button" data-agent-export-checkpoint="${escapeHtml(item.path || '')}" data-agent-export-default-name="${escapeHtml((job.run_name || job.run_id || 'agent') + '_' + String(item.step || 'step'))}">Export Model</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" class="muted">No checkpoint saved yet.</td></tr>';
  $('agentCheckpointModal').classList.remove('hidden');
}

async function openAgentRunManage(runId) {
  try {
    const res = await apiGet(`/api/agent/runs/${runId}?records=0`);
    renderAgentRunManageModal(res.data || {});
  } catch (err) {
    alert(`Load checkpoints failed: ${err.message}`);
  }
}

async function viewAgentRun(runId) {
  state.agentRunId = runId;
  showAgentTab('monitor');
  startPollingAgentRun(runId);
}

function openAgentMonitorWindow() {
  const run = state.agentRunId ? `&run=${encodeURIComponent(state.agentRunId)}` : '';
  window.open(`/?section=agent&agentTab=monitor${run}`, 'pocketagent-agent-monitor', 'width=1440,height=960');
}

function captureVisualizationModeSnapshot() {
  const visualization = state.visualization;
  return {
    payload: visualization.payload,
    selectedFreq: visualization.selectedFreq,
    windowOffset: visualization.windowOffset,
    totalRows: visualization.totalRows,
    viewStart: visualization.viewStart,
    viewEnd: visualization.viewEnd,
    hoverIndex: visualization.hoverIndex,
    hoverPanel: visualization.hoverPanel,
    referenceGuides: { ...(visualization.referenceGuides || emptyVisualizationReferenceGuides()) },
    mainOverlays: [...visualization.mainOverlays],
    mainOverlayPreferenceExplicit: visualization.mainOverlayPreferenceExplicit,
    panelSelections: [...visualization.panelSelections],
    panelSelectionsByFreq: Object.fromEntries(
      Object.entries(visualization.panelSelectionsByFreq || {}).map(([freq, selections]) => [freq, [...selections]])
    ),
    inspectorVisible: visualization.inspectorVisible,
    pendingZoom: visualization.pendingZoom ? { ...visualization.pendingZoom } : null,
  };
}

function saveVisualizationModeSnapshot(mode) {
  const payload = state.visualization.payload;
  if (mode === 'market' && payload && payload.mode !== 'evaluation') {
    state.visualization.modeSnapshots.market = captureVisualizationModeSnapshot();
  }
  if (mode === 'evaluation' && payload?.mode === 'evaluation') {
    state.visualization.modeSnapshots.evaluation = captureVisualizationModeSnapshot();
  }
}

function restoreVisualizationModeSnapshot(mode) {
  const snapshot = state.visualization.modeSnapshots?.[mode];
  if (!snapshot?.payload) return false;
  Object.assign(state.visualization, {
    payload: snapshot.payload,
    selectedFreq: snapshot.selectedFreq,
    windowOffset: snapshot.windowOffset,
    totalRows: snapshot.totalRows,
    viewStart: snapshot.viewStart,
    viewEnd: snapshot.viewEnd,
    hoverIndex: snapshot.hoverIndex,
    hoverPanel: snapshot.hoverPanel,
    referenceGuides: { ...(snapshot.referenceGuides || emptyVisualizationReferenceGuides()) },
    mainOverlays: new Set(snapshot.mainOverlays || []),
    mainOverlayPreferenceExplicit: Boolean(snapshot.mainOverlayPreferenceExplicit),
    panelSelections: [...(snapshot.panelSelections || state.visualization.panelSelections)],
    panelSelectionsByFreq: Object.fromEntries(
      Object.entries(snapshot.panelSelectionsByFreq || {}).map(([freq, selections]) => [freq, [...selections]])
    ),
    inspectorVisible: snapshot.inspectorVisible !== false,
    pendingZoom: snapshot.pendingZoom ? { ...snapshot.pendingZoom } : null,
  });
  if (mode === 'market') renderVisualizationControls({ keepSymbol: true, keepFreq: true });
  initializeVisualizationFeatureControls(snapshot.payload);
  setVisualizationQuote(snapshot.payload);
  drawVisualizationPayload(snapshot.payload);
  return true;
}

function showVisualTab(tab) {
  const selected = tab === 'live' || tab === 'evaluation' ? tab : 'market';
  const previousTab = state.evaluationReplay.activeTab || 'market';
  saveVisualizationModeSnapshot(previousTab);
  state.evaluationReplay.activeTab = selected;
  if (selected !== 'evaluation') stopEvaluationReplayPolling();
  document.querySelectorAll('.visual-tab').forEach(button => {
    button.classList.toggle('active', button.dataset.visualTab === selected);
  });
  const shell = $('visualShell');
  if (shell) {
    shell.classList.remove('visual-mode-market', 'visual-mode-evaluation', 'visual-mode-live');
    shell.classList.add(`visual-mode-${selected}`);
  }
  $('visualEvaluationPane')?.classList.toggle('hidden', selected !== 'evaluation');
  $('visualLivePane')?.classList.toggle('hidden', selected !== 'live');
  $('evaluationStatsPanel')?.classList.toggle('hidden', selected !== 'evaluation');
  $('visualFeatureInspector')?.classList.toggle('hidden', selected === 'evaluation' || !state.visualization.inspectorVisible);
  if (selected === 'evaluation') {
    restoreVisualizationModeSnapshot('evaluation');
    setVisualStatus('');
    loadEvaluationRuns({ silent: true })
      .then(() => loadEvaluationReplay($('visualEvaluationRunSelect')?.value || state.evaluationRunId))
      .catch(() => {
        if (state.evaluationReplay.activeTab === 'evaluation') drawEvaluationReplay();
      });
  } else if (selected === 'market') {
    renderEvaluationStats(null);
    if (restoreVisualizationModeSnapshot('market')) return;
    if (state.visualization.payload?.mode === 'evaluation') {
      state.visualization.selectedFreq = 'daily';
      renderVisualizationControls({ keepSymbol: true, keepFreq: false });
      void loadVisualizationChart({ anchor: 'latest' });
    } else drawVisualizationPayload(state.visualization.payload);
  }
}

function stopEvaluationReplayPolling() {
  if (state.evaluationReplay.pollTimer) {
    clearTimeout(state.evaluationReplay.pollTimer);
    state.evaluationReplay.pollTimer = null;
  }
}

async function loadEvaluationModels() {
  try {
    const res = await apiGet('/api/evaluation/models');
    state.evaluationModels = res.data || [];
    renderEvaluationModels();
  } catch (err) {
    if ($('evaluationModelsTableBody')) $('evaluationModelsTableBody').innerHTML = `<tr><td colspan="4" class="error-cell">Models failed: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function renderEvaluationModels() {
  const select = $('evaluationModelSelect');
  if (select) {
    select.innerHTML = state.evaluationModels.map(model => `<option value="${escapeHtml(model.name)}">${escapeHtml(model.name)}</option>`).join('') || '<option value="">No exported models</option>';
  }
  const body = $('evaluationModelsTableBody');
  if (body) {
    body.innerHTML = state.evaluationModels.map(model => `
      <tr><td>${escapeHtml(model.name)}</td><td>${escapeHtml(model.source_run_id || '-')}</td><td>${escapeHtml(model.source_step ?? '-')}</td><td>${escapeHtml(model.exported_at || '-')}</td></tr>
    `).join('') || '<tr><td colspan="4" class="muted">No exported models.</td></tr>';
  }
}

function isEvaluationRunActive(run) {
  return ['queued', 'running'].includes(String(run?.status || '').toLowerCase());
}

function selectedEvaluationRun() {
  const runId = state.evaluationRunId || $('visualEvaluationRunSelect')?.value || '';
  return state.evaluationRuns.find(run => run.run_id === runId) || null;
}

function updateEvaluationStopButtons() {
  const run = selectedEvaluationRun();
  const active = isEvaluationRunActive(run);
  if ($('stopEvaluationBtn')) $('stopEvaluationBtn').disabled = !active;
  if ($('stopVisualEvaluationBtn')) $('stopVisualEvaluationBtn').disabled = !active;
}

function getEvaluationPayload() {
  return {
    model_name: $('evaluationModelSelect')?.value || '',
    feature_dataset_dir: $('evaluationFeatureDataset')?.value.trim() || 'runtime_layer/reports/feature_dataset',
    symbol: $('evaluationSymbol')?.value.trim().toUpperCase() || '',
    start: $('evaluationStart')?.value || '',
    end: $('evaluationEnd')?.value || '',
    device: $('evaluationDevice')?.value || 'cpu',
    initial_cash: Number($('evaluationInitialCash')?.value || 1000000),
    lot_size: Number($('evaluationLotSize')?.value || 100),
    max_position_ratio: Number($('evaluationMaxPosition')?.value || 0.2),
    commission_rate: Number($('evaluationCommission')?.value || 0.0003),
    minimum_commission: Number($('evaluationMinCommission')?.value || 5),
    base_slippage_rate: Number($('evaluationSlippage')?.value || 0.0002),
  };
}

async function startEvaluation() {
  try {
    const payload = getEvaluationPayload();
    if (!payload.model_name) return alert('Export and select an evaluation model first.');
    if (!payload.symbol || !payload.start || !payload.end) return alert('Symbol, start date, and end date are required.');
    $('evaluationStartStatus').textContent = 'Evaluation queued.';
    const res = await apiPost('/api/evaluation/start', payload);
    state.evaluationRunId = res.data.run_id;
    $('evaluationStartStatus').textContent = `Evaluation started: ${res.data.run_id}`;
    await loadEvaluationRuns();
    updateEvaluationStopButtons();
    showSection('visualization');
    showVisualTab('evaluation');
    setVisualEvaluationRun(res.data.run_id);
    startEvaluationReplayPolling(res.data.run_id);
  } catch (err) {
    $('evaluationStartStatus').textContent = `Start failed: ${err.message}`;
    alert(`Start Evaluation failed: ${err.message}`);
  }
}

async function stopEvaluation(runId = null) {
  const target = runId || state.evaluationRunId || $('visualEvaluationRunSelect')?.value;
  if (!target) return alert('Select an evaluation run first.');
  try {
    if ($('stopEvaluationBtn')) $('stopEvaluationBtn').disabled = true;
    if ($('stopVisualEvaluationBtn')) $('stopVisualEvaluationBtn').disabled = true;
    $('evaluationStartStatus').textContent = `Stopping evaluation: ${target}`;
    const res = await apiPost(`/api/evaluation/runs/${encodeURIComponent(target)}/stop`, {});
    $('evaluationStartStatus').textContent = res.data?.message || `Stop requested: ${target}`;
    await loadEvaluationRuns({ silent: true });
    await loadEvaluationReplay(target);
  } catch (err) {
    $('evaluationStartStatus').textContent = `Stop failed: ${err.message}`;
    alert(`Stop Evaluation failed: ${err.message}`);
  } finally {
    updateEvaluationStopButtons();
  }
}

async function loadEvaluationRuns({ silent = false } = {}) {
  try {
    const res = await apiGet('/api/evaluation/runs');
    state.evaluationRuns = res.data || [];
    renderEvaluationRuns();
    renderVisualEvaluationRunSelect();
    updateEvaluationStopButtons();
  } catch (err) {
    if (!silent && $('evaluationRunsTableBody')) $('evaluationRunsTableBody').innerHTML = `<tr><td colspan="8" class="error-cell">Runs failed: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function renderEvaluationRuns() {
  const body = $('evaluationRunsTableBody');
  if (!body) return;
  body.innerHTML = state.evaluationRuns.map(run => `
    <tr>
      <td><div class="agent-run-name">${escapeHtml(run.run_id)}</div><div class="agent-run-meta">${escapeHtml(run.path || '')}</div></td>
      <td>${escapeHtml(run.model_name || '-')}</td>
      <td>${escapeHtml(run.symbol || '-')}</td>
      <td><span class="status-pill ${run.status === 'completed' ? 'ok' : run.status === 'failed' ? 'bad' : 'warn'}">${escapeHtml(run.status || '-')}</span></td>
      <td>${formatPercent(run.total_return)}</td>
      <td>${formatPercent(run.maximum_drawdown)}</td>
      <td>${escapeHtml(run.updated_at || '-')}</td>
      <td>
        <button class="small secondary" data-evaluation-view-run="${escapeHtml(run.run_id)}" type="button">View</button>
        ${isEvaluationRunActive(run) ? `<button class="small secondary danger-text" data-evaluation-stop-run="${escapeHtml(run.run_id)}" type="button">Stop</button>` : ''}
      </td>
    </tr>
  `).join('') || '<tr><td colspan="8" class="muted">No evaluation runs.</td></tr>';
}

function renderVisualEvaluationRunSelect() {
  const select = $('visualEvaluationRunSelect');
  if (!select) return;
  const current = select.value || state.evaluationRunId || state.evaluationRuns[0]?.run_id || '';
  select.innerHTML = state.evaluationRuns.map(run => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} / ${escapeHtml(run.symbol || '-')} / ${escapeHtml(run.status || '-')}</option>`).join('') || '<option value="">No evaluation runs</option>';
  select.value = state.evaluationRuns.some(run => run.run_id === current) ? current : (state.evaluationRuns[0]?.run_id || '');
}

function setVisualEvaluationRun(runId) {
  state.evaluationRunId = runId;
  state.evaluationReplay.runId = runId;
  if ($('visualEvaluationRunSelect')) $('visualEvaluationRunSelect').value = runId || '';
  updateEvaluationStopButtons();
}

async function loadEvaluationReplay(runId = null) {
  const selected = runId || $('visualEvaluationRunSelect')?.value || state.evaluationRunId;
  if (!selected) {
    state.evaluationReplay.payload = null;
    if (state.evaluationReplay.activeTab === 'evaluation') drawEvaluationReplay();
    return;
  }
  setVisualEvaluationRun(selected);
  try {
    const res = await apiGet(`/api/evaluation/runs/${encodeURIComponent(selected)}?event_limit=10000`);
    state.evaluationReplay.payload = res.data || null;
    if (state.evaluationReplay.activeTab === 'evaluation') drawEvaluationReplay();
    updateEvaluationStopButtons();
    const status = res.data?.status || {};
    if (state.evaluationReplay.activeTab === 'evaluation' && ['queued', 'running'].includes(status.status)) {
      startEvaluationReplayPolling(selected);
    }
  } catch (err) {
    if (state.evaluationReplay.activeTab === 'evaluation' && $('visualEvaluationStatus')) {
      $('visualEvaluationStatus').textContent = `Replay load failed: ${err.message}`;
    }
  }
}

function startEvaluationReplayPolling(runId) {
  if (state.evaluationReplay.pollTimer) clearTimeout(state.evaluationReplay.pollTimer);
  const generationRun = runId;
  async function poll() {
    if (state.evaluationReplay.runId !== generationRun) return;
    if (state.evaluationReplay.activeTab !== 'evaluation') {
      state.evaluationReplay.pollTimer = null;
      return;
    }
    await loadEvaluationReplay(generationRun);
    const status = state.evaluationReplay.payload?.status?.status;
    if (state.evaluationReplay.activeTab === 'evaluation' && ['queued', 'running'].includes(status)) {
      state.evaluationReplay.pollTimer = setTimeout(poll, 1500);
    } else {
      state.evaluationReplay.pollTimer = null;
      await loadEvaluationRuns({ silent: true });
    }
  }
  state.evaluationReplay.pollTimer = setTimeout(poll, 1000);
}

function evaluationFeatureColor(index) {
  const palette = ['#60a5fa', '#f59e0b', '#a78bfa', '#34d399', '#f472b6', '#22d3ee', '#fb7185', '#eab308'];
  return palette[index % palette.length];
}

function buildEvaluationVisualizationIndicators(events) {
  const keys = [];
  const seen = new Set();
  events.forEach(event => {
    Object.entries(event.features || {}).forEach(([freq, values]) => {
      Object.keys(values || {}).forEach(key => {
        const id = `${freq}__${key}`;
        if (seen.has(id)) return;
        seen.add(id);
        keys.push({ id, freq, key, label: `${freq} ${key}` });
      });
    });
  });
  return keys.map((item, index) => {
    const values = events.map(event => {
      const value = event.features?.[item.freq]?.[item.key];
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    });
    return {
      id: `eval_feature_${item.id.replace(/[^A-Za-z0-9_:-]/g, '_')}`,
      label: item.label,
      render_target: 'sub_panel',
      default_visible: index < 3,
      axis: { mode: 'auto' },
      display_series: [{
        id: 'value',
        label: item.label,
        model_field: 'value',
        style: 'line',
        color: evaluationFeatureColor(index),
      }],
      model_series: [{
        id: 'value',
        label: item.label,
        values,
      }],
    };
  });
}

function buildEvaluationVisualizationPayload(payload) {
  const events = payload?.events || [];
  if (!payload || !events.length) return null;
  const bars = events.map(event => {
    const bar = event.bar || {};
    return {
      datetime: bar.datetime || event.time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume: bar.volume,
      amount: bar.amount,
      pctChg: bar.pctChg,
      turn: bar.turn,
      is_st: bar.is_st,
      _evaluationEvent: event,
    };
  });
  const last = bars[bars.length - 1] || {};
  const previous = bars.length > 1 ? bars[bars.length - 2] : null;
  const close = Number(last.close);
  const previousClose = Number(previous?.close);
  const change = Number.isFinite(close) && Number.isFinite(previousClose) ? close - previousClose : null;
  const pctChg = Number.isFinite(close) && Number.isFinite(previousClose) && previousClose !== 0 ? close / previousClose - 1 : last.pctChg;
  return {
    mode: 'evaluation',
    run_id: payload.run_id,
    status: payload.status,
    symbol: payload.config?.symbol || payload.status?.symbol || events[0]?.symbol || '-',
    freq: 'evaluation',
    adjust: payload.config?.model_name || 'model replay',
    rows: bars.length,
    total_rows: bars.length,
    bars,
    summary: {
      symbol: payload.config?.symbol || events[0]?.symbol || '-',
      latest_datetime: last.datetime,
      latest_close: close,
      change,
      pctChg,
      total_return: payload.summary?.metrics?.total_return,
      maximum_drawdown: payload.summary?.metrics?.maximum_drawdown,
    },
    features: {
      indicators: buildEvaluationVisualizationIndicators(events),
    },
    evaluation: payload,
  };
}

function applyEvaluationVisualizationPayload(payload) {
  const visualPayload = buildEvaluationVisualizationPayload(payload);
  if (!visualPayload) {
    state.visualization.payload = null;
    state.visualization.viewStart = 0;
    state.visualization.viewEnd = 0;
    setVisualizationQuote(null);
    drawVisualizationPayload(null);
    renderEvaluationStats(null);
    return;
  }
  const previousRunId = state.evaluationReplay.appliedRunId;
  const previousCount = state.evaluationReplay.appliedEventCount || 0;
  const atRightEdge = previousCount > 0 && state.visualization.viewEnd >= previousCount - 2;
  const windowSize = Math.max(20, state.visualization.viewEnd - state.visualization.viewStart || VISUAL_DEFAULT_VISIBLE_BARS);
  const shouldReset = previousRunId !== visualPayload.run_id || state.visualization.payload?.mode !== 'evaluation';
  state.visualization.payload = visualPayload;
  state.visualization.windowOffset = 0;
  state.visualization.totalRows = visualPayload.bars.length;
  state.visualization.selectedFreq = 'evaluation';
  state.evaluationReplay.appliedRunId = visualPayload.run_id;
  state.evaluationReplay.appliedEventCount = visualPayload.bars.length;
  initializeVisualizationFeatureControls(visualPayload);
  if (shouldReset) {
    resetVisualizationView(visualPayload.bars, 'latest', Math.min(VISUAL_DEFAULT_VISIBLE_BARS, visualPayload.bars.length));
  } else if (atRightEdge || state.visualization.viewEnd > visualPayload.bars.length) {
    state.visualization.viewEnd = visualPayload.bars.length;
    state.visualization.viewStart = Math.max(0, visualPayload.bars.length - windowSize);
  } else {
    clampVisualizationView(state.visualization.viewStart, state.visualization.viewEnd);
  }
  setVisualizationQuote(visualPayload);
  drawVisualizationPayload(visualPayload);
}

function drawEvaluationReplay() {
  const payload = state.evaluationReplay.payload;
  const statusElement = $('visualEvaluationStatus');
  const events = payload?.events || [];
  const status = payload?.status || {};
  if (statusElement) {
    statusElement.textContent = payload
      ? `${payload.run_id} / ${status.status || '-'} / ${Math.round(Number(status.progress || 0) * 100)}% / ${status.message || ''}`
      : 'Select an evaluation run.';
  }
  if (!payload || !events.length) {
    drawCanvasMessage($('visualPriceCanvas'), payload ? 'No evaluation events yet.' : 'Select an evaluation run.');
    for (let slot = 0; slot < 3; slot += 1) {
      drawCanvasMessage($(`visualPanelCanvas${slot}`), 'No evaluation data');
      const legend = $(`visualPanelLegend${slot}`);
      if (legend) legend.innerHTML = '-';
    }
    $('visualPriceLegend').innerHTML = '-';
    renderEvaluationStats(null);
    return;
  }
  applyEvaluationVisualizationPayload(payload);
}

function drawEvaluationAxes(ctx, rows, pad, width, height, minPrice, maxPrice) {
  ctx.strokeStyle = '#263448';
  ctx.fillStyle = '#8ba3c7';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i += 1) {
    const ratio = i / 4;
    const y = pad.top + (height - pad.top - pad.bottom) * ratio;
    const value = maxPrice - (maxPrice - minPrice) * ratio;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(formatNumber(value, 2), pad.left - 8, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const labels = Math.min(6, rows.length);
  for (let i = 0; i < labels; i += 1) {
    const index = Math.floor((rows.length - 1) * (i / Math.max(1, labels - 1)));
    const x = pad.left + (width - pad.left - pad.right) * (index / Math.max(1, rows.length - 1));
    ctx.fillText(String(rows[index]?.time || '').slice(5, 10), x, height - pad.bottom + 8);
  }
}

function renderEvaluationStats(event) {
  const time = $('evaluationReplayTime');
  const content = $('evaluationStatsContent');
  if (!content || !time) return;
  if (!event) {
    time.textContent = '-';
    content.innerHTML = '<div class="muted">No evaluation data.</div>';
    return;
  }
  time.textContent = String(event.time || '-').slice(0, 19);
  const fillInfo = evaluationFillInfo(event);
  const fill = fillInfo.primary || {};
  const execution = event.execution || {};
  const account = event.account || {};
  const performance = event.performance || {};
  const bar = event.bar || {};
  const executionStatus = execution.executed ? 'executed' : execution.blocked ? 'blocked' : '-';
  const tradeText = fill.status
    ? `${fill.side || event.decision?.action || '-'} ${fill.shares || 0} @ ${formatNumber(fill.price ?? fill.reference_price, 3)} (${fill.status})`
    : '-';
  const modelInput = event.model_input || {};
  const features = modelInput.market_sequences || event.features || {};
  const groupHtml = (title, rows) => `
    <section class="visual-inspector-group">
      <h3>${escapeHtml(title)}</h3>
      ${rows.map(([label, value]) => `<div class="visual-inspector-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? '-')}</strong></div>`).join('')}
    </section>
  `;
  const featureGroupItems = Object.entries(features).slice(0, 6)
    .filter(([, values]) => Object.keys(values || {}).length)
    .map(([freq, values]) => groupHtml(`${freq} Features`, Object.entries(values || {}).slice(0, 10).map(([key, value]) => [key, formatFeatureValue(value)])));
  const decisionContextRows = Object.entries(modelInput.decision_context || {}).slice(0, 12).map(([key, value]) => [key, formatFeatureValue(value)]);
  const runtimeStateRows = Object.entries(modelInput.runtime_state || {}).slice(0, 12).map(([key, value]) => [key, formatFeatureValue(value)]);
  const constraintRows = Object.entries(modelInput.constraints || {}).map(([key, value]) => [key, value === true ? 'true' : value === false ? 'false' : value]);
  if (decisionContextRows.length) featureGroupItems.push(groupHtml('Decision Context', decisionContextRows));
  if (runtimeStateRows.length) featureGroupItems.push(groupHtml('Runtime State', runtimeStateRows));
  if (constraintRows.length) featureGroupItems.push(groupHtml('Constraints', constraintRows));
  const featureGroups = featureGroupItems.join('');
  content.innerHTML = `
    ${groupHtml('Current Bar', [
      ['Symbol', event.symbol],
      ['Datetime', String(event.time || bar.datetime || '-').slice(0, 19)],
      ['Open / High', `${formatNumber(bar.open, 3)} / ${formatNumber(bar.high, 3)}`],
      ['Low / Close', `${formatNumber(bar.low, 3)} / ${formatNumber(bar.close, 3)}`],
      ['Volume', formatCompactNumber(bar.volume)],
      ['Amount', formatNumber(bar.amount, 2)],
    ])}
    ${groupHtml('Model Decision', [
      ['Action', evaluationActionLabel(event)],
      ['Raw / Final', `${event.decision?.raw_action || event.decision?.action || '-'} / ${event.decision?.final_action || execution.final_action || '-'}`],
      ['Size', formatNumber(event.decision?.size, 4)],
      ['Prob S/H/B', `${formatPercent(event.decision?.prob_sell)} / ${formatPercent(event.decision?.prob_hold)} / ${formatPercent(event.decision?.prob_buy)}`],
      ['Value', formatNumber(event.decision?.value, 5)],
    ])}
    ${groupHtml('Trade Execution', [
      ['Status', executionStatus],
      ['Blocked Reason', execution.blocked_reason || (fillInfo.blocked ? fillInfo.blocked.reason : '-')],
      ['Trade', tradeText],
      ['Shares', formatNumber(execution.shares ?? fill.shares, 0)],
      ['Price', formatNumber(execution.price ?? fill.price ?? fill.reference_price, 3)],
      ['Amount', formatNumber(execution.amount ?? fill.amount, 2)],
      ['Fee', formatNumber(execution.fee ?? fill.fee, 2)],
      ['Slippage', formatNumber(execution.slippage ?? fill.slippage, 4)],
      ['Position Change', formatPercent(execution.position_change_ratio ?? fill.position_change_ratio)],
    ])}
    ${groupHtml('Account', [
      ['Cash', formatNumber(account.cash, 2)],
      ['Position Shares', formatNumber(account.position_shares, 0)],
      ['Position Value', formatNumber(account.position_value, 2)],
      ['Total Asset', formatNumber(account.total_asset, 2)],
      ['Position Ratio', formatPercent(account.position_ratio)],
      ['Total Fees', formatNumber(account.total_fees, 2)],
      ['Total Trades', formatNumber(performance.total_trades ?? performance.orders, 0)],
    ])}
    ${groupHtml('Performance', [
      ['Return', formatPercent(performance.return)],
      ['Drawdown', formatPercent(performance.drawdown)],
      ['Reward', formatNumber(performance.reward, 6)],
      ['Total Actions', formatNumber(performance.total_actions, 0)],
      ['Blocked Actions', formatNumber(performance.blocked_actions ?? performance.blocked_orders, 0)],
      ['Blocked Buy / Sell', `${performance.blocked_buys ?? 0} / ${performance.blocked_sells ?? 0}`],
    ])}
    ${featureGroups || '<section class="visual-inspector-group"><h3>Model Input Features</h3><div class="muted">No feature snapshot.</div></section>'}
  `;
}

function exportCheckpointButtonPayload(button) {
  return {
    checkpoint: button.dataset.agentExportCheckpoint || '',
    defaultName: button.dataset.agentExportDefaultName || '',
  };
}

function closeAgentExportConflictModal(action = 'cancel') {
  $('agentExportConflictModal')?.classList.add('hidden');
  if (agentExportConflictResolver) {
    const resolve = agentExportConflictResolver;
    agentExportConflictResolver = null;
    resolve(action);
  }
}

function chooseAgentExportConflictAction(modelName) {
  const modal = $('agentExportConflictModal');
  const message = $('agentExportConflictMessage');
  if (!modal || !message) {
    return Promise.resolve(window.confirm(`Evaluation model "${modelName}" already exists. Overwrite it?`) ? 'overwrite' : 'cancel');
  }
  message.textContent = `Evaluation model "${modelName}" already exists. Choose Rename, Overwrite, or Cancel.`;
  modal.classList.remove('hidden');
  return new Promise(resolve => {
    agentExportConflictResolver = resolve;
  });
}

async function exportAgentCheckpoint(button) {
  const payload = exportCheckpointButtonPayload(button);
  if (!state.agentSelectedManageRun || !payload.checkpoint) return;
  let modelName = prompt('Export model name', payload.defaultName || 'agent_model');
  if (!modelName) return;
  async function submit(overwrite = false) {
    return apiPost(`/api/agent/runs/${encodeURIComponent(state.agentSelectedManageRun)}/export-model`, {
      checkpoint: payload.checkpoint,
      model_name: modelName,
      overwrite,
    });
  }
  try {
    await submit(false);
    alert(`Model exported: ${modelName}`);
    await loadEvaluationModels();
  } catch (err) {
    if (String(err.message || '').includes('already exists')) {
      const action = await chooseAgentExportConflictAction(modelName);
      if (action === 'overwrite') {
        await submit(true);
        alert(`Model overwritten: ${modelName}`);
      } else if (action === 'rename') {
        const renamed = prompt('New model name', `${modelName}_v2`);
        if (!renamed) return;
        modelName = renamed;
        await submit(false);
        alert(`Model exported: ${modelName}`);
      } else {
        return;
      }
      await loadEvaluationModels();
    } else {
      alert(`Export failed: ${err.message}`);
    }
  }
}

function setDownloadSymbolsPath(path) {
  const value = (path || DEFAULT_DOWNLOAD_SYMBOLS_FILE).trim();
  state.configSymbolPath = value;
  ['downloadSymbolsFile', 'coverageSymbolsFile', 'universeCandidates', 'materializeSymbolsFile', 'configUniversePath'].forEach(id => {
    if ($(id)) $(id).value = value;
  });
  if ($('downloadSymbolSummary')) {
    $('downloadSymbolSummary').textContent = `Using ${value}`;
  }
}

async function loadConfigSymbolsFromFile({ silent = false } = {}) {
  const path = $('configUniversePath').value.trim();
  if (!path) {
    if (!silent) alert('Enter a universe file path.');
    return;
  }
  try {
    const res = await apiGet(`/api/data/symbols/file?path=${encodeURIComponent(path)}`);
    $('configSymbolsText').value = res.data.text || '';
    $('configSymbolsStatus').textContent = `${res.data.count} symbols loaded from ${res.data.path || path}.`;
    setDownloadSymbolsPath(path);
  } catch (err) {
    $('configSymbolsStatus').textContent = `Load failed: ${err.message}`;
    if (!silent) alert(`Load symbols failed: ${err.message}`);
  }
}

async function saveConfigSymbolsToFile() {
  const path = $('configUniversePath').value.trim();
  const symbols = parseSymbols($('configSymbolsText').value);
  if (!path) return alert('Enter a universe file path.');
  try {
    const res = await apiPost('/api/data/symbols/file', { path, symbols });
    $('configSymbolsStatus').textContent = `Saved ${res.data.count} symbols to ${res.data.path}.`;
    setDownloadSymbolsPath(path);
  } catch (err) {
    $('configSymbolsStatus').textContent = `Save failed: ${err.message}`;
    alert(`Save symbols failed: ${err.message}`);
  }
}

function useConfigSymbolsForDownload() {
  setDownloadSymbolsPath($('configUniversePath').value.trim());
  $('downloadSymbolSummary').textContent = `Using ${$('downloadSymbolsFile').value.trim()}`;
}

function applyConfigDefaultsToDownload() {
  setSharedDbPath($('configDefaultDbPath').value);
  $('downloadStart').value = $('configDefaultStart').value;
  $('downloadEnd').value = $('configDefaultEnd').value;
  $('downloadSleep').value = $('configDefaultSleep').value;
  if ($('downloadStorageRoot') && $('configDefaultStorageRoot')) $('downloadStorageRoot').value = $('configDefaultStorageRoot').value;
  if ($('downloadWorkers') && $('configDefaultWorkers')) $('downloadWorkers').value = $('configDefaultWorkers').value;
  $('downloadSkipExisting').checked = $('configDefaultSkipExisting').checked;
  $('downloadReplace').checked = $('configDefaultReplace').checked;
  setCheckedValues('.downloadFreq', getCheckedValues('.configDefaultFreq'));
  setCheckedValues('.downloadAdjustflag', getCheckedValues('.configDefaultAdjustflag'));
  setDownloadSymbolsPath($('configUniversePath').value.trim());
}

function renderModalRows() {
  const rows = state.detail.rows;
  renderModalTableHead();
  if (!rows.length) {
    setModalTableMessage('No rows.');
    return;
  }

  const dailyCells = row => shouldShowTurnColumn()
    ? `<td>${escapeHtml(formatNumber(row.turn, 4))}</td><td>${row.isST === true ? 'ST' : row.isST === false ? 'Normal' : '-'}</td>`
    : '';

  $('modalKlineTableBody').innerHTML = rows.map(row => `
    <tr>
      <td>${escapeHtml(String(row.datetime ?? '-').slice(0, 19))}</td>
      <td>${escapeHtml(formatNumber(row.open, 2))}</td>
      <td>${escapeHtml(formatNumber(row.high, 2))}</td>
      <td>${escapeHtml(formatNumber(row.low, 2))}</td>
      <td>${escapeHtml(formatNumber(row.close, 2))}</td>
      <td>${escapeHtml(formatNumber(row.volume, 0))}</td>
      <td>${escapeHtml(formatNumber(row.amount, 2))}</td>
      ${dailyCells(row)}
      <td>${escapeHtml(formatPercent(row.pctChg))}</td>
    </tr>
  `).join('');
}

function getDetailFreqs() {
  const values = [];
  for (const slice of state.detail.slices || []) {
    const value = slice.freq || '-';
    if (!values.includes(value)) values.push(value);
  }
  return sortFrequencies(values);
}

function getDetailAdjustsForFreq(freq) {
  const values = [];
  for (const slice of state.detail.slices || []) {
    const sliceFreq = slice.freq || '-';
    if (sliceFreq !== freq) continue;
    const value = slice.adjust || '-';
    if (!values.includes(value)) values.push(value);
  }
  return values;
}

function preferDetailFreq(freqs) {
  return sortFrequencies(freqs)[0] || null;
}

function preferDetailAdjust(adjusts) {
  const order = ['none', 'pre', 'post'];
  for (const item of order) {
    if (adjusts.includes(item)) return item;
  }
  return adjusts[0] || null;
}

function getCurrentDetailSlice() {
  const freq = state.detail.selectedFreq;
  const adjust = state.detail.selectedAdjust;
  return (
    (state.detail.slices || []).find(slice => (slice.freq || '-') === freq && (slice.adjust || '-') === adjust)
    || state.detail.slices[0]
    || { freq: null, adjust: null }
  );
}

function renderModalDetailControls({ keepAdjust = true } = {}) {
  const freqSelect = $('modalFreqSelect');
  const adjustSelect = $('modalAdjustSelect');
  const freqs = getDetailFreqs();

  if (!freqs.length) {
    freqSelect.innerHTML = '<option value="">none</option>';
    adjustSelect.innerHTML = '<option value="">none</option>';
    state.detail.selectedFreq = null;
    state.detail.selectedAdjust = null;
    return;
  }

  if (!state.detail.selectedFreq || !freqs.includes(state.detail.selectedFreq)) {
    state.detail.selectedFreq = preferDetailFreq(freqs);
  }

  const adjusts = getDetailAdjustsForFreq(state.detail.selectedFreq);
  if (!keepAdjust || !state.detail.selectedAdjust || !adjusts.includes(state.detail.selectedAdjust)) {
    state.detail.selectedAdjust = preferDetailAdjust(adjusts);
  }

  freqSelect.innerHTML = freqs.map(freq => {
    const totalRows = (state.detail.slices || [])
      .filter(slice => (slice.freq || '-') === freq)
      .reduce((sum, slice) => sum + Number(slice.rows || 0), 0);
    const selected = freq === state.detail.selectedFreq ? 'selected' : '';
    return `<option value="${escapeHtml(freq)}" ${selected}>${escapeHtml(freq)} / ${totalRows} rows</option>`;
  }).join('');

  adjustSelect.innerHTML = adjusts.map(adjust => {
    const slice = (state.detail.slices || []).find(item =>
      (item.freq || '-') === state.detail.selectedFreq && (item.adjust || '-') === adjust
    );
    const selected = adjust === state.detail.selectedAdjust ? 'selected' : '';
    return `<option value="${escapeHtml(adjust)}" ${selected}>${escapeHtml(adjust)} / ${slice?.rows || 0} rows</option>`;
  }).join('');
}

async function loadDetailPage(reset = false) {
  if (state.detail.loading) return;
  if (!reset && !state.detail.hasMore) return;
  state.detail.loading = true;
  $('modalLoadStatus').textContent = 'Loading...';
  try {
    const db = encodeURIComponent(getDataDbPath());
    const { symbol, offset, limit } = state.detail;
    const slice = getCurrentDetailSlice();
    const freq = normalizeOptionalValue(slice.freq);
    const adjust = normalizeOptionalValue(slice.adjust);
    const url = `/api/data/symbol-detail?db=${db}&symbol=${encodeURIComponent(symbol)}&freq=${encodeURIComponent(freq || '')}&adjust=${encodeURIComponent(adjust || '')}&offset=${offset}&limit=${limit}`;
    const res = await apiGet(url);
    const data = res.data;
    if (reset) {
      state.detail.rows = [];
      renderSummaryCards('modalSummary', [
        { label: 'Symbol', value: data.symbol || '-' },
        { label: 'Freq', value: data.freq || '-' },
        { label: 'Adjust', value: data.adjust || '-' },
        { label: 'Rows', value: data.rows ?? 0 },
      ]);
      renderSummaryCards('modalDateSummary', [
        { label: 'Start', value: data.start_datetime || '-' },
        { label: 'End', value: data.end_datetime || '-' },
      ]);
      $('modalTableWrap').scrollTop = 0;
    }
    const items = data.items || [];
    state.detail.rows.push(...items);
    state.detail.offset += items.length;
    state.detail.hasMore = Boolean(data.has_more);
    renderModalRows();
    $('modalLoadStatus').textContent = state.detail.hasMore ? 'Scroll to load more.' : 'All rows loaded.';
  } catch (err) {
    $('modalLoadStatus').textContent = `Load failed: ${err.message}`;
  } finally {
    state.detail.loading = false;
  }
}

async function reloadDetailForSelectedFilters({ freqChanged = false } = {}) {
  state.detail.selectedFreq = $('modalFreqSelect').value || null;
  if (freqChanged) {
    renderModalDetailControls({ keepAdjust: false });
  } else {
    state.detail.selectedAdjust = $('modalAdjustSelect').value || null;
    renderModalDetailControls({ keepAdjust: true });
  }

  state.detail.offset = 0;
  state.detail.rows = [];
  state.detail.hasMore = true;
  renderModalRows();
  await loadDetailPage(true);
}

async function showSymbolDetail(symbol, options = {}) {
  const group = getInventoryGroups().find(item => item.symbol === String(symbol).toUpperCase());
  if (!group || !group.slices.length) return alert(`No local data for ${symbol}.`);
  state.detail = {
    symbol: group.symbol,
    slices: group.slices,
    selectedFreq: options.freq || null,
    selectedAdjust: options.adjust || null,
    offset: 0,
    limit: 100,
    rows: [],
    hasMore: true,
    loading: false,
  };
  $('detailModalTitle').textContent = `${group.symbol} K-line Data`;
  $('detailModalSubtitle').textContent = 'Core bars with locally calculated pctChg.';
  renderModalDetailControls({ keepAdjust: Boolean(options.adjust) });
  setModalTableMessage('Loading...');
  $('detailModal').classList.remove('hidden');
  await loadDetailPage(true);
}

async function openVisualizationDetail() {
  const symbol = getVisualSymbolValue();
  if (!symbol) return alert('No local symbol selected.');
  await showSymbolDetail(symbol, {
    freq: getCurrentVisualizationFreq(),
    adjust: $('visualAdjustSelect').value || null,
  });
}

function closeDetailModal() {
  $('detailModal').classList.add('hidden');
}

async function deleteSymbolData(symbol, freq = null, adjust = null) {
  const label = [symbol, freq, adjust].filter(Boolean).join(' / ');
  if (!confirm(`Delete local K-line data for ${label}?`)) return;
  try {
    const payload = { db_path: getDataDbPath(), symbol, freq, adjust };
    const res = await apiPost('/api/data/delete-symbol', payload);
    alert(`Deleted ${res.data.deleted_rows} rows.`);
    state.selectedInventoryKeys.delete(inventoryKey({ symbol, freq: freq || '-', adjust: adjust || '-' }));
    await refreshInventory();
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
}

async function deleteSelectedInventory() {
  const visibleKeys = new Set(getVisibleInventorySliceRows().map(inventoryKey));
  const keys = [...state.selectedInventoryKeys].filter(key => visibleKeys.has(key));
  if (!keys.length) return alert('Select visible inventory slices first.');
  const items = keys.map(inventoryItemFromKey).filter(item => item.symbol);
  if (!confirm(`Delete ${items.length} selected K-line slices?`)) return;
  try {
    const res = await apiPost('/api/data/delete-symbols', { db_path: getDataDbPath(), items });
    alert(`Deleted ${res.data.deleted_rows} rows.`);
    state.selectedInventoryKeys.clear();
    await refreshInventory();
  } catch (err) {
    alert(`Batch delete failed: ${err.message}`);
  }
}

function handleInventoryAction(event) {
  const checkbox = event.target.closest('input.inventory-select');
  if (checkbox) {
    if (checkbox.checked) state.selectedInventoryKeys.add(checkbox.dataset.key);
    else state.selectedInventoryKeys.delete(checkbox.dataset.key);
    renderInventoryTable();
    return;
  }

  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const symbol = button.dataset.symbol;
  const freq = normalizeOptionalValue(button.dataset.freq);
  const adjust = normalizeOptionalValue(button.dataset.adjust);
  if (button.dataset.action === 'detail') showSymbolDetail(symbol, { freq, adjust });
  else if (button.dataset.action === 'manage') {
    const normalized = String(symbol || '').toUpperCase();
    if (state.expandedInventorySymbols.has(normalized)) state.expandedInventorySymbols.delete(normalized);
    else state.expandedInventorySymbols.add(normalized);
    renderInventoryTable();
  }
  else if (button.dataset.action === 'delete') deleteSymbolData(symbol, freq, adjust);
}

function toggleSelectAllInventory() {
  const rows = getVisibleInventorySliceRows();
  const checked = $('selectAllInventory').checked;
  rows.forEach(row => {
    const key = inventoryKey(row);
    if (checked) state.selectedInventoryKeys.add(key);
    else state.selectedInventoryKeys.delete(key);
  });
  renderInventoryTable();
}


function handleFeatureDatasetAction(event) {
  const button = event.target.closest('#preflightFeatureDatasetBtn, #previewFeatureDatasetBtn, #buildFeatureDatasetBtn, #stopFeatureDatasetBtn');
  if (!button || button.disabled) return;
  event.preventDefault();
  if (button.id === 'preflightFeatureDatasetBtn') preflightFeatureDataset();
  else if (button.id === 'previewFeatureDatasetBtn') previewFeatureDataset();
  else if (button.id === 'buildFeatureDatasetBtn') buildFeatureDataset();
  else if (button.id === 'stopFeatureDatasetBtn') stopFeatureDataset();
}

function bindEvents() {
  document.addEventListener('click', handleFeatureDatasetAction);
  document.querySelectorAll('.nav-item').forEach(btn => btn.addEventListener('click', () => showSection(btn.dataset.section)));
  document.querySelectorAll('.visual-tab').forEach(btn => btn.addEventListener('click', () => showVisualTab(btn.dataset.visualTab || 'market')));
  $('refreshVisualizationBtn').addEventListener('click', refreshInventory);
  $('visualDbPath').addEventListener('change', () => {
    setSharedDbPath($('visualDbPath').value);
    refreshInventory();
  });
  $('visualSymbolInput').addEventListener('change', async () => {
    renderVisualizationControls({ keepSymbol: true, keepFreq: false });
    await loadVisualizationChart();
  });
  $('visualSymbolInput').addEventListener('keydown', async event => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    renderVisualizationControls({ keepSymbol: true, keepFreq: false });
    await loadVisualizationChart();
  });
  $('visualFreqButtons').addEventListener('click', async event => {
    const button = event.target.closest('button[data-freq]');
    if (!button || button.disabled) return;
    state.visualization.selectedFreq = button.dataset.freq;
    renderVisualizationControls({ keepSymbol: true, keepFreq: true });
    await loadVisualizationChart();
  });
  $('visualAdjustSelect').addEventListener('change', loadVisualizationChart);
  $('visualSubPanelCount').addEventListener('change', () => {
    updateVisualizationPanelVisibility();
    drawVisualizationPayload(state.visualization.payload);
  });
  $('visualIndicatorMenuBtn').addEventListener('click', event => {
    event.stopPropagation();
    $('visualIndicatorMenu').classList.toggle('hidden');
  });
  $('visualIndicatorMenu').addEventListener('change', event => {
    const id = event.target.dataset.overlayId;
    if (!id) return;
    if (event.target.checked) state.visualization.mainOverlays.add(id);
    else state.visualization.mainOverlays.delete(id);
    state.visualization.mainOverlayPreferenceExplicit = true;
    persistVisualizationPreferences();
    drawVisualizationPayload(state.visualization.payload);
  });
  $('visualInspectorToggleBtn').addEventListener('click', () => {
    state.visualization.inspectorVisible = !state.visualization.inspectorVisible;
    $('visualFeatureInspector').classList.toggle('hidden', !state.visualization.inspectorVisible);
    persistVisualizationPreferences();
    drawVisualizationPayload(state.visualization.payload);
  });
  $('visualChartGrid').addEventListener('change', event => {
    const select = event.target.closest('.visualPanelSelect');
    if (!select) return;
    const slot = Number(select.dataset.panelSlot || 0);
    const selected = select.value;
    state.visualization.panelSelections.forEach((value, index) => {
      if (index !== slot && selected !== 'none' && value === selected) {
        state.visualization.panelSelections[index] = 'none';
      }
    });
    state.visualization.panelSelections[slot] = selected;
    renderVisualizationPanelSelectors();
    persistVisualizationPreferences();
    drawVisualizationPayload(state.visualization.payload);
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.visual-indicator-picker')) $('visualIndicatorMenu').classList.add('hidden');
  });
  $('visualChartGrid').addEventListener('wheel', event => {
    event.preventDefault();
    zoomVisualization(event.deltaY, event.clientX);
  }, { passive: false });
  $('visualChartGrid').addEventListener('pointerdown', event => {
    if (hasVisualizationReferenceGuide()) {
      event.preventDefault();
      return;
    }
    const canvas = getVisualizationCanvasFromTarget(event.target);
    if (!canvas) return;
    event.preventDefault();
    startVisualizationDrag(event.clientX);
  });
  $('visualChartGrid').addEventListener('click', event => {
    if (!hasVisualizationReferenceGuide() || event.detail !== 1) return;
    event.preventDefault();
    clearPendingVisualizationReferenceClick();
    visualReferenceClickTimer = window.setTimeout(() => {
      visualReferenceClickTimer = null;
      clearVisualizationReferenceGuide();
    }, 260);
  });
  $('visualChartGrid').addEventListener('dblclick', event => {
    clearPendingVisualizationReferenceClick();
    if (hasVisualizationReferenceGuide()) {
      event.preventDefault();
      clearVisualizationReferenceGuide();
      return;
    }
    const canvas = getVisualizationCanvasFromTarget(event.target);
    if (!canvas) return;
    event.preventDefault();
    activateVisualizationReferenceGuide(canvas, event.clientX, event.clientY);
  });
  window.addEventListener('pointermove', event => {
    if (state.visualization.drag.active) {
      moveVisualizationDrag(event.clientX);
      return;
    }
    const canvas = getVisualizationCanvasFromTarget(event.target);
    if (canvas) {
      updateVisualizationHover(canvas, event.clientX, event.clientY);
    }
  });
  window.addEventListener('pointerup', endVisualizationDrag);
  window.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (!$('agentExportConflictModal')?.classList.contains('hidden')) {
      closeAgentExportConflictModal('cancel');
      return;
    }
    $('visualIndicatorMenu').classList.add('hidden');
    clearVisualizationReferenceGuide();
  });
  $('visualChartGrid').addEventListener('pointerleave', () => {
    if (!state.visualization.drag.active && !hasVisualizationReferenceGuide()) {
      clearVisualizationHover();
      drawVisualizationPayload(state.visualization.payload);
    }
  });
  $('openVisualDetailBtn').addEventListener('click', openVisualizationDetail);
  $('visualGoDataBtn').addEventListener('click', () => showSection('data'));
  $('refreshInventoryBtn').addEventListener('click', refreshInventory);
  $('dataDbPath').addEventListener('change', () => {
    setSharedDbPath($('dataDbPath').value);
    refreshInventory();
  });
  $('featureDbPath').addEventListener('change', () => setSharedDbPath($('featureDbPath').value));
  $('inventorySearch').addEventListener('input', renderInventoryTable);
  $('inventoryFreqFilter').addEventListener('change', renderInventoryTable);
  $('inventoryAdjustFilter').addEventListener('change', renderInventoryTable);
  document.querySelectorAll('[data-inventory-mode]').forEach(button => {
    button.addEventListener('click', () => {
      state.inventoryViewMode = button.dataset.inventoryMode || 'group';
      renderInventoryTable();
    });
  });
  $('deleteSelectedBtn').addEventListener('click', deleteSelectedInventory);
  $('inventoryTableBody').addEventListener('click', handleInventoryAction);
  $('useConfigSymbolsBtn').addEventListener('click', useConfigSymbolsForDownload);
  $('editConfigSymbolsBtn').addEventListener('click', () => showSection('config'));
  $('configUniversePreset').addEventListener('change', async () => {
    $('configUniversePath').value = $('configUniversePreset').value;
    await loadConfigSymbolsFromFile({ silent: true });
  });
  $('configUniversePath').addEventListener('change', () => setDownloadSymbolsPath($('configUniversePath').value));
  $('loadConfigSymbolsBtn').addEventListener('click', () => loadConfigSymbolsFromFile());
  $('saveConfigSymbolsBtn').addEventListener('click', saveConfigSymbolsToFile);
  $('setDownloadSymbolsBtn').addEventListener('click', useConfigSymbolsForDownload);
  $('applyConfigDefaultsBtn').addEventListener('click', applyConfigDefaultsToDownload);
  $('startDownloadBtn').addEventListener('click', startDownload);
  $('stopDownloadBtn').addEventListener('click', stopDownload);
  $('materializeBarsBtn').addEventListener('click', materializeBars);
  $('stopMaterializeBtn').addEventListener('click', stopMaterializeBars);
  $('materializeBaseFreq').addEventListener('change', syncMaterializeTargetsForBase);
  $('exportBundleBtn').addEventListener('click', exportBundle);
  $('importBundleBtn').addEventListener('click', importBundle);
  $('inspectBundleBtn').addEventListener('click', inspectBundle);
  $('stopBundleBtn').addEventListener('click', stopBundleJob);
  $('buildAgentCacheBtn').addEventListener('click', buildAgentCache);
  if ($('stopAgentCacheBtn')) $('stopAgentCacheBtn').addEventListener('click', stopAgentCacheBuild);
  $('inspectAgentCacheBtn').addEventListener('click', inspectAgentCache);
  $('useAgentCachePathBtn').addEventListener('click', useAgentCachePathForTraining);
  $('agentPreflightBtn').addEventListener('click', preflightAgent);
  $('startAgentTrainingBtn').addEventListener('click', startAgentTraining);
  $('stopAgentTrainingBtn').addEventListener('click', stopAgentTraining);
  $('pauseAgentRunBtn').addEventListener('click', () => controlAgentRun('pause'));
  $('resumeAgentRunBtn').addEventListener('click', () => controlAgentRun('resume'));
  $('checkpointAgentRunBtn').addEventListener('click', () => controlAgentRun('checkpoint'));
  $('refreshAgentRunsBtn').addEventListener('click', loadAgentRuns);
  $('agentPopoutBtn').addEventListener('click', openAgentMonitorWindow);
  readAgentChartGroupPrefs();
  applyAgentChartGroupState();
  document.querySelectorAll('[data-agent-toggle-group]').forEach(button => {
    button.addEventListener('click', () => {
      const group = button.dataset.agentToggleGroup;
      state.agentChartGroupsCollapsed[group] = !state.agentChartGroupsCollapsed[group];
      persistAgentChartGroupPrefs();
      applyAgentChartGroupState();
      drawAllAgentCharts();
    });
  });
  document.querySelectorAll('.agent-metric-chart').forEach(canvas => {
    canvas.addEventListener('pointermove', event => {
      const layout = canvas.__agentChartLayout;
      if (!layout || !layout.rowsCount) return;
      const rect = canvas.getBoundingClientRect();
      const localX = event.clientX - rect.left;
      let index = 0;
      if (Array.isArray(layout.pointXs) && layout.pointXs.length) {
        let bestDistance = Infinity;
        layout.pointXs.forEach((pointX, pointIndex) => {
          const distance = Math.abs(pointX - localX);
          if (distance < bestDistance) {
            bestDistance = distance;
            index = pointIndex;
          }
        });
      } else {
        const ratio = Math.max(0, Math.min(1, (localX - layout.pad.left) / Math.max(1, layout.plotW)));
        index = Math.round(ratio * Math.max(0, layout.rowsCount - 1));
      }
      state.agentChartHover[canvas.dataset.agentChartId || canvas.id] = { index };
      drawAllAgentCharts();
    });
    canvas.addEventListener('pointerleave', () => {
      delete state.agentChartHover[canvas.dataset.agentChartId || canvas.id];
      drawAllAgentCharts();
    });
  });
  document.addEventListener('click', event => {
    const legend = event.target.closest('[data-agent-legend-key]');
    if (!legend) return;
    const key = legend.dataset.agentLegendKey;
    state.agentChartHidden[key] = !state.agentChartHidden[key];
    drawAllAgentCharts();
  });
  $('closeAgentCheckpointModalBtn')?.addEventListener('click', closeAgentRunManageModal);
  $('agentCheckpointModalBackdrop')?.addEventListener('click', closeAgentRunManageModal);
  $('agentCheckpointModalTableBody')?.addEventListener('click', event => {
    const button = event.target.closest('[data-agent-export-checkpoint]');
    if (button) exportAgentCheckpoint(button);
  });
  $('agentExportRenameBtn')?.addEventListener('click', () => closeAgentExportConflictModal('rename'));
  $('agentExportOverwriteBtn')?.addEventListener('click', () => closeAgentExportConflictModal('overwrite'));
  $('agentExportCancelBtn')?.addEventListener('click', () => closeAgentExportConflictModal('cancel'));
  $('agentExportConflictBackdrop')?.addEventListener('click', () => closeAgentExportConflictModal('cancel'));
  $('agentProfile').addEventListener('change', () => applyAgentProfile($('agentProfile').value));
  document.querySelectorAll('.agentFrequency').forEach(input => input.addEventListener('change', renderAgentContract));
  document.querySelectorAll('.agent-tab').forEach(button => {
    button.addEventListener('click', () => showAgentTab(button.dataset.agentTab || 'setup'));
  });
  $('agentRunsTableBody').addEventListener('click', event => {
    const view = event.target.closest('[data-agent-view-run]');
    const manage = event.target.closest('[data-agent-manage-run]');
    const resume = event.target.closest('[data-agent-resume-run]');
    if (view) viewAgentRun(view.dataset.agentViewRun);
    if (manage) openAgentRunManage(manage.dataset.agentManageRun);
    if (resume) {
      state.agentRunId = resume.dataset.agentResumeRun;
      controlAgentRun('resume');
      showAgentTab('monitor');
    }
  });
  $('refreshEvaluationModelsBtn')?.addEventListener('click', loadEvaluationModels);
  $('startEvaluationBtn')?.addEventListener('click', startEvaluation);
  $('stopEvaluationBtn')?.addEventListener('click', () => stopEvaluation());
  $('refreshEvaluationRunsBtn')?.addEventListener('click', () => loadEvaluationRuns());
  $('refreshVisualEvaluationRunsBtn')?.addEventListener('click', () => loadEvaluationRuns());
  $('refreshVisualEvaluationReplayBtn')?.addEventListener('click', () => loadEvaluationReplay());
  $('stopVisualEvaluationBtn')?.addEventListener('click', () => stopEvaluation());
  $('visualEvaluationRunSelect')?.addEventListener('change', event => {
    setVisualEvaluationRun(event.target.value);
    loadEvaluationReplay(event.target.value);
  });
  $('evaluationRunsTableBody')?.addEventListener('click', event => {
    const stop = event.target.closest('[data-evaluation-stop-run]');
    if (stop) {
      stopEvaluation(stop.dataset.evaluationStopRun);
      return;
    }
    const view = event.target.closest('[data-evaluation-view-run]');
    if (!view) return;
    showSection('visualization');
    showVisualTab('evaluation');
    setVisualEvaluationRun(view.dataset.evaluationViewRun);
    loadEvaluationReplay(view.dataset.evaluationViewRun);
  });
  $('addFeatureIndicatorBtn').addEventListener('click', addFeatureIndicator);
  $('saveFeatureIndicatorsBtn').addEventListener('click', saveFeatureIndicators);
  $('featureIndicatorEditor').addEventListener('click', event => {
    const button = event.target.closest('.indicatorDelete');
    if (!button) return;
    state.featureIndicatorConfig.indicators = collectFeatureIndicators();
    const editor = button.closest('.indicator-editor');
    const index = Number(editor?.dataset.indicatorIndex);
    if (Number.isInteger(index)) {
      state.featureIndicatorConfig.indicators.splice(index, 1);
      renderFeatureIndicatorEditor();
    }
  });
  $('featureIndicatorEditor').addEventListener('change', event => {
    if (!event.target.classList.contains('indicatorKind')) return;
    state.featureIndicatorConfig.indicators = collectFeatureIndicators();
    const editor = event.target.closest('.indicator-editor');
    const index = Number(editor?.dataset.indicatorIndex);
    const item = state.featureIndicatorConfig?.indicators?.[index];
    if (!item) return;
    item.kind = event.target.value;
    item.render_target = item.kind === 'ema_channel' ? 'main_overlay' : 'sub_panel';
    item.params = Object.fromEntries((INDICATOR_PARAM_LABELS[item.kind] || []).map(([name]) => [name, name === 'slow' ? 26 : 9]));
    if (item.kind === 'ema_channel') item.params = { fast: 13, slow: 21 };
    if (item.kind === 'efi') item.params = { fast: 2, slow: 13, baseline: 20 };
    if (item.kind === 'kd') item.params = { lookback: 9, smooth_k: 3, smooth_d: 3 };
    if (item.kind === 'macd') item.params = { fast: 12, slow: 26, signal: 9 };
    renderFeatureIndicatorEditor();
  });
  $('addModelInputFeatureBtn').addEventListener('click', () => openModelInputCatalog(null));
  $('addModelInputGroupBtn').addEventListener('click', () => insertModelInputItem({
    id: modelInputItemId('group'), type: 'group', label: 'New Group',
  }));
  $('addModelInputCommentBtn').addEventListener('click', () => insertModelInputItem({
    id: modelInputItemId('comment'), type: 'comment', text: 'Add a note about the channels below.',
  }));
  $('resetModelInputBlueprintBtn').addEventListener('click', resetModelInputBlueprint);
  $('saveModelInputBlueprintBtn').addEventListener('click', saveModelInputBlueprint);
  $('closeModelInputCatalogBtn').addEventListener('click', () => $('modelInputCatalog').classList.add('hidden'));
  $('modelInputCatalogSearch').addEventListener('input', renderModelInputCatalog);
  $('modelInputCatalogGroups').addEventListener('click', event => {
    const button = event.target.closest('[data-model-input-catalog-name]');
    if (button && !button.disabled) addModelInputCatalogFeature(button.dataset.modelInputCatalogName);
  });
  $('modelInputList').addEventListener('click', event => {
    const button = event.target.closest('[data-model-input-action]');
    const row = event.target.closest('[data-model-input-index]');
    if (!button || !row) return;
    const index = Number(row.dataset.modelInputIndex);
    const items = state.modelInputBlueprint?.items || [];
    if (button.dataset.modelInputAction === 'up') moveModelInputItem(index, index - 1);
    if (button.dataset.modelInputAction === 'down') moveModelInputItem(index, index + 1);
    if (button.dataset.modelInputAction === 'insert') openModelInputCatalog(index + 1);
    if (button.dataset.modelInputAction === 'delete') {
      items.splice(index, 1);
      markModelInputDirty();
    }
  });
  $('modelInputList').addEventListener('input', event => {
    const row = event.target.closest('[data-model-input-index]');
    const field = event.target.dataset.modelInputText;
    if (!row || !field) return;
    const item = state.modelInputBlueprint?.items?.[Number(row.dataset.modelInputIndex)];
    if (!item) return;
    item[field] = event.target.value;
    markModelInputDirty({ rerender: false });
  });
  $('modelInputList').addEventListener('change', event => {
    const row = event.target.closest('[data-model-input-index]');
    if (!row) return;
    const item = state.modelInputBlueprint?.items?.[Number(row.dataset.modelInputIndex)];
    if (!item) return;
    if (event.target.hasAttribute('data-model-input-enabled')) {
      item.enabled = event.target.checked;
      markModelInputDirty();
    }
    const frequency = event.target.dataset.modelInputFrequency;
    if (frequency) {
      const selected = new Set(item.frequencies || []);
      if (event.target.checked) selected.add(frequency);
      else selected.delete(frequency);
      item.frequencies = (modelInputCatalogItem(item.name)?.available_frequencies || [])
        .filter(value => selected.has(value));
      markModelInputDirty();
    }
  });
  $('modelInputList').addEventListener('dragstart', event => {
    const row = event.target.closest('[data-model-input-index]');
    if (!row) return;
    state.modelInputDragIndex = Number(row.dataset.modelInputIndex);
    row.classList.add('dragging');
    event.dataTransfer.effectAllowed = 'move';
  });
  $('modelInputList').addEventListener('dragover', event => {
    const row = event.target.closest('[data-model-input-index]');
    if (!row) return;
    event.preventDefault();
    document.querySelectorAll('.model-input-row.drag-over').forEach(item => item.classList.remove('drag-over'));
    row.classList.add('drag-over');
  });
  $('modelInputList').addEventListener('drop', event => {
    const row = event.target.closest('[data-model-input-index]');
    if (!row || state.modelInputDragIndex === null) return;
    event.preventDefault();
    const from = state.modelInputDragIndex;
    let to = Number(row.dataset.modelInputIndex);
    if (from < to) to -= 1;
    moveModelInputItem(from, to);
  });
  $('modelInputList').addEventListener('dragend', () => {
    state.modelInputDragIndex = null;
    document.querySelectorAll('.model-input-row.dragging, .model-input-row.drag-over')
      .forEach(item => item.classList.remove('dragging', 'drag-over'));
  });
  $('checkCoverageBtn').addEventListener('click', checkCoverage);
  $('buildUniverseBtn').addEventListener('click', buildUniverse);
  $('closeDetailModalBtn').addEventListener('click', closeDetailModal);
  $('detailModalBackdrop').addEventListener('click', closeDetailModal);
  $('modalFreqSelect').addEventListener('change', () => reloadDetailForSelectedFilters({ freqChanged: true }));
  $('modalAdjustSelect').addEventListener('change', () => reloadDetailForSelectedFilters({ freqChanged: false }));
  $('modalTableWrap').addEventListener('scroll', () => {
    const el = $('modalTableWrap');
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 80) loadDetailPage(false);
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  document.body.dataset.section = 'visualization';
  bindEvents();
  setupLanguageToggle();
  syncMaterializeTargetsForBase();
  setSharedDbPath(DEFAULT_DB_PATH);
  setDownloadRunning(false);
  setDownloadSymbolsPath(DEFAULT_DOWNLOAD_SYMBOLS_FILE);
  await loadFeatureSpec();
  await loadFeatureIndicators();
  await loadModelInputBlueprint();
  await loadAgentSpec();
  await loadEvaluationModels();
  await loadEvaluationRuns({ silent: true });
  await loadConfigSymbolsFromFile({ silent: true });
  await refreshInventory();
  await resumeActiveDownloadJob();
  applyLanguage(currentLanguage());
  const query = new URLSearchParams(window.location.search);
  if (query.get('section') === 'agent') showSection('agent');
  if (query.get('section') === 'evaluation') showSection('evaluation');
  if (query.get('section') === 'live') showSection('live');
  if (query.get('agentTab')) showAgentTab(query.get('agentTab'));
  if (query.get('run')) viewAgentRun(query.get('run'));
  if (query.get('evaluationRun')) { showSection('visualization'); showVisualTab('evaluation'); setVisualEvaluationRun(query.get('evaluationRun')); loadEvaluationReplay(query.get('evaluationRun')); }
});

window.addEventListener('beforeunload', event => {
  if (!state.modelInputDirty) return;
  event.preventDefault();
  event.returnValue = '';
});

window.addEventListener('resize', () => {
  if ($('visualization')?.classList.contains('active')) {
    if (state.evaluationReplay.activeTab === 'evaluation') drawEvaluationReplay();
    else drawVisualizationPayload(state.visualization.payload);
  }
  drawAllAgentCharts();
});
