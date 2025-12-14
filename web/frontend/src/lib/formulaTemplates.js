const PRICE_SOURCES = [
  { label: '收盘价 CLOSE', value: 'CLOSE' },
  { label: '开盘价 OPEN', value: 'OPEN' },
  { label: '最高价 HIGH', value: 'HIGH' },
  { label: '最低价 LOW', value: 'LOW' },
  { label: '成交量 VOL', value: 'VOL' },
];

const ensurePositiveInt = (value, fallback) => {
  const num = parseInt(value, 10);
  if (!Number.isFinite(num) || num <= 0) return fallback;
  return num;
};

const ensureNumber = (value, fallback) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const withSemicolon = (text) => {
  const trimmed = (text || '').trim();
  if (!trimmed) return '';
  return trimmed.endsWith(';') ? trimmed : `${trimmed};`;
};

export const indicatorBlocks = [
  {
    id: 'ema',
    label: 'EMA 指标',
    description: '变量 := EMA(数据源, 周期)',
    fields: [
      {
        key: 'alias',
        label: '变量名',
        type: 'text',
        default: 'EMA12',
        hint: '用于后续引用的变量名，可自定义（如 EMA4、EMA20）',
      },
      { key: 'source', label: '数据源', type: 'select', options: PRICE_SOURCES, default: 'CLOSE' },
      { key: 'period', label: '周期', type: 'number', min: 1, default: 12, hint: '计算窗口长度（天/根数）' },
    ],
    build: (values) => {
      const alias = values.alias?.trim() || 'EMA12';
      const source = values.source || 'CLOSE';
      const period = ensurePositiveInt(values.period, 12);
      return withSemicolon(`${alias} := EMA(${source},${period})`);
    },
  },
  {
    id: 'ma',
    label: 'MA 指标',
    description: '变量 := MA(数据源, 周期)',
    fields: [
      {
        key: 'alias',
        label: '变量名',
        type: 'text',
        default: 'MA5',
        hint: '命名仅供引用，习惯用 MA5 表示 5 日均线，改为 MA4 即自定义变量名',
      },
      { key: 'source', label: '数据源', type: 'select', options: PRICE_SOURCES, default: 'CLOSE' },
      { key: 'period', label: '周期', type: 'number', min: 1, default: 5, hint: '计算该均线所用的窗口数' },
    ],
    build: (values) => {
      const alias = values.alias?.trim() || 'MA5';
      const source = values.source || 'CLOSE';
      const period = ensurePositiveInt(values.period, 5);
      return withSemicolon(`${alias} := MA(${source},${period})`);
    },
  },
  {
    id: 'macd',
    label: 'MACD 套件',
    description: '自动生成 SHORT/LONG/DIF/DEA/MACD',
    fields: [
      { key: 'source', label: '数据源', type: 'select', options: PRICE_SOURCES, default: 'CLOSE' },
      { key: 'shortPeriod', label: '短周期', type: 'number', default: 12, min: 1 },
      { key: 'longPeriod', label: '长周期', type: 'number', default: 26, min: 1 },
      { key: 'deaPeriod', label: 'DEA 平滑', type: 'number', default: 9, min: 1 },
      { key: 'shortAlias', label: '短 EMA 变量', type: 'text', default: 'SHORT' },
      { key: 'longAlias', label: '长 EMA 变量', type: 'text', default: 'LONG' },
      { key: 'difAlias', label: 'DIF 变量', type: 'text', default: 'DIF' },
      { key: 'deaAlias', label: 'DEA 变量', type: 'text', default: 'DEA' },
      { key: 'macdAlias', label: 'MACD 变量', type: 'text', default: 'MACD' },
    ],
    build: (values) => {
      const source = values.source || 'CLOSE';
      const shortPeriod = ensurePositiveInt(values.shortPeriod, 12);
      const longPeriod = ensurePositiveInt(values.longPeriod, 26);
      const deaPeriod = ensurePositiveInt(values.deaPeriod, 9);
      const shortAlias = values.shortAlias?.trim() || 'SHORT';
      const longAlias = values.longAlias?.trim() || 'LONG';
      const difAlias = values.difAlias?.trim() || 'DIF';
      const deaAlias = values.deaAlias?.trim() || 'DEA';
      const macdAlias = values.macdAlias?.trim() || 'MACD';
      return [
        withSemicolon(`${shortAlias} := EMA(${source},${shortPeriod})`),
        withSemicolon(`${longAlias} := EMA(${source},${longPeriod})`),
        withSemicolon(`${difAlias} := ${shortAlias} - ${longAlias}`),
        withSemicolon(`${deaAlias} := EMA(${difAlias},${deaPeriod})`),
        withSemicolon(`${macdAlias} := 2 * (${difAlias} - ${deaAlias})`),
      ];
    },
  },
  {
    id: 'vol_ma',
    label: '量能均线',
    description: '变量 := MA(VOL, 周期)',
    fields: [
      { key: 'alias', label: '变量名', type: 'text', default: 'VOLMA5' },
      { key: 'period', label: '周期', type: 'number', min: 1, default: 5 },
    ],
    build: (values) => {
      const alias = values.alias?.trim() || 'VOLMA5';
      const period = ensurePositiveInt(values.period, 5);
      return withSemicolon(`${alias} := MA(VOL,${period})`);
    },
  },
  {
    id: 'custom',
    label: '自定义语句',
    description: '直接输入任意赋值或函数调用',
    fields: [{ key: 'script', label: '语句', type: 'textarea', rows: 2, default: '' }],
    build: (values) => withSemicolon(values.script || ''),
  },
];

const OPERATOR_OPTIONS = [
  { label: '大于 (>)', value: '>' },
  { label: '大于等于 (>=)', value: '>=' },
  { label: '小于 (<)', value: '<' },
  { label: '小于等于 (<=)', value: '<=' },
  { label: '等于 (=)', value: '=' },
  { label: '不等于 (<>)', value: '<>' },
];

export const conditionBlocks = [
  {
    id: 'compare',
    label: '数值比较',
    description: '左侧与右侧进行比较，如 CLOSE > MA(CLOSE,5)',
    fields: [
      { key: 'left', label: '左侧表达式', type: 'text', default: 'CLOSE' },
      { key: 'operator', label: '比较符', type: 'select', options: OPERATOR_OPTIONS, default: '>' },
      { key: 'right', label: '右侧表达式', type: 'text', default: 'MA(CLOSE,5)' },
    ],
    build: (values) => {
      const left = values.left?.trim() || 'CLOSE';
      const operator = values.operator || '>';
      const right = values.right?.trim() || 'MA(CLOSE,5)';
      return `${left} ${operator} ${right}`;
    },
  },
  {
    id: 'range',
    label: '区间判断',
    description: '判断表达式是否位于区间内',
    fields: [
      { key: 'expr', label: '表达式', type: 'text', default: 'CLOSE' },
      { key: 'min', label: '下限', type: 'text', default: 'MA(CLOSE,34)' },
      { key: 'max', label: '上限', type: 'text', default: 'MA(CLOSE,5)' },
    ],
    build: (values) => {
      const expr = values.expr?.trim() || 'CLOSE';
      const min = values.min?.trim() || 'MA(CLOSE,34)';
      const max = values.max?.trim() || 'MA(CLOSE,5)';
      return `${min} < ${expr} AND ${expr} < ${max}`;
    },
  },
  {
    id: 'ref_compare',
    label: '与历史比较',
    description: '表达式与 REF(expr, N) 比较，可用于“上涨/回落”',
    fields: [
      { key: 'expr', label: '表达式', type: 'text', default: 'CLOSE' },
      { key: 'operator', label: '比较符', type: 'select', options: OPERATOR_OPTIONS, default: '>' },
      { key: 'bars', label: '对比周期', type: 'number', min: 1, default: 1 },
    ],
    build: (values) => {
      const expr = values.expr?.trim() || 'CLOSE';
      const operator = values.operator || '>';
      const bars = ensurePositiveInt(values.bars, 1);
      return `${expr} ${operator} REF(${expr},${bars})`;
    },
  },
  {
    id: 'cross',
    label: 'CROSS 穿越',
    description: '使用 CROSS 检测快线向上/向下穿越慢线',
    fields: [
      { key: 'fast', label: '快线表达式', type: 'text', default: 'MA(CLOSE,5)' },
      { key: 'slow', label: '慢线表达式', type: 'text', default: 'MA(CLOSE,13)' },
      {
        key: 'direction',
        label: '方向',
        type: 'select',
        options: [
          { label: '快线上穿慢线', value: 'up' },
          { label: '快线下穿慢线', value: 'down' },
        ],
        default: 'up',
      },
    ],
    build: (values) => {
      const fast = values.fast?.trim() || 'MA(CLOSE,5)';
      const slow = values.slow?.trim() || 'MA(CLOSE,13)';
      return values.direction === 'down' ? `CROSS(${slow},${fast})` : `CROSS(${fast},${slow})`;
    },
  },
  {
    id: 'volume_spike',
    label: '放量条件',
    description: '成交量 > 均量 × 倍数',
    fields: [
      { key: 'maPeriod', label: '均量周期', type: 'number', min: 1, default: 5 },
      { key: 'multiplier', label: '倍数', type: 'number', step: 0.1, default: 1.2 },
    ],
    build: (values) => {
      const period = ensurePositiveInt(values.maPeriod, 5);
      const multiplier = ensureNumber(values.multiplier, 1.2);
      const rounded = Number(multiplier.toFixed(4));
      return `VOL > MA(VOL,${period}) * ${rounded}`;
    },
  },
  {
    id: 'custom',
    label: '自定义条件',
    description: '直接输入任意布尔表达式',
    fields: [{ key: 'expression', label: '表达式', type: 'textarea', rows: 2, default: '' }],
    build: (values) => values.expression?.trim() || '',
  },
];
