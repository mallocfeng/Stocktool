<script setup>
import { reactive, ref, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import axios from 'axios';
import flatpickr from 'flatpickr';
import { Mandarin as flatpickrZh } from 'flatpickr/dist/l10n/zh.js';
import 'flatpickr/dist/themes/dark.css';
import FormulaBuilder from './FormulaBuilder.vue';

flatpickr.localize(flatpickrZh);

const props = defineProps({
  busy: { type: Boolean, default: false },
});

const emit = defineEmits(['run', 'block', 'unblock']);
let importTicket = 0;

const DEFAULT_FORMULA = `SHORT := EMA(CLOSE,12);
LONG := EMA(CLOSE,26);
DIF := SHORT - LONG;
DEA := EMA(DIF,9);
MACD := 2 * (DIF - DEA);
MA5 := MA(CLOSE,5);
MA13 := MA(CLOSE,13);
MA34 := MA(CLOSE,34);
VOLMA5 := MA(VOL,5);
VOLMA20 := MA(VOL,20);
B_COND := C > MA5 AND MA5 > MA13 AND MA13 > MA34 AND C > REF(C,1) AND VOL > VOLMA5 * 1.2 AND MACD > 0 AND DIF > DEA AND DIF > REF(DIF,1);
S_COND := C < MA5 OR MACD < 0 OR DIF < DEA;`;

const config = reactive({
  csv_path: '',
  formula: DEFAULT_FORMULA,
  initial_capital: 100000,
  fee_rate: 0.0005,
  multi_freqs: 'D,W,M',
  strategy_text: '',
  strategies: {
    fixed: { enabled: true, periods: '5,10,20' },
    tpsl: { enabled: true, tp: 10, sl: 5 },
    dca: { enabled: false, size: 5, target: 20 },
    grid: { enabled: false, pct: 5, cash: 1000, limit: '', accumulate: true },
    dynamic: {
      enabled: true,
      lossStepAmount: 10,
      maxAddSteps: 5,
      maxInvestmentLimit: 50000,
      resetOnWin: true,
      maxDrawdownLimit: '',
      enableHedge: true,
      hedgeInitialInvestment: 5000,
      hedgeLossStepAmount: 5,
      hedgeMaxAddSteps: 5,
    },
  },
});

const fileInput = ref(null);
const uploading = ref(false);
const stockCode = ref('');
const fetching = ref(false);
const assetLabel = ref('');
const builderVisible = ref(false);
const dateRange = reactive({ start: '', end: '' });
const selectedPreset = ref('all');
const startPickerEl = ref(null);
const endPickerEl = ref(null);
let startPicker = null;
let endPicker = null;
const rangePresetOptions = [
  { key: 'all', label: '全部' },
  { key: '1M', label: '近1月' },
  { key: '3M', label: '近3月' },
  { key: '6M', label: '近6月' },
  { key: '1Y', label: '近1年' },
];
let presetApplying = false;

const formatLocalIso = (dateObj, defaults) => {
  const pad = (n) => String(n).padStart(2, '0');
  const yyyy = dateObj.getFullYear();
  const mm = pad(dateObj.getMonth() + 1);
  const dd = pad(dateObj.getDate());
  let hh = pad(dateObj.getHours());
  let min = pad(dateObj.getMinutes());
  if (defaults) {
    hh = pad(defaults.hours);
    min = pad(defaults.minutes);
  }
  return `${yyyy}-${mm}-${dd}T${hh}:${min}`;
};

const applyRangePreset = (key) => {
  presetApplying = true;
  if (key === 'all') {
    dateRange.start = '';
    dateRange.end = '';
  } else {
    const now = new Date();
    const endStr = formatLocalIso(now, { hours: 15, minutes: 0 });
    let start = new Date(now);
    if (key.endsWith('M')) {
      start.setMonth(start.getMonth() - parseInt(key, 10));
    } else if (key.endsWith('Y')) {
      start.setFullYear(start.getFullYear() - parseInt(key, 10));
    }
    dateRange.start = formatLocalIso(start, { hours: 9, minutes: 0 });
    dateRange.end = endStr;
  }
  selectedPreset.value = key;
  nextTick(() => {
    presetApplying = false;
  });
};

watch(
  () => [dateRange.start, dateRange.end],
  () => {
    if (presetApplying) return;
    selectedPreset.value = dateRange.start || dateRange.end ? 'custom' : 'all';
  }
);

const toLocalIso = (date) => {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(
    date.getMinutes()
  )}`;
};

const syncPickerFromModel = (picker, value) => {
  if (!picker) return;
  if (!value) {
    if (picker.selectedDates.length) picker.clear();
    return;
  }
  const nextDate = new Date(value);
  if (!picker.selectedDates.length || Math.abs(picker.selectedDates[0].getTime() - nextDate.getTime()) > 60000) {
    picker.setDate(nextDate, false);
  }
};

watch(
  () => dateRange.start,
  (val) => syncPickerFromModel(startPicker, val)
);

watch(
  () => dateRange.end,
  (val) => syncPickerFromModel(endPicker, val)
);

const buildPicker = (elRef, onChange) => {
  if (!elRef?.value) return null;
  return flatpickr(elRef.value, {
    enableTime: true,
    time_24hr: true,
    minuteIncrement: 30,
    dateFormat: 'Y-m-d H:i',
    allowInput: true,
    wrap: false,
    onChange: (selectedDates) => {
      onChange(selectedDates?.[0] ? toLocalIso(selectedDates[0]) : '');
    },
  });
};

onMounted(() => {
  startPicker = buildPicker(startPickerEl, (val) => {
    dateRange.start = val;
  });
  endPicker = buildPicker(endPickerEl, (val) => {
    dateRange.end = val;
  });
  syncPickerFromModel(startPicker, dateRange.start);
  syncPickerFromModel(endPicker, dateRange.end);
});

onBeforeUnmount(() => {
  startPicker?.destroy();
  endPicker?.destroy();
});

const triggerFileSelect = () => {
  fileInput.value?.click();
};

const handleFileUpload = async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const ticket = ++importTicket;
  const formData = new FormData();
  formData.append('file', file);
  uploading.value = true;
  emit('block', {
    title: '正在上传 CSV 文件',
    detail: '大文件上传可能需要较长时间，请勿刷新或重复操作…',
  });
  try {
    const res = await axios.post('/upload', formData);
    if (ticket !== importTicket) return;
    config.csv_path = res.data.path;
    assetLabel.value = file.name || '';
    stockCode.value = '';
    autoRunBacktest();
  } catch (e) {
    alert('上传失败：' + (e.response?.data?.detail || e.message));
  } finally {
    uploading.value = false;
    emit('unblock');
  }
};

const handleFetchFromSina = async () => {
  const code = stockCode.value.trim();
  if (!code) {
    alert('请输入股票代码，例如 600519 或 sh600519');
    return;
  }
  const ticket = ++importTicket;
  fetching.value = true;
  emit('block', {
    title: '正在从新浪下载行情',
    detail: '网络行情加载需要一些时间，请耐心等待完成…',
  });
  try {
    const res = await axios.post('/import/sina', { symbol: code });
    if (ticket !== importTicket) return;
    config.csv_path = res.data.path;
    assetLabel.value = res.data.label || (res.data.symbol || code).toUpperCase();
    autoRunBacktest();
  } catch (e) {
    alert('读取失败：' + (e.response?.data?.detail || e.message));
  } finally {
    fetching.value = false;
    emit('unblock');
  }
};

const checkFormula = () => {
  if (!config.formula || !config.formula.toUpperCase().includes('B_COND')) {
    alert('公式中必须包含 B_COND := ... 请补充买入条件。');
  } else {
    alert('公式已包含 B_COND，可运行回测。');
  }
};

const convertFromText = async () => {
  const desc = config.strategy_text.trim();
  if (!desc) {
    alert('请输入策略描述后再转换。');
    return;
  }
  try {
    const res = await axios.post('/analytics/nlp_formula', { text: desc });
    config.formula = res.data.formula || config.formula;
  } catch (e) {
    alert('转换失败：' + (e.response?.data?.detail || e.message));
  }
};

const buildStrategiesPayload = () => {
  const strategies = {};
  const toNumber = (value, fallback = 0) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  };
  const toRatio = (value) => {
    const num = Number(value);
    if (Number.isNaN(num) || num <= 0) {
      return 0;
    }
    return num > 1 ? num / 100 : num;
  };
  const parseDrawdownLimit = (input) => {
    if (input === null || input === undefined) return null;
    if (typeof input === 'string') {
      const trimmed = input.trim();
      if (!trimmed) return null;
      if (trimmed.endsWith('%')) {
        const num = Number(trimmed.slice(0, -1));
        return Number.isFinite(num) ? num / 100 : null;
      }
      const num = Number(trimmed);
      return Number.isFinite(num) ? num : null;
    }
    const num = Number(input);
    return Number.isFinite(num) ? num : null;
  };
  if (config.strategies.fixed.enabled) {
    const parts = config.strategies.fixed.periods
      .split(',')
      .map((p) => parseInt(p.trim(), 10))
      .filter((n) => !Number.isNaN(n) && n > 0);
    if (parts.length) strategies.fixed = parts;
  }
  if (config.strategies.tpsl.enabled) {
    strategies.tpsl = {
      tp: toRatio(config.strategies.tpsl.tp),
      sl: toRatio(config.strategies.tpsl.sl),
    };
  }
  if (config.strategies.dca.enabled) {
    strategies.dca = {
      size: toRatio(config.strategies.dca.size),
      target: toRatio(config.strategies.dca.target),
    };
  }
  if (config.strategies.grid.enabled) {
    const limitRaw = (config.strategies.grid.limit ?? '').toString().trim();
    const parsedLimit = limitRaw && limitRaw.toLowerCase() !== 'none' ? parseInt(limitRaw, 10) : null;
    strategies.grid = {
      grid_pct: toRatio(config.strategies.grid.pct),
      single_cash: Number(config.strategies.grid.cash),
      max_grids: Number.isNaN(parsedLimit) ? null : parsedLimit,
      accumulate: Boolean(config.strategies.grid.accumulate),
    };
  }
  if (config.strategies.dynamic?.enabled) {
    const drawdownLimit = parseDrawdownLimit(config.strategies.dynamic.maxDrawdownLimit);
    const dynamicCfg = {
      initialInvestment: toNumber(config.initial_capital),
      lossStepAmount: toNumber(config.strategies.dynamic.lossStepAmount),
      maxAddSteps: parseInt(config.strategies.dynamic.maxAddSteps, 10) || 0,
      maxInvestmentLimit: toNumber(config.strategies.dynamic.maxInvestmentLimit),
      resetOnWin: Boolean(config.strategies.dynamic.resetOnWin),
      enableHedge: Boolean(config.strategies.dynamic.enableHedge),
      hedgeInitialInvestment: toNumber(config.strategies.dynamic.hedgeInitialInvestment),
      hedgeLossStepAmount: toNumber(config.strategies.dynamic.hedgeLossStepAmount),
      hedgeMaxAddSteps: parseInt(config.strategies.dynamic.hedgeMaxAddSteps, 10) || 0,
      maxDrawdownLimit: drawdownLimit,
    };
    if (drawdownLimit === null) {
      delete dynamicCfg.maxDrawdownLimit;
    }
    strategies.dynamic = dynamicCfg;
  }
  return strategies;
};

const autoRunBacktest = () => {
  if (props.busy) return;
  if (!config.csv_path) return;
  if (!config.formula || !config.formula.toUpperCase().includes('B_COND')) return;
  const strategies = buildStrategiesPayload();
  if (!Object.keys(strategies).length) return;
  runBacktest();
};

const handleApplyFormula = (formula) => {
  if (typeof formula === 'string' && formula.trim()) {
    config.formula = formula;
  }
  builderVisible.value = false;
};

const runBacktest = () => {
  if (!config.csv_path) {
    alert('请先上传行情 CSV 文件。');
    return;
  }
  if (!config.formula || !config.formula.toUpperCase().includes('B_COND')) {
    alert('请先准备包含 B_COND 的通达信公式。');
    return;
  }
  const strategies = buildStrategiesPayload();
  if (!Object.keys(strategies).length) {
    alert('请至少启用一个策略模块。');
    return;
  }
  const payload = {
    csv_path: config.csv_path,
    formula: config.formula,
    initial_capital: Number(config.initial_capital),
    fee_rate: Number(config.fee_rate),
    strategies,
    date_start: dateRange.start || null,
    date_end: dateRange.end || null,
  };
  emit('run', {
    payload,
    meta: {
      initialCapital: Number(config.initial_capital),
      multiFreqs: config.multi_freqs,
      assetLabel: assetLabel.value || (config.csv_path?.split('/').pop() || ''),
    },
  });
};
</script>

<template>
  <div class="card config-panel">
    <h2 class="panel-title">策略配置</h2>

    <section class="form-section">
      <div class="section-title">行情数据</div>
      <div class="file-row">
        <input class="text-input" type="text" :value="config.csv_path" placeholder="请上传 CSV 文件" readonly />
        <button class="secondary" type="button" @click="triggerFileSelect" :disabled="uploading">
          {{ uploading ? '上传中…' : '选择文件' }}
        </button>
        <input ref="fileInput" type="file" class="hidden-input" accept=".csv" @change="handleFileUpload" />
      </div>
      <div class="file-row">
        <input
          class="text-input"
          type="text"
          v-model="stockCode"
          placeholder="输入股票代码，例如 600519 或 sh600519"
        />
        <button class="secondary" type="button" @click="handleFetchFromSina" :disabled="fetching">
          {{ fetching ? '读取中…' : '读取在线数据' }}
        </button>
      </div>
    </section>

    <section class="form-section">
      <div class="section-title">时间范围</div>
      <div class="range-card">
        <div class="range-presets">
          <button
            v-for="preset in rangePresetOptions"
            :key="preset.key"
            type="button"
            class="chip-button"
            :class="{ active: selectedPreset === preset.key }"
            @click="applyRangePreset(preset.key)"
          >
            {{ preset.label }}
          </button>
          <button
            type="button"
            class="chip-button ghost"
            :class="{ active: selectedPreset === 'custom' }"
            @click="selectedPreset = 'custom'"
          >
            自定义
          </button>
        </div>
        <div class="range-inputs">
          <label class="field">
            <span>起始时间</span>
            <input type="text" ref="startPickerEl" class="dt-input" placeholder="选择开始时间" />
          </label>
          <label class="field">
            <span>结束时间</span>
            <input type="text" ref="endPickerEl" class="dt-input" placeholder="选择结束时间" />
          </label>
        </div>
        <small class="field-hint">若留空则使用完整数据；时间精确到小时。</small>
      </div>
    </section>

    <section class="form-section grid">
      <label class="field">
        <span>初始资金</span>
        <input type="number" v-model="config.initial_capital" min="0" />
      </label>
      <label class="field">
        <span>手续费（单边）</span>
        <input type="number" v-model="config.fee_rate" step="0.0001" />
      </label>
      <label class="field">
        <span>多周期 (如 D,W,M)</span>
        <input type="text" v-model="config.multi_freqs" />
        <small class="field-hint">
          输入单个周期（<code>D</code>、<code>1H</code>、<code>7m</code> 等）并用逗号分隔。
          想要“高周期看趋势、低周期找入场”时，可写 <code>15m&gt;5m</code>（或 <code>15m-&gt;5m@趋势+入场</code>），表示：
          先计算 15 分钟信号判断多空，只在 15 分钟保持多头时才考虑 5 分钟买入。行情粒度不足的周期会被自动忽略并给出提示。
        </small>
      </label>
    </section>

    <section class="form-section">
      <div class="section-title">通达信公式</div>
      <textarea v-model="config.formula" rows="6" class="code-area"></textarea>
      <div class="button-row">
        <button type="button" class="secondary" @click="builderVisible = true">公式向导</button>
        <button type="button" class="secondary" @click="checkFormula">检查公式</button>
        <button type="button" class="secondary" @click="convertFromText">自然语言转公式</button>
      </div>
      <input type="text" class="text-input" placeholder="例如：5日均线金叉20日配合放量" v-model="config.strategy_text" />
    </section>

    <section class="form-section">
      <div class="section-title">策略模块</div>
      <div class="strategy-card dynamic-card">
        <label class="checkbox-row">
          <input type="checkbox" v-model="config.strategies.dynamic.enabled" />
          <span>动态资金管理</span>
        </label>
        <div class="sub-grid" v-if="config.strategies.dynamic.enabled">
          <label class="field">
            <span>亏损加注（手）</span>
            <input type="number" v-model="config.strategies.dynamic.lossStepAmount" min="0" />
            <small class="field-hint">单位为手，1 手 = 100 股</small>
          </label>
          <label class="field">
            <span>连续加注次数</span>
            <input type="number" v-model="config.strategies.dynamic.maxAddSteps" min="0" />
          </label>
          <label class="field">
            <span>单笔资金上限</span>
            <input type="number" v-model="config.strategies.dynamic.maxInvestmentLimit" min="0" />
          </label>
          <label class="field">
            <span>最大允许回撤</span>
            <input
              type="text"
              v-model="config.strategies.dynamic.maxDrawdownLimit"
              placeholder="如 50000 或 15%"
            />
          </label>
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.dynamic.resetOnWin" />
            <span>盈利后恢复初始金额</span>
          </label>
          <div class="sub-section">
            <label class="checkbox-row">
              <input type="checkbox" v-model="config.strategies.dynamic.enableHedge" />
              <span>启用反向对冲</span>
            </label>
            <div class="sub-grid" v-if="config.strategies.dynamic.enableHedge">
              <label class="field">
                <span>对冲初始金额</span>
                <input type="number" v-model="config.strategies.dynamic.hedgeInitialInvestment" min="0" />
              </label>
              <label class="field">
                <span>对冲亏损加注（手）</span>
                <input type="number" v-model="config.strategies.dynamic.hedgeLossStepAmount" min="0" />
                <small class="field-hint">单位为手，1 手 = 100 股</small>
              </label>
              <label class="field">
                <span>对冲加注上限</span>
                <input type="number" v-model="config.strategies.dynamic.hedgeMaxAddSteps" min="0" />
              </label>
            </div>
          </div>
        </div>
      </div>
      <div class="strategy-card" v-for="module in ['fixed','tpsl','dca','grid']" :key="module">
        <template v-if="module==='fixed'">
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.fixed.enabled" />
            <span>固定周期持有</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.fixed.enabled">
            <label class="field">
              <span>周期（逗号分隔）</span>
              <input type="text" v-model="config.strategies.fixed.periods" />
            </label>
          </div>
        </template>
        <template v-else-if="module==='tpsl'">
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.tpsl.enabled" />
            <span>止盈 / 止损</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.tpsl.enabled">
            <label class="field">
              <span>止盈 %</span>
              <input type="number" v-model="config.strategies.tpsl.tp" />
            </label>
            <label class="field">
              <span>止损 %</span>
              <input type="number" v-model="config.strategies.tpsl.sl" />
            </label>
          </div>
        </template>
        <template v-else-if="module==='dca'">
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.dca.enabled" />
            <span>定投模式</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.dca.enabled">
            <label class="field">
              <span>定投比例 %</span>
              <input type="number" v-model="config.strategies.dca.size" />
            </label>
            <label class="field">
              <span>目标收益 %</span>
              <input type="number" v-model="config.strategies.dca.target" />
            </label>
          </div>
        </template>
        <template v-else>
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.grid.enabled" />
            <span>网格策略</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.grid.enabled">
            <label class="field">
              <span>网格 %</span>
              <input type="number" v-model="config.strategies.grid.pct" />
            </label>
            <label class="field">
              <span>单网资金</span>
              <input type="number" v-model="config.strategies.grid.cash" />
            </label>
            <label class="field">
              <span>最大网格数</span>
              <input type="number" v-model="config.strategies.grid.limit" />
            </label>
            <label class="checkbox-row">
              <input type="checkbox" v-model="config.strategies.grid.accumulate" />
              <span>累积份额</span>
            </label>
          </div>
        </template>
      </div>
    </section>

    <button class="primary run-button" type="button" @click="runBacktest" :disabled="props.busy">
      {{ props.busy ? '回测执行中…' : '开始回测' }}
    </button>
  </div>

  <FormulaBuilder
    :show="builderVisible"
    :initialFormula="config.formula"
    @close="builderVisible = false"
    @apply="handleApplyFormula"
  />
</template>
