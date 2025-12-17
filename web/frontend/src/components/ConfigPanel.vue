<script setup>
import { reactive, ref, watch, nextTick, onMounted, onBeforeUnmount, computed } from 'vue';
import axios from 'axios';
import flatpickr from 'flatpickr';
import { Mandarin as flatpickrZh } from 'flatpickr/dist/l10n/zh.js';
import 'flatpickr/dist/themes/dark.css';
import FormulaBuilder from './FormulaBuilder.vue';
import TdxCodeEditor from './TdxCodeEditor.vue';

flatpickr.localize(flatpickrZh);

const props = defineProps({
  busy: { type: Boolean, default: false },
  isMobile: { type: Boolean, default: false },
});

const emit = defineEmits(['run', 'block', 'unblock']);
let importTicket = 0;

const formulaLintLogs = ref([]);
const formulaLintSummary = ref('');

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

const LOT_SIZE = 100;

const config = reactive({
  csv_path: '',
  formula: DEFAULT_FORMULA,
  initial_capital: 100000,
  fee_rate: 0.0005,
  multi_freqs: 'D,W,M',
  strategy_text: '',
  strategies: {
    fixed: { enabled: true, periods: '5,10,20' },
    tpsl: { enabled: false, tp: 10, sl: 5 },
    dca: { enabled: false, size: 5, target: 20 },
    grid: { enabled: false, pct: 5, cash: 1000, limit: '', accumulate: true },
    dynamic: {
      enabled: false,
      lossStepAmount: 10,
      maxAddSteps: 5,
      maxInvestmentLimit: 50000,
      singleInvestmentLimit: '',
      forceOneLotEntry: true,
      allowSingleLimitOverride: false,
      resetOnWin: true,
      maxDrawdownLimit: '',
      enableHedge: true,
      hedgeInitialInvestment: 5000,
      hedgeLossStepAmount: 5,
      hedgeMaxAddSteps: 5,
    },
    buyHedge: {
      enabled: false,
      hedge: {
        enabled: false,
        mode: 'full',
      },
      allowRepeatAfterExit: true,
      stepMode: 'fixed',
      stepFixedType: 'percent',
      stepFixedValue: 3,
      stepAbsoluteValue: 1,
      stepAbsoluteRounding: 'round',
      stepAuto: {
        method: 'atr',
        atrPeriod: 14,
        atrMultiplier: 1.0,
        avgRangeLength: 5,
        avgRangeMultiplier: 1.0,
        stdPeriod: 20,
        stdMultiplier: 1.0,
        maFast: 5,
        maSlow: 20,
        maGapPct: 1,
      },
      growthMode: 'equal',
      growthEqualHands: 1,
      growthIncrementVariant: 'arithmetic',
      growthIncrementBase: 1,
      growthIncrementStep: 1,
      growthIncrementFlexibleStep: 50,
      growthDoubleBase: 1,
      positionMode: 'fixed',
      positionFixedPct: '',
      positionIncrementStartPct: '',
      positionIncrementStepPct: '',
      entryMode: 'none',
      entryMaFast: 5,
      entryMaSlow: 10,
      entryMaPeriod: 20,
      entryProgressiveCount: 3,
      profitMode: 'percent',
      profitTargetPercent: 5,
      profitTargetAbsolute: 0,
      profitBase: 'overall',
      profitBatch: false,
      reverse: {
        enabled: false,
        indicator: 'rsi',
        interval: 5,
        filterMode: 'consecutive',
        filterValue: 3,
        minHits: 2,
        action: 'exit',
        profitType: 'percent',
        profitValue: 2,
        threshold: 30,
      },
      capitalMode: 'unlimited',
      capitalFixedAmount: '',
      capitalFixedPercent: '',
      capitalIncrementStart: '',
      capitalIncrementStep: '',
      exitMode: 'batch',
      exitBatchPct: 50,
      exitBatchStrategy: 'per_batch',
      exitBatchStepPct: 25,
      exitSingleType: 'market',
      limits: {
        limitBuyPrice: '',
        limitSellPrice: '',
        minPrice: '',
        stopAddingAtMin: false,
      },
      baseInitialHands: '',
      baseReferencePrice: '',
      baseReferenceSource: 'first',
      maxAddCount: 5,
      reference: 'last',
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
const strategyDrawerOpen = ref(false);
const strategyPanelRef = ref(null);
const dataReady = computed(() => Boolean(config.csv_path));
const strategyManifest = [
  { key: 'dynamic', label: '动态资金管理' },
  { key: 'buyHedge', label: '买入对冲' },
  { key: 'fixed', label: '固定周期持有' },
  { key: 'tpsl', label: '止盈 / 止损' },
  { key: 'dca', label: '定投模式' },
  { key: 'grid', label: '网格策略' },
];
const enabledStrategies = computed(() =>
  strategyManifest.filter(({ key }) => config.strategies?.[key]?.enabled).map((item) => item.label)
);
const strategySummaryText = computed(() => {
  const list = enabledStrategies.value;
  if (!list.length) return '当前未启用模块';
  if (list.length <= 2) return `已启用：${list.join('、')}`;
  const preview = list.slice(0, 3).join('、');
  return `已启用 ${list.length} 项：${preview}${list.length > 3 ? '…' : ''}`;
});

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

const resetViewportZoom = () => {
  if (typeof document === 'undefined') return;
  const active = document.activeElement;
  if (active && typeof active.blur === 'function') {
    active.blur();
  }
  const viewport = document.querySelector('meta[name="viewport"]');
  if (!viewport) return;
  const original = viewport.dataset.originalContent || viewport.getAttribute('content') || '';
  if (!viewport.dataset.originalContent) {
    viewport.dataset.originalContent = original;
  }
  viewport.setAttribute('content', 'width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no');
  setTimeout(() => {
    viewport.setAttribute('content', viewport.dataset.originalContent || 'width=device-width, initial-scale=1');
  }, 300);
};

const handleFetchFromSina = async () => {
  const code = stockCode.value.trim();
  if (!code) {
    alert('请输入股票代码，例如 600519 或 sh600519');
    return;
  }
  resetViewportZoom();
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

const scrollToDrawer = () => {
  nextTick(() => {
    if (!strategyPanelRef.value) return;
    const block = props.isMobile ? 'start' : 'center';
    strategyPanelRef.value.scrollIntoView({ behavior: 'smooth', block });
  });
};

const openStrategyDrawer = () => {
  if (!dataReady.value) return;
  strategyDrawerOpen.value = true;
  scrollToDrawer();
};
const closeStrategyDrawer = () => {
  strategyDrawerOpen.value = false;
};
const toggleStrategyDrawer = () => {
  if (!dataReady.value) return;
  const opening = !strategyDrawerOpen.value;
  strategyDrawerOpen.value = opening;
  if (opening) scrollToDrawer();
};

const validateFormula = async () => {
  if (!config.formula || !config.formula.trim()) {
    alert('请先输入通达信公式。');
    return;
  }
  if (!config.csv_path) {
    alert('请先上传/选择 CSV 行情数据后再校验公式。');
    return;
  }
  formulaLintSummary.value = '校验中…';
  formulaLintLogs.value = [];
  try {
    const res = await axios.post('/formula/validate', {
      csv_path: config.csv_path,
      formula: config.formula,
      date_start: dateRange.start || null,
      date_end: dateRange.end || null,
    });
    const logs = res.data?.logs || [];
    formulaLintLogs.value = Array.isArray(logs) ? logs : [String(logs)];
    const buyCount = res.data?.buy_count ?? 0;
    const sellCount = res.data?.sell_count ?? 0;
    formulaLintSummary.value = `校验完成：买入信号 ${buyCount}，卖出信号 ${sellCount}`;
    if (!String(config.formula || '').toUpperCase().includes('B_COND')) {
      alert('校验完成，但公式中缺少 B_COND（买入条件）。');
    } else {
      alert(formulaLintSummary.value);
    }
  } catch (e) {
    formulaLintSummary.value = '校验失败';
    formulaLintLogs.value = [e.response?.data?.detail || e.message];
    alert('校验失败：' + (e.response?.data?.detail || e.message));
  }
};

const copyTextToClipboard = async (text) => {
  const raw = String(text ?? '');
  if (!raw) return false;
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(raw);
      return true;
    }
  } catch {
    // fallback below
  }
  try {
    if (typeof document === 'undefined') return false;
    const textarea = document.createElement('textarea');
    textarea.value = raw;
    textarea.setAttribute('readonly', 'true');
    textarea.style.position = 'fixed';
    textarea.style.top = '-1000px';
    textarea.style.left = '-1000px';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);
    const ok = document.execCommand('copy');
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
};

const clearFormula = async () => {
  const current = String(config.formula || '');
  if (!current.trim()) {
    alert('当前公式为空，无需清空。');
    return;
  }
  const ok = confirm('确认清空当前公式吗？系统会先把原内容复制到剪贴板，方便你随时粘贴恢复。');
  if (!ok) return;
  const copied = await copyTextToClipboard(current);
  if (!copied) {
    alert('复制到剪贴板失败，为了安全已取消清空。请手动复制后再清空。');
    return;
  }
  config.formula = '';
  formulaLintLogs.value = [];
  formulaLintSummary.value = '';
  alert('已清空公式，原内容已复制到剪贴板。');
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
  const parseOptionalNumber = (input) => {
    if (input === null || input === undefined) return null;
    if (typeof input === 'string' && !input.trim()) return null;
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
        singleInvestmentLimit: toNumber(config.strategies.dynamic.singleInvestmentLimit),
        resetOnWin: Boolean(config.strategies.dynamic.resetOnWin),
        enableHedge: Boolean(config.strategies.dynamic.enableHedge),
        hedgeInitialInvestment: toNumber(config.strategies.dynamic.hedgeInitialInvestment),
        hedgeLossStepAmount: toNumber(config.strategies.dynamic.hedgeLossStepAmount),
        hedgeMaxAddSteps: parseInt(config.strategies.dynamic.hedgeMaxAddSteps, 10) || 0,
        forceOneLotEntry: Boolean(config.strategies.dynamic.forceOneLotEntry),
        allowSingleLimitOverride: Boolean(config.strategies.dynamic.allowSingleLimitOverride),
        maxDrawdownLimit: drawdownLimit,
      };
      if (drawdownLimit === null) {
      delete dynamicCfg.maxDrawdownLimit;
    }
    strategies.dynamic = dynamicCfg;
  }
  if (config.strategies.buyHedge?.enabled) {
    const bh = config.strategies.buyHedge;
    const normalizeHands = (value) => {
      const num = Number(value);
      if (!Number.isFinite(num)) return 0;
      return Math.max(0, Math.floor(num));
    };
    const toShares = (hands) => {
      const normalized = normalizeHands(hands);
      return normalized <= 0 ? 0 : normalized * LOT_SIZE;
    };
    const growthMode = bh.growthMode || 'equal';
    let startHands = 0;
    if (growthMode === 'equal') {
      startHands = toNumber(bh.growthEqualHands);
    } else if (growthMode === 'increment') {
      startHands = toNumber(bh.growthIncrementBase);
    } else {
      startHands = toNumber(bh.growthDoubleBase);
    }
    let incrementHands = 0;
    if (growthMode === 'increment') {
      incrementHands = toNumber(bh.growthIncrementStep);
    } else if (growthMode === 'double') {
      incrementHands = toNumber(bh.growthDoubleBase);
    }
    const startShares = toShares(startHands);
    const incrementShares = toShares(incrementHands);
    if (startShares >= LOT_SIZE) {
      const stepType = bh.stepFixedType || 'percent';
      const stepPct = stepType === 'percent' ? toRatio(bh.stepFixedValue) : 0;
      const stepAbs = stepType === 'absolute' ? toNumber(bh.stepAbsoluteValue) : 0;
      const buyHedgeCfg = {
        hedge: {
          enabled: Boolean(bh.hedge?.enabled),
          mode: bh.hedge?.mode || 'full',
        },
        allow_repeat: Boolean(bh.allowRepeatAfterExit),
        step_mode: bh.stepMode || 'fixed',
        step_type: stepType,
        step_pct: stepPct,
        step_abs: stepAbs,
        step_rounding: bh.stepAbsoluteRounding,
        step_auto: {
          method: bh.stepAuto?.method || 'atr',
          atr_period: parseOptionalNumber(bh.stepAuto?.atrPeriod) ?? 0,
          atr_multiplier: parseOptionalNumber(bh.stepAuto?.atrMultiplier) ?? 1,
          avg_range_length: parseOptionalNumber(bh.stepAuto?.avgRangeLength) ?? 0,
          avg_range_multiplier: parseOptionalNumber(bh.stepAuto?.avgRangeMultiplier) ?? 1,
          std_period: parseOptionalNumber(bh.stepAuto?.stdPeriod) ?? 0,
          std_multiplier: parseOptionalNumber(bh.stepAuto?.stdMultiplier) ?? 1,
          ma_fast: parseOptionalNumber(bh.stepAuto?.maFast) ?? 0,
          ma_slow: parseOptionalNumber(bh.stepAuto?.maSlow) ?? 0,
          ma_gap_pct: parseOptionalNumber(bh.stepAuto?.maGapPct) ?? 0,
        },
        growth: {
          mode: growthMode,
          equal_hands: toNumber(bh.growthEqualHands),
          increment_variant: bh.growthIncrementVariant || 'arithmetic',
          increment_base: toNumber(bh.growthIncrementBase),
          increment_step: toNumber(bh.growthIncrementStep),
          increment_flexible: toNumber(bh.growthIncrementFlexibleStep),
          double_base: toNumber(bh.growthDoubleBase),
        },
        position: {
          mode: bh.positionMode || 'fixed',
          fixed_pct: toNumber(bh.positionFixedPct),
          inc_start_pct: toNumber(bh.positionIncrementStartPct),
          inc_step_pct: toNumber(bh.positionIncrementStepPct),
        },
        entry: {
          mode: bh.entryMode || 'none',
          ma_period: toNumber(bh.entryMaPeriod),
          ma_fast: toNumber(bh.entryMaFast),
          ma_slow: toNumber(bh.entryMaSlow),
          progressive_count: parseInt(bh.entryProgressiveCount, 10) || 0,
        },
        profit: {
          mode: bh.profitMode || 'percent',
          target_pct: toRatio(bh.profitTargetPercent),
          target_abs: toNumber(bh.profitTargetAbsolute),
          reference: bh.profitBase || 'overall',
          per_batch: Boolean(bh.profitBatch),
        },
        reverse: {
          enabled: Boolean(bh.reverse?.enabled),
          indicator: bh.reverse?.indicator || 'rsi',
          interval: parseInt(bh.reverse?.interval, 10) || 0,
          filter_mode: bh.reverse?.filterMode || 'consecutive',
          filter_value: parseInt(bh.reverse?.filterValue, 10) || 0,
          min_hits: parseInt(bh.reverse?.minHits, 10) || 0,
          threshold: toNumber(bh.reverse?.threshold),
          action: bh.reverse?.action || 'exit',
          profit_type: bh.reverse?.profitType || 'percent',
          profit_value: toNumber(bh.reverse?.profitValue),
        },
        capital: {
          mode: bh.capitalMode || 'unlimited',
          fixed_amount: parseOptionalNumber(bh.capitalFixedAmount),
          fixed_percent: parseOptionalNumber(bh.capitalFixedPercent),
          increment_start: parseOptionalNumber(bh.capitalIncrementStart),
          increment_step: parseOptionalNumber(bh.capitalIncrementStep),
        },
        exit: {
          mode: bh.exitMode || 'batch',
          batch_pct: toNumber(bh.exitBatchPct),
          batch_strategy: bh.exitBatchStrategy || 'per_batch',
          batch_step_pct: toNumber(bh.exitBatchStepPct),
          single_type: bh.exitSingleType || 'market',
        },
        limits: {
          limit_buy_price: parseOptionalNumber(bh.limits?.limitBuyPrice),
          limit_sell_price: parseOptionalNumber(bh.limits?.limitSellPrice),
          min_price: parseOptionalNumber(bh.limits?.minPrice),
          stop_adding_at_min: Boolean(bh.limits?.stopAddingAtMin),
        },
        base: {
          initial_hands: toNumber(bh.baseInitialHands),
          reference_price: toNumber(bh.baseReferencePrice),
          reference_source: bh.baseReferenceSource || 'first',
        },
        start_position: startShares,
        increment_unit: incrementShares,
        max_adds: parseInt(bh.maxAddCount, 10) || 0,
        reference: bh.reference || 'last',
      };
      strategies.buy_hedge = buyHedgeCfg;
    }
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
  strategyDrawerOpen.value = false;
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
        <button class="fetch-online-button" type="button" @click="handleFetchFromSina" :disabled="fetching">
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
      <TdxCodeEditor
        v-model="config.formula"
        :lintLogs="formulaLintLogs"
        minHeight="220px"
        placeholder="输入通达信公式脚本，例如：B_COND := ...; S_COND := ...;"
      />
      <div v-if="formulaLintSummary" class="formula-summary">{{ formulaLintSummary }}</div>
      <div v-if="formulaLintLogs.length" class="formula-log">
        <div v-for="(msg, idx) in formulaLintLogs" :key="idx">{{ msg }}</div>
      </div>
      <div class="button-row">
        <button type="button" class="secondary" @click="builderVisible = true">公式向导</button>
        <button type="button" class="secondary" @click="validateFormula">检查公式</button>
        <button type="button" class="secondary danger" @click="clearFormula">清空公式</button>
        <button type="button" class="secondary" @click="convertFromText">自然语言转公式</button>
      </div>
      <input type="text" class="text-input" placeholder="例如：5日均线金叉20日配合放量" v-model="config.strategy_text" />
    </section>

    <section class="form-section strategy-launch">
      <div class="strategy-launch-header">
        <div class="strategy-launch-info">
          <div class="section-title">策略模块</div>
          <span class="summary-label">已启用</span>
          <div class="summary-content">
            <template v-if="enabledStrategies.length">
              <span
                v-for="mod in enabledStrategies"
                :key="mod"
                class="summary-pill"
              >
                {{ mod }}
              </span>
            </template>
            <span v-else>未启用任何模块</span>
          </div>
        </div>
        <button type="button" class="secondary strategy-toggle" @click="toggleStrategyDrawer" :disabled="!dataReady">
          {{ strategyDrawerOpen ? '收起模块' : '展开设置' }}
        </button>
      </div>
      <ul class="strategy-hints">
        <li>点击展开抽屉即可管理所有模块</li>
        <li>关闭抽屉可腾出更多空间查看结果</li>
      </ul>
    </section>

    <transition name="drawer-slide">
      <div
        v-if="strategyDrawerOpen"
        :class="props.isMobile ? 'strategy-inline-wrapper' : 'strategy-drawer-layer'"
      >
        <div v-if="!props.isMobile" class="drawer-backdrop" @click="closeStrategyDrawer"></div>
        <div
          :class="['strategy-panel', props.isMobile ? 'strategy-inline-panel' : 'strategy-drawer-panel']"
          :role="props.isMobile ? 'region' : 'dialog'"
          :aria-modal="props.isMobile ? null : 'true'"
          ref="strategyPanelRef"
        >
          <div :class="['drawer-header', { 'inline-drawer-header': props.isMobile }]">
            <div>
              <h3>策略模块配置</h3>
              <p>勾选并调整想要启用的模块</p>
            </div>
            <button class="plain-icon close-btn" type="button" @click="closeStrategyDrawer" aria-label="关闭策略模块">
              <span aria-hidden="true"></span>
            </button>
          </div>
          <div class="drawer-body">
            <div class="strategy-group">
              <div class="strategy-card dynamic-card">
                <label class="checkbox-row dynamic-card-toggle">
                  <input type="checkbox" v-model="config.strategies.dynamic.enabled" />
                  <span>动态资金管理</span>
                </label>
                <div class="dynamic-card-body" v-if="config.strategies.dynamic.enabled">
                  <div class="dynamic-grid">
                    <label class="field">
                      <span>亏损加注（手）</span>
                      <input type="number" v-model="config.strategies.dynamic.lossStepAmount" min="0" />
                      <small class="field-hint">单位为手，1 手 = 100 股</small>
                    </label>
                    <label class="field">
                      <span>连续加注次数</span>
                      <input type="number" v-model="config.strategies.dynamic.maxAddSteps" min="0" />
                    </label>
                  </div>
                  <div class="dynamic-note">
                    <p>首单金额由市场价格自动推算（1 手成本 + 手续费），与任何金额限制无关；“单笔上限 / 总上限”只约束后续加仓或总暴露。</p>
                    <p>若 1 手都超出限额，可选择“允许忽略一次”或禁止加仓并提示原因。</p>
                  </div>
                  <div class="dynamic-grid dynamic-grid--limits">
                    <div class="dynamic-limit-toggle">
                      <label class="checkbox-row">
                        <input type="checkbox" v-model="config.strategies.dynamic.forceOneLotEntry" />
                        <span>启用 1 手首单</span>
                      </label>
                      <small class="field-hint">首单固定 1 手，后续加仓再受金额限制</small>
                    </div>
                    <label class="field">
                      <span>单笔加仓上限</span>
                      <input type="number" v-model="config.strategies.dynamic.singleInvestmentLimit" min="0" />
                      <small class="field-hint">用于限制每次加仓金额，首单 1 手不受限</small>
                    </label>
                  </div>
                  <label class="checkbox-row">
                    <input type="checkbox" v-model="config.strategies.dynamic.allowSingleLimitOverride" />
                    <span>当单笔上限低于 1 手时仍允许加仓</span>
                  </label>
                  <div class="dynamic-grid dynamic-grid--limits">
                    <label class="field">
                      <span>总资金上限</span>
                      <input type="number" v-model="config.strategies.dynamic.maxInvestmentLimit" min="0" />
                      <small class="field-hint">当前持仓市值 + 加仓不得超过此额度</small>
                    </label>
                    <label class="field">
                      <span>最大允许回撤</span>
                      <input
                        type="text"
                        v-model="config.strategies.dynamic.maxDrawdownLimit"
                        placeholder="如 50000 或 15%"
                      />
                    </label>
                  </div>
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
                    <div class="sub-grid">
                      <label class="field">
                        <span>最大加仓次数</span>
                        <input type="number" min="0" v-model="config.strategies.buyHedge.maxAddCount" />
                        <small class="field-hint">0 表示不限制加仓层数</small>
                      </label>
                      <label class="field">
                        <span>触发基准</span>
                        <select v-model="config.strategies.buyHedge.reference">
                          <option value="last">以上一笔买入价</option>
                          <option value="first">以首次买入价</option>
                        </select>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
              <div class="module-divider" role="presentation"></div>
              <div class="strategy-card buy-hedge-card">
                <label class="checkbox-row">
                  <input type="checkbox" v-model="config.strategies.buyHedge.enabled" />
                  <span>买入对冲（逢跌加仓）</span>
                </label>
                <div class="buyhedge-sections" v-if="config.strategies.buyHedge.enabled">
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">对冲 & 重启</div>
                    <div class="sub-grid">
                      <label class="checkbox-row">
                        <input type="checkbox" v-model="config.strategies.buyHedge.hedge.enabled" />
                        <span>启用对冲行为（反向头寸或止损/退出）</span>
                      </label>
                      <label class="field">
                        <span>对冲模式</span>
                        <select v-model="config.strategies.buyHedge.hedge.mode">
                          <option value="full">反向仓全面对冲</option>
                          <option value="weak">仅反向止损 / 反向退出</option>
                        </select>
                      </label>
                      <label class="checkbox-row">
                        <input type="checkbox" v-model="config.strategies.buyHedge.allowRepeatAfterExit" />
                        <span>清仓后允许从开仓条件重新开始</span>
                      </label>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">步长设置</div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>步长来源</span>
                        <select v-model="config.strategies.buyHedge.stepMode">
                          <option value="fixed">固定</option>
                          <option value="auto">自动</option>
                        </select>
                      </label>
                    </div>
                    <div class="sub-grid" v-if="config.strategies.buyHedge.stepMode === 'fixed'">
                      <label class="field">
                        <span>步长单位</span>
                        <select v-model="config.strategies.buyHedge.stepFixedType">
                          <option value="percent">百分比</option>
                          <option value="absolute">绝对价差</option>
                        </select>
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.stepFixedType === 'percent'">
                        <span>步长 (%)</span>
                        <input
                          type="number"
                          v-model="config.strategies.buyHedge.stepFixedValue"
                          min="0"
                          step="0.1"
                        />
                      </label>
                      <label class="field" v-else>
                        <span>步长（价差）</span>
                        <input
                          type="number"
                          v-model="config.strategies.buyHedge.stepAbsoluteValue"
                          min="0"
                          step="0.01"
                        />
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.stepFixedType === 'absolute'">
                        <span>价差取整规则</span>
                        <select v-model="config.strategies.buyHedge.stepAbsoluteRounding">
                          <option value="round">四舍五入</option>
                          <option value="floor">向下取整</option>
                          <option value="ceil">向上取整</option>
                        </select>
                      </label>
                      <p class="field-note">A 股按最小变动单位（分）取整，避免出现不可委托价。</p>
                    </div>
                    <div class="sub-grid" v-else>
                      <label class="field">
                        <span>自动算法</span>
                        <select v-model="config.strategies.buyHedge.stepAuto.method">
                          <option value="atr">ATR</option>
                          <option value="avg_range">最近 N 根平均真实波幅</option>
                          <option value="stddev">标准差</option>
                          <option value="ma_gap">均线乖离</option>
                        </select>
                      </label>
                      <template v-if="config.strategies.buyHedge.stepAuto.method === 'atr'">
                        <label class="field">
                          <span>ATR 周期</span>
                          <input type="number" min="1" v-model="config.strategies.buyHedge.stepAuto.atrPeriod" />
                        </label>
                        <label class="field">
                          <span>倍数</span>
                          <input
                            type="number"
                            min="0"
                            step="0.1"
                            v-model="config.strategies.buyHedge.stepAuto.atrMultiplier"
                          />
                        </label>
                      </template>
                      <template v-else-if="config.strategies.buyHedge.stepAuto.method === 'avg_range'">
                        <label class="field">
                          <span>平均真实波幅长度 N</span>
                          <input
                            type="number"
                            min="1"
                            v-model="config.strategies.buyHedge.stepAuto.avgRangeLength"
                          />
                        </label>
                        <label class="field">
                          <span>乘数</span>
                          <input
                            type="number"
                            min="0"
                            step="0.1"
                            v-model="config.strategies.buyHedge.stepAuto.avgRangeMultiplier"
                          />
                        </label>
                      </template>
                      <template v-else-if="config.strategies.buyHedge.stepAuto.method === 'stddev'">
                        <label class="field">
                          <span>标准差周期</span>
                          <input type="number" min="1" v-model="config.strategies.buyHedge.stepAuto.stdPeriod" />
                        </label>
                        <label class="field">
                          <span>倍数</span>
                          <input
                            type="number"
                            min="0"
                            step="0.1"
                            v-model="config.strategies.buyHedge.stepAuto.stdMultiplier"
                          />
                        </label>
                      </template>
                      <template v-else>
                        <label class="field">
                          <span>快线周期</span>
                          <input
                            type="number"
                            min="1"
                            v-model="config.strategies.buyHedge.stepAuto.maFast"
                          />
                        </label>
                        <label class="field">
                          <span>慢线周期</span>
                          <input
                            type="number"
                            min="1"
                            v-model="config.strategies.buyHedge.stepAuto.maSlow"
                          />
                        </label>
                        <label class="field">
                          <span>乖离阈值 (%)</span>
                          <input
                            type="number"
                            min="0"
                            step="0.1"
                            v-model="config.strategies.buyHedge.stepAuto.maGapPct"
                          />
                        </label>
                      </template>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">仓位增长 & 资金占比</div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>增长模式</span>
                        <select v-model="config.strategies.buyHedge.growthMode">
                          <option value="equal">等长（每次买入固定仓位）</option>
                          <option value="increment">递增</option>
                          <option value="double">加倍</option>
                        </select>
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.growthMode === 'equal'">
                        <span>固定仓位（手）</span>
                        <input type="number" min="0" v-model="config.strategies.buyHedge.growthEqualHands" />
                      </label>
                      <template v-else-if="config.strategies.buyHedge.growthMode === 'increment'">
                        <label class="field">
                          <span>递增类型</span>
                          <select v-model="config.strategies.buyHedge.growthIncrementVariant">
                            <option value="arithmetic">等差（base + k × step）</option>
                            <option value="flexible">等差（可配置增量）</option>
                          </select>
                        </label>
                        <label class="field">
                          <span>起始仓位（手）</span>
                          <input
                            type="number"
                            min="0"
                            v-model="config.strategies.buyHedge.growthIncrementBase"
                          />
                        </label>
                        <label class="field">
                          <span>递增单位（手）</span>
                          <input
                            type="number"
                            min="0"
                            v-model="config.strategies.buyHedge.growthIncrementStep"
                          />
                        </label>
                        <label class="field" v-if="config.strategies.buyHedge.growthIncrementVariant === 'flexible'">
                          <span>可调增量（手）</span>
                          <input
                            type="number"
                            min="0"
                            v-model="config.strategies.buyHedge.growthIncrementFlexibleStep"
                          />
                        </label>
                      </template>
                      <label class="field" v-else>
                        <span>加倍基础仓位（手）</span>
                        <input type="number" min="0" v-model="config.strategies.buyHedge.growthDoubleBase" />
                      </label>
                    </div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>仓位量模式</span>
                        <select v-model="config.strategies.buyHedge.positionMode">
                          <option value="fixed">固定百分比仓位</option>
                          <option value="increment">递增百分比仓位</option>
                        </select>
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.positionMode === 'fixed'">
                        <span>仓位占比 (%)</span>
                        <input
                          type="number"
                          min="0"
                          max="100"
                          v-model="config.strategies.buyHedge.positionFixedPct"
                        />
                      </label>
                      <template v-else>
                        <label class="field">
                          <span>起始仓位 (%)</span>
                          <input type="number" min="0" max="100" v-model="config.strategies.buyHedge.positionIncrementStartPct" />
                        </label>
                        <label class="field">
                          <span>递增 (%)</span>
                          <input type="number" min="0" max="100" v-model="config.strategies.buyHedge.positionIncrementStepPct" />
                        </label>
                      </template>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">开仓条件</div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>开仓方式</span>
                        <select v-model="config.strategies.buyHedge.entryMode">
                          <option value="none">无 MA，任意满足策略即可首单</option>
                          <option value="ma">MA 指标金叉 / 上穿</option>
                          <option value="ma_progressive">MA 指标累进满足</option>
                        </select>
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.entryMode !== 'none'">
                        <span>MA 时间窗口 N</span>
                        <input type="number" min="1" v-model="config.strategies.buyHedge.entryMaPeriod" />
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.entryMode !== 'none'">
                        <span>快/慢线周期</span>
                        <input
                          type="number"
                          min="1"
                          v-model="config.strategies.buyHedge.entryMaFast"
                        />
                        <input
                          type="number"
                          min="1"
                          v-model="config.strategies.buyHedge.entryMaSlow"
                        />
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.entryMode === 'ma_progressive'">
                        <span>连续满足次数（5 分钟 K）</span>
                        <input
                          type="number"
                          min="1"
                          v-model="config.strategies.buyHedge.entryProgressiveCount"
                        />
                      </label>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">止盈方式</div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>止盈类型</span>
                        <select v-model="config.strategies.buyHedge.profitMode">
                          <option value="percent">百分比止盈</option>
                          <option value="absolute">固定价差止盈</option>
                        </select>
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.profitMode === 'percent'">
                        <span>阈值 (%)</span>
                        <input
                          type="number"
                          min="0"
                          step="0.1"
                          v-model="config.strategies.buyHedge.profitTargetPercent"
                        />
                      </label>
                      <label class="field" v-else>
                        <span>阈值（价差）</span>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          v-model="config.strategies.buyHedge.profitTargetAbsolute"
                        />
                      </label>
                      <label class="field">
                        <span>参考价</span>
                        <select v-model="config.strategies.buyHedge.profitBase">
                          <option value="overall">整体持仓均价</option>
                          <option value="last">最后一笔</option>
                          <option value="batch">分批单独计算</option>
                        </select>
                      </label>
                      <label class="checkbox-row">
                        <input type="checkbox" v-model="config.strategies.buyHedge.profitBatch" />
                        <span>分批逐笔止盈</span>
                      </label>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">反转指标</div>
                    <div class="sub-grid">
                      <label class="checkbox-row">
                        <input type="checkbox" v-model="config.strategies.buyHedge.reverse.enabled" />
                        <span>启用反转信号</span>
                      </label>
                      <label class="field">
                        <span>指标</span>
                        <select v-model="config.strategies.buyHedge.reverse.indicator">
                          <option value="rsi">RSI</option>
                          <option value="macd">MACD</option>
                          <option value="kdj">KDJ</option>
                          <option value="ma_turn">均线拐头</option>
                          <option value="price_pattern">价格形态（新低失败）</option>
                        </select>
                      </label>
                    </div>
                    <div class="sub-grid" v-if="config.strategies.buyHedge.reverse.enabled">
                      <label class="field">
                        <span>计算周期 N（5 分钟 K）</span>
                        <input type="number" min="1" v-model="config.strategies.buyHedge.reverse.interval" />
                      </label>
                      <label class="field">
                        <span>过滤方式</span>
                        <select v-model="config.strategies.buyHedge.reverse.filterMode">
                          <option value="consecutive">连续 N 根满足</option>
                          <option value="at_least">过去 N 根至少满足 X 次</option>
                        </select>
                      </label>
                      <label class="field">
                        <span>阈值 / 可视次数</span>
                        <input type="number" min="1" v-model="config.strategies.buyHedge.reverse.filterValue" />
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.reverse.filterMode === 'at_least'">
                        <span>至少满足次数</span>
                        <input type="number" min="1" v-model="config.strategies.buyHedge.reverse.minHits" />
                      </label>
                      <label class="field">
                        <span>指标触发阈值</span>
                        <input type="number" min="0" v-model="config.strategies.buyHedge.reverse.threshold" />
                      </label>
                      <label class="field">
                        <span>触发后的操作</span>
                        <select v-model="config.strategies.buyHedge.reverse.action">
                          <option value="exit">立刻离场</option>
                          <option value="adjust">切换止盈 / 启动“反转盈利”</option>
                        </select>
                      </label>
                      <label class="field">
                        <span>反转盈利类型</span>
                        <select v-model="config.strategies.buyHedge.reverse.profitType">
                          <option value="percent">百分比</option>
                          <option value="absolute">固定价差</option>
                        </select>
                      </label>
                      <label class="field">
                        <span>反转盈利阈值</span>
                        <input
                          type="number"
                          min="0"
                          step="0.1"
                          v-model="config.strategies.buyHedge.reverse.profitValue"
                        />
                      </label>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">资金控制 & 离场</div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>资金模式</span>
                        <select v-model="config.strategies.buyHedge.capitalMode">
                          <option value="unlimited">不限</option>
                          <option value="fixed">固定</option>
                          <option value="increment">递增</option>
                        </select>
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.capitalMode === 'fixed'">
                        <span>固定资金 (元)</span>
                        <input type="number" min="0" v-model="config.strategies.buyHedge.capitalFixedAmount" />
                      </label>
                      <label class="field" v-if="config.strategies.buyHedge.capitalMode === 'fixed'">
                        <span>或占比 (%)</span>
                        <input
                          type="number"
                          min="0"
                          max="100"
                          v-model="config.strategies.buyHedge.capitalFixedPercent"
                        />
                      </label>
                      <template v-else-if="config.strategies.buyHedge.capitalMode === 'increment'">
                        <label class="field">
                          <span>起始仓位 (%)</span>
                          <input
                            type="number"
                            min="0"
                            max="100"
                            v-model="config.strategies.buyHedge.capitalIncrementStart"
                          />
                        </label>
                        <label class="field">
                          <span>递增 (%)</span>
                          <input
                            type="number"
                            min="0"
                            max="100"
                            v-model="config.strategies.buyHedge.capitalIncrementStep"
                          />
                        </label>
                      </template>
                    </div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>离场模式</span>
                        <select v-model="config.strategies.buyHedge.exitMode">
                          <option value="batch">分批离场</option>
                          <option value="single">带单 / 一次性清仓</option>
                        </select>
                      </label>
                      <template v-if="config.strategies.buyHedge.exitMode === 'batch'">
                        <label class="field">
                          <span>每次卖出占持仓 (%)</span>
                          <input
                            type="number"
                            min="0"
                            max="100"
                            v-model="config.strategies.buyHedge.exitBatchPct"
                          />
                        </label>
                        <label class="field">
                          <span>策略</span>
                          <select v-model="config.strategies.buyHedge.exitBatchStrategy">
                            <option value="per_batch">按照持仓批次</option>
                            <option value="per_step">每上涨一个步长</option>
                            <option value="ratio">按比例（固定百分比）</option>
                          </select>
                        </label>
                        <label class="field">
                          <span>步长触发时卖出 (%)</span>
                          <input
                            type="number"
                            min="0"
                            max="100"
                            v-model="config.strategies.buyHedge.exitBatchStepPct"
                          />
                        </label>
                      </template>
                      <template v-else>
                        <label class="field">
                          <span>带单类型</span>
                          <select v-model="config.strategies.buyHedge.exitSingleType">
                            <option value="market">一次性市价清仓</option>
                            <option value="limit">一次性限价清仓</option>
                          </select>
                        </label>
                      </template>
                    </div>
                  </div>
                  <div class="buyhedge-section">
                    <div class="buyhedge-section__title">限制 & 初始底仓</div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>限买价格</span>
                        <input
                          type="number"
                          min="0"
                          v-model="config.strategies.buyHedge.limits.limitBuyPrice"
                        />
                      </label>
                      <label class="field">
                        <span>限平价格</span>
                        <input
                          type="number"
                          min="0"
                          v-model="config.strategies.buyHedge.limits.limitSellPrice"
                        />
                      </label>
                      <label class="field">
                        <span>最低价格</span>
                        <input
                          type="number"
                          min="0"
                          v-model="config.strategies.buyHedge.limits.minPrice"
                        />
                      </label>
                      <label class="checkbox-row">
                        <input type="checkbox" v-model="config.strategies.buyHedge.limits.stopAddingAtMin" />
                        <span>达到最低价时停止加仓 / 强制退出</span>
                      </label>
                    </div>
                    <div class="sub-grid">
                      <label class="field">
                        <span>初始底仓（手）</span>
                        <input type="number" min="0" v-model="config.strategies.buyHedge.baseInitialHands" />
                      </label>
                      <label class="field">
                        <span>底仓参考价</span>
                        <input
                          type="number"
                          min="0"
                          v-model="config.strategies.buyHedge.baseReferencePrice"
                        />
                      </label>
                      <label class="field">
                        <span>参考价来源</span>
                        <select v-model="config.strategies.buyHedge.baseReferenceSource">
                          <option value="first">首次成交价</option>
                          <option value="last">当前基准价</option>
                          <option value="custom">自定义</option>
                        </select>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
              <div class="module-divider" role="presentation"></div>
              <template v-for="(module, idx) in ['fixed','tpsl','dca','grid']" :key="module">
                <div class="strategy-card">
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
                <div v-if="idx < 3" class="module-divider" role="presentation"></div>
              </template>
            </div>
          </div>
          <div class="drawer-footer">
            <button type="button" class="secondary" @click="closeStrategyDrawer">取消</button>
            <button type="button" class="secondary" @click="closeStrategyDrawer">完成设置</button>
            <button type="button" class="primary" @click="runBacktest" :disabled="props.busy">保存并回测</button>
          </div>
        </div>
      </div>
    </transition>

    <button class="primary run-button" type="button" @click="runBacktest" :disabled="props.busy || !dataReady">
      {{ props.busy ? '回测执行中…' : '开始回测' }}
    </button>
  </div>

  <FormulaBuilder
    :show="builderVisible"
    :initialFormula="config.formula"
    :csvPath="config.csv_path"
    @close="builderVisible = false"
    @apply="handleApplyFormula"
  />
</template>
