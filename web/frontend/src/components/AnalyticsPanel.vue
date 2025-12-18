<script setup>
import { ref, watch, nextTick, onBeforeUnmount, onMounted, computed } from 'vue';
import axios from 'axios';
import * as echarts from 'echarts';
import SignalChart from './SignalChart.vue';
import { resolveEchartTheme } from '../lib/echartTheme';

const props = defineProps({
  results: { type: Array, default: () => [] },
  hasData: { type: Boolean, default: false },
  initialCapital: { type: Number, default: 100000 },
  multiFreqs: { type: String, default: 'D,W,M' },
  theme: { type: String, default: 'dark' },
  isMobile: { type: Boolean, default: false },
  enabledStrategies: { type: Array, default: () => [] },
});

const emit = defineEmits(['selectStrategy']);

const baseCategories = [
  { key: 'results', title: '回测结果', desc: '策略收益与回撤概览' },
  { key: 'scores', title: '指标评分', desc: '多维指标综合得分' },
  { key: 'plan', title: '仓位计划', desc: '加仓/再平衡建议' },
  { key: 'stop', title: '止盈止损', desc: 'ATR 建议价位' },
  { key: 'risk', title: '风控模板', desc: '多场景止盈止损' },
  { key: 'heatmap', title: '收益热力图', desc: '持有周期 VS 收益' },
  { key: 'multi', title: '多周期信号', desc: '不同周期买卖提示' },
  { key: 'report', title: '专业回测报告', desc: '高级绩效指标' },
  { key: 'brief', title: '复盘摘要', desc: '自动生成复盘语句' },
  { key: 'dynamic', title: '资金管理', desc: '投入/对冲轨迹' },
  { key: 'buyhedge', title: '买入对冲', desc: '逢跌加仓表现' },
];

const riskProfiles = [
  {
    key: 'conservative',
    label: '保守 · 快速止损',
    description: '止损近、止盈稍远，适合锁定回撤',
    stopMultiplier: 1.1,
    rewardMultiplier: 1.4,
  },
  {
    key: 'balanced',
    label: '平衡 · 标准策略',
    description: '止损/止盈配比平衡，适合常规持股',
    stopMultiplier: 1.6,
    rewardMultiplier: 2,
  },
  {
    key: 'aggressive',
    label: '进攻 · 跟踪止盈',
    description: '止损稍远、止盈更远，适合趋势向上加仓',
    stopMultiplier: 2.2,
    rewardMultiplier: 3,
  },
];

const riskToleranceOptions = [
  { key: 'conservative', label: '低风险' },
  { key: 'balanced', label: '中性' },
  { key: 'aggressive', label: '高风险' },
];
const riskToleranceFactors = {
  conservative: 0.9,
  balanced: 1,
  aggressive: 1.1,
};
const simulatorConfig = ref({
  targetReturn: 10,
  riskTolerance: 'balanced',
});
const lotSize = 100;
const baseCapital = computed(() => Math.max(0, Number(props.initialCapital || 0)));
const dynamicAvailable = computed(() =>
  props.results.some(
    (entry) =>
      entry.name === 'dynamic_capital' || Boolean(entry.result?.dynamicSummary) || Boolean(entry.result?.equityCurveWithDynamicFund),
  ),
);
const buyHedgeAvailable = computed(() =>
  props.results.some(
    (entry) => entry.name === 'buy_hedge' || Boolean(entry.result?.buyHedgeSummary) || Boolean(entry.result?.buyHedgeEvents?.length),
  ),
);
const visibleCategories = computed(() =>
  baseCategories.filter((item) => {
    if (item.key === 'dynamic') return dynamicAvailable.value;
    if (item.key === 'buyhedge') return buyHedgeAvailable.value;
    return true;
  }),
);

const moduleDescriptions = {
  fixed: {
    title: '固定周期持有',
    detail: '按照配置周期（如 5,10,20）均匀触发持仓，展示不同周期选股的收益对比与周期贡献。',
  },
  tpsl: {
    title: '止盈 / 止损',
    detail: '展示止盈/止损阈值如何影响止盈几率与最大回撤，助你校准盈亏比。',
  },
  dca: {
    title: '定投模式',
    detail: '突出逐步投入与目标收益关系，同时显示定投仓位的渐进变化。',
  },
  grid: {
    title: '网格策略',
    detail: '呈现网格间距、单网资金与加仓频率，让你评估网格资金占用情况。',
  },
};

const activeTab = ref('results');
const loading = ref(false);
const tabContentRef = ref(null);
const selectedResult = ref('');
const detailEntry = ref(null);
const detailTrades = computed(() => detailEntry.value?.result?.trades || []);
const scores = ref([]);
const positionPlan = ref([]);
const planMessage = ref('');
const stopSuggestion = ref(null);
const simulatorPlans = computed(() => {
  const stop = stopSuggestion.value;
  if (!stop) return [];
  const price = Number(stop.last_price);
  const atr = Number(stop.atr);
  if (!Number.isFinite(price) || !Number.isFinite(atr)) return [];
  const capital = baseCapital.value;
  if (!capital) return [];
  const targetReturnPct = Math.max(0, Number(simulatorConfig.value.targetReturn) || 0);
  if (!targetReturnPct) return [];
  const toleranceFactor = riskToleranceFactors[simulatorConfig.value.riskTolerance] || 1;
  const shares = price > 0 ? capital / price : 0;
  const targetProfit = (capital * targetReturnPct) / 100;
  const shareStepRaw = Math.max(shares, 1) * 0.25;
  const shareStep = Math.max(lotSize, Math.round(shareStepRaw / lotSize) * lotSize || lotSize);
  return riskProfiles.map((profile) => {
    const stopDistance = profile.stopMultiplier * atr * toleranceFactor;
    const takeDistance = profile.rewardMultiplier * atr * toleranceFactor;
    const stopPrice = Math.max(price - stopDistance, 0);
    const takePrice = price + takeDistance;
    const profitFromPosition = shares * takeDistance;
    const unmetProfit = Math.max(0, targetProfit - profitFromPosition);
    const extraSharesNeededBase = takeDistance > 0 ? Math.max(0, Math.ceil(unmetProfit / takeDistance)) : 0;
    const extraSharesNeeded = Math.ceil(extraSharesNeededBase / lotSize) * lotSize;
    const addSteps = extraSharesNeeded > 0 ? Math.ceil(extraSharesNeeded / shareStep) : 0;
    const extraCapitalNeeded = extraSharesNeeded * price;
    const riskLabel =
      unmetProfit <= 0
        ? '当前仓位即可实现'
        : `需补仓 ${addSteps} 步`;
    return {
      ...profile,
      stopPrice,
      takePrice,
      stopDistance,
      takeDistance,
      profitFromPosition,
      unmetProfit,
      extraCapitalNeeded,
      addSteps,
      shareStep,
      riskLabel,
    };
  });
});

const heatmapData = ref(null);
const heatmapLookup = ref({});
const heatmapMessage = ref('');
const heatmapMaxAbs = ref(0);
const dailyBrief = ref('');
const multiSignals = ref([]);
const multiMessage = ref('');
const scoreChartRef = ref(null);
const dynamicEquityChartRef = ref(null);
const dynamicInvestmentChartRef = ref(null);
const buyHedgePriceChartRef = ref(null);
const buyHedgeImpactChartRef = ref(null);
const reportData = ref(null);
const reportLoading = ref(false);
const reportError = ref('');
const reportChartRef = ref(null);
let scoreChartInstance = null;
let dynamicEquityChartInstance = null;
let dynamicInvestmentChartInstance = null;
let reportChartInstance = null;
let buyHedgePriceChartInstance = null;
let buyHedgeImpactChartInstance = null;
const selectedResultIndex = ref(0);
const reportDirty = ref(true);

const resolveCssVar = (name, fallback) => {
  if (typeof window === 'undefined') return fallback;
  const styles = getComputedStyle(document.documentElement);
  const value = styles.getPropertyValue(name);
  return value ? value.trim() : fallback;
};

const resolveCardBackground = () => resolveCssVar('--card-bg', '#1e293b');
const resolveBorderColor = () => resolveCssVar('--border', '#334155');
const currentThemeName = computed(() => resolveEchartTheme(props.theme));

const normalizeFreqToken = (value) => {
  if (!value) return '';
  const trimmed = value.trim();
  if (!trimmed) return '';
  const compact = trimmed.replace(/\s+/g, '');
  if (!compact) return '';
  return /^\d/.test(compact) ? compact.toLowerCase() : compact.toUpperCase();
};

const buildMultiRequestPayload = () => {
  const raw = props.multiFreqs || 'D,W,M';
  const tokens = raw.split(',').map((item) => item.trim()).filter(Boolean);
  const freqSet = new Set();
  const freqLabels = {};
  const pairs = [];
  tokens.forEach((token) => {
    const cleaned = token.replace(/\s+/g, '');
    if (!cleaned) return;
    const [expr, label] = cleaned.split('@');
    const parts = expr.split(/(?:->|→|＞|>)/i).map((part) => normalizeFreqToken(part));
    if (parts.length === 2 && parts[0] && parts[1]) {
      freqSet.add(parts[0]);
      freqSet.add(parts[1]);
      const pairItem = { trend: parts[0], entry: parts[1] };
      if (label) pairItem.label = label.trim();
      pairs.push(pairItem);
    } else {
      const freq = normalizeFreqToken(expr);
      if (freq) {
        freqSet.add(freq);
        if (label) freqLabels[freq] = label.trim();
      }
    }
  });
  if (!freqSet.size) ['D', 'W', 'M'].forEach((f) => freqSet.add(f));
  return { freqs: Array.from(freqSet), pairs, labels: freqLabels };
};

const selectResult = (entry) => {
  selectedResult.value = entry.name;
  const idx = props.results.findIndex((item) => item.name === entry.name);
  if (idx >= 0) {
    selectedResultIndex.value = idx;
    reportData.value = null;
    reportError.value = '';
    if (activeTab.value === 'report' && props.hasData) {
      fetchReport();
    }
  }
  emit('selectStrategy', entry);
};

const selectBuyHedgeEntryIfAvailable = () => {
  if (buyHedgeSummary.value) return;
  const entry = props.results.find((item) => item.name === 'buy_hedge');
  if (entry && currentEntry.value?.name !== entry.name) {
    selectResult(entry);
  }
};

const currentEntry = computed(() => props.results.find((entry) => entry.name === selectedResult.value) || null);
const baseStaticEntry = computed(() => {
  const nonDynamic = props.results.find((entry) => entry.name !== 'dynamic_capital');
  return nonDynamic || props.results[0] || null;
});
const dynamicEntry = computed(() => {
  const explicit = props.results.find((entry) => entry.name === 'dynamic_capital');
  if (explicit) return explicit;
  if (currentEntry.value?.result?.dynamicSummary) return currentEntry.value;
  return null;
});
const investmentSeriesMain = computed(
  () =>
    dynamicEntry.value?.result?.investmentCurveMain ||
    dynamicEntry.value?.result?.investmentAmount ||
    currentEntry.value?.result?.investmentCurveMain ||
    currentEntry.value?.result?.investmentAmount ||
    []
);
const investmentSeriesHedge = computed(
  () =>
    dynamicEntry.value?.result?.investmentCurveHedge ||
    dynamicEntry.value?.result?.hedgeInvestmentAmount ||
    currentEntry.value?.result?.investmentCurveHedge ||
    currentEntry.value?.result?.hedgeInvestmentAmount ||
    []
);
const dynamicDetails = computed(() => dynamicEntry.value?.result?.positionDetail || []);
const dynamicTrades = computed(() => dynamicEntry.value?.result?.trades || []);
const dynamicSummary = computed(() => dynamicEntry.value?.result?.dynamicSummary || null);
const dynamicTradeSummary = computed(() => {
  const trades = dynamicTrades.value || [];
  if (!trades.length) return null;
  let totalOpen = 0;
  let totalClose = 0;
  let totalProfit = 0;
  trades.forEach((trade) => {
    const qty = Number(trade.adjusted_quantity ?? 0);
    const openValue = Number(trade.entry_price ?? 0) * qty;
    const closeValue = Number(trade.exit_price ?? 0) * qty;
    totalOpen += isFinite(openValue) ? openValue : 0;
    totalClose += isFinite(closeValue) ? closeValue : 0;
    const pnl = Number(trade.pnl_with_dynamic_fund ?? 0);
    totalProfit += Number.isFinite(pnl) ? pnl : 0;
  });
  return {
    totalOpen,
    totalClose,
    totalProfit,
  };
});
const buyHedgeSummary = computed(() => currentEntry.value?.result?.buyHedgeSummary || null);
const buyHedgeTrades = computed(() => currentEntry.value?.result?.buyHedgeTrades || []);
const buyHedgeEvents = computed(() => currentEntry.value?.result?.buyHedgeEvents || []);
const equityCurveStatic = computed(() => {
  const explicitBaseline = dynamicEntry.value?.result?.equityCurveOriginal;
  if (Array.isArray(explicitBaseline) && explicitBaseline.length) return explicitBaseline;
  return baseStaticEntry.value?.result?.equity_curve || [];
});
const equityCurveDynamic = computed(() => dynamicEntry.value?.result?.equityCurveWithDynamicFund || []);
const dynamicForceStop = computed(
  () => dynamicEntry.value?.result?.forceStopByDrawdown ?? dynamicEntry.value?.result?.forceStop ?? false
);
const latestInvestment = computed(() =>
  investmentSeriesMain.value.length ? investmentSeriesMain.value[investmentSeriesMain.value.length - 1][1] : 0
);
const latestHedge = computed(() =>
  investmentSeriesHedge.value.length ? investmentSeriesHedge.value[investmentSeriesHedge.value.length - 1][1] : 0
);
const isDynamicEntry = computed(() => Boolean(dynamicSummary.value));
const dynamicGateMessage = computed(() => {
  if (!props.results.length) return '请先运行并选择回测结果';
  if (!dynamicSummary.value) return '当前策略未启用动态资金管理';
  return '';
});


const currentFloatingPnl = computed(() => {
  const val = dynamicSummary.value?.currentFloatingPnl;
  if (val === null || val === undefined) return 0;
  const num = Number(val);
  return Number.isFinite(num) ? num : 0;
});

const lastDynamicTrade = computed(() => {
  const trades = dynamicTrades.value;
  if (!trades || !trades.length) return null;
  return trades[trades.length - 1];
});

const lastClosedPnl = computed(() => {
  const last = lastDynamicTrade.value;
  if (!last) return null;
  return Number(last.pnl_with_dynamic_fund ?? last.pnl ?? null);
});

const lastAddStatus = computed(() => {
  const pnl = lastClosedPnl.value;
  if (pnl == null || Number.isNaN(pnl)) return '-';
  if (pnl > 0) return '盈利';
  if (pnl < 0) return '亏损';
  return '持平';
});

const renderBuyHedgePriceChart = () => {
  const formatPrice = (val) => {
    const num = Number(val);
    if (!Number.isFinite(num)) return '--';
    return num.toFixed(2);
  };
  const dom = buyHedgePriceChartRef.value;
  if (!dom || !buyHedgeEvents.value.length) {
    buyHedgePriceChartInstance?.dispose();
    buyHedgePriceChartInstance = null;
    return;
  }
  if (buyHedgePriceChartInstance && buyHedgePriceChartInstance.getDom() !== dom) {
    buyHedgePriceChartInstance.dispose();
    buyHedgePriceChartInstance = null;
  }
  if (!buyHedgePriceChartInstance) {
    buyHedgePriceChartInstance = echarts.init(dom, currentThemeName.value);
  }
  const sortedEvents = [...buyHedgeEvents.value].sort((a, b) => String(a.date || '').localeCompare(String(b.date || '')));
  const dates = sortedEvents.map((evt) => evt.date);
  const priceLine = sortedEvents.map((evt) => (evt.price ?? null));
  const totalShares = sortedEvents.map((evt) =>
    evt.total_shares != null ? Math.round(evt.total_shares) : null
  );
  let lastCost = null;
  const costLine = sortedEvents.map((evt, idx) => {
    const raw = evt.avg_cost ?? null;
    if (raw != null) {
      lastCost = raw;
      return raw;
    }
    const qty = totalShares[idx];
    if (qty != null && qty > 0 && lastCost != null) return lastCost;
    return null;
  });
  const triggerLine = sortedEvents.map((evt) => {
    if (evt.trigger_price != null) return evt.trigger_price;
    return null;
  });
  const scatterPoints = sortedEvents
    .filter((evt) => evt.type && evt.type !== 'record')
    .map((evt) => ({
      value: [evt.date, evt.price],
      layer: evt.layer,
      type: buyHedgeEventLabel(evt.type),
    }));
  const eventColors = {
    首次买入: '#60a5fa',
    加仓: '#22c55e',
    跳过: '#f97316',
    退出: '#f87171',
    记录: '#94a3b8',
  };
  const eventMarkers = scatterPoints.map((pt) => ({
    coord: pt.value,
    itemStyle: { color: eventColors[pt.type] || '#60a5fa' },
    label: { show: false },
  }));
  const priceDiff = priceLine.map((price, idx) => {
    const cost = costLine[idx];
    if (price == null || cost == null) return null;
    return Number(((price - cost) * 100).toFixed(2));
  });
  buyHedgePriceChartInstance.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (items = []) => {
        if (!items.length) return '';
        const label = items[0].axisValue;
        const lines = [label];
        const scatterItem = scatterPoints.find((pt) => pt.value[0] === label);
        if (scatterItem) {
          lines.push(`事件：${scatterItem.type}`);
          if (scatterItem.layer != null) lines.push(`层级：${scatterItem.layer}`);
        }
        items.forEach((item) => {
          const val = Array.isArray(item.value) ? item.value[1] : item.value;
          lines.push(`${item.marker} ${item.seriesName}：${formatPrice(val)}`);
        });
        const qty = totalShares[items[0].dataIndex];
        if (qty != null) lines.push(`持仓：${qty} 手`);
        return lines.join('<br/>');
      },
    },
    legend: { data: ['触发价', '事件价格', '持仓均价', '价差(分)'], bottom: 12 },
    grid: { left: 50, right: 20, top: 52, bottom: 110 },
    xAxis: { type: 'category', boundaryGap: false, data: dates, axisLabel: { margin: 18 } },
    yAxis: [
      { type: 'value', name: '价格', scale: true },
      {
        type: 'value',
        name: '持仓（手）',
        scale: true,
        splitLine: { show: false },
        position: 'right',
        nameGap: 40,
        nameLocation: 'middle',
      },
      {
        type: 'value',
        name: '价差(分)',
        scale: true,
        splitLine: { show: false },
        position: 'right',
        nameGap: 70,
        nameLocation: 'middle',
      },
    ],
    series: [
      { name: '触发价', type: 'line', data: triggerLine, smooth: true },
      {
        name: '事件价格',
        type: 'line',
        data: priceLine,
        smooth: true,
        markPoint: { symbol: 'circle', symbolSize: 10, data: eventMarkers },
      },
      { name: '持仓均价', type: 'line', data: costLine, smooth: true },
      {
        name: '价差(分)',
        type: 'line',
        yAxisIndex: 2,
        data: priceDiff,
        smooth: true,
        showSymbol: false,
        connectNulls: true,
        lineStyle: { type: 'dashed' },
      },
    ],
  });
  buyHedgePriceChartInstance.resize();
};

const renderBuyHedgeImpactChart = () => {
  const dom = buyHedgeImpactChartRef.value;
  if (!dom || !buyHedgeTrades.value.length) {
    buyHedgeImpactChartInstance?.dispose();
    buyHedgeImpactChartInstance = null;
    return;
  }
  if (buyHedgeImpactChartInstance && buyHedgeImpactChartInstance.getDom() !== dom) {
    buyHedgeImpactChartInstance.dispose();
    buyHedgeImpactChartInstance = null;
  }
  if (!buyHedgeImpactChartInstance) {
    buyHedgeImpactChartInstance = echarts.init(dom, currentThemeName.value);
  }
  const categories = buyHedgeTrades.value.map((trade, idx) => {
    const base = trade.entry_date || `#${idx + 1}`;
    const suffix = trade.trade_id != null ? `#${trade.trade_id}` : `#${idx + 1}`;
    return `${base} ${suffix}`;
  });
  const costImpact = buyHedgeTrades.value.map((trade) => (trade.avg_cost_delta_pct ?? 0) * 100);
  const pnlImpact = buyHedgeTrades.value.map((trade) => (trade.return_pct ?? 0) * 100);
  buyHedgeImpactChartInstance.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (items = []) => {
        if (!items.length) return '';
        const label = items[0].axisValue;
        const lines = [label];
        items.forEach((item) => {
          lines.push(`${item.marker} ${item.seriesName}：${item.value.toFixed(2)}%`);
        });
        return lines.join('<br/>');
      },
    },
    legend: { data: ['摊低成本', '收益率'], bottom: 12 },
    grid: { left: 50, right: 20, top: 20, bottom: 120 },
    xAxis: { type: 'category', data: categories, axisLabel: { rotate: 35 } },
    yAxis: { type: 'value', name: '%', scale: true },
    series: [
      { name: '摊低成本', type: 'bar', data: costImpact, itemStyle: { opacity: 0.8 } },
      { name: '收益率', type: 'line', data: pnlImpact, smooth: true },
    ],
  });
  buyHedgeImpactChartInstance.resize();
};

const renderDynamicEquityChart = () => {
  const dom = dynamicEquityChartRef.value;
  if (!dom || !dynamicSummary.value) {
    dynamicEquityChartInstance?.dispose();
    dynamicEquityChartInstance = null;
    return;
  }
  if (dynamicEquityChartInstance && dynamicEquityChartInstance.getDom() !== dom) {
    dynamicEquityChartInstance.dispose();
    dynamicEquityChartInstance = null;
  }
  if (!dynamicEquityChartInstance) {
    dynamicEquityChartInstance = echarts.init(dom, currentThemeName.value);
  }
  const staticSeries = equityCurveStatic.value?.map(([ts, val]) => [ts, val]) || [];
  const dynamicSeriesArr = equityCurveDynamic.value?.map(([ts, val]) => [ts, val]) || [];
  const toMap = (series) => new Map(series.map((item) => [item[0], item[1]]));
  const staticMap = toMap(staticSeries);
  const dynamicMap = toMap(dynamicSeriesArr);
  const dates = Array.from(new Set([...staticMap.keys(), ...dynamicMap.keys()])).sort();
  const staticValues = dates.map((d) => (staticMap.has(d) ? staticMap.get(d) : null));
  const dynamicValues = dates.map((d) => (dynamicMap.has(d) ? dynamicMap.get(d) : null));
  dynamicEquityChartInstance.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['原始权益', '动态权益'], bottom: 12 },
    grid: { left: 40, right: 20, top: 20, bottom: 92 },
    xAxis: {
      type: 'category',
      data: dates,
      boundaryGap: false,
      axisLabel: { margin: 12 },
    },
    yAxis: { type: 'value', scale: true },
    series: [
      { name: '原始权益', type: 'line', smooth: true, data: staticValues },
      { name: '动态权益', type: 'line', smooth: true, data: dynamicValues },
    ],
  });
  dynamicEquityChartInstance.resize();
};

const renderDynamicInvestmentChart = () => {
  const dom = dynamicInvestmentChartRef.value;
  if (!dom || !dynamicSummary.value) {
    dynamicInvestmentChartInstance?.dispose();
    dynamicInvestmentChartInstance = null;
    return;
  }
  if (dynamicInvestmentChartInstance && dynamicInvestmentChartInstance.getDom() !== dom) {
    dynamicInvestmentChartInstance.dispose();
    dynamicInvestmentChartInstance = null;
  }
  if (!dynamicInvestmentChartInstance) {
    dynamicInvestmentChartInstance = echarts.init(dom, currentThemeName.value);
  }
  const investMain = investmentSeriesMain.value || [];
  const investHedge = investmentSeriesHedge.value || [];
  const toMap = (series) => new Map(series.map((item) => [item[0], item[1]]));
  const mainMap = toMap(investMain);
  const hedgeMap = toMap(investHedge);
  const dates = Array.from(new Set([...mainMap.keys(), ...hedgeMap.keys()])).sort();
  const formatAmount = (val) => {
    const num = Number(val);
    if (!Number.isFinite(num)) return '--';
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  };
  dynamicInvestmentChartInstance.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (items = []) => {
        if (!items.length) return '';
        const label = items[0]?.axisValue ?? '';
        const lines = [label];
        items.forEach((item) => {
          const value = Array.isArray(item.value) ? item.value[1] : item.value;
          lines.push(`${item.marker} ${item.seriesName}：${formatAmount(value)}`);
        });
        return lines.join('<br/>');
      },
    },
    legend: { data: ['主方向投入', '对冲投入'], bottom: 12 },
    grid: { left: 40, right: 20, top: 20, bottom: 90 },
    xAxis: {
      type: 'category',
      boundaryGap: true,
      data: dates,
      axisLabel: { hideOverlap: true, margin: 12 },
    },
    yAxis: { type: 'value', scale: true },
    series: [
      {
        name: '主方向投入',
        type: 'scatter',
        symbolSize: 10,
        itemStyle: { color: '#60a5fa' },
        data: dates
          .filter((d) => mainMap.has(d))
          .map((d) => ({
            value: [d, mainMap.get(d) ?? 0],
          })),
      },
      {
        name: '对冲投入',
        type: 'scatter',
        symbolSize: 10,
        itemStyle: { color: '#facc15' },
        data: dates
          .filter((d) => hedgeMap.has(d))
          .map((d) => ({
            value: [d, hedgeMap.get(d) ?? 0],
          })),
      },
    ],
  });
  dynamicInvestmentChartInstance.resize();
};

watch(
  () => props.results,
  (entries) => {
    if (entries && entries.length) {
      const dynamicEntry = entries.find((item) => item.name === 'dynamic_capital');
      const target = dynamicEntry || entries[0];
      selectedResult.value = target.name;
      selectedResultIndex.value = entries.indexOf(target);
      if (selectedResultIndex.value < 0) selectedResultIndex.value = 0;
      reportData.value = null;
      reportError.value = '';
      reportDirty.value = true;
      dailyBrief.value = '';
    }
  },
  { immediate: true }
);

const fetchScores = async () => {
  loading.value = true;
  try {
    const res = await axios.post('/analytics/scores');
    scores.value = res.data || [];
    await nextTick();
    if (scores.value.length) renderScoreChart();
  } catch (e) {
    console.error(e);
  } finally {
    loading.value = false;
  }
};

const fetchBrief = async () => {
  loading.value = true;
  try {
    const res = await axios.post('/analytics/daily_brief');
    dailyBrief.value = res.data?.text || '';
  } catch (e) {
    console.error(e);
  } finally {
    loading.value = false;
  }
};

const fetchPlan = async () => {
  loading.value = true;
  planMessage.value = '';
  try {
    const payload = { capital: props.initialCapital || 100000 };
    const res = await axios.post('/analytics/position_plan', payload);
    if (Array.isArray(res.data)) {
      positionPlan.value = res.data;
      planMessage.value = '';
    } else {
      positionPlan.value = [];
      planMessage.value = res.data?.message || '暂无计划';
    }
  } catch (e) {
    console.error(e);
    planMessage.value = '请求失败';
  } finally {
    loading.value = false;
  }
};

const fetchReport = async () => {
  reportLoading.value = true;
  reportError.value = '';
  try {
    const res = await axios.post('/analytics/performance_report', {
      strategy_index: selectedResultIndex.value || 0,
    });
    reportData.value = res.data || null;
    reportDirty.value = false;
  } catch (e) {
    console.error(e);
    reportError.value = e.response?.data?.detail || '报告获取失败';
    reportDirty.value = true;
  } finally {
    reportLoading.value = false;
    await nextTick();
    if (activeTab.value === 'report' && !reportError.value && reportData.value) {
      renderReportDrawdown();
    }
  }
};

const fetchStop = async () => {
  loading.value = true;
  try {
    const res = await axios.post('/analytics/stop_suggestion');
    stopSuggestion.value = res.data;
  } catch (e) {
    console.error(e);
  } finally {
    loading.value = false;
  }
};

const fetchHeatmap = async () => {
  loading.value = true;
  heatmapMessage.value = '';
  try {
    const res = await axios.post('/analytics/heatmap');
    if (res.data?.data) {
      heatmapData.value = res.data;
      const map = {};
      res.data.data.forEach(([x, y, val]) => {
        map[`${x}-${y}`] = val;
      });
      heatmapLookup.value = map;
      const maxVal = Math.max(
        0,
        ...res.data.data.map(([, , val]) =>
          Number.isFinite(Number(val)) ? Math.abs(Number(val)) : 0,
        ),
      );
      heatmapMaxAbs.value = Math.max(maxVal, 0.01);
      heatmapMessage.value = '';
    } else {
      heatmapData.value = null;
      heatmapLookup.value = {};
      heatmapMessage.value = res.data?.message || '数据不足';
    }
  } catch (e) {
    console.error(e);
    heatmapMessage.value = '请求失败';
  } finally {
    loading.value = false;
  }
};

const fetchMulti = async () => {
  loading.value = true;
  multiMessage.value = '';
  try {
    const payload = buildMultiRequestPayload();
    const res = await axios.post('/analytics/multi_timeframe', payload);
    const series = Array.isArray(res.data?.series) ? res.data.series : [];
    const pairSeries = Array.isArray(res.data?.pairs) ? res.data.pairs : [];
    const combined = [...series, ...pairSeries];
    multiSignals.value = combined;
    const meta = res.data?.meta || {};
    const notes = [];
    if (meta.base_interval) {
      notes.push(`当前数据最小周期约 ${meta.base_interval}`);
    }
    if (Array.isArray(meta.skipped_freqs) && meta.skipped_freqs.length) {
      notes.push(
        `以下周期因数据限制被忽略：${meta.skipped_freqs
          .map((item) => item.freq)
          .filter(Boolean)
          .join('、')}`,
      );
    }
    if (Array.isArray(meta.skipped_pairs) && meta.skipped_pairs.length) {
      notes.push(
        `以下组合未生成：${meta.skipped_pairs
          .map((item) => item.pair)
          .filter(Boolean)
          .join('、')}`,
      );
    }
    if (!combined.length) {
      if (!notes.length) {
        notes.push('暂无信号数据');
      }
      if (Array.isArray(meta.recommended_freqs) && meta.recommended_freqs.length) {
        notes.push(`推荐周期：${meta.recommended_freqs.join(' / ')}`);
      }
    }
    multiMessage.value = notes.join('；');
  } catch (e) {
    console.error(e);
    multiSignals.value = [];
    multiMessage.value = '请求失败';
  } finally {
    loading.value = false;
  }
};

const renderScoreChart = () => {
  if (!scoreChartRef.value) return;
  if (scoreChartInstance && scoreChartInstance.getDom() !== scoreChartRef.value) {
    scoreChartInstance.dispose();
    scoreChartInstance = null;
  }
  if (!scoreChartInstance) {
    scoreChartInstance = echarts.init(scoreChartRef.value, currentThemeName.value);
  }
  const dates = scores.value.map((row) => row.date);
  const values = scores.value.map((row) => row.total_score);
  scoreChartInstance.setOption({
    backgroundColor: resolveCardBackground(),
    grid: { left: 45, right: 20, top: 15, bottom: 25 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: resolveBorderColor() } } },
    series: [
      {
        type: 'line',
        data: values,
        smooth: true,
        areaStyle: { opacity: 0.1 },
        lineStyle: { color: '#60a5fa' },
      },
    ],
  });
  scoreChartInstance.resize();
};

const renderReportDrawdown = () => {
  const dom = reportChartRef.value;
  const series = reportDrawdownSeries.value || [];
  if (!dom || !series.length) {
    reportChartInstance?.dispose();
    reportChartInstance = null;
    return;
  }
  if (reportChartInstance && reportChartInstance.getDom() !== dom) {
    reportChartInstance.dispose();
    reportChartInstance = null;
  }
  if (!reportChartInstance) {
    reportChartInstance = echarts.init(dom, currentThemeName.value);
  }
  const dates = series.map((item) => item.date);
  const values = series.map((item) => item.value);
  reportChartInstance.setOption({
    grid: { left: 45, right: 10, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis', valueFormatter: (val) => `${val?.toFixed?.(2) ?? val}%` },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: '{value}%' },
      splitLine: { lineStyle: { color: resolveBorderColor() } },
    },
    series: [
      {
        type: 'line',
        data: values,
        smooth: true,
        lineStyle: { color: '#f87171' },
        areaStyle: { opacity: 0.2, color: '#f87171' },
      },
    ],
  });
  reportChartInstance.resize();
};

watch(
  () => [currentEntry.value, investmentSeriesMain.value, investmentSeriesHedge.value, equityCurveStatic.value, equityCurveDynamic.value],
  () => {
    if (activeTab.value === 'dynamic') {
      nextTick(() => {
        renderDynamicEquityChart();
        renderDynamicInvestmentChart();
      });
    }
  }
);

watch(
  () => props.multiFreqs,
  () => {
    multiSignals.value = [];
    multiMessage.value = '';
    if (activeTab.value === 'multi' && props.hasData) {
      fetchMulti();
    }
  }
);

watch(
  () => props.results,
  () => {
    multiSignals.value = [];
    multiMessage.value = '';
    if (activeTab.value === 'multi' && props.hasData) {
      fetchMulti();
    }
    reportData.value = null;
    reportError.value = '';
    reportDirty.value = true;
    if (activeTab.value === 'report' && props.hasData) {
      fetchReport();
    }
    if (activeTab.value === 'buyhedge') {
      nextTick(() => {
        selectBuyHedgeEntryIfAvailable();
        renderBuyHedgePriceChart();
        renderBuyHedgeImpactChart();
      });
    }
  }
);

watch(
  () => [buyHedgeEvents.value, props.theme],
  () => {
    if (activeTab.value === 'buyhedge') {
      nextTick(() => renderBuyHedgePriceChart());
    }
  }
);

watch(
  () => [buyHedgeTrades.value, props.theme],
  () => {
    if (activeTab.value === 'buyhedge') {
      nextTick(() => renderBuyHedgeImpactChart());
    }
  }
);

watch(
  () => selectedResultIndex.value,
  () => {
    reportData.value = null;
    reportError.value = '';
    reportDirty.value = true;
    if (activeTab.value === 'report' && props.hasData) {
      fetchReport();
    }
  }
);

const disposeCharts = () => {
  scoreChartInstance?.dispose();
  dynamicEquityChartInstance?.dispose();
  dynamicInvestmentChartInstance?.dispose();
  reportChartInstance?.dispose();
  buyHedgePriceChartInstance?.dispose();
  buyHedgeImpactChartInstance?.dispose();
  scoreChartInstance = null;
  dynamicEquityChartInstance = null;
  dynamicInvestmentChartInstance = null;
  reportChartInstance = null;
  buyHedgePriceChartInstance = null;
  buyHedgeImpactChartInstance = null;
};

watch(
  () => props.theme,
  () => {
    disposeCharts();
    nextTick(() => {
      if (activeTab.value === 'scores' && scores.value.length) {
        renderScoreChart();
      }
      if (activeTab.value === 'dynamic') {
        renderDynamicEquityChart();
        renderDynamicInvestmentChart();
      }
      if (activeTab.value === 'buyhedge') {
        renderBuyHedgePriceChart();
        renderBuyHedgeImpactChart();
      }
      if (activeTab.value === 'report' && reportDrawdownSeries.value.length) {
        renderReportDrawdown();
      }
    });
  }
);

const handleResize = () => {
  scoreChartInstance?.resize();
  dynamicEquityChartInstance?.resize();
  dynamicInvestmentChartInstance?.resize();
  reportChartInstance?.resize();
  buyHedgePriceChartInstance?.resize();
  buyHedgeImpactChartInstance?.resize();
};

const toggleTradeDetail = (entry) => {
  detailEntry.value = detailEntry.value?.name === entry.name ? null : entry;
  if (detailEntry.value && typeof window !== 'undefined') {
    nextTick(() => {
      const container = document.querySelector('.trade-detail-panel');
      if (!container) return;
      container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
};

const scrollToTabContentIfMobile = () => {
  if (!props.isMobile) return;
  if (typeof window === 'undefined') return;
  nextTick(() => {
    const anchor = tabContentRef.value;
    if (!anchor) return;
    const targetY = anchor.getBoundingClientRect().top + window.scrollY;
    const offset = Math.max(targetY - 80, 0);
    window.scrollTo({ top: offset, left: 0, behavior: 'smooth' });
  });
};

onMounted(() => {
  window.addEventListener('resize', handleResize);
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize);
  disposeCharts();
});

const switchTab = (tab) => {
  activeTab.value = tab;
  scrollToTabContentIfMobile();
  if (tab === 'buyhedge') {
    selectBuyHedgeEntryIfAvailable();
    nextTick(() => {
      renderBuyHedgePriceChart();
      renderBuyHedgeImpactChart();
    });
  }
  if (!props.hasData || tab === 'results') return;
  if (tab === 'scores') {
    if (!scores.value.length) {
      fetchScores();
    } else {
      nextTick(() => {
        renderScoreChart();
      });
    }
  }
  if (tab === 'brief' && !dailyBrief.value) fetchBrief();
  if (tab === 'plan' && !positionPlan.value.length) fetchPlan();
  if (tab === 'stop' && !stopSuggestion.value) fetchStop();
  if (tab === 'risk' && !stopSuggestion.value) fetchStop();
  if (tab === 'heatmap' && !heatmapData.value) fetchHeatmap();
  if (tab === 'multi' && !multiSignals.value.length) fetchMulti();
  if (tab === 'dynamic') {
    nextTick(() => {
      renderDynamicEquityChart();
      renderDynamicInvestmentChart();
    });
  }
  if (tab === 'report') {
    if ((reportDirty.value || !reportData.value) && !reportLoading.value) {
      fetchReport();
    } else {
      nextTick(() => renderReportDrawdown());
    }
  }
};

const heatmapValue = (xIdx, yIdx) => heatmapLookup.value[`${xIdx}-${yIdx}`] ?? 0;
const heatmapColor = (value) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return 'rgba(15, 23, 42, 0.4)';
  const base = heatmapMaxAbs.value || 0.01;
  const normalized = Math.min(Math.abs(num) / base, 1);
  if (num >= 0) {
    const saturation = 65 + normalized * 10;
    const lightness = 55 - normalized * 15;
    const alpha = 0.25 + normalized * 0.6;
    return `hsla(140, ${saturation}%, ${lightness}%, ${alpha})`;
  }
  const saturation = 70 + normalized * 10;
  const lightness = 75 - normalized * 40;
  const alpha = 0.2 + normalized * 0.7;
  return `hsla(0, ${saturation}%, ${lightness}%, ${alpha})`;
};
const formatAmount = (val) =>
  Number.isFinite(Number(val)) ? Number(val).toLocaleString('zh-CN', { maximumFractionDigits: 2 }) : '-';
const formatDateTime = (val) => {
  if (!val) return '-';
  const date = new Date(val);
  if (!Number.isFinite(date.getTime())) return String(val);
  return date.toLocaleString('zh-CN', { hour12: false });
};
const formatPercent = (val) => {
  const num = Number(val);
  if (!Number.isFinite(num)) return '-';
  const ratio = num > 1 || num < -1 ? num : num * 100;
  return `${ratio.toFixed(2)}%`;
};
const formatRiskReward = (val) =>
  Number.isFinite(Number(val)) ? Number(val).toFixed(2) : '-';
const formatATRMultiple = (val) =>
  Number.isFinite(Number(val)) ? `${Number(val).toFixed(2)}×ATR` : '-';
const formatDrawdown = (val) => {
  if (val === null || val === undefined || val === '') return '-';
  const num = Number(val);
  if (!Number.isFinite(num)) return '-';
  if (num > 0 && num <= 1) return `${(num * 100).toFixed(2)}%`;
  return formatAmount(num);
};
const formatBoolean = (val) => (val ? '是' : '否');
const displayPercent = (val) => (Number.isFinite(Number(val)) ? `${Number(val).toFixed(2)}%` : '--');
const formatFloat = (val) => (Number.isFinite(Number(val)) ? Number(val).toFixed(2) : '--');
const formatHands = (val) => {
  const num = Number(val);
  if (!Number.isFinite(num)) return '--';
  return `${Math.round(num)} 手`;
};
const formatHandsValue = (val) => {
  const num = Number(val);
  if (!Number.isFinite(num)) return '--';
  return Math.round(num);
};
const formatMaybeHands = (val) => {
  const num = Number(val);
  if (!Number.isFinite(num)) return null;
  return formatHands(num);
};
const formatSharesToHands = (shares) => {
  const num = Number(shares);
  if (!Number.isFinite(num)) return null;
  return formatHands(num / lotSize);
};
const chooseHandsLabel = (primary, fallbackShares) => {
  return formatMaybeHands(primary) ?? formatSharesToHands(fallbackShares) ?? '--';
};
const getSummaryValue = (summary, ...keys) => {
  if (!summary) return undefined;
  for (const key of keys) {
    if (summary[key] !== undefined && summary[key] !== null) {
      return summary[key];
    }
  }
  return undefined;
};
const indicatorLabelMap = {
  rsi: 'RSI',
  macd: 'MACD',
  kdj: 'KDJ',
  ma_turn: '均线拐头',
  price_pattern: '价格形态',
};
const buyHedgeStepLabel = (summary) => {
  if (!summary) return '--';
  const stepMode = getSummaryValue(summary, 'step_mode', 'stepMode') || 'fixed';
  if (stepMode === 'auto') {
    const auto = summary.step_auto || summary.stepAuto || {};
    const method = auto.method || 'atr';
    const methodNames = {
      atr: 'ATR',
      avg_range: '平均真实波幅',
      stddev: '标准差',
      ma_gap: '均线乖离',
    };
    const period =
      getSummaryValue(auto, 'atr_period', 'atrPeriod') ||
      getSummaryValue(auto, 'avg_range_length', 'avgRangeLength') ||
      getSummaryValue(auto, 'std_period', 'stdPeriod') ||
      getSummaryValue(auto, 'ma_gap_period', 'maGapPeriod');
    const multiplier =
      getSummaryValue(auto, 'atr_multiplier', 'atrMultiplier') ||
      getSummaryValue(auto, 'avg_range_multiplier', 'avgRangeMultiplier') ||
      getSummaryValue(auto, 'std_multiplier', 'stdMultiplier') ||
      getSummaryValue(auto, 'ma_gap_pct', 'maGapPct');
    const parts = [];
    if (period) parts.push(`N=${period}`);
    if (multiplier != null) parts.push(`×${multiplier}`);
    return `${methodNames[method] || method}${parts.length ? ` (${parts.join(' · ')})` : ''}`;
  }
  const stepType = getSummaryValue(summary, 'step_type', 'stepType') || 'percent';
  if (stepType === 'absolute') {
    const abs = getSummaryValue(summary, 'step_abs', 'stepAbs');
    const rounding = getSummaryValue(summary, 'step_rounding', 'stepRounding');
    return `${abs != null ? formatAmount(abs) : '--'}${rounding ? ` (${rounding})` : ''}`;
  }
  const pct = getSummaryValue(summary, 'step_pct', 'stepPct');
  return pct != null ? formatPercent(pct) : '--';
};
const buyHedgeGrowthLabel = (summary) => {
  if (!summary) return '--';
  const growth = summary.growth || {};
  const mode =
    getSummaryValue(growth, 'mode', 'growthMode', 'growth_mode') ||
    getSummaryValue(summary, 'mode') ||
    'equal';
  const modeLabels = { equal: '等长', increment: '递增', double: '加倍' };
  if (mode === 'increment') {
    const base = getSummaryValue(growth, 'increment_base', 'incrementBase') ?? getSummaryValue(summary, 'start_position', 'startPosition');
    const step = getSummaryValue(growth, 'increment_step', 'incrementStep') ?? getSummaryValue(summary, 'increment_unit', 'incrementUnit');
    const baseLabel = chooseHandsLabel(getSummaryValue(growth, 'increment_base', 'incrementBase'), getSummaryValue(summary, 'start_position', 'startPosition'));
    const stepLabel = chooseHandsLabel(getSummaryValue(growth, 'increment_step', 'incrementStep'), getSummaryValue(summary, 'increment_unit', 'incrementUnit'));
    return `${modeLabels.increment} ${baseLabel} 起 + ${stepLabel} 递增`;
  }
  if (mode === 'double') {
    const baseLabel = chooseHandsLabel(getSummaryValue(growth, 'double_base', 'doubleBase'), getSummaryValue(summary, 'start_position', 'startPosition'));
    return `${modeLabels.double} ${baseLabel} 起`;
  }
  const equalLabel = chooseHandsLabel(getSummaryValue(growth, 'equal_hands', 'equalHands'), getSummaryValue(summary, 'start_position', 'startPosition'));
  return `${modeLabels.equal} ${equalLabel}`;
};
const isZeroPercent = (val) => {
  if (val == null) return true;
  const num = Number(val);
  return !Number.isFinite(num) || Math.abs(num) < 1e-9;
};

const buyHedgePositionLabel = (summary) => {
  if (!summary) return '--';
  const position = summary.position || {};
  const mode = getSummaryValue(position, 'mode') || 'fixed';
  if (mode === 'increment') {
    return `递增 ${formatPercent(getSummaryValue(position, 'inc_start_pct', 'incStartPct'))} 起 + ${formatPercent(
      getSummaryValue(position, 'inc_step_pct', 'incStepPct')
    )} 递增`;
  }
  const fixedPct = getSummaryValue(position, 'fixed_pct', 'fixedPct');
  if (isZeroPercent(fixedPct)) {
    return '未设置仓位';
  }
  return `固定 ${formatPercent(fixedPct)}`;
};
const buyHedgeEntryLabel = (summary) => {
  if (!summary) return '--';
  const entry = summary.entry || {};
  const mode = getSummaryValue(entry, 'mode') || 'none';
  const entryFast = getSummaryValue(entry, 'ma_fast', 'maFast', 'fast');
  const entrySlow = getSummaryValue(entry, 'ma_slow', 'maSlow', 'slow');
  const viewPair = `${entryFast || '--'}/${entrySlow || '--'}`;
  const progressiveCount = getSummaryValue(entry, 'progressive_count', 'progressiveCount') ?? 0;
  if (mode === 'ma_progressive') {
    return `MA ${viewPair} 连续 ${progressiveCount} 根`;
  }
  if (mode === 'ma') {
    return `MA ${viewPair} 上穿`;
  }
  return '无 MA 限制';
};
const buyHedgeProfitBaseLabel = (key) => {
  if (key === 'last') return '最后一笔';
  if (key === 'batch') return '分批单';
  return '整体均价';
};
const buyHedgeProfitLabel = (summary) => {
  if (!summary) return '--';
  const profit = summary.profit || {};
  const mode = getSummaryValue(profit, 'mode') || 'percent';
  const reference = getSummaryValue(profit, 'reference') || 'overall';
  const targetPct = getSummaryValue(profit, 'target_pct', 'targetPct');
  const targetAbs = getSummaryValue(profit, 'target_abs', 'targetAbs');
  const value = mode === 'percent' ? formatPercent(targetPct ?? 0) : formatAmount(targetAbs ?? 0);
  return `${mode === 'percent' ? '百分比' : '绝对价差'} ${value} (${buyHedgeProfitBaseLabel(reference)})`;
};
const buyHedgeReverseLabel = (summary) => {
  if (!summary) return '--';
  const reverse = summary.reverse || {};
  if (!reverse.enabled) return '未启用';
  const indicator = indicatorLabelMap[getSummaryValue(reverse, 'indicator')] || getSummaryValue(reverse, 'indicator') || '反转指标';
  const action = getSummaryValue(reverse, 'action') === 'adjust' ? '调整止盈' : '立即离场';
  const filterMode = getSummaryValue(reverse, 'filter_mode', 'filterMode');
  const filterValue = getSummaryValue(reverse, 'filter_value', 'filterValue') ?? getSummaryValue(reverse, 'interval') ?? 0;
  const minHits = getSummaryValue(reverse, 'min_hits', 'minHits') ?? 0;
  const filterDesc =
    filterMode === 'at_least' ? `过去 ${filterValue} 根至少 ${minHits} 次` : `连续 ${filterValue} 根`;
  const profitValue =
    getSummaryValue(reverse, 'profit_type', 'profitType') === 'absolute'
      ? formatAmount(getSummaryValue(reverse, 'profit_value', 'profitValue'))
      : formatPercent(getSummaryValue(reverse, 'profit_value', 'profitValue'));
  const threshold = getSummaryValue(reverse, 'threshold');
  return `${indicator} ${action} / ${filterDesc} / 阈值 ${threshold ?? '-'} / 反转盈利 ${profitValue}`;
};
const buyHedgeHedgeLabel = (summary) => {
  if (!summary) return '--';
  const hedge = summary.hedge || {};
  const mode = getSummaryValue(hedge, 'mode') || 'full';
  const status = hedge.enabled
    ? mode === 'weak'
      ? '弱对冲（仅止损/退出）'
      : '反向仓对冲'
    : '未启用对冲';
  const repeat = getSummaryValue(summary, 'allow_repeat', 'allowRepeat') ? '清仓后可重启' : '达成后不重启';
  return `${status} / ${repeat}`;
};
const buyHedgeCapitalLabel = (summary) => {
  if (!summary) return '--';
  const capital = summary.capital || {};
  const mode = getSummaryValue(capital, 'mode') || 'unlimited';
  if (mode === 'fixed') {
    const amount = getSummaryValue(capital, 'fixed_amount', 'fixedAmount');
    const percent = getSummaryValue(capital, 'fixed_percent', 'fixedPercent');
    const amountText = amount != null ? formatAmount(amount) : '--';
    const percentText = percent != null ? formatPercent(percent) : null;
    return `固定 ${amountText}${percentText ? ` / ${percentText}` : ''}`;
  }
  if (mode === 'increment') {
    const startPct = formatPercent(getSummaryValue(capital, 'increment_start', 'incrementStart') ?? 0);
    const stepPct = formatPercent(getSummaryValue(capital, 'increment_step', 'incrementStep') ?? 0);
    return `递增 ${startPct} 起 + ${stepPct} 递增`;
  }
  return '不限';
};
const buyHedgeExitLabel = (summary) => {
  if (!summary) return '--';
  const exit = summary.exit || {};
  const mode = getSummaryValue(exit, 'mode') || 'batch';
  if (mode === 'single') {
    const type = getSummaryValue(exit, 'single_type', 'singleType');
    return `带单 ${type === 'limit' ? '限价' : '市价'} 清仓`;
  }
  const pct = formatPercent(getSummaryValue(exit, 'batch_pct', 'batchPct') ?? 0);
  const strategy = getSummaryValue(exit, 'batch_strategy', 'batchStrategy');
  const stepPct = formatPercent(getSummaryValue(exit, 'batch_step_pct', 'batchStepPct') ?? 0);
  const strategyLabel =
    strategy === 'per_step' ? '每上涨一个步长' : strategy === 'ratio' ? '固定比例' : '按持仓批次';
  return `分批 ${pct} / ${strategyLabel} / 每步卖出 ${stepPct}`;
};
const buyHedgeLimitsLabel = (summary) => {
  if (!summary) return '--';
  const limits = summary.limits || {};
  const parts = [];
  const buyPrice = getSummaryValue(limits, 'limit_buy_price', 'limitBuyPrice');
  const sellPrice = getSummaryValue(limits, 'limit_sell_price', 'limitSellPrice');
  const minPrice = getSummaryValue(limits, 'min_price', 'minPrice');
  if (buyPrice != null) parts.push(`限买 ≤ ${formatAmount(buyPrice)}`);
  if (sellPrice != null) parts.push(`限平 ≥ ${formatAmount(sellPrice)}`);
  if (minPrice != null) parts.push(`最低价 ${formatAmount(minPrice)}`);
  return parts.length ? parts.join(' / ') : '无';
};
const formatBuyHedgeTradeTag = (row) => {
  if (!row) return '--';
  const tags = [];
  const hedgeActive = Boolean(row.hedge_active);
  const mode = (row.hedge_mode || 'full').toLowerCase();
  if (hedgeActive) {
    const label = mode === 'weak' ? '弱对冲' : '反向对冲';
    tags.push(label);
  } else {
    tags.push('主方向');
  }
  if (Boolean(row.allow_repeat)) {
    tags.push('清仓后重启');
  }
  return tags.join(' / ');
};
const summaryTone = (val, invert = false) => {
  const num = Number(val);
  if (!Number.isFinite(num) || num === 0) return '';
  if (invert) return num > 0 ? 'negative' : 'positive';
  return num > 0 ? 'positive' : 'negative';
};
const displayMonthlyValue = (val) => {
  const num = Number(val);
  if (!Number.isFinite(num)) return '--';
  return `${num.toFixed(2)}%`;
};
const getMonthlyValue = (year, month) => {
  const matrix = reportData.value?.monthly_matrix || {};
  if (!matrix[year]) return undefined;
  return matrix[year][month];
};
const buildDrawdownFromPairs = (pairs) => {
  if (!Array.isArray(pairs) || !pairs.length) return [];
  const subset = pairs.slice(-600);
  let runningMax = -Infinity;
  return subset
    .map(([date, raw]) => {
      const value = Number(raw);
      if (!Number.isFinite(value)) return null;
      runningMax = Math.max(runningMax, value);
      if (!Number.isFinite(runningMax) || runningMax <= 0) {
        return { date, value: 0 };
      }
      const drawdown = (value / runningMax - 1) * 100;
      return { date, value: Number(drawdown.toFixed(2)) };
    })
    .filter(Boolean);
};
const planHeaderMap = {
  index: '序号',
  step: '步骤',
  target_price: '目标价',
  buy_shares: '买入股数',
  cost: '总成本',
  remaining_cash: '剩余现金',
  avg_cost: '平均成本',
};

const reportSummaryList = computed(() => {
  const summary = reportData.value?.summary;
  if (!summary) return [];
  return [
    { key: 'total', label: '总收益', value: displayPercent(summary.total_return_pct), tone: summaryTone(summary.total_return_pct) },
    { key: 'annual', label: '年化收益', value: displayPercent(summary.annualized_return_pct), tone: summaryTone(summary.annualized_return_pct) },
    { key: 'sharpe', label: 'Sharpe', value: formatFloat(summary.sharpe), tone: summaryTone(summary.sharpe) },
    { key: 'sortino', label: 'Sortino', value: formatFloat(summary.sortino), tone: summaryTone(summary.sortino) },
    { key: 'maxdd', label: '最大回撤', value: displayPercent(summary.max_drawdown_pct), tone: summaryTone(summary.max_drawdown_pct, true) },
    { key: 'winrate', label: '胜率', value: displayPercent(summary.win_rate_pct), tone: summaryTone(summary.win_rate_pct) },
    {
      key: 'pf',
      label: '盈亏比',
      value: formatFloat(summary.profit_factor),
      tone: Number(summary.profit_factor) >= 1 ? 'positive' : Number(summary.profit_factor) > 0 ? 'negative' : '',
    },
    { key: 'avgwin', label: '平均盈利', value: displayPercent(summary.avg_win_pct), tone: summaryTone(summary.avg_win_pct) },
    { key: 'avgloss', label: '平均亏损', value: displayPercent(summary.avg_loss_pct), tone: summaryTone(summary.avg_loss_pct) },
  ];
});

const reportAnnualRows = computed(() => reportData.value?.annual_returns || []);
const reportAnnualLookup = computed(() => {
  const rows = reportData.value?.annual_returns || [];
  const map = {};
  rows.forEach((item) => {
    if (item?.year != null) map[item.year] = item.return_pct;
  });
  return map;
});
const reportMonthlyYears = computed(() => {
  const matrix = reportData.value?.monthly_matrix || {};
  return Object.keys(matrix).sort();
});
const reportMonthlyOrder = computed(
  () => reportData.value?.monthly_order || ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
);
const reportDrawdownSeries = computed(() => {
  const backendSeries = reportData.value?.drawdown_series;
  if (Array.isArray(backendSeries) && backendSeries.length) {
    return backendSeries;
  }
  const equityPairs = currentEntry.value?.result?.equity_curve;
  if (!Array.isArray(equityPairs) || !equityPairs.length) return [];
  return buildDrawdownFromPairs(equityPairs);
});

const moduleResults = {
  fixed: (entry) => entry.name?.startsWith('fixed'),
  tpsl: (entry) => entry.name?.startsWith('tpsl'),
  dca: (entry) => entry.name?.startsWith('dca'),
  grid: (entry) => entry.name?.startsWith('grid'),
};

const activeModuleCards = computed(() => {
  if (!props.enabledStrategies?.length) return [];
  return props.enabledStrategies
    .map((key, idx) => {
      const detail = moduleDescriptions[key];
      if (!detail) return null;
      const entry = props.results.find((item) => moduleResults[key]?.(item));
      if (!entry) return null;
      return {
        id: `${key}-${idx}`,
        ...detail,
        entry,
      };
    })
    .filter(Boolean);
});

watch(
  () => reportDrawdownSeries.value,
  (series) => {
    if (activeTab.value === 'report' && Array.isArray(series)) {
      nextTick(() => renderReportDrawdown());
    }
  }
);

const buyHedgeEventLabel = (type) => {
  const map = { entry: '首次买入', add: '加仓', skip: '跳过' };
  return map[type] || '记录';
};
const isCategoryDisabled = (key) => {
  if (key === 'results') return false;
  if (!props.hasData) return true;
  return false;
};

</script>

<template>
  <div class="analytics-panel card">
    <div class="category-row">
        <button
          v-for="item in visibleCategories"
          :key="item.key"
          class="category-card"
          :class="{ active: activeTab === item.key }"
          :data-key="item.key"
          :disabled="isCategoryDisabled(item.key)"
          @click="switchTab(item.key)"
        >
        <span class="category-label">{{ item.title }}</span>
        <p>{{ item.desc }}</p>
      </button>
    </div>
    <section v-if="activeModuleCards.length" class="module-report-grid">
      <article v-for="card in activeModuleCards" :key="card.id" class="module-report-card">
        <div class="module-report-card__icon" aria-hidden="true"></div>
        <div class="module-report-card__body">
          <strong>{{ card.title }}</strong>
          <p>{{ card.detail }}</p>
          <div class="module-report-card__metrics">
            <span>
              总收益
              <strong :class="card.entry.result.total_return >= 0 ? 'positive' : 'negative'">
                {{ formatPercent(card.entry.result.total_return) }}
              </strong>
            </span>
            <span>胜率 <strong>{{ formatPercent(card.entry.result.win_rate ?? 0) }}</strong></span>
            <span>回撤 <strong>{{ formatPercent(card.entry.result.max_drawdown ?? 0) }}</strong></span>
          </div>
        </div>
      </article>
    </section>

    <div class="tab-content" ref="tabContentRef">
      <div v-if="loading" class="loading">数据加载中…</div>

    <div v-if="activeTab === 'results'" class="results-note-wrapper">
      <p class="results-note">点击右侧“笔交易”可展开逐笔明细。</p>
    </div>
    <section v-if="activeTab === 'results'" class="results-view">
        <div v-if="!results || !results.length" class="empty">请先运行一次回测</div>
        <div v-else class="result-list">
            <article
              v-for="entry in results"
              :key="entry.name"
              class="result-card"
              :class="{ selected: selectedResult === entry.name }"
              @click="selectResult(entry)"
            >
              <header>
                <h3>{{ entry.title }}</h3>
              <button
                type="button"
                class="tag tag-button"
                @click.stop.prevent="toggleTradeDetail(entry)"
              >
                {{ entry.result.trades?.length || 0 }} 笔交易
              </button>
              </header>
              <dl>
                <div><dt>总收益</dt><dd :class="entry.result.total_return >= 0 ? 'positive' : 'negative'">{{ formatPercent(entry.result.total_return) }}</dd></div>
              <div><dt>年化</dt><dd>{{ formatPercent(entry.result.annualized_return) }}</dd></div>
              <div><dt>最大回撤</dt><dd>{{ formatPercent(entry.result.max_drawdown) }}</dd></div>
              <div><dt>胜率</dt><dd>{{ formatPercent(entry.result.win_rate) }}</dd></div>
            </dl>
          </article>
        </div>
        <div v-if="detailEntry" class="trade-detail-panel">
          <div class="trade-detail-head">
            <div>
              <h4>{{ detailEntry.title }}</h4>
              <p>以下为该策略的逐笔交易（动态资金可选）</p>
            </div>
            <button type="button" class="detail-close" @click="toggleTradeDetail(detailEntry)">收起</button>
          </div>
          <div v-if="!detailTrades.length" class="empty small">暂无交易记录</div>
          <div v-else class="table-wrapper compact-table">
            <table class="modern-table">
              <thead>
                <tr>
                  <th>开仓时间</th>
                  <th>平仓时间</th>
                  <th>持仓天数</th>
                  <th>入场价</th>
                  <th>出场价</th>
                  <th>盈亏 (%)</th>
                  <th>投资金额</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="trade in detailTrades" :key="`${trade.entry_date}-${trade.exit_date}-${trade.entry_price}`">
                  <td>{{ formatDateTime(trade.entry_date) }}</td>
                  <td>{{ formatDateTime(trade.exit_date) }}</td>
                  <td>{{ trade.holding_days ?? '-' }}</td>
                  <td>{{ formatAmount(trade.entry_price ?? '-') }}</td>
                  <td>{{ formatAmount(trade.exit_price ?? '-') }}</td>
                  <td :class="(trade.return_pct ?? 0) >= 0 ? 'positive' : 'negative'">
                    {{ formatPercent(trade.return_pct ?? null) }}
                  </td>
                  <td>{{ formatAmount(trade.investment_amount ?? '-') }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'scores'">
        <div v-if="scores.length" class="scores-view">
          <div ref="scoreChartRef" class="score-chart"></div>
          <div class="table-wrapper">
            <table class="modern-table">
              <thead><tr><th>日期</th><th>综合评分</th></tr></thead>
              <tbody>
                <tr v-for="row in scores.slice(-80).reverse()" :key="row.date">
                  <td>{{ row.date }}</td>
                  <td :class="row.total_score >= 0 ? 'positive' : 'negative'">{{ row.total_score.toFixed(2) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div v-else class="empty">暂无评分数据</div>
      </section>

      <section v-else-if="activeTab === 'plan'">
        <div v-if="!positionPlan.length" class="empty">{{ planMessage || '条件不足或尚未加载' }}</div>
        <div v-else class="table-wrapper">
          <table class="modern-table">
            <thead>
              <tr>
                <th v-for="(value, key) in positionPlan[0]" :key="key">{{ planHeaderMap[key] || key }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in positionPlan" :key="idx">
                <td v-for="(value, key) in row" :key="key">{{ typeof value === 'number' ? value.toFixed(2) : value }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section v-else-if="activeTab === 'stop'">
        <div v-if="!stopSuggestion" class="empty">点击标签即可获取最新建议</div>
        <div v-else>
          <div class="stop-hint">
            <p>
              当前根据最新价（{{ formatAmount(stopSuggestion.last_price) }}）与 ATR 预估出各类价位：
              “止损/止盈距离”显示距离最新价的差值及所占百分比，风险回报 = 止盈距离 ÷ 止损距离。
              可据此快速判断盈亏比是否符合策略要求。
            </p>
          </div>
          <div class="stop-grid">
          <div class="stop-card">
            <span>最新价</span>
            <strong>{{ formatAmount(stopSuggestion.last_price) }}</strong>
          </div>
          <div class="stop-card">
            <span>ATR (14)</span>
            <strong>{{ formatAmount(stopSuggestion.atr) }}</strong>
          </div>
          <div class="stop-card">
            <span>建议止损</span>
            <strong class="negative">{{ formatAmount(stopSuggestion.suggest_stop_loss) }}</strong>
          </div>
          <div class="stop-card">
            <span>止损距离 / 占比</span>
            <strong>
              {{ formatAmount(stopSuggestion.stop_loss_distance) }}
              <em>{{ formatPercent(stopSuggestion.stop_loss_pct) }}</em>
            </strong>
            <small>{{ formatATRMultiple(stopSuggestion.stop_loss_atr) }}</small>
          </div>
          <div class="stop-card">
            <span>建议止盈</span>
            <strong class="positive">{{ formatAmount(stopSuggestion.suggest_take_profit) }}</strong>
          </div>
          <div class="stop-card">
            <span>止盈距离 / 占比</span>
            <strong>
              {{ formatAmount(stopSuggestion.take_profit_distance) }}
              <em>{{ formatPercent(stopSuggestion.take_profit_pct) }}</em>
            </strong>
            <small>{{ formatATRMultiple(stopSuggestion.take_profit_atr) }}</small>
          </div>
          <div class="stop-card">
            <span>跟踪止损</span>
            <strong>{{ formatAmount(stopSuggestion.trailing_stop) }}</strong>
          </div>
          <div class="stop-card">
            <span>风险回报</span>
            <strong>{{ formatRiskReward(stopSuggestion.risk_reward) }}</strong>
          </div>
        </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'risk'">
        <div v-if="!stopSuggestion" class="empty">等待止盈止损建议以生成风控模拟...</div>
        <div v-else>
          <div class="risk-intro">
            <p>
              当前持仓基于最新已知价格（{{ formatAmount(stopSuggestion.last_price) }}）和 ATR，
              可选择不同强度的风险承受度来生成加码/止盈策略。系统会按照初始资金和目标收益给出
              后续加码所需资金、每步份额与风险回报预估。
            </p>
            <p>
              上方“风险承受”开关用于调节整体止盈/止损距离——从保守到激进分别对应不同的 ATR 倍数。
              下方的保守/平衡/进攻卡片是在当前风险级别下的三种执行模板，方便你快速比较仓位调度、
              加码步数与潜在收益，选择最贴合交易计划的方案。
            </p>
          </div>
          <div class="simulator-panel">
            <div class="simulator-form">
              <div class="form-field">
                <label>目标收益 (%)</label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  v-model.number="simulatorConfig.targetReturn"
                />
              </div>
              <div class="form-field">
                <label>风险承受</label>
                <div class="radio-group">
                  <button
                    v-for="opt in riskToleranceOptions"
                    :key="opt.key"
                    type="button"
                    :class="{ active: simulatorConfig.riskTolerance === opt.key }"
                    @click="simulatorConfig.riskTolerance = opt.key"
                  >
                    {{ opt.label }}
                  </button>
                </div>
              </div>
            </div>
            <div v-if="simulatorPlans.length" class="simulator-results">
              <article v-for="plan in simulatorPlans" :key="plan.key" class="simulator-card">
                <header>
                  <div>
                    <strong>{{ plan.label }}</strong>
                    <p>{{ plan.description }}</p>
                  </div>
                  <span class="simulator-tag">{{ plan.riskLabel }}</span>
                </header>
                <div class="simulator-grid">
                  <div>
                    <dt>止损价格</dt>
                    <dd>{{ formatAmount(plan.stopPrice) }}</dd>
                  </div>
                  <div>
                    <dt>止盈价格</dt>
                    <dd>{{ formatAmount(plan.takePrice) }}</dd>
                  </div>
                  <div>
                    <dt>预期利润</dt>
                    <dd>{{ formatAmount(plan.profitFromPosition) }}</dd>
                  </div>
                  <div>
                    <dt>需追加资金</dt>
                    <dd>{{ plan.extraCapitalNeeded ? formatAmount(plan.extraCapitalNeeded) : '无需' }}</dd>
                  </div>
                  <div>
                    <dt>加码步数</dt>
                    <dd>{{ plan.addSteps || '当前仓位' }}</dd>
                  </div>
                  <div>
                    <dt>每步份额</dt>
                    <dd>{{ plan.shareStep }} 股（约）</dd>
                  </div>
                </div>
              </article>
            </div>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'dynamic'">
        <div v-if="dynamicGateMessage" class="empty">{{ dynamicGateMessage }}</div>
        <div v-else class="dynamic-view">
          <div class="dynamic-charts">
            <div class="dynamic-chart-card">
              <div class="chart-header">
                <h4>资金曲线对比</h4>
                <span>动态 vs 原始权益</span>
              </div>
              <div ref="dynamicEquityChartRef" class="mini-chart"></div>
            </div>
            <div class="dynamic-chart-card">
              <div class="chart-header">
                <h4>投入金额变化</h4>
                <span>主方向 / 对冲</span>
              </div>
              <div ref="dynamicInvestmentChartRef" class="mini-chart"></div>
            </div>
          </div>
          <div class="dynamic-summary">
            <div class="summary-block">
              <h4>运行快照</h4>
              <ul>
                <li><span>当前投资金额</span><strong>{{ formatAmount(latestInvestment) }}</strong></li>
                <li><span>当前对冲金额</span><strong>{{ formatAmount(latestHedge) }}</strong></li>
                <li>
                  <span>当前持仓浮动盈亏</span>
                  <strong :class="currentFloatingPnl >= 0 ? 'positive' : 'negative'">
                    {{ Number.isFinite(currentFloatingPnl) ? formatAmount(currentFloatingPnl) : '0.00' }}
                  </strong>
                </li>
                <li>
                  <span>最近平仓盈亏</span>
                  <strong
                    :class="lastClosedPnl != null ? (lastClosedPnl >= 0 ? 'positive' : 'negative') : ''"
                  >
                    {{ lastClosedPnl != null ? formatAmount(lastClosedPnl) : '-' }}
                  </strong>
                </li>
                <li>
                  <span>最近一次加仓状态</span>
                  <strong>{{ lastAddStatus }}</strong>
                </li>
                <li>
                  <span>是否强制停止</span>
                  <strong :class="dynamicForceStop ? 'negative' : 'positive'">{{ formatBoolean(dynamicForceStop) }}</strong>
                </li>
              </ul>
            </div>
            <div class="summary-block">
              <h4>主方向配置</h4>
              <ul>
                <li><span>初始投资</span><strong>{{ formatAmount(dynamicSummary.initialInvestment) }}</strong></li>
                <li><span>亏损加注</span><strong>{{ formatHands(dynamicSummary.lossStepAmount) }}</strong></li>
                <li><span>最大加注次数</span><strong>{{ dynamicSummary.maxAddSteps }}</strong></li>
                <li><span>投资上限</span><strong>{{ formatAmount(dynamicSummary.maxInvestmentLimit) }}</strong></li>
                <li><span>盈利后重置</span><strong>{{ formatBoolean(dynamicSummary.resetOnWin) }}</strong></li>
                <li><span>实际最大连亏</span><strong>{{ dynamicSummary.maxLossStreakUsed }}</strong></li>
                <li><span>实际最大投资</span><strong>{{ formatAmount(dynamicSummary.maxInvestmentUsed) }}</strong></li>
              </ul>
            </div>
            <div class="summary-block" v-if="dynamicSummary.enableHedge">
              <h4>对冲配置</h4>
              <ul>
                <li><span>启用对冲</span><strong>{{ formatBoolean(dynamicSummary.enableHedge) }}</strong></li>
                <li><span>对冲初始</span><strong>{{ formatAmount(dynamicSummary.hedgeInitialInvestment) }}</strong></li>
                <li><span>对冲加注</span><strong>{{ formatHands(dynamicSummary.hedgeLossStepAmount) }}</strong></li>
                <li><span>最大对冲加注数</span><strong>{{ dynamicSummary.hedgeMaxAddSteps }}</strong></li>
                <li><span>实际最大对冲连亏</span><strong>{{ dynamicSummary.hedgeMaxLossStreakUsed ?? '-' }}</strong></li>
                <li><span>实际最大对冲投资</span><strong>{{ formatAmount(dynamicSummary.hedgeMaxInvestmentUsed) }}</strong></li>
              </ul>
            </div>
            <div class="summary-block">
              <h4>风控</h4>
              <ul>
                <li><span>最大允许回撤</span><strong>{{ formatDrawdown(dynamicSummary.maxDrawdownLimitInput ?? dynamicSummary.maxDrawdownLimitValue) }}</strong></li>
                <li><span>触发回撤保护</span><strong>{{ formatBoolean(dynamicSummary.forceStopByDrawdown) }}</strong></li>
              </ul>
            </div>
          </div>
          <div class="dynamic-trades">
            <h4>交易明细（含动态资金指标）</h4>
            <div v-if="!dynamicTrades.length" class="empty small">暂无交易记录</div>
            <div v-else class="table-wrapper compact-table">
              <table class="modern-table">
                <thead>
                    <tr>
                      <th>开仓时间</th>
                      <th>平仓时间</th>
                      <th>入场价</th>
                      <th>出场价</th>
                      <th>投资金额</th>
                      <th>连续亏损</th>
                      <th>数量（手）</th>
                      <th>动态盈亏</th>
                      <th>对冲投资</th>
                      <th>对冲连亏</th>
                      <th>对冲数量（手）</th>
                      <th>对冲盈亏</th>
                    </tr>
                </thead>
                <tbody>
                  <tr v-for="trade in dynamicTrades" :key="trade.entry_date + trade.exit_date">
                    <td>{{ trade.entry_date }}</td>
                    <td>{{ trade.exit_date }}</td>
                    <td>{{ formatAmount(trade.entry_price ?? '-') }}</td>
                    <td>{{ formatAmount(trade.exit_price ?? '-') }}</td>
                    <td>{{ formatAmount(trade.investment_amount ?? '-') }}</td>
                    <td>{{ trade.loss_streak ?? '-' }}</td>
                    <td>{{ trade.adjusted_quantity != null ? Math.round(trade.adjusted_quantity / 100) : '-' }}</td>
                    <td :class="(trade.pnl_with_dynamic_fund ?? 0) >= 0 ? 'positive' : 'negative'">
                      {{ formatAmount(trade.pnl_with_dynamic_fund ?? '-') }}
                    </td>
                    <td>{{ formatAmount(trade.hedge_investment_amount ?? '-') }}</td>
                    <td>{{ trade.hedge_loss_streak ?? '-' }}</td>
                    <td>{{ trade.hedge_adjusted_quantity != null ? Math.round(trade.hedge_adjusted_quantity / 100) : '-' }}</td>
                    <td :class="(trade.hedge_pnl_with_dynamic_fund ?? 0) >= 0 ? 'positive' : 'negative'">
                      {{ trade.hedge_pnl_with_dynamic_fund != null ? formatAmount(trade.hedge_pnl_with_dynamic_fund) : '-' }}
                    </td>
                  </tr>
                </tbody>
              </table>
              <div v-if="dynamicTradeSummary" class="trade-summary-row">
                <div>
                  <span>开仓金额</span>
                  <strong>{{ formatAmount(dynamicTradeSummary.totalOpen) }}</strong>
                </div>
                <div>
                  <span>平仓金额</span>
                  <strong>{{ formatAmount(dynamicTradeSummary.totalClose) }}</strong>
                </div>
                <div>
                  <span>平仓盈亏</span>
                  <strong>{{ formatAmount(dynamicTradeSummary.totalProfit) }}</strong>
                </div>
              </div>
            </div>
          </div>
          <div class="dynamic-detail">
            <h4>投资金额序列（最近 120 条）</h4>
            <div v-if="!dynamicDetails.length" class="empty small">暂无投资序列记录</div>
            <div v-else class="table-wrapper compact-table">
              <table class="modern-table">
                <thead>
                  <tr>
                    <th>日期</th>
                    <th>投资金额</th>
                    <th>数量（手）</th>
                    <th>连亏次数</th>
                    <th>对冲金额</th>
                    <th>对冲数量（手）</th>
                    <th>对冲连亏</th>
                    <th>止损触发</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in dynamicDetails.slice(-120)" :key="row.date + (row.investmentAmount ?? 0)">
                    <td>{{ row.date }}</td>
                    <td>{{ formatAmount(row.investmentAmount) }}</td>
                    <td>{{ row.quantity != null ? Math.round(row.quantity / 100) : '-' }}</td>
                    <td>{{ row.lossStreak }}</td>
                    <td>{{ formatAmount(row.hedgeInvestmentAmount) }}</td>
                    <td>{{ row.hedgeQuantity != null ? Math.round(row.hedgeQuantity / 100) : '-' }}</td>
                    <td>{{ row.hedgeLossStreak }}</td>
                    <td>{{ formatBoolean(row.forceStop) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
      <section v-else-if="activeTab === 'buyhedge'">
        <div v-if="!buyHedgeSummary" class="empty">当前策略未启用买入对冲模块</div>
        <div v-else class="buyhedge-view">
          <div class="buyhedge-charts">
            <div class="buyhedge-chart-card">
              <div class="panel-header">
                <h4>买入轨迹</h4>
                <span>事件价格 / 持仓均价 / 持仓数量</span>
              </div>
              <div ref="buyHedgePriceChartRef" class="buyhedge-chart"></div>
            </div>
            <div class="buyhedge-chart-card">
              <div class="panel-header">
                <h4>加仓效果</h4>
                <span>摊低成本 vs 收益率</span>
              </div>
              <div ref="buyHedgeImpactChartRef" class="buyhedge-chart"></div>
            </div>
          </div>
          <div class="buyhedge-summary-grid">
            <div class="summary-card">
              <h4>参数配置</h4>
              <ul>
                <li><span>步长</span><strong>{{ buyHedgeStepLabel(buyHedgeSummary) }}</strong></li>
                <li><span>增长模式</span><strong>{{ buyHedgeGrowthLabel(buyHedgeSummary) }}</strong></li>
                <li><span>仓位设置</span><strong>{{ buyHedgePositionLabel(buyHedgeSummary) }}</strong></li>
                <li><span>开仓条件</span><strong>{{ buyHedgeEntryLabel(buyHedgeSummary) }}</strong></li>
                <li><span>止盈参考</span><strong>{{ buyHedgeProfitLabel(buyHedgeSummary) }}</strong></li>
                <li><span>反转指标</span><strong>{{ buyHedgeReverseLabel(buyHedgeSummary) }}</strong></li>
                <li><span>对冲 / 重启</span><strong>{{ buyHedgeHedgeLabel(buyHedgeSummary) }}</strong></li>
                <li><span>资金 & 限制</span><strong>{{ buyHedgeCapitalLabel(buyHedgeSummary) }}</strong></li>
                <li><span>离场策略</span><strong>{{ buyHedgeExitLabel(buyHedgeSummary) }}</strong></li>
                <li><span>限价 / 最低价</span><strong>{{ buyHedgeLimitsLabel(buyHedgeSummary) }}</strong></li>
              </ul>
            </div>
            <div class="summary-card">
              <h4>执行统计</h4>
              <ul>
                <li><span>交易笔数</span><strong>{{ buyHedgeSummary.trade_count || 0 }}</strong></li>
                <li><span>加仓次数</span><strong>{{ buyHedgeSummary.total_adds || 0 }}</strong></li>
                <li>
                  <span>单笔平均加仓</span><strong>{{ (buyHedgeSummary.avg_adds_per_trade || 0).toFixed(2) }}</strong>
                </li>
                <li><span>最大层数</span><strong>{{ buyHedgeSummary.max_layers || 0 }}</strong></li>
                <li><span>平均摊低成本</span><strong>{{ formatPercent(buyHedgeSummary.avg_cost_reduction_pct || 0) }}</strong></li>
                <li><span>最大资金占用</span><strong>{{ formatAmount(buyHedgeSummary.max_capital_used || 0) }}</strong></li>
                <li>
                  <span>跳过记录</span>
                  <strong>
                    规则 {{ buyHedgeSummary.skipped_by_rule || 0 }} /
                    资金 {{ buyHedgeSummary.skipped_by_limit || 0 }} /
                    现金 {{ buyHedgeSummary.skipped_by_cash || 0 }}
                  </strong>
                </li>
              </ul>
            </div>
          </div>
          <div class="buyhedge-panels">
            <div class="table-wrapper compact-table">
              <div class="panel-header">
                <h4>交易层级统计</h4>
                <span>含每笔加仓详情</span>
              </div>
              <div v-if="!buyHedgeTrades.length" class="empty small">暂无交易记录</div>
              <div v-else>
                <table class="modern-table">
                  <thead>
                    <tr>
                      <th>开仓时间</th>
                      <th>平仓时间</th>
                      <th>开仓价</th>
                      <th>平仓价</th>
                      <th>数量（手）</th>
                      <th>加仓次数</th>
                      <th>平均成本</th>
                      <th>成交金额</th>
                      <th>平均摊低</th>
                      <th>盈亏</th>
                      <th>收益率</th>
                      <th>标签</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in buyHedgeTrades" :key="row.trade_id || row.entry_date">
                      <td>{{ row.entry_date }}</td>
                      <td>{{ row.exit_date }}</td>
                      <td>{{ formatAmount(row.entry_price) }}</td>
                      <td>{{ formatAmount(row.exit_price) }}</td>
                      <td>{{ formatHandsValue(row.total_shares) }}</td>
                      <td>{{ row.adds ?? 0 }}</td>
                      <td>{{ formatAmount(row.avg_cost) }}</td>
                      <td>{{ formatAmount(row.capital_used) }}</td>
                      <td>{{ formatPercent(row.avg_cost_delta_pct ?? 0) }}</td>
                      <td :class="(row.pnl ?? 0) >= 0 ? 'positive' : 'negative'">{{ formatAmount(row.pnl) }}</td>
                      <td :class="(row.return_pct ?? 0) >= 0 ? 'positive' : 'negative'">
                        {{ formatPercent(row.return_pct ?? 0) }}
                      </td>
                      <td>{{ formatBuyHedgeTradeTag(row) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="table-wrapper compact-table">
              <div class="panel-header">
                <h4>买入事件</h4>
                <span>含首次买入 / 加仓 / 跳过</span>
              </div>
              <div v-if="!buyHedgeEvents.length" class="empty small">尚无事件记录</div>
              <div v-else>
                <table class="modern-table">
                  <thead>
                    <tr>
                      <th>时间</th>
                      <th>类型</th>
                      <th>价格</th>
                      <th>数量（手）</th>
                      <th>累计持仓（手）</th>
                      <th>平均成本</th>
                      <th>触发价</th>
                      <th>层级</th>
                      <th>备注</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in buyHedgeEvents.slice(-200)" :key="row.date + (row.type || '') + (row.trigger_price || 0)">
                      <td>{{ row.date }}</td>
                      <td>{{ buyHedgeEventLabel(row.type) }}</td>
                      <td>{{ formatAmount(row.price) }}</td>
                      <td>{{ formatHandsValue(row.shares) }}</td>
                      <td>{{ formatHandsValue(row.total_shares) }}</td>
                      <td>{{ formatAmount(row.avg_cost) }}</td>
                      <td>{{ row.trigger_price != null ? formatAmount(row.trigger_price) : '-' }}</td>
                      <td>{{ row.layer != null ? row.layer : '-' }}</td>
                      <td>{{ row.note || '-' }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'heatmap'">
        <div v-if="!heatmapData || !heatmapData.data" class="empty">{{ heatmapMessage || '暂无可用数据' }}</div>
        <div v-else class="heatmap-table">
          <p class="heatmap-tip">
            每行代表一次入场，列是持有天数。颜色越深，代表收益越大（绿色）或亏损越严重（红色），颜色越浅说明幅度较小。忽略具体数值，通过色彩即可快速判断组合表现。
          </p>
          <div class="heatmap-row header">
            <div class="y-label">日期 \\ 持有</div>
            <div class="x-list">
              <span v-for="(label, xIdx) in heatmapData.x_labels" :key="label">{{ label }}</span>
            </div>
          </div>
          <div v-for="(label, yIdx) in heatmapData.y_labels" :key="label" class="heatmap-row">
            <div class="y-label">{{ label }}</div>
            <div class="x-list">
              <span
                v-for="(xLabel, xIdx) in heatmapData.x_labels"
                :key="xLabel + yIdx"
                :style="{ backgroundColor: heatmapColor(heatmapValue(xIdx, yIdx)) }"
                class="heatmap-cell"
              ></span>
            </div>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'multi'">
        <p class="multi-hint">
          示例：<code>D,W,M</code> 表示单周期；<code>15m&gt;5m</code> 表示“15m 判断趋势 &amp; 5m 精确买点”，可用
          <code>@备注</code> 自定义标题。
        </p>
        <div v-if="!multiSignals.length" class="empty">{{ multiMessage || '点击标签以加载多周期信号' }}</div>
        <div v-else>
          <div class="multi-charts">
            <SignalChart v-for="item in multiSignals" :key="item.freq" :dataset="item" :height="220" :theme="theme" />
          </div>
          <div v-if="multiMessage" class="empty small">{{ multiMessage }}</div>
        </div>
      </section>

      <section v-else-if="activeTab === 'report'" class="report-view">
        <div v-if="reportLoading" class="loading small">专业报告生成中…</div>
        <div v-else-if="reportError" class="empty">{{ reportError }}</div>
        <div v-else-if="!reportData" class="empty">点击标签以生成报告</div>
        <div v-else class="report-body">
          <header class="report-header">
            <div>
              <h3>{{ reportData.strategy?.title || reportData.strategy?.name || '专业回测报告' }}</h3>
              <p>自动汇总关键绩效指标、年度收益及回撤轨迹，帮助快速评估策略质量。</p>
            </div>
            <button class="ghost-btn" @click="fetchReport">重新生成</button>
          </header>
          <div class="report-summary-grid">
            <div
              v-for="item in reportSummaryList"
              :key="item.key"
              class="report-kpi"
              :class="item.tone"
            >
              <span class="label">{{ item.label }}</span>
              <strong class="value">{{ item.value }}</strong>
            </div>
          </div>
          <div class="report-panels">
            <div class="report-panel">
              <div class="panel-header">
                <h4>年度收益</h4>
                <span>按自然年统计</span>
              </div>
              <div v-if="!reportAnnualRows.length" class="empty small">年度收益数据不足</div>
              <div v-else class="table-wrapper compact-table">
                <table class="modern-table report-table">
                  <thead>
                    <tr><th>年份</th><th>收益率</th></tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in reportAnnualRows" :key="row.year">
                      <td>{{ row.year }}</td>
                      <td :class="row.return_pct >= 0 ? 'positive' : 'negative'">{{ displayMonthlyValue(row.return_pct) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="report-panel">
              <div class="panel-header">
                <h4>回撤轨迹</h4>
                <span>最近 600 个点</span>
              </div>
              <div
                v-if="!reportDrawdownSeries.length"
                class="empty small"
              >
                暂无回撤数据
              </div>
              <div v-else ref="reportChartRef" class="report-chart"></div>
            </div>
          </div>
          <div class="report-panel full-width">
            <div class="panel-header">
              <h4>月度收益矩阵</h4>
              <span>绿色代表盈利，红色代表亏损</span>
            </div>
            <div v-if="!reportMonthlyYears.length" class="empty small">月度收益数据不足</div>
            <div v-else class="monthly-matrix">
              <table>
                <thead>
                  <tr>
                    <th>年份/月</th>
                    <th v-for="month in reportMonthlyOrder" :key="month">{{ month }}</th>
                    <th>全年</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="year in reportMonthlyYears" :key="year">
                    <td class="year-label">{{ year }}</td>
                    <td
                      v-for="month in reportMonthlyOrder"
                      :key="year + month"
                      :class="[
                        'month-cell',
                        getMonthlyValue(year, month) > 0 ? 'positive' : getMonthlyValue(year, month) < 0 ? 'negative' : '',
                      ]"
                    >
                      {{ getMonthlyValue(year, month) != null ? displayMonthlyValue(getMonthlyValue(year, month)) : '--' }}
                    </td>
                    <td class="year-total" :class="reportAnnualLookup[year] >= 0 ? 'positive' : 'negative'">
                      {{ reportAnnualLookup[year] != null ? displayMonthlyValue(reportAnnualLookup[year]) : '--' }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'brief'" class="brief-view">
        <div class="brief-card" v-if="dailyBrief">
          <div class="brief-line" v-for="(line, idx) in dailyBrief.split('\n')" :key="idx">
            <span v-if="idx === 0" class="tag">复盘摘要</span>
            <p>{{ idx === 0 ? line.replace('【复盘摘要】', '') : line }}</p>
          </div>
        </div>
        <div v-else class="empty">点击加载复盘摘要</div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.dynamic-view {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.dynamic-charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}
.dynamic-chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  height: 240px;
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}
.chart-header h4 {
  margin: 0;
  font-size: 14px;
}
.chart-header span {
  font-size: 12px;
  color: var(--text-secondary);
}
.mini-chart {
  width: 100%;
  height: 220px;
  flex: 1;
}
.dynamic-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}
.summary-block {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
}
.summary-block h4 {
  margin: 0 0 8px;
  font-size: 14px;
}
.summary-block ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.summary-block li {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}
.dynamic-trades h4,
.dynamic-detail h4 {
  margin-bottom: 8px;
}
.dynamic-trades .table-wrapper,
.dynamic-detail .table-wrapper {
  margin-top: 6px;
}
.dynamic-trades th,
.dynamic-detail th {
  font-size: 0.78rem;
  letter-spacing: 0.02em;
}
.dynamic-trades td,
.dynamic-detail td {
  font-size: 0.78rem;
}
.compact-table {
  padding: 0;
}
.brief-view {
  margin-top: 12px;
}
.brief-card {
  background: rgba(var(--surface-rgb), 0.85);
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  padding: 16px 18px;
  box-shadow: 0 4px 12px rgba(2, 6, 23, 0.08);
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.brief-line {
  display: flex;
  align-items: center;
  gap: 10px;
  line-height: 1.4;
}
.brief-line .tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  text-transform: uppercase;
  background: linear-gradient(120deg, rgba(59, 130, 246, 0.35), rgba(16, 185, 129, 0.35));
  border: 1px solid rgba(148, 163, 184, 0.2);
}
.brief-line p {
  margin: 0;
  font-size: 0.95rem;
  color: var(--text-primary);
}
.multi-hint {
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin: 0 0 8px;
}
.multi-hint code {
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 0.8rem;
  color: var(--text-primary);
  background: rgba(148, 163, 184, 0.18);
  padding: 0 4px;
  border-radius: 4px;
}
.empty.small {
  font-size: 12px;
  padding: 8px 0;
}

.report-view {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.report-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: rgba(var(--surface-rgb), 0.85);
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  padding: 18px;
  box-shadow: 0 4px 12px rgba(2, 6, 23, 0.08);
}
.report-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.report-header h3 {
  margin: 0;
  font-size: 1.1rem;
}
.report-header p {
  margin: 4px 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.ghost-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-primary);
  border-radius: 999px;
  padding: 6px 14px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
}
.ghost-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.report-summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}
.report-kpi {
  background: var(--card-bg);
  border-radius: 14px;
  padding: 12px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: transform 0.2s ease;
}
.report-kpi .label {
  font-size: 0.8rem;
  color: var(--text-secondary);
}
.report-kpi .value {
  font-size: 1.15rem;
}
.report-kpi.positive .value {
  color: #34d399;
}
.report-kpi.negative .value {
  color: #f87171;
}
.report-panels {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}
.report-panel {
  background: var(--card-bg);
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.15);
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.report-panel.full-width {
  width: 100%;
  box-sizing: border-box;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}
.panel-header h4 {
  margin: 0;
  font-size: 0.95rem;
}
.panel-header span {
  color: var(--text-secondary);
  font-size: 0.75rem;
}
.report-chart {
  width: 100%;
  height: 220px;
}
.report-table td,
.report-table th {
  text-align: left;
}
.monthly-matrix {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  display: block;
  padding-bottom: 8px;
}
.monthly-matrix table {
  width: 100%;
  min-width: 900px;
  border-collapse: collapse;
  font-size: 0.85rem;
  table-layout: fixed;
}
.monthly-matrix th,
.monthly-matrix td {
  padding: 6px 8px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  text-align: center;
}
.monthly-matrix thead {
  background: rgba(var(--surface-rgb), 0.08);
}
.month-cell {
  font-variant-numeric: tabular-nums;
}
.month-cell.positive {
  color: #34d399;
}
.month-cell.negative {
  color: #f87171;
}
.year-label {
  text-align: left;
  color: var(--text-primary);
  font-weight: 600;
}
.year-total {
  font-weight: 600;
}
.buyhedge-view {
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.buyhedge-summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}
.buyhedge-view .summary-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
}
.buyhedge-view .summary-card h4 {
  margin: 0 0 8px;
  font-size: 0.95rem;
}
.buyhedge-view .summary-card ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.buyhedge-view .summary-card li {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
}
.buyhedge-panels {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}
.buyhedge-view .table-wrapper {
  padding: 12px;
}
.buyhedge-view .table-wrapper table {
  margin-top: 8px;
}
.buyhedge-charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}
.buyhedge-chart-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  background: var(--card-bg);
  min-height: 360px;
}
.buyhedge-chart {
  width: 100%;
  height: 320px;
}
</style>
