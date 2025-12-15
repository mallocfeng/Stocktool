<script setup>
import { onMounted, ref, watch, nextTick, onBeforeUnmount, computed } from 'vue';
import * as echarts from 'echarts';
import { resolveEchartTheme } from '../lib/echartTheme';

const props = defineProps({
  marketData: { type: Object, default: () => ({}) },
  equityData: { type: Array, default: () => [] },
  dynamicEquity: { type: Array, default: () => [] },
  investmentMain: { type: Array, default: () => [] },
  investmentHedge: { type: Array, default: () => [] },
  theme: { type: String, default: 'dark' },
});

const chartContainer = ref(null);
let chartInstance = null;
let resizeObserver = null;
let klineRawCache = [];

const hasKlineData = computed(() => Array.isArray(props.marketData?.kline) && props.marketData.kline.length > 0);
const themeClass = computed(() => `theme-${props.theme || 'dark'}`);

const readCssVar = (name, fallback) => {
  if (typeof window === 'undefined') return fallback;
  const styles = getComputedStyle(document.documentElement);
  const value = styles.getPropertyValue(name);
  return value ? value.trim() : fallback;
};

const resolveBackgroundColor = () => {
  const fallback = readCssVar('--card-bg', '#1e293b');
  return readCssVar('--chart-panel-canvas', fallback);
};

const formatTooltipValue = (val) => {
  const num = Number(val);
  if (!Number.isFinite(num)) return val ?? '--';
  return num.toFixed(2);
};
const toNumber = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const buildOption = () => {
  const tooltipFormatter = (params = []) => {
    if (!params.length) return '';
    const lines = [];
    const axisLabel = params[0]?.axisValueLabel || params[0]?.axisValue || '';
    if (axisLabel) lines.push(axisLabel);
    params.forEach((item) => {
      if (item.seriesType === 'candlestick') {
        const raw = klineRawCache[item.dataIndex] || {};
        const openVal = raw.open ?? item.value?.[0];
        const closeVal = raw.close ?? item.value?.[1];
        const lowVal = raw.low ?? item.value?.[2];
        const highVal = raw.high ?? item.value?.[3];
        lines.push(`${item.marker} ${item.seriesName}`);
        lines.push(`&nbsp;&nbsp;开盘：${formatTooltipValue(openVal)}`);
        lines.push(`&nbsp;&nbsp;收盘：${formatTooltipValue(closeVal)}`);
        lines.push(`&nbsp;&nbsp;最低：${formatTooltipValue(lowVal)}`);
        lines.push(`&nbsp;&nbsp;最高：${formatTooltipValue(highVal)}`);
      } else {
        const value = Array.isArray(item.value) ? item.value[1] ?? item.value[0] : item.data ?? item.value;
        lines.push(`${item.marker} ${item.seriesName}：${formatTooltipValue(value)}`);
      }
    });
    return lines.join('<br/>');
  };
  const upColor = readCssVar('--danger', '#ef4444');
  const downColor = readCssVar('--success', '#10b981');
  const primaryLine = readCssVar('--accent', '#3b82f6');
  const dynamicLine = readCssVar('--accent-secondary', '#a855f7');
  const investMainColor = readCssVar('--warning', '#facc15');
  const investHedgeColor = readCssVar('--success', '#34d399');
  const buyColor = readCssVar('--success', '#22c55e');
  const sellColor = readCssVar('--danger', '#f87171');
  return {
    backgroundColor: resolveBackgroundColor(),
    tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'cross' },
    formatter: tooltipFormatter,
  },
  grid: [
    { left: '4%', right: '4%', top: 10, height: '55%' },
    { left: '4%', right: '4%', top: '68%', height: '25%' },
  ],
  xAxis: [
    { type: 'category', data: [], boundaryGap: false, min: 'dataMin', max: 'dataMax' },
    { type: 'category', gridIndex: 1, data: [], boundaryGap: false, min: 'dataMin', max: 'dataMax', axisLabel: { show: false } },
  ],
  yAxis: [
    { scale: true, splitArea: { show: true } },
    { scale: true, gridIndex: 1, splitNumber: 3, axisLabel: { show: false }, splitLine: { show: false } },
  ],
  dataZoom: [
    { type: 'inside', xAxisIndex: [0, 1], start: 40, end: 100 },
    { type: 'slider', xAxisIndex: [0, 1], start: 40, end: 100 },
  ],
  series: [
    {
      name: 'K线',
      type: 'candlestick',
      data: [],
      itemStyle: {
        color: upColor,
        color0: downColor,
        borderColor: upColor,
        borderColor0: downColor,
      },
    },
    {
      name: '原始权益',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: true,
      lineStyle: { opacity: 0.8, color: primaryLine },
    },
    {
      name: '动态权益',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: true,
      lineStyle: { opacity: 0.8, color: dynamicLine },
    },
    {
      name: '主方向投入',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: false,
      showSymbol: false,
      lineStyle: { color: investMainColor, width: 1.5 },
    },
    {
      name: '对冲投入',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: false,
      showSymbol: false,
      lineStyle: { color: investHedgeColor, width: 1.2, type: 'dashed' },
    },
    {
      name: '买入信号',
      type: 'scatter',
      data: [],
      symbol: 'triangle',
      symbolSize: 10,
      itemStyle: { color: buyColor },
    },
    {
      name: '卖出信号',
      type: 'scatter',
      data: [],
      symbol: 'triangle',
      symbolSize: 10,
      symbolRotate: 180,
      itemStyle: { color: sellColor },
    },
  ],
  };
};

const initChart = () => {
  if (!chartContainer.value) return;
  chartInstance?.dispose();
  const themeName = resolveEchartTheme(props.theme);
  chartInstance = echarts.init(chartContainer.value, themeName);
  chartInstance.setOption(buildOption());
  updateChart();
};

const updateChart = () => {
  if (!chartInstance || !props.marketData || !props.marketData.kline) return;
  const { kline, buy_signals = [], sell_signals = [] } = props.marketData;
  klineRawCache = kline.map((item) => ({
    date: item.date,
    open: toNumber(item.open),
    close: toNumber(item.close),
    low: toNumber(item.low),
    high: toNumber(item.high),
  }));
  const klineData = klineRawCache.map((item) => [item.open, item.close, item.low, item.high]);
  const dates = kline.map((item) => item.date);
  const dateIndex = new Map(dates.map((d, idx) => [d, idx]));
  const buyPoints = [];
  const sellPoints = [];
  buy_signals.forEach((d) => {
    const idx = dateIndex.get(d);
    if (idx !== undefined) {
      const row = kline[idx];
      buyPoints.push([d, row.low * 0.98]);
    }
  });
  sell_signals.forEach((d) => {
    const idx = dateIndex.get(d);
    if (idx !== undefined) {
      const row = kline[idx];
      sellPoints.push([d, row.high * 1.02]);
    }
  });
  chartInstance.setOption({
    xAxis: [{ data: dates }, { data: dates }],
    series: [
      { data: klineData },
      { data: props.equityData || [] },
      { data: props.dynamicEquity || [] },
      { data: props.investmentMain || [] },
      { data: props.investmentHedge || [] },
      { data: buyPoints },
      { data: sellPoints },
    ],
  });
  chartInstance.resize();
};

watch(
  () => [props.marketData, props.equityData, props.dynamicEquity, props.investmentMain, props.investmentHedge],
  () => {
    nextTick(() => updateChart());
  },
  { deep: true }
);

watch(
  () => props.theme,
  () => {
    nextTick(() => initChart());
  }
);

onMounted(() => {
  initChart();
  resizeObserver = new ResizeObserver(() => chartInstance?.resize());
  if (chartContainer.value) {
    resizeObserver.observe(chartContainer.value);
  }
  setTimeout(() => updateChart(), 400);
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  chartInstance?.dispose();
});
</script>

<template>
  <div class="chart-panel-wrapper" :class="themeClass">
    <div v-if="!hasKlineData" class="no-data-overlay">
      还未加载数据，请先加载 CSV 或在线读取股票数据
    </div>
    <div class="chart-panel" ref="chartContainer"></div>
  </div>
</template>

<style scoped>
.chart-panel-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 360px;
  padding: 18px;
  border-radius: 26px;
  background: var(--chart-panel-bg, rgba(var(--surface-rgb), 0.85));
  border: 1px solid var(--chart-panel-border, rgba(201, 217, 255, 0.85));
  box-shadow: var(--chart-panel-shadow, 0 20px 50px rgba(0, 0, 0, 0.18));
  box-sizing: border-box;
  display: flex;
}

.chart-panel {
  width: 100%;
  flex: 1;
  height: 100%;
  border-radius: 18px;
  background: var(--chart-panel-canvas, var(--card-bg));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
  overflow: hidden;
}

.no-data-overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 5;
  max-width: min(360px, calc(100% - 32px));
  padding: 10px 14px;
  font-size: 15px;
  line-height: 1.35;
  text-align: center;
  border-radius: 10px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  pointer-events: none;
}

.chart-panel-wrapper.theme-dark .no-data-overlay {
  color: rgba(255, 255, 255, 0.92);
  background: rgba(15, 23, 42, 0.74);
  border: 1px solid rgba(148, 163, 184, 0.25);
}

.chart-panel-wrapper.theme-light .no-data-overlay {
  color: rgba(15, 23, 42, 0.9);
  background:
    linear-gradient(135deg, rgba(var(--tone-1-rgb), 0.16), rgba(var(--tone-4-rgb), 0.08)),
    rgba(var(--surface-rgb), 0.9);
  border: 1px solid rgba(15, 23, 42, 0.12);
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
}

.chart-panel-wrapper.theme-morandi .no-data-overlay {
  color: var(--text-primary);
  background:
    linear-gradient(
      135deg,
      rgba(var(--tone-1-rgb), 0.22),
      rgba(var(--tone-3-rgb), 0.16),
      rgba(var(--tone-6-rgb), 0.14)
    ),
    rgba(var(--surface-rgb), 0.86);
  border: 1px solid rgba(var(--overlay-rgb), 0.14);
  box-shadow: 0 12px 30px rgba(var(--overlay-rgb), 0.16);
}
</style>
