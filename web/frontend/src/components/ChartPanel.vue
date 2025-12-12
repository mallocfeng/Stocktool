<script setup>
import { onMounted, ref, watch, nextTick, onBeforeUnmount } from 'vue';
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

const readCssVar = (name, fallback) => {
  if (typeof window === 'undefined') return fallback;
  const styles = getComputedStyle(document.documentElement);
  const value = styles.getPropertyValue(name);
  return value ? value.trim() : fallback;
};

const resolveBackgroundColor = () => readCssVar('--card-bg', '#1e293b');

const buildOption = () => {
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
  const klineData = kline.map((item) => [item.open, item.close, item.low, item.high]);
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
  <div class="chart-panel" ref="chartContainer"></div>
</template>

<style scoped>
.chart-panel {
  width: 100%;
  height: 100%;
  min-height: 360px;
}
</style>
