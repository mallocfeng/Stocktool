<script setup>
import { onMounted, ref, watch, nextTick, onBeforeUnmount } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  marketData: { type: Object, default: () => ({}) },
  equityData: { type: Array, default: () => [] },
  dynamicEquity: { type: Array, default: () => [] },
  investmentMain: { type: Array, default: () => [] },
  investmentHedge: { type: Array, default: () => [] },
});

const chartContainer = ref(null);
let chartInstance = null;
let resizeObserver = null;

const buildOption = () => ({
  backgroundColor: '#1e293b',
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
        color: '#ef4444',
        color0: '#10b981',
        borderColor: '#ef4444',
        borderColor0: '#10b981',
      },
    },
    {
      name: '原始权益',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: true,
      lineStyle: { opacity: 0.8, color: '#3b82f6' },
    },
    {
      name: '动态权益',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: true,
      lineStyle: { opacity: 0.8, color: '#a855f7' },
    },
    {
      name: '主方向投入',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: false,
      showSymbol: false,
      lineStyle: { color: '#facc15', width: 1.5 },
    },
    {
      name: '对冲投入',
      type: 'line',
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: [],
      smooth: false,
      showSymbol: false,
      lineStyle: { color: '#34d399', width: 1.2, type: 'dashed' },
    },
    {
      name: '买入信号',
      type: 'scatter',
      data: [],
      symbol: 'triangle',
      symbolSize: 10,
      itemStyle: { color: '#22c55e' },
    },
    {
      name: '卖出信号',
      type: 'scatter',
      data: [],
      symbol: 'triangle',
      symbolSize: 10,
      symbolRotate: 180,
      itemStyle: { color: '#f87171' },
    },
  ],
});

const initChart = () => {
  if (!chartContainer.value) return;
  chartInstance?.dispose();
  chartInstance = echarts.init(chartContainer.value, 'dark');
  chartInstance.setOption(buildOption());
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
