<script setup>
import { ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import * as echarts from 'echarts';
import { resolveEchartTheme } from '../lib/echartTheme';

const props = defineProps({
  dataset: { type: Object, required: true },
  height: { type: Number, default: 240 },
  theme: { type: String, default: 'dark' },
});

const container = ref(null);
let instance = null;
let instanceTheme = null;

const resolveCssVar = (name, fallback) => {
  if (typeof window === 'undefined') return fallback;
  const styles = getComputedStyle(document.documentElement);
  const value = styles.getPropertyValue(name);
  return value ? value.trim() : fallback;
};

const cardBackground = () => resolveCssVar('--card-bg', '#1e293b');
const textPrimary = () => resolveCssVar('--text-primary', '#e2e8f0');
const accentColor = () => resolveCssVar('--accent', '#3b82f6');
const buyColor = () => resolveCssVar('--success', '#34d399');
const sellColor = () => resolveCssVar('--danger', '#f87171');

const renderChart = () => {
  if (!container.value || !props.dataset) return;
  const themeName = resolveEchartTheme(props.theme);
  if (!instance || instanceTheme !== themeName) {
    instance?.dispose();
    instance = echarts.init(container.value, themeName);
    instanceTheme = themeName;
  }
  const dates = props.dataset.kline?.map((row) => row.date) || [];
  const closeSeries = props.dataset.kline?.map((row) => row.close) || [];
  const priceMap = new Map(props.dataset.kline?.map((row) => [row.date, row.close]) || []);
  const buildScatter = (list, adjust = 0) =>
    (list || []).map((d) => [d, (priceMap.get(d) || 0) * (1 + adjust)]);
  const title = props.dataset.label || `${props.dataset.freq} 周期`;
  instance.setOption({
    backgroundColor: cardBackground(),
    tooltip: { trigger: 'axis' },
    title: { show: false },
    grid: { left: 45, right: 16, top: 20, bottom: 30 },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: { type: 'value', scale: true },
    series: [
      {
        name: 'Close',
        type: 'line',
        data: closeSeries,
        smooth: true,
        lineStyle: { color: accentColor() },
        areaStyle: { opacity: 0.08 },
      },
      {
        name: 'Buy',
        type: 'scatter',
        data: buildScatter(props.dataset.buy_signals, -0.01),
        symbol: 'triangle',
        symbolSize: 9,
        itemStyle: { color: buyColor() },
      },
      {
        name: 'Sell',
        type: 'scatter',
        data: buildScatter(props.dataset.sell_signals, 0.01),
        symbol: 'triangle',
        symbolRotate: 180,
        symbolSize: 9,
        itemStyle: { color: sellColor() },
      },
    ],
  });
  instance.resize();
};

watch(
  () => [props.dataset, props.theme],
  () => {
    nextTick(() => renderChart());
  },
  { deep: true }
);

onMounted(() => {
  renderChart();
});

onBeforeUnmount(() => {
  instance?.dispose();
  instance = null;
  instanceTheme = null;
});
</script>

<template>
  <div class="signal-chart" :style="{ height: height + 'px' }">
    <div class="chart-title">{{ dataset.label || dataset.freq }}</div>
    <div class="chart-body" ref="container"></div>
  </div>
</template>

<style scoped>
.signal-chart {
  width: 100%;
  display: flex;
  flex-direction: column;
}
.chart-title {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 8px 0 8px;
  text-align: center;
}
.chart-body {
  flex: 1;
}
</style>
