<script setup>
import { ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue';
import * as echarts from 'echarts';

const props = defineProps({
  dataset: { type: Object, required: true },
  height: { type: Number, default: 240 },
});

const container = ref(null);
let instance = null;

const renderChart = () => {
  if (!container.value || !props.dataset) return;
  if (!instance) {
    instance = echarts.init(container.value, 'dark');
  }
  const dates = props.dataset.kline?.map((row) => row.date) || [];
  const closeSeries = props.dataset.kline?.map((row) => row.close) || [];
  const priceMap = new Map(props.dataset.kline?.map((row) => [row.date, row.close]) || []);
  const buildScatter = (list, adjust = 0) =>
    (list || []).map((d) => [d, (priceMap.get(d) || 0) * (1 + adjust)]);
  const title = props.dataset.label || `${props.dataset.freq} 周期`;
  instance.setOption({
    backgroundColor: '#1e293b',
    tooltip: { trigger: 'axis' },
    grid: { left: 45, right: 16, top: 20, bottom: 30 },
    title: { text: title, textStyle: { color: '#e2e8f0', fontSize: 13 } },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: { type: 'value', scale: true },
    series: [
      {
        name: 'Close',
        type: 'line',
        data: closeSeries,
        smooth: true,
        lineStyle: { color: '#38bdf8' },
        areaStyle: { opacity: 0.08 },
      },
      {
        name: 'Buy',
        type: 'scatter',
        data: buildScatter(props.dataset.buy_signals, -0.01),
        symbol: 'triangle',
        symbolSize: 9,
        itemStyle: { color: '#34d399' },
      },
      {
        name: 'Sell',
        type: 'scatter',
        data: buildScatter(props.dataset.sell_signals, 0.01),
        symbol: 'triangle',
        symbolRotate: 180,
        symbolSize: 9,
        itemStyle: { color: '#f87171' },
      },
    ],
  });
  instance.resize();
};

watch(
  () => props.dataset,
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
});
</script>

<template>
  <div class="signal-chart" :style="{ height: height + 'px' }" ref="container"></div>
</template>

<style scoped>
.signal-chart {
  width: 100%;
}
</style>
