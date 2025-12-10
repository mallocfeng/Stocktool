<script setup>
import { ref } from 'vue';
import axios from 'axios';
import ConfigPanel from './components/ConfigPanel.vue';
import ChartPanel from './components/ChartPanel.vue';
import AnalyticsPanel from './components/AnalyticsPanel.vue';

const marketData = ref({ kline: [], buy_signals: [], sell_signals: [] });
const backtestResults = ref([]);
const selectedEquity = ref([]);
const logs = ref([]);
const hasData = ref(false);
const isRunning = ref(false);
const lastConfigMeta = ref({ initialCapital: 100000, multiFreqs: 'D,W,M' });

const handleRun = async (configPayload) => {
  if (!configPayload || !configPayload.payload) return;
  const { payload, meta } = configPayload;
  lastConfigMeta.value = {
    initialCapital: meta?.initialCapital ?? lastConfigMeta.value.initialCapital,
    multiFreqs: meta?.multiFreqs ?? lastConfigMeta.value.multiFreqs,
  };
  logs.value = ['正在发送请求…'];
  isRunning.value = true;
  try {
    const res = await axios.post('/run_backtest', payload);
    if (res.data.status === 'success') {
      backtestResults.value = res.data.entries;
      logs.value = res.data.logs || [];
      if (res.data.entries.length) {
        selectedEquity.value = res.data.entries[0].result.equity_curve;
      }
      const marketRes = await axios.get('/market_data_chart');
      marketData.value = marketRes.data;
      hasData.value = true;
    } else {
      logs.value = res.data.logs || [res.data.error];
      alert('回测失败：' + (res.data.error || '未知错误'));
    }
  } catch (e) {
    logs.value = [...logs.value, '请求失败：' + e.message];
    alert('请求错误：' + e.message);
  } finally {
    isRunning.value = false;
  }
};

const handleSelectStrategy = (entry) => {
  if (!entry) return;
  selectedEquity.value = entry.result?.equity_curve || [];
};
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <h1>StockTool 云端量化</h1>
        <p>一键上传 CSV · 自动回测 · 智能洞察</p>
      </div>
      <span class="status" :class="{ running: isRunning }">{{ isRunning ? '回测执行中…' : '就绪' }}</span>
    </header>

    <main class="app-main">
      <aside class="sidebar">
        <ConfigPanel :busy="isRunning" @run="handleRun" />
        <section class="card log-panel">
          <div class="panel-header">
            <h3>运行日志</h3>
          </div>
          <div class="log-view">
            <template v-if="logs.length">
              <div v-for="(log, idx) in logs" :key="idx">{{ log }}</div>
            </template>
            <div v-else class="empty">暂无日志</div>
          </div>
        </section>
      </aside>

      <section class="content-column">
        <div class="card chart-card">
          <div class="panel-header">
            <h3>行情与权益走势</h3>
            <span v-if="backtestResults.length" class="tag">{{ backtestResults[0].title }}</span>
          </div>
          <ChartPanel :marketData="marketData" :equityData="selectedEquity" />
        </div>
        <AnalyticsPanel
          :results="backtestResults"
          :hasData="hasData"
          :initialCapital="lastConfigMeta.initialCapital"
          :multiFreqs="lastConfigMeta.multiFreqs"
          @select-strategy="handleSelectStrategy"
        />
      </section>
    </main>
  </div>
</template>
