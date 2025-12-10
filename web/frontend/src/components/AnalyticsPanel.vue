<script setup>
import { ref, watch, nextTick, onBeforeUnmount } from 'vue';
import axios from 'axios';
import * as echarts from 'echarts';
import SignalChart from './SignalChart.vue';

const props = defineProps({
  results: { type: Array, default: () => [] },
  hasData: { type: Boolean, default: false },
  initialCapital: { type: Number, default: 100000 },
  multiFreqs: { type: String, default: 'D,W,M' },
});

const emit = defineEmits(['selectStrategy']);

const categories = [
  { key: 'results', title: '回测结果', desc: '策略收益与回撤概览' },
  { key: 'scores', title: '指标评分', desc: '多维指标综合得分' },
  { key: 'stress', title: '情景压测', desc: '历史极端行情表现' },
  { key: 'plan', title: '仓位计划', desc: '加仓/再平衡建议' },
  { key: 'stop', title: '止盈止损', desc: 'ATR 建议价位' },
  { key: 'heatmap', title: '收益热力图', desc: '持有周期 VS 收益' },
  { key: 'multi', title: '多周期信号', desc: '不同周期买卖提示' },
  { key: 'brief', title: '复盘摘要', desc: '自动生成复盘语句' },
];

const activeTab = ref('results');
const loading = ref(false);
const selectedResult = ref('');
const scores = ref([]);
const stressTest = ref([]);
const stressMessage = ref('');
const positionPlan = ref([]);
const planMessage = ref('');
const stopSuggestion = ref(null);
const heatmapData = ref(null);
const heatmapLookup = ref({});
const heatmapMessage = ref('');
const dailyBrief = ref('');
const multiSignals = ref([]);
const multiMessage = ref('');
const scoreChartRef = ref(null);
let scoreChartInstance = null;

const parseFreqs = () =>
  (props.multiFreqs || 'D,W,M')
    .split(',')
    .map((f) => f.trim().toUpperCase())
    .filter((f) => f.length);

const selectResult = (entry) => {
  selectedResult.value = entry.name;
  emit('selectStrategy', entry);
};

watch(
  () => props.results,
  (entries) => {
    if (entries && entries.length) {
      selectedResult.value = entries[0].name;
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

const fetchStress = async () => {
  loading.value = true;
  stressMessage.value = '';
  try {
    const res = await axios.post('/analytics/stress');
    if (Array.isArray(res.data)) {
      stressTest.value = res.data;
      stressMessage.value = '';
    } else {
      stressTest.value = [];
      stressMessage.value = res.data?.message || '暂无数据';
    }
  } catch (e) {
    console.error(e);
    stressMessage.value = '获取失败';
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
    const res = await axios.post('/analytics/multi_timeframe', { freqs: parseFreqs() });
    if (Array.isArray(res.data) && res.data.length) {
      multiSignals.value = res.data;
      multiMessage.value = '';
    } else {
      multiSignals.value = [];
      multiMessage.value = '暂无信号数据';
    }
  } catch (e) {
    console.error(e);
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
    scoreChartInstance = echarts.init(scoreChartRef.value, 'dark');
  }
  const dates = scores.value.map((row) => row.date);
  const values = scores.value.map((row) => row.total_score);
  scoreChartInstance.setOption({
    backgroundColor: '#1e293b',
    grid: { left: 45, right: 20, top: 15, bottom: 25 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: dates, boundaryGap: false },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#374151' } } },
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

onBeforeUnmount(() => {
  scoreChartInstance?.dispose();
});

const switchTab = (tab) => {
  activeTab.value = tab;
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
  if (tab === 'stress' && !stressTest.value.length) fetchStress();
  if (tab === 'brief' && !dailyBrief.value) fetchBrief();
  if (tab === 'plan' && !positionPlan.value.length) fetchPlan();
  if (tab === 'stop' && !stopSuggestion.value) fetchStop();
  if (tab === 'heatmap' && !heatmapData.value) fetchHeatmap();
  if (tab === 'multi' && !multiSignals.value.length) fetchMulti();
};

const heatmapValue = (xIdx, yIdx) => heatmapLookup.value[`${xIdx}-${yIdx}`] ?? 0;
const formatPercent = (val) => `${(val * 100).toFixed(2)}%`;
const planHeaderMap = {
  index: '序号',
  step: '步骤',
  target_price: '目标价',
  buy_shares: '买入股数',
  cost: '总成本',
  remaining_cash: '剩余现金',
  avg_cost: '平均成本',
};
</script>

<template>
  <div class="analytics-panel card">
    <div class="category-row">
      <button
        v-for="item in categories"
        :key="item.key"
        class="category-card"
        :class="{ active: activeTab === item.key }"
        :disabled="item.key !== 'results' && !hasData"
        @click="switchTab(item.key)"
      >
        <span class="category-label">{{ item.title }}</span>
        <p>{{ item.desc }}</p>
      </button>
    </div>

    <div class="tab-content">
      <div v-if="loading" class="loading">数据加载中…</div>

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
              <span class="tag">{{ entry.result.trades?.length || 0 }} 笔交易</span>
            </header>
            <dl>
              <div><dt>总收益</dt><dd :class="entry.result.total_return >= 0 ? 'positive' : 'negative'">{{ formatPercent(entry.result.total_return) }}</dd></div>
              <div><dt>年化</dt><dd>{{ formatPercent(entry.result.annualized_return) }}</dd></div>
              <div><dt>最大回撤</dt><dd>{{ formatPercent(entry.result.max_drawdown) }}</dd></div>
              <div><dt>胜率</dt><dd>{{ formatPercent(entry.result.win_rate) }}</dd></div>
            </dl>
          </article>
        </div>
      </section>

      <section v-else-if="activeTab === 'scores'">
        <div v-if="scores.length" class="scores-view">
          <div ref="scoreChartRef" class="score-chart"></div>
          <table>
            <thead><tr><th>日期</th><th>综合评分</th></tr></thead>
            <tbody>
              <tr v-for="row in scores.slice(-80).reverse()" :key="row.date">
                <td>{{ row.date }}</td>
                <td :class="row.total_score >= 0 ? 'positive' : 'negative'">{{ row.total_score.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="empty">暂无评分数据</div>
      </section>

      <section v-else-if="activeTab === 'stress'">
        <div v-if="!stressTest.length" class="empty">{{ stressMessage || '暂无情景覆盖' }}</div>
        <table v-else>
          <thead>
            <tr>
              <th v-for="(value, key) in stressTest[0]" :key="key">{{ key }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, idx) in stressTest" :key="idx">
              <td v-for="(value, key) in row" :key="key">{{ typeof value === 'number' ? value.toFixed(2) : value }}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section v-else-if="activeTab === 'plan'">
        <div v-if="!positionPlan.length" class="empty">{{ planMessage || '条件不足或尚未加载' }}</div>
        <table v-else>
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
      </section>

      <section v-else-if="activeTab === 'stop'">
        <div v-if="!stopSuggestion" class="empty">点击标签即可获取最新建议</div>
        <div v-else class="stop-grid">
          <div>
            <span>最新价</span>
            <strong>{{ stopSuggestion.last_price?.toFixed(2) }}</strong>
          </div>
          <div>
            <span>ATR</span>
            <strong>{{ stopSuggestion.atr?.toFixed(2) }}</strong>
          </div>
          <div>
            <span>建议止损</span>
            <strong class="negative">{{ stopSuggestion.suggest_stop_loss?.toFixed(2) }}</strong>
          </div>
          <div>
            <span>建议止盈</span>
            <strong class="positive">{{ stopSuggestion.suggest_take_profit?.toFixed(2) }}</strong>
          </div>
          <div>
            <span>跟踪止损</span>
            <strong>{{ stopSuggestion.trailing_stop?.toFixed(2) }}</strong>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'heatmap'">
        <div v-if="!heatmapData || !heatmapData.data" class="empty">{{ heatmapMessage || '暂无可用数据' }}</div>
        <div v-else class="heatmap-table">
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
                :style="{ backgroundColor: heatmapValue(xIdx, yIdx) > 0 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)' }"
              >
                {{ (heatmapValue(xIdx, yIdx) * 100).toFixed(0) }}
              </span>
            </div>
          </div>
        </div>
      </section>

      <section v-else-if="activeTab === 'multi'">
        <div v-if="!multiSignals.length" class="empty">{{ multiMessage || '点击标签以加载多周期信号' }}</div>
        <div v-else class="multi-charts">
          <SignalChart v-for="item in multiSignals" :key="item.freq" :dataset="item" :height="220" />
        </div>
      </section>

      <section v-else-if="activeTab === 'brief'" class="brief-view">
        <pre>{{ dailyBrief || '点击加载复盘摘要' }}</pre>
      </section>
    </div>
  </div>
</template>
