<script setup>
import { ref, watch, nextTick, onBeforeUnmount, computed } from 'vue';
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
  { key: 'plan', title: '仓位计划', desc: '加仓/再平衡建议' },
  { key: 'stop', title: '止盈止损', desc: 'ATR 建议价位' },
  { key: 'heatmap', title: '收益热力图', desc: '持有周期 VS 收益' },
  { key: 'multi', title: '多周期信号', desc: '不同周期买卖提示' },
  { key: 'brief', title: '复盘摘要', desc: '自动生成复盘语句' },
  { key: 'dynamic', title: '资金管理', desc: '投入/对冲轨迹' },
];

const activeTab = ref('results');
const loading = ref(false);
const selectedResult = ref('');
const scores = ref([]);
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
const dynamicEquityChartRef = ref(null);
const dynamicInvestmentChartRef = ref(null);
let scoreChartInstance = null;
let dynamicEquityChartInstance = null;
let dynamicInvestmentChartInstance = null;

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
  emit('selectStrategy', entry);
};

const currentEntry = computed(() => props.results.find((entry) => entry.name === selectedResult.value) || null);
const investmentSeriesMain = computed(
  () => currentEntry.value?.result?.investmentCurveMain || currentEntry.value?.result?.investmentAmount || []
);
const investmentSeriesHedge = computed(
  () => currentEntry.value?.result?.investmentCurveHedge || currentEntry.value?.result?.hedgeInvestmentAmount || []
);
const dynamicDetails = computed(() => currentEntry.value?.result?.positionDetail || []);
const dynamicTrades = computed(() => currentEntry.value?.result?.trades || []);
const dynamicSummary = computed(() => currentEntry.value?.result?.dynamicSummary || null);
const equityCurveStatic = computed(() => currentEntry.value?.result?.equity_curve || []);
const equityCurveDynamic = computed(() => currentEntry.value?.result?.equityCurveWithDynamicFund || []);
const dynamicForceStop = computed(
  () => currentEntry.value?.result?.forceStopByDrawdown ?? currentEntry.value?.result?.forceStop ?? false
);
const latestInvestment = computed(() =>
  investmentSeriesMain.value.length ? investmentSeriesMain.value[investmentSeriesMain.value.length - 1][1] : 0
);
const latestHedge = computed(() =>
  investmentSeriesHedge.value.length ? investmentSeriesHedge.value[investmentSeriesHedge.value.length - 1][1] : 0
);
const isDynamicEntry = computed(() => Boolean(dynamicSummary.value));
const dynamicGateMessage = computed(() => {
  if (!currentEntry.value) return '请先运行并选择回测结果';
  if (!dynamicSummary.value) return '当前策略未启用动态资金管理';
  return '';
});

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
    dynamicEquityChartInstance = echarts.init(dom, 'dark');
  }
  const staticSeries = equityCurveStatic.value?.map(([ts, val]) => [ts, val]) || [];
  const dynamicSeriesArr = equityCurveDynamic.value?.map(([ts, val]) => [ts, val]) || [];
  dynamicEquityChartInstance.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['原始权益', '动态权益'], bottom: 0 },
    grid: { left: 40, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: staticSeries.map((item) => item[0]), boundaryGap: false },
    yAxis: { type: 'value', scale: true },
    series: [
      { name: '原始权益', type: 'line', smooth: true, data: staticSeries.map((item) => item[1]) },
      { name: '动态权益', type: 'line', smooth: true, data: dynamicSeriesArr.map((item) => item[1]) },
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
    dynamicInvestmentChartInstance = echarts.init(dom, 'dark');
  }
  const investMain = investmentSeriesMain.value || [];
  const investHedge = investmentSeriesHedge.value || [];
  dynamicInvestmentChartInstance.setOption({
    tooltip: { trigger: 'item' },
    legend: { data: ['主方向投入', '对冲投入'], bottom: 0 },
    grid: { left: 40, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: 'category',
      boundaryGap: true,
      data: investMain.map((item, idx) => item[0] || `T${idx + 1}`),
      axisLabel: { hideOverlap: true },
    },
    yAxis: { type: 'value', scale: true },
    series: [
      {
        name: '主方向投入',
        type: 'scatter',
        symbolSize: 10,
        itemStyle: { color: '#60a5fa' },
        data: investMain.map((item, idx) => ({ value: item[1] ?? 0, name: item[0] || `主-${idx + 1}` })),
      },
      {
        name: '对冲投入',
        type: 'scatter',
        symbolSize: 10,
        itemStyle: { color: '#facc15' },
        data: investHedge.map((item, idx) => ({ value: item[1] ?? 0, name: item[0] || `对冲-${idx + 1}` })),
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
      selectedResult.value = (dynamicEntry || entries[0]).name;
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
  }
);

const disposeCharts = () => {
  scoreChartInstance?.dispose();
  dynamicEquityChartInstance?.dispose();
  dynamicInvestmentChartInstance?.dispose();
  scoreChartInstance = null;
  dynamicEquityChartInstance = null;
  dynamicInvestmentChartInstance = null;
};

onBeforeUnmount(() => {
  disposeCharts();
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
  if (tab === 'brief' && !dailyBrief.value) fetchBrief();
  if (tab === 'plan' && !positionPlan.value.length) fetchPlan();
  if (tab === 'stop' && !stopSuggestion.value) fetchStop();
  if (tab === 'heatmap' && !heatmapData.value) fetchHeatmap();
  if (tab === 'multi' && !multiSignals.value.length) fetchMulti();
  if (tab === 'dynamic') {
    nextTick(() => {
      renderDynamicEquityChart();
      renderDynamicInvestmentChart();
    });
  }
};

const heatmapValue = (xIdx, yIdx) => heatmapLookup.value[`${xIdx}-${yIdx}`] ?? 0;
const formatPercent = (val) => `${(val * 100).toFixed(2)}%`;
const formatAmount = (val) => (Number.isFinite(Number(val)) ? Number(val).toLocaleString('zh-CN', { maximumFractionDigits: 2 }) : '-');
const formatDrawdown = (val) => {
  if (val === null || val === undefined || val === '') return '-';
  const num = Number(val);
  if (!Number.isFinite(num)) return '-';
  if (num > 0 && num <= 1) return `${(num * 100).toFixed(2)}%`;
  return formatAmount(num);
};
const formatBoolean = (val) => (val ? '是' : '否');
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
        :data-key="item.key"
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
                  <span>是否强制停止</span>
                  <strong :class="dynamicForceStop ? 'negative' : 'positive'">{{ formatBoolean(dynamicForceStop) }}</strong>
                </li>
              </ul>
            </div>
            <div class="summary-block">
              <h4>主方向配置</h4>
              <ul>
                <li><span>初始投资</span><strong>{{ formatAmount(dynamicSummary.initialInvestment) }}</strong></li>
                <li><span>亏损加注</span><strong>{{ formatAmount(dynamicSummary.lossStepAmount) }}</strong></li>
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
                <li><span>对冲加注</span><strong>{{ formatAmount(dynamicSummary.hedgeLossStepAmount) }}</strong></li>
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
                    <th>投资金额</th>
                    <th>连续亏损</th>
                    <th>数量</th>
                    <th>动态盈亏</th>
                    <th>对冲投资</th>
                    <th>对冲连亏</th>
                    <th>对冲数量</th>
                    <th>对冲盈亏</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="trade in dynamicTrades" :key="trade.entry_date + trade.exit_date">
                    <td>{{ trade.entry_date }}</td>
                    <td>{{ trade.exit_date }}</td>
                    <td>{{ formatAmount(trade.investment_amount ?? '-') }}</td>
                    <td>{{ trade.loss_streak ?? '-' }}</td>
                    <td>{{ trade.adjusted_quantity ?? '-' }}</td>
                    <td :class="(trade.pnl_with_dynamic_fund ?? 0) >= 0 ? 'positive' : 'negative'">
                      {{ formatAmount(trade.pnl_with_dynamic_fund ?? '-') }}
                    </td>
                    <td>{{ formatAmount(trade.hedge_investment_amount ?? '-') }}</td>
                    <td>{{ trade.hedge_loss_streak ?? '-' }}</td>
                    <td>{{ trade.hedge_adjusted_quantity ?? '-' }}</td>
                    <td :class="(trade.hedge_pnl_with_dynamic_fund ?? 0) >= 0 ? 'positive' : 'negative'">
                      {{ trade.hedge_pnl_with_dynamic_fund != null ? formatAmount(trade.hedge_pnl_with_dynamic_fund) : '-' }}
                    </td>
                  </tr>
                </tbody>
              </table>
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
                    <th>连亏次数</th>
                    <th>对冲金额</th>
                    <th>对冲连亏</th>
                    <th>止损触发</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in dynamicDetails.slice(-120).reverse()" :key="row.date + (row.investmentAmount ?? 0)">
                    <td>{{ row.date }}</td>
                    <td>{{ formatAmount(row.investmentAmount) }}</td>
                    <td>{{ row.lossStreak }}</td>
                    <td>{{ formatAmount(row.hedgeInvestmentAmount) }}</td>
                    <td>{{ row.hedgeLossStreak }}</td>
                    <td>{{ formatBoolean(row.forceStop) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
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
        <p class="multi-hint">
          示例：<code>D,W,M</code> 表示单周期；<code>15m&gt;5m</code> 表示“15m 判断趋势 &amp; 5m 精确买点”，可用
          <code>@备注</code> 自定义标题。
        </p>
        <div v-if="!multiSignals.length" class="empty">{{ multiMessage || '点击标签以加载多周期信号' }}</div>
        <div v-else>
          <div class="multi-charts">
            <SignalChart v-for="item in multiSignals" :key="item.freq" :dataset="item" :height="220" />
          </div>
          <div v-if="multiMessage" class="empty small">{{ multiMessage }}</div>
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
  background: #0f172a;
  border: 1px solid rgba(148, 163, 184, 0.2);
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
  color: #94a3b8;
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
  background: #0f172a;
  border: 1px solid rgba(148, 163, 184, 0.2);
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
  background: rgba(15, 23, 42, 0.85);
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.45);
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
</style>
