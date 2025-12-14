<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import axios from 'axios';
import { marked } from 'marked';
import { useRouter } from 'vue-router';
import ConfigPanel from '../components/ConfigPanel.vue';
import ChartPanel from '../components/ChartPanel.vue';
import AnalyticsPanel from '../components/AnalyticsPanel.vue';
import { useAuth } from '../lib/useAuth';
import { useTheme } from '../lib/useTheme';
import { disableAuth } from '../lib/authConfig';

marked.setOptions({
  breaks: true,
  gfm: true,
});

const AI_STORAGE_KEY = 'stocktool_ai_insight';

const marketData = ref({ kline: [], buy_signals: [], sell_signals: [] });
const backtestResults = ref([]);
const selectedEquity = ref([]);
const dynamicEquity = ref([]);
const investmentCurveMain = ref([]);
const investmentCurveHedge = ref([]);
const logs = ref([]);
const hasData = ref(false);
const isRunning = ref(false);
const lastConfigMeta = ref({ initialCapital: 100000, multiFreqs: 'D,W,M', assetLabel: '' });
const aiResult = ref(null);
const aiLoading = ref(false);
const aiError = ref('');
const datasetSignature = ref('');
const cachedSignature = ref('');
const aiCardExpanded = ref(false);
let aiHoverTimer = null;
const overlayBlocking = ref(false);
const overlayMessage = ref({ title: '', detail: '' });
const showPasswordModal = ref(false);
const passwordForm = ref({ currentPassword: '', newPassword: '', confirmPassword: '' });
const passwordModalError = ref('');
const passwordModalLoading = ref(false);
let aiTicket = 0;
const aiEnabled = ref(false);
const router = useRouter();
const auth = useAuth();
const currentUser = auth.currentUser;
const showUserInfo = computed(() => !disableAuth && Boolean(currentUser.value));
const showAdminLink = computed(() => !disableAuth && currentUser.value?.role === 'admin');
const isAdminUser = computed(() => currentUser.value?.role === 'admin');
const usernameDisplay = computed(() => currentUser.value?.username || '未命名');
const goAdmin = () => {
  router.push({ path: '/admin' });
};
const { themeMode, resolvedTheme, themeOptions, setThemeMode } = useTheme();

const overlayActive = computed(() => isRunning.value || overlayBlocking.value);
const overlayTitle = computed(() => {
  if (isRunning.value) return '正在处理 CSV / 回测';
  return overlayMessage.value.title || '数据处理中';
});
const overlayDetail = computed(() => {
  if (isRunning.value) {
    return '大型文件上传和计算可能需要一点时间，请勿刷新或操作页面…';
  }
  return overlayMessage.value.detail || '请勿刷新或关闭页面…';
});

const aiRenderedHtml = computed(() => {
  const text = aiResult.value?.analysis?.trim();
  if (!text) return '';
  return marked.parse(text);
});
const computeDatasetSignature = (data) => {
  if (!data || !Array.isArray(data.kline) || !data.kline.length) return '';
  const scope = data.kline.slice(-120);
  const sample = {
    len: data.kline.length,
    tail: scope.map((row) => [row.date, row.close, row.high, row.low]),
    buy: data.buy_signals?.slice(-40) || [],
    sell: data.sell_signals?.slice(-40) || [],
  };
  const raw = JSON.stringify(sample);
  let hash = 0;
  for (let i = 0; i < raw.length; i += 1) {
    hash = (hash * 31 + raw.charCodeAt(i)) >>> 0;
  }
  return `${hash.toString(16)}-${sample.len}`;
};

const loadPersistedAIInsight = () => {
  if (typeof window === 'undefined') return;
  try {
    const cached = localStorage.getItem(AI_STORAGE_KEY);
    if (!cached) return;
    const payload = JSON.parse(cached);
    if (payload?.result) {
      aiResult.value = payload.result;
      cachedSignature.value = payload.datasetSignature || '';
    }
  } catch (err) {
    console.warn('Failed to load AI insight cache', err);
  }
};

const persistAIInsight = (signature, result) => {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(
      AI_STORAGE_KEY,
      JSON.stringify({
        datasetSignature: signature,
        result,
      }),
    );
    cachedSignature.value = signature;
  } catch (err) {
    console.warn('Failed to persist AI insight cache', err);
  }
};

const handleDatasetSignature = (data) => {
  const signature = computeDatasetSignature(data);
  datasetSignature.value = signature;
  if (!signature) return;
  if (!aiEnabled.value) return;
  if (signature !== cachedSignature.value) {
    requestAIInsight();
  }
};

watch(hasData, (value) => {
  if (value) {
    showTopTip.value = false;
  }
});

onMounted(() => {
  loadPersistedAIInsight();
});

const handleOverlayBlock = (info) => {
  const payload = typeof info === 'object' && info !== null ? info : {};
  overlayBlocking.value = true;
  overlayMessage.value = {
    title: payload.title || '数据处理中',
    detail: payload.detail || '请勿刷新或关闭页面…',
  };
};

const handleOverlayUnblock = () => {
  overlayBlocking.value = false;
  overlayMessage.value = { title: '', detail: '' };
};

const handleLogout = async () => {
  try {
    await auth.logout();
  } finally {
    router.replace('/login');
  }
};

const openPasswordModal = () => {
  passwordModalError.value = '';
  passwordModalLoading.value = false;
  passwordForm.value = { currentPassword: '', newPassword: '', confirmPassword: '' };
  showPasswordModal.value = true;
};

const closePasswordModal = () => {
  if (passwordModalLoading.value) return;
  showPasswordModal.value = false;
};

const submitPasswordChange = async () => {
  passwordModalError.value = '';
  if (!passwordForm.value.currentPassword) {
    passwordModalError.value = '请输入当前密码';
    return;
  }
  if (passwordForm.value.newPassword.length < 6) {
    passwordModalError.value = '新密码至少需要 6 个字符';
    return;
  }
  if (passwordForm.value.newPassword !== passwordForm.value.confirmPassword) {
    passwordModalError.value = '两次输入的新密码不一致';
    return;
  }
  passwordModalLoading.value = true;
  try {
    await axios.post('/me/change-password', {
      current_password: passwordForm.value.currentPassword,
      new_password: passwordForm.value.newPassword,
    });
    passwordForm.value = { currentPassword: '', newPassword: '', confirmPassword: '' };
    await handleLogout();
    showPasswordModal.value = false;
    return;
  } catch (err) {
    const detail = err?.response?.data?.detail;
    passwordModalError.value = detail || err?.message || '修改密码失败，请稍后再试';
  } finally {
    passwordModalLoading.value = false;
  }
};

const showTopTip = ref(true);

const handleAIToggle = () => {
  aiEnabled.value = !aiEnabled.value;
  if (!aiEnabled.value) {
    aiResult.value = null;
    aiError.value = '';
    aiLoading.value = false;
    datasetSignature.value = '';
    cachedSignature.value = '';
    if (typeof window !== 'undefined') {
      localStorage.removeItem(AI_STORAGE_KEY);
    }
  }
};

const handleRun = async (configPayload) => {
  if (!configPayload || !configPayload.payload) return;
  const { payload, meta } = configPayload;
  if (typeof window !== 'undefined') {
    requestAnimationFrame(() => {
      const target = document.querySelector('.analytics-panel');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  }
  lastConfigMeta.value = {
    initialCapital: meta?.initialCapital ?? lastConfigMeta.value.initialCapital,
    multiFreqs: meta?.multiFreqs ?? lastConfigMeta.value.multiFreqs,
    assetLabel: meta?.assetLabel ?? lastConfigMeta.value.assetLabel,
  };
  logs.value = ['正在发送请求…'];
  isRunning.value = true;
  aiError.value = '';
  aiTicket += 1;
  aiLoading.value = false;
  try {
    const res = await axios.post('/run_backtest', payload);
    if (res.data.status === 'success') {
      backtestResults.value = res.data.entries;
      logs.value = res.data.logs || [];
      if (res.data.entries.length) {
        const first = res.data.entries[0];
        selectedEquity.value = first.result.equity_curve;
        dynamicEquity.value = first.result.equityCurveWithDynamicFund || [];
        investmentCurveMain.value = first.result.investmentCurveMain || first.result.investmentAmount || [];
        investmentCurveHedge.value = first.result.investmentCurveHedge || first.result.hedgeInvestmentAmount || [];
      }
      const marketRes = await axios.get('/market_data_chart');
      marketData.value = marketRes.data;
      hasData.value = true;
      handleDatasetSignature(marketRes.data);
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
  dynamicEquity.value = entry.result?.equityCurveWithDynamicFund || [];
  investmentCurveMain.value = entry.result?.investmentCurveMain || entry.result?.investmentAmount || [];
  investmentCurveHedge.value = entry.result?.investmentCurveHedge || entry.result?.hedgeInvestmentAmount || [];
};

const requestAIInsight = async (force = false) => {
  if (!aiEnabled.value) {
    aiError.value = '';
    return;
  }
  if (!hasData.value) {
    aiError.value = '请先加载行情数据并运行回测';
    return;
  }
  if (!datasetSignature.value) {
    aiError.value = '缺少行情指纹，请先加载行情数据';
    return;
  }
  if (!force && datasetSignature.value === cachedSignature.value && aiResult.value) {
    return;
  }
  const currentTicket = ++aiTicket;
  aiLoading.value = true;
  aiError.value = '';
  try {
    const res = await axios.post('/analytics/ai_insight', {
      asset_label: lastConfigMeta.value.assetLabel || '',
    });
    if (currentTicket !== aiTicket || !aiEnabled.value) return;
    aiResult.value = res.data;
    persistAIInsight(datasetSignature.value, res.data);
  } catch (e) {
    if (currentTicket !== aiTicket || !aiEnabled.value) return;
    if (e.response?.status === 403) {
      aiError.value =
        '免费额度已用完：https://mallocfeng1982.win/v1/chat/completions，请联系管理员';
    } else {
      aiError.value = e.response?.data?.detail || e.message;
    }
  } finally {
    if (currentTicket === aiTicket) {
      aiLoading.value = false;
    }
  }
};

const handleAIMouseEnter = () => {
  clearTimeout(aiHoverTimer);
  aiHoverTimer = setTimeout(() => {
    aiCardExpanded.value = true;
  }, 1000);
};

const handleAIMouseLeave = () => {
  clearTimeout(aiHoverTimer);
  aiHoverTimer = null;
  aiCardExpanded.value = false;
};
</script>

<template>
  <div class="app-shell">
    <transition name="fade">
      <div v-if="overlayActive" class="global-overlay" aria-live="assertive">
        <div class="overlay-card">
          <div class="overlay-spinner" aria-hidden="true"></div>
          <div class="overlay-text">
            <strong>{{ overlayTitle }}</strong>
            <p>{{ overlayDetail }}</p>
          </div>
        </div>
      </div>
    </transition>

    <header class="app-header">
      <div>
        <h1>StockTool 云端量化 V1.0.6</h1>
        <p>一键上传 CSV · 自动回测 · 智能洞察</p>
      </div>
      <div class="header-actions">
        <div v-if="showUserInfo" class="user-actions">
          <div class="user-info">
            <span class="user-info-label">登录用户</span>
            <strong>
              {{ usernameDisplay }}
              <span class="user-role" v-if="isAdminUser">（管理员）</span>
            </strong>
          </div>
          <button class="secondary" type="button" @click="openPasswordModal">修改密码</button>
          <button class="secondary danger" type="button" @click="handleLogout">退出登录</button>
          <button v-if="showAdminLink" class="secondary" type="button" @click="goAdmin">管理员后台</button>
        </div>
        <div class="theme-toggle" role="group" aria-label="颜色模式">
          <div class="theme-options">
            <button
              v-for="option in themeOptions"
              :key="option.value"
              type="button"
              class="theme-chip"
              :class="{ active: themeMode === option.value }"
              :aria-pressed="themeMode === option.value"
              @click="setThemeMode(option.value)"
            >
              {{ option.label }}
            </button>
          </div>
        </div>
      </div>
    </header>
    <transition name="fade">
      <div
        v-if="showPasswordModal"
        class="password-modal-backdrop"
        @click.self="closePasswordModal"
        role="presentation"
      >
        <div
          class="password-modal-card"
          role="dialog"
          aria-modal="true"
          aria-labelledby="password-modal-title"
        >
          <header class="password-modal-header">
            <h3 id="password-modal-title">修改登录密码</h3>
            <button class="secondary" type="button" aria-label="关闭" @click="closePasswordModal">
              ✕
            </button>
          </header>
          <form class="password-modal-form" @submit.prevent="submitPasswordChange">
            <label>
              <span>当前密码</span>
              <input
                type="password"
                v-model="passwordForm.currentPassword"
                placeholder="请输入当前密码"
                autocomplete="current-password"
                :disabled="passwordModalLoading"
              />
            </label>
            <label>
              <span>新密码</span>
              <input
                type="password"
                v-model="passwordForm.newPassword"
                placeholder="至少 6 个字符"
                autocomplete="new-password"
                :disabled="passwordModalLoading"
              />
            </label>
            <label>
              <span>确认新密码</span>
              <input
                type="password"
                v-model="passwordForm.confirmPassword"
                placeholder="再次输入新密码"
                autocomplete="new-password"
                :disabled="passwordModalLoading"
              />
            </label>
            <div class="password-modal-actions">
              <button class="secondary" type="button" @click="closePasswordModal" :disabled="passwordModalLoading">
                取消
              </button>
              <button class="primary" type="submit" :disabled="passwordModalLoading">
                {{ passwordModalLoading ? '保存中…' : '保存' }}
              </button>
            </div>
            <p v-if="passwordModalError" class="password-modal-error" role="alert">
              {{ passwordModalError }}
            </p>
          </form>
        </div>
      </div>
    </transition>
    <transition name="fade">
      <div v-if="showTopTip" class="app-tip-wrapper">
        <section class="app-tip-card">
        <div class="app-tip-content">
          <strong>版本更新提醒</strong>
          <ul class="tip-list">
            <li>动态资金模块默认启用，首单保持 1 手并在界面直接说明各类金额上限的作用。</li>
            <li>“当前持仓浮动盈亏”与对冲统计改用真实的持仓成本和市值差值。</li>
            <li>百分比类指标（如加仓步长、平均摊低）修正为 100 倍显示，避免误导。</li>
          </ul>
        </div>
        <button type="button" class="tip-close" @click="showTopTip = false" aria-label="关闭提示">✕</button>
      </section>
    </div>
    </transition>

    <main class="app-main">
      <aside class="sidebar">
        <ConfigPanel
          :busy="isRunning"
          @run="handleRun"
          @block="handleOverlayBlock"
          @unblock="handleOverlayUnblock"
        />
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
            <div>
              <h3>行情与权益走势</h3>
              <p v-if="lastConfigMeta.assetLabel" class="panel-subtitle">标的：{{ lastConfigMeta.assetLabel }}</p>
            </div>
            <span v-if="backtestResults.length" class="tag">{{ backtestResults[0].title }}</span>
          </div>
          <ChartPanel
            :marketData="marketData"
            :equityData="selectedEquity"
            :dynamicEquity="dynamicEquity"
            :investmentMain="investmentCurveMain"
            :investmentHedge="investmentCurveHedge"
            :theme="resolvedTheme"
          />
        </div>
        <div
          class="card ai-card"
          :class="{ expanded: aiCardExpanded }"
          @mouseenter="handleAIMouseEnter"
          @mouseleave="handleAIMouseLeave"
          @focusin="handleAIMouseEnter"
          @focusout="handleAIMouseLeave"
        >
          <div class="panel-header">
            <div>
              <h3>当前股票趋势解读</h3>
              <p class="panel-subtitle">深度分析 + 未来走势预测</p>
            </div>
            <div class="ai-panel-actions">
              <button class="secondary" type="button" @click="requestAIInsight(true)" :disabled="aiLoading || !hasData || !aiEnabled">
                {{ aiLoading ? '分析中…' : '重新分析' }}
              </button>
              <div class="ai-toggle" :class="{ 'ai-toggle--active': aiEnabled }">
                <label class="ai-switch" aria-label="切换 AI 趋势解读">
                  <input type="checkbox" :checked="aiEnabled" @change="handleAIToggle" />
                  <span class="ai-slider"></span>
                </label>
                <span>{{ aiEnabled ? 'AI 已启用' : 'AI 已关闭' }}</span>
              </div>
            </div>
          </div>
          <div class="ai-body">
            <div v-if="!aiEnabled" class="ai-disabled-message">
              股票趋势解读为关闭状态，如需使用请打开卡片右上角开关。
            </div>
            <template v-else>
              <div v-if="!hasData" class="ai-empty">请先上传行情数据并运行一次回测</div>
              <div v-else-if="aiLoading" class="ai-loading">
                <span class="spinner" aria-hidden="true"></span>
                正在分析当前股票，请稍等…
              </div>
              <div v-else-if="aiError" class="ai-error">AI 调用失败：{{ aiError }}</div>
              <div v-else-if="aiResult?.analysis" class="ai-content">
                <ul class="ai-stats" v-if="aiResult.stats">
                  <li>区间：{{ aiResult.stats.date_range }}</li>
                  <li>区间涨跌：{{ aiResult.stats.total_return_pct }}%</li>
                  <li>波动率：{{ aiResult.stats.volatility_pct }}%</li>
                  <li>平均成交量：{{ aiResult.stats.avg_volume }}</li>
                </ul>
                <div
                  class="ai-analysis-text markdown-body"
                  v-if="aiRenderedHtml"
                  v-html="aiRenderedHtml"
                ></div>
                <div v-else class="ai-empty">暂无分析内容</div>
              </div>
              <div v-else class="ai-empty">暂无分析结果</div>
            </template>
          </div>
        </div>
        <AnalyticsPanel
          :results="backtestResults"
          :hasData="hasData"
          :initialCapital="lastConfigMeta.initialCapital"
          :multiFreqs="lastConfigMeta.multiFreqs"
          :theme="resolvedTheme"
          @select-strategy="handleSelectStrategy"
        />
      </section>
    </main>
  </div>
</template>
