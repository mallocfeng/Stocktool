<script setup>
import { reactive, ref } from 'vue';
import axios from 'axios';

const props = defineProps({
  busy: { type: Boolean, default: false },
});

const emit = defineEmits(['run']);

const config = reactive({
  csv_path: '',
  formula: 'B_COND := C > REF(C, 1);',
  initial_capital: 100000,
  fee_rate: 0.0005,
  multi_freqs: 'D,W,M',
  strategy_text: '',
  strategies: {
    fixed: { enabled: true, periods: '5,10,20' },
    tpsl: { enabled: true, tp: 10, sl: 5 },
    dca: { enabled: false, size: 5, target: 20 },
    grid: { enabled: false, pct: 5, cash: 1000, limit: '', accumulate: true },
  },
});

const fileInput = ref(null);
const uploading = ref(false);

const triggerFileSelect = () => {
  fileInput.value?.click();
};

const handleFileUpload = async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  uploading.value = true;
  try {
    const res = await axios.post('http://127.0.0.1:8000/upload', formData);
    config.csv_path = res.data.path;
    autoRunBacktest();
  } catch (e) {
    alert('上传失败：' + (e.response?.data?.detail || e.message));
  } finally {
    uploading.value = false;
  }
};

const checkFormula = () => {
  if (!config.formula || !config.formula.toUpperCase().includes('B_COND')) {
    alert('公式中必须包含 B_COND := ... 请补充买入条件。');
  } else {
    alert('公式已包含 B_COND，可运行回测。');
  }
};

const convertFromText = async () => {
  const desc = config.strategy_text.trim();
  if (!desc) {
    alert('请输入策略描述后再转换。');
    return;
  }
  try {
    const res = await axios.post('http://127.0.0.1:8000/analytics/nlp_formula', { text: desc });
    config.formula = res.data.formula || config.formula;
  } catch (e) {
    alert('转换失败：' + (e.response?.data?.detail || e.message));
  }
};

const buildStrategiesPayload = () => {
  const strategies = {};
  const toRatio = (value) => {
    const num = Number(value);
    if (Number.isNaN(num) || num <= 0) {
      return 0;
    }
    return num > 1 ? num / 100 : num;
  };
  if (config.strategies.fixed.enabled) {
    const parts = config.strategies.fixed.periods
      .split(',')
      .map((p) => parseInt(p.trim(), 10))
      .filter((n) => !Number.isNaN(n) && n > 0);
    if (parts.length) strategies.fixed = parts;
  }
  if (config.strategies.tpsl.enabled) {
    strategies.tpsl = {
      tp: toRatio(config.strategies.tpsl.tp),
      sl: toRatio(config.strategies.tpsl.sl),
    };
  }
  if (config.strategies.dca.enabled) {
    strategies.dca = {
      size: toRatio(config.strategies.dca.size),
      target: toRatio(config.strategies.dca.target),
    };
  }
  if (config.strategies.grid.enabled) {
    const limitRaw = (config.strategies.grid.limit ?? '').toString().trim();
    const parsedLimit = limitRaw && limitRaw.toLowerCase() !== 'none' ? parseInt(limitRaw, 10) : null;
    strategies.grid = {
      grid_pct: toRatio(config.strategies.grid.pct),
      single_cash: Number(config.strategies.grid.cash),
      max_grids: Number.isNaN(parsedLimit) ? null : parsedLimit,
      accumulate: Boolean(config.strategies.grid.accumulate),
    };
  }
  return strategies;
};

const autoRunBacktest = () => {
  if (props.busy) return;
  if (!config.csv_path) return;
  if (!config.formula || !config.formula.toUpperCase().includes('B_COND')) return;
  const strategies = buildStrategiesPayload();
  if (!Object.keys(strategies).length) return;
  runBacktest();
};

const runBacktest = () => {
  if (!config.csv_path) {
    alert('请先上传行情 CSV 文件。');
    return;
  }
  if (!config.formula || !config.formula.toUpperCase().includes('B_COND')) {
    alert('请先准备包含 B_COND 的通达信公式。');
    return;
  }
  const strategies = buildStrategiesPayload();
  if (!Object.keys(strategies).length) {
    alert('请至少启用一个策略模块。');
    return;
  }
  const payload = {
    csv_path: config.csv_path,
    formula: config.formula,
    initial_capital: Number(config.initial_capital),
    fee_rate: Number(config.fee_rate),
    strategies,
  };
  emit('run', {
    payload,
    meta: {
      initialCapital: Number(config.initial_capital),
      multiFreqs: config.multi_freqs,
    },
  });
};
</script>

<template>
  <div class="card config-panel">
    <h2 class="panel-title">策略配置</h2>

    <section class="form-section">
      <div class="section-title">行情数据</div>
      <div class="file-row">
        <input class="text-input" type="text" :value="config.csv_path" placeholder="请上传 CSV 文件" readonly />
        <button class="secondary" type="button" @click="triggerFileSelect" :disabled="uploading">
          {{ uploading ? '上传中…' : '选择文件' }}
        </button>
        <input ref="fileInput" type="file" class="hidden-input" accept=".csv" @change="handleFileUpload" />
      </div>
    </section>

    <section class="form-section grid">
      <label class="field">
        <span>初始资金</span>
        <input type="number" v-model="config.initial_capital" min="0" />
      </label>
      <label class="field">
        <span>手续费（单边）</span>
        <input type="number" v-model="config.fee_rate" step="0.0001" />
      </label>
      <label class="field">
        <span>多周期 (如 D,W,M)</span>
        <input type="text" v-model="config.multi_freqs" />
      </label>
    </section>

    <section class="form-section">
      <div class="section-title">通达信公式</div>
      <textarea v-model="config.formula" rows="6" class="code-area"></textarea>
      <div class="button-row">
        <button type="button" class="secondary" @click="checkFormula">检查公式</button>
        <button type="button" class="secondary" @click="convertFromText">自然语言转公式</button>
      </div>
      <input type="text" class="text-input" placeholder="例如：5日均线金叉20日配合放量" v-model="config.strategy_text" />
    </section>

    <section class="form-section">
      <div class="section-title">策略模块</div>
      <div class="strategy-card" v-for="module in ['fixed','tpsl','dca','grid']" :key="module">
        <template v-if="module==='fixed'">
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.fixed.enabled" />
            <span>固定周期持有</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.fixed.enabled">
            <label class="field">
              <span>周期（逗号分隔）</span>
              <input type="text" v-model="config.strategies.fixed.periods" />
            </label>
          </div>
        </template>
        <template v-else-if="module==='tpsl'">
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.tpsl.enabled" />
            <span>止盈 / 止损</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.tpsl.enabled">
            <label class="field">
              <span>止盈 %</span>
              <input type="number" v-model="config.strategies.tpsl.tp" />
            </label>
            <label class="field">
              <span>止损 %</span>
              <input type="number" v-model="config.strategies.tpsl.sl" />
            </label>
          </div>
        </template>
        <template v-else-if="module==='dca'">
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.dca.enabled" />
            <span>定投模式</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.dca.enabled">
            <label class="field">
              <span>定投比例 %</span>
              <input type="number" v-model="config.strategies.dca.size" />
            </label>
            <label class="field">
              <span>目标收益 %</span>
              <input type="number" v-model="config.strategies.dca.target" />
            </label>
          </div>
        </template>
        <template v-else>
          <label class="checkbox-row">
            <input type="checkbox" v-model="config.strategies.grid.enabled" />
            <span>网格策略</span>
          </label>
          <div class="sub-grid" v-if="config.strategies.grid.enabled">
            <label class="field">
              <span>网格 %</span>
              <input type="number" v-model="config.strategies.grid.pct" />
            </label>
            <label class="field">
              <span>单网资金</span>
              <input type="number" v-model="config.strategies.grid.cash" />
            </label>
            <label class="field">
              <span>最大网格数</span>
              <input type="number" v-model="config.strategies.grid.limit" />
            </label>
            <label class="checkbox-row">
              <input type="checkbox" v-model="config.strategies.grid.accumulate" />
              <span>累积份额</span>
            </label>
          </div>
        </template>
      </div>
    </section>

    <button class="primary run-button" type="button" @click="runBacktest" :disabled="props.busy">
      {{ props.busy ? '回测执行中…' : '开始回测' }}
    </button>
  </div>
</template>
