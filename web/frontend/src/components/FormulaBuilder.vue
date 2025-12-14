<script setup>
import { computed, reactive, ref, watch } from 'vue';
import axios from 'axios';
import { indicatorBlocks, conditionBlocks } from '../lib/formulaTemplates';
import TdxCodeEditor from './TdxCodeEditor.vue';

const props = defineProps({
  show: { type: Boolean, default: false },
  initialFormula: { type: String, default: '' },
  csvPath: { type: String, default: '' },
});

const emit = defineEmits(['close', 'apply']);

const indicatorMap = Object.fromEntries(indicatorBlocks.map((block) => [block.id, block]));
const conditionMap = Object.fromEntries(conditionBlocks.map((block) => [block.id, block]));

const createEntryValues = (fields = []) => {
  const values = {};
  fields.forEach((field) => {
    if (field.default !== undefined) {
      values[field.key] = field.default;
    } else if (field.type === 'number') {
      values[field.key] = field.min ?? 0;
    } else {
      values[field.key] = '';
    }
  });
  return values;
};

const uniqId = (() => {
  let seed = 0;
  return () => {
    seed += 1;
    return `entry_${Date.now()}_${seed}`;
  };
})();

const builderState = reactive({
  indicators: [],
  buyConditions: [],
  sellConditions: [],
  buyJoiner: 'AND',
  sellJoiner: 'OR',
  extraScript: '',
  customBuyExpr: '',
  customSellExpr: '',
});

const validationLogs = ref([]);
const validationSummary = ref('');

const indicatorSelection = ref(indicatorBlocks[0]?.id || '');
const buyConditionSelection = ref(conditionBlocks[0]?.id || '');
const sellConditionSelection = ref(conditionBlocks[0]?.id || '');

const resetState = () => {
  builderState.indicators = [];
  builderState.buyConditions = [];
  builderState.sellConditions = [];
  builderState.buyJoiner = 'AND';
  builderState.sellJoiner = 'OR';
  builderState.extraScript = '';
  builderState.customBuyExpr = '';
  builderState.customSellExpr = '';
};

const extractExpr = (raw, key) => {
  const pattern = new RegExp(`${key}\\s*:?=\\s*([\\s\\S]*?);`, 'i');
  const match = raw.match(pattern);
  return match ? match[1].trim() : '';
};

const hydrateFromFormula = () => {
  resetState();
  validationLogs.value = [];
  validationSummary.value = '';
  const raw = props.initialFormula || '';
  if (!raw.trim()) return;
  builderState.customBuyExpr = extractExpr(raw, 'B_COND');
  builderState.customSellExpr = extractExpr(raw, 'S_COND');
  const lines = raw.split(/\r?\n/);
  const extraLines = lines.filter((line) => {
    const trimmed = line.trim().toUpperCase();
    if (!trimmed) return false;
    if (trimmed.startsWith('B_COND')) return false;
    if (trimmed.startsWith('S_COND')) return false;
    return true;
  });
  builderState.extraScript = extraLines.join('\n').trim();
};

watch(
  () => props.show,
  (visible) => {
    if (visible) {
      hydrateFromFormula();
    }
  }
);

watch(
  () => props.initialFormula,
  () => {
    if (props.show) {
      hydrateFromFormula();
    }
  }
);

const addIndicatorEntry = () => {
  const block = indicatorMap[indicatorSelection.value];
  if (!block) return;
  builderState.indicators.push({
    uid: uniqId(),
    blockId: block.id,
    values: createEntryValues(block.fields),
  });
};

const addConditionEntry = (target) => {
  const selection = target === 'sell' ? sellConditionSelection : buyConditionSelection;
  const block = conditionMap[selection.value];
  if (!block) return;
  const entry = {
    uid: uniqId(),
    blockId: block.id,
    values: createEntryValues(block.fields),
  };
  if (target === 'sell') {
    builderState.sellConditions.push(entry);
  } else {
    builderState.buyConditions.push(entry);
  }
};

const removeEntry = (collection, uid) => {
  const idx = collection.findIndex((item) => item.uid === uid);
  if (idx >= 0) {
    collection.splice(idx, 1);
  }
};

const sanitizeExpression = (expr) => expr.trim().replace(/;$/g, '').trim();

const ensureStatement = (text) => {
  const trimmed = text.trim();
  if (!trimmed) return '';
  return trimmed.endsWith(';') ? trimmed : `${trimmed};`;
};

const toStatementLines = (rendered) => {
  const arr = Array.isArray(rendered) ? rendered : [rendered];
  return arr
    .map((line) => (typeof line === 'string' ? line.trim() : ''))
    .filter((line) => Boolean(line));
};

const indicatorLines = computed(() =>
  builderState.indicators.flatMap((entry) => {
    const block = indicatorMap[entry.blockId];
    if (!block) return [];
    return toStatementLines(block.build(entry.values)).map((line) => ensureStatement(line));
  })
);



const buildConditionExpression = (entry) => {
  const block = conditionMap[entry.blockId];
  if (!block) return '';
  const expr = block.build(entry.values) || '';
  return sanitizeExpression(expr);
};

const joinConditions = (items, joiner, fallback) => {
  if (!items.length) return fallback;
  const clauses = items
    .map((entry) => buildConditionExpression(entry))
    .filter(Boolean)
    .map((expr) => (expr.includes(' AND ') || expr.includes(' OR ') ? `(${expr})` : expr));
  if (!clauses.length) return fallback;
  return clauses.join(` ${joiner} `);
};

const buyExpression = computed(() => {
  const custom = sanitizeExpression(builderState.customBuyExpr || '');
  if (custom) return custom;
  const fallback = builderState.buyJoiner === 'AND' ? 'TRUE' : 'FALSE';
  return joinConditions(builderState.buyConditions, builderState.buyJoiner, fallback);
});

const sellExpression = computed(() => {
  const custom = sanitizeExpression(builderState.customSellExpr || '');
  if (custom) return custom;
  const fallback = builderState.sellJoiner === 'AND' ? 'TRUE' : 'FALSE';
  return joinConditions(builderState.sellConditions, builderState.sellJoiner, fallback);
});

const formulaPreview = computed(() => {
  const sections = [];
  if (builderState.extraScript.trim()) {
    sections.push(builderState.extraScript.trim());
  }
  if (indicatorLines.value.length) {
    sections.push(indicatorLines.value.join('\n'));
  }
  sections.push(`B_COND := ${buyExpression.value};`);
  sections.push(`S_COND := ${sellExpression.value};`);
  return sections.join('\n');
});

const getIndicatorPreview = (entry) => {
  const block = indicatorMap[entry.blockId];
  if (!block) return '';
  return toStatementLines(block.build(entry.values)).join('\n');
};

const getConditionPreview = (entry) => buildConditionExpression(entry);

const closeBuilder = () => emit('close');
const applyFormula = () => emit('apply', formulaPreview.value);

const copyHint = ref('');
const copyFormula = async () => {
  if (typeof navigator === 'undefined' || !navigator.clipboard) {
    copyHint.value = '浏览器无法复制';
    setTimeout(() => (copyHint.value = ''), 1500);
    return;
  }
  try {
    await navigator.clipboard.writeText(formulaPreview.value);
    copyHint.value = '已复制';
  } catch (err) {
    console.error(err);
    copyHint.value = '复制失败';
  } finally {
    setTimeout(() => (copyHint.value = ''), 1800);
  }
};

const validateFormula = async () => {
  if (!props.csvPath) {
    alert('请先上传/选择 CSV 行情数据后再校验公式。');
    return;
  }
  validationSummary.value = '校验中…';
  validationLogs.value = [];
  try {
    const res = await axios.post('/formula/validate', {
      csv_path: props.csvPath,
      formula: formulaPreview.value,
    });
    const logs = res.data?.logs || [];
    validationLogs.value = Array.isArray(logs) ? logs : [String(logs)];
    const buyCount = res.data?.buy_count ?? 0;
    const sellCount = res.data?.sell_count ?? 0;
    validationSummary.value = `校验完成：买入信号 ${buyCount}，卖出信号 ${sellCount}`;
  } catch (e) {
    validationSummary.value = '校验失败';
    validationLogs.value = [e.response?.data?.detail || e.message];
  }
};

const previewDoc = computed({
  get: () => formulaPreview.value,
  set: () => {},
});
</script>

<template>
  <teleport to="body">
    <div v-if="props.show" class="modal-overlay" @click.self="closeBuilder">
      <div class="formula-builder card">
        <header class="builder-header">
          <div>
            <h3>通达信公式向导</h3>
            <p>自由组合指标与买卖条件，系统自动拼装为合法脚本。</p>
          </div>
          <button type="button" class="secondary" @click="closeBuilder">关闭</button>
        </header>

        <div class="builder-body">
          <section class="builder-section">
            <h4>其他脚本（可选）</h4>
            <TdxCodeEditor
              v-model="builderState.extraScript"
              minHeight="180px"
              placeholder="在此粘贴任意赋值语句、函数等。B_COND/S_COND 不必写在这里。"
            />
          </section>

          <section class="builder-section">
            <div class="section-head">
              <h4>指标定义</h4>
              <div class="builder-toolbar">
                <select v-model="indicatorSelection">
                  <option v-for="block in indicatorBlocks" :key="block.id" :value="block.id">
                    {{ block.label }}
                  </option>
                </select>
                <button type="button" class="secondary" @click="addIndicatorEntry">添加指标</button>
              </div>
            </div>
            <div v-if="!builderState.indicators.length" class="empty-hint">尚未添加指标，可直接点击上方按钮。</div>
            <div v-else class="entry-list">
              <div v-for="entry in builderState.indicators" :key="entry.uid" class="entry-card">
                <header class="entry-header">
                  <div>
                    <strong>{{ indicatorMap[entry.blockId]?.label || '未命名指标' }}</strong>
                    <p class="entry-desc">{{ indicatorMap[entry.blockId]?.description }}</p>
                  </div>
                  <button type="button" class="secondary" @click="removeEntry(builderState.indicators, entry.uid)">移除</button>
                </header>
                <div class="entry-fields">
                  <div v-for="field in indicatorMap[entry.blockId]?.fields || []" :key="field.key" class="field">
                    <span>{{ field.label }}</span>
                    <template v-if="field.type === 'select'">
                      <select v-model="entry.values[field.key]">
                        <option v-for="opt in field.options" :key="opt.value" :value="opt.value">
                          {{ opt.label }}
                        </option>
                      </select>
                    </template>
                    <template v-else-if="field.type === 'textarea'">
                      <textarea :rows="field.rows || 2" v-model="entry.values[field.key]"></textarea>
                    </template>
                    <template v-else>
                      <input
                        :type="field.type === 'number' ? 'number' : 'text'"
                        v-model="entry.values[field.key]"
                        :min="field.min"
                        :step="field.step"
                        :placeholder="field.placeholder"
                      />
                    </template>
                    <small v-if="field.hint" class="field-hint">{{ field.hint }}</small>
                  </div>
                </div>
                <pre class="entry-preview">{{ getIndicatorPreview(entry) || '（暂无输出）' }}</pre>
              </div>
            </div>
          </section>

          <section class="builder-section">
            <div class="section-head">
              <h4>买入条件</h4>
              <div class="joiner-row">
                <label>条件连接</label>
                <select v-model="builderState.buyJoiner">
                  <option value="AND">所有条件同时满足 (AND)</option>
                  <option value="OR">任意条件满足 (OR)</option>
                </select>
              </div>
            </div>
            <div class="builder-toolbar">
              <select v-model="buyConditionSelection">
                <option v-for="block in conditionBlocks" :key="block.id" :value="block.id">{{ block.label }}</option>
              </select>
              <button type="button" class="secondary" @click="addConditionEntry('buy')">添加条件</button>
            </div>
            <div v-if="!builderState.buyConditions.length" class="empty-hint">暂无条件，可通过自定义表达式补充。</div>
            <div class="entry-list" v-else>
              <div v-for="entry in builderState.buyConditions" :key="entry.uid" class="entry-card">
                <header class="entry-header">
                  <div>
                    <strong>{{ conditionMap[entry.blockId]?.label || '未命名' }}</strong>
                    <p class="entry-desc">{{ conditionMap[entry.blockId]?.description }}</p>
                  </div>
                  <button type="button" class="secondary" @click="removeEntry(builderState.buyConditions, entry.uid)">移除</button>
                </header>
                <div class="entry-fields">
                  <div v-for="field in conditionMap[entry.blockId]?.fields || []" :key="field.key" class="field">
                    <span>{{ field.label }}</span>
                    <template v-if="field.type === 'select'">
                      <select v-model="entry.values[field.key]">
                        <option v-for="opt in field.options" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
                      </select>
                    </template>
                    <template v-else-if="field.type === 'textarea'">
                      <textarea :rows="field.rows || 2" v-model="entry.values[field.key]"></textarea>
                    </template>
                    <template v-else>
                      <input
                        :type="field.type === 'number' ? 'number' : 'text'"
                        v-model="entry.values[field.key]"
                        :min="field.min"
                        :step="field.step"
                        :placeholder="field.placeholder"
                      />
                    </template>
                    <small v-if="field.hint" class="field-hint">{{ field.hint }}</small>
                  </div>
                </div>
                <pre class="entry-preview">{{ getConditionPreview(entry) || '（未生成条件表达式）' }}</pre>
              </div>
            </div>
            <TdxCodeEditor
              v-model="builderState.customBuyExpr"
              minHeight="120px"
              placeholder="可选：直接输入完整 B_COND 表达式。若填写，则忽略上方所有条件。"
            />
          </section>

          <section class="builder-section">
            <div class="section-head">
              <h4>卖出条件</h4>
              <div class="joiner-row">
                <label>条件连接</label>
                <select v-model="builderState.sellJoiner">
                  <option value="AND">所有条件同时满足 (AND)</option>
                  <option value="OR">任意条件满足 (OR)</option>
                </select>
              </div>
            </div>
            <div class="builder-toolbar">
              <select v-model="sellConditionSelection">
                <option v-for="block in conditionBlocks" :key="block.id" :value="block.id">{{ block.label }}</option>
              </select>
              <button type="button" class="secondary" @click="addConditionEntry('sell')">添加条件</button>
            </div>
            <div v-if="!builderState.sellConditions.length" class="empty-hint">暂无条件，可保持为空表示“不卖出”。</div>
            <div class="entry-list" v-else>
              <div v-for="entry in builderState.sellConditions" :key="entry.uid" class="entry-card">
                <header class="entry-header">
                  <div>
                    <strong>{{ conditionMap[entry.blockId]?.label || '未命名' }}</strong>
                    <p class="entry-desc">{{ conditionMap[entry.blockId]?.description }}</p>
                  </div>
                  <button type="button" class="secondary" @click="removeEntry(builderState.sellConditions, entry.uid)">移除</button>
                </header>
                <div class="entry-fields">
                  <div v-for="field in conditionMap[entry.blockId]?.fields || []" :key="field.key" class="field">
                    <span>{{ field.label }}</span>
                    <template v-if="field.type === 'select'">
                      <select v-model="entry.values[field.key]">
                        <option v-for="opt in field.options" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
                      </select>
                    </template>
                    <template v-else-if="field.type === 'textarea'">
                      <textarea :rows="field.rows || 2" v-model="entry.values[field.key]"></textarea>
                    </template>
                    <template v-else>
                      <input
                        :type="field.type === 'number' ? 'number' : 'text'"
                        v-model="entry.values[field.key]"
                        :min="field.min"
                        :step="field.step"
                        :placeholder="field.placeholder"
                      />
                    </template>
                    <small v-if="field.hint" class="field-hint">{{ field.hint }}</small>
                  </div>
                </div>
                <pre class="entry-preview">{{ getConditionPreview(entry) || '（未生成条件表达式）' }}</pre>
              </div>
            </div>
            <TdxCodeEditor
              v-model="builderState.customSellExpr"
              minHeight="120px"
              placeholder="可选：直接输入完整 S_COND 表达式。若填写，则忽略上方所有条件。"
            />
          </section>

          <section class="builder-section preview-section">
            <div class="preview-header">
              <h4>公式预览</h4>
              <span class="preview-hint">{{ validationSummary || copyHint || '可复制或直接写回配置面板' }}</span>
            </div>
            <TdxCodeEditor
              v-model="previewDoc"
              :lintLogs="validationLogs"
              minHeight="220px"
              readOnly
            />
            <div v-if="validationLogs.length" class="builder-log">
              <div v-for="(msg, idx) in validationLogs" :key="idx">{{ msg }}</div>
            </div>
            <div class="builder-actions">
              <button type="button" class="secondary" @click="validateFormula">校验</button>
              <button type="button" class="secondary" @click="copyFormula">复制</button>
              <button type="button" class="primary" @click="applyFormula">填入配置</button>
            </div>
          </section>
        </div>
      </div>
    </div>
  </teleport>
</template>
