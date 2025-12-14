<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLineGutter } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { autocompletion } from '@codemirror/autocomplete';
import { lintGutter, linter } from '@codemirror/lint';
import { StreamLanguage } from '@codemirror/language';
import { placeholder as cmPlaceholder } from '@codemirror/view';

const props = defineProps({
  modelValue: { type: String, default: '' },
  placeholder: { type: String, default: '' },
  minHeight: { type: String, default: '140px' },
  lintLogs: { type: Array, default: () => [] },
  readOnly: { type: Boolean, default: false },
});

const emit = defineEmits(['update:modelValue']);

const host = ref(null);
let view = null;

const KEYWORDS = ['AND', 'OR', 'NOT'];
const BUILTINS = [
  'CLOSE',
  'OPEN',
  'HIGH',
  'LOW',
  'VOL',
  'EMA',
  'MA',
  'REF',
  'COUNT',
  'LLV',
  'HHV',
  'CROSS',
  'BARSLAST',
  'IF',
  'B_COND',
  'S_COND',
];

const completionSource = (context) => {
  const word = context.matchBefore(/[A-Za-z_][A-Za-z0-9_]*/);
  if (!word || (word.from === word.to && !context.explicit)) return null;
  const options = [...KEYWORDS, ...BUILTINS]
    .map((label) => ({
      label,
      type: KEYWORDS.includes(label) ? 'keyword' : 'function',
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
  return {
    from: word.from,
    options,
  };
};

const parseLintLogs = (logs = []) => {
  const out = [];
  logs.forEach((raw) => {
    const text = String(raw || '').trim();
    if (!text) return;
    const lineMatch = text.match(/第\\s*(\\d+)\\s*行/);
    const line = lineMatch ? Number(lineMatch[1]) : null;
    const severity = text.includes('警告') ? 'warning' : text.includes('错误') ? 'error' : 'info';
    out.push({ line, message: text, severity });
  });
  return out;
};

const lintExtension = linter((viewInstance) => {
  const diagnostics = [];
  const parsed = parseLintLogs(props.lintLogs);
  const doc = viewInstance.state.doc;
  parsed.forEach((item) => {
    if (!item.line || item.line <= 0) return;
    if (item.line > doc.lines) return;
    const ln = doc.line(item.line);
    diagnostics.push({
      from: ln.from,
      to: ln.to,
      severity: item.severity,
      message: item.message,
    });
  });
  return diagnostics;
});

const tdxMode = StreamLanguage.define(
  {
    startState() {
      return { inComment: false };
    },
    token(stream, _state) {
      if (stream.match('//')) {
        stream.skipToEnd();
        return 'comment';
      }
      if (stream.eatSpace()) return null;
      const ch = stream.peek();
      if (ch === '"' || ch === "'") {
        stream.next();
        while (!stream.eol()) {
          const c = stream.next();
          if (c === ch) break;
          if (c === '\\\\') stream.next();
        }
        return 'string';
      }
      if (stream.match(/(:=|<=|>=|==|!=|[+*/(),;^%<>=.\\[\\]\\-])/)) {
        return 'operator';
      }
      if (stream.match(/\\d+(?:\\.\\d+)?/)) return 'number';
      if (stream.match(/[A-Za-z_][A-Za-z0-9_]*/)) {
        const cur = stream.current();
        const upper = cur.toUpperCase();
        if (KEYWORDS.includes(upper)) return 'keyword';
        if (BUILTINS.includes(upper)) return 'variableName.special';
        return 'variableName';
      }
      stream.next();
      return null;
    },
    languageData: {
      commentTokens: { line: '//' },
    },
  },
);

const theme = EditorView.theme(
  {
    '&': {
      background: 'rgba(var(--surface-rgb), 0.45)',
      border: '1px solid rgba(148, 163, 184, 0.18)',
      borderRadius: '10px',
      fontFamily: "'SFMono-Regular', Consolas, monospace",
      fontSize: '0.82rem',
      minHeight: props.minHeight,
    },
    '.cm-scroller': {
      overflow: 'auto',
    },
    '.cm-content': {
      padding: '10px 12px',
    },
    '.cm-gutters': {
      background: 'transparent',
      borderRight: '1px solid rgba(148, 163, 184, 0.12)',
      color: 'var(--text-secondary)',
    },
    '.cm-activeLineGutter': {
      backgroundColor: 'rgba(59, 130, 246, 0.08)',
    },
    '.cm-activeLine': {
      backgroundColor: 'rgba(59, 130, 246, 0.06)',
    },
    '.cm-keyword': {
      color: 'var(--tone-2, var(--accent-secondary))',
      fontWeight: '600',
    },
    '.cm-comment': {
      color: 'rgba(148, 163, 184, 0.85)',
      fontStyle: 'italic',
    },
    '.cm-number': {
      color: 'var(--tone-5, #f59e0b)',
    },
    '.cm-string': {
      color: 'var(--tone-6, #34d399)',
    },
    '.cm-operator': {
      color: 'rgba(226, 232, 240, 0.92)',
    },
    '.cm-variableName': {
      color: 'var(--text-primary)',
    },
    '.cm-variableName-special': {
      color: 'var(--tone-1, var(--accent))',
      fontWeight: '600',
    },
    '.cm-tooltip': {
      background: 'rgba(var(--overlay-rgb), 0.92)',
      border: '1px solid rgba(148, 163, 184, 0.25)',
      color: 'var(--text-primary)',
    },
    '.cm-diagnostic': {
      padding: '6px 10px',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
      fontSize: '0.85rem',
      maxWidth: '420px',
    },
  },
  { dark: true },
);

const setDoc = (text) => {
  if (!view) return;
  const current = view.state.doc.toString();
  if (current === text) return;
  view.dispatch({
    changes: { from: 0, to: current.length, insert: text },
  });
};

onMounted(() => {
  const startState = EditorState.create({
    doc: props.modelValue || '',
    extensions: [
      lineNumbers(),
      highlightActiveLineGutter(),
      keymap.of([indentWithTab, ...defaultKeymap, ...historyKeymap]),
      history(),
      autocompletion({ override: [completionSource] }),
      lintGutter(),
      lintExtension,
      tdxMode,
      theme,
      EditorView.editable.of(!props.readOnly),
      props.placeholder ? cmPlaceholder(props.placeholder) : [],
      EditorView.updateListener.of((update) => {
        if (!update.docChanged) return;
        emit('update:modelValue', update.state.doc.toString());
      }),
    ],
  });
  view = new EditorView({
    state: startState,
    parent: host.value,
  });
});

onBeforeUnmount(() => {
  if (view) {
    view.destroy();
    view = null;
  }
});

watch(
  () => props.modelValue,
  (val) => {
    setDoc(val || '');
  },
);

watch(
  () => props.lintLogs,
  () => {
    if (view) view.dispatch({});
  },
  { deep: true },
);
</script>

<template>
  <div ref="host"></div>
</template>
