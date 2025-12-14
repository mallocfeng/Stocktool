import { ref, computed, watch } from 'vue';

const THEME_STORAGE_KEY = 'stocktool_theme_mode';
const themeOptions = [
  { label: '自动', value: 'auto' },
  { label: '浅色', value: 'light' },
  { label: '深色', value: 'dark' },
  { label: '莫兰迪', value: 'morandi' },
];

const themeMode = ref('auto');
const systemPrefersDark = ref(true);
const resolvedTheme = computed(() => {
  if (themeMode.value === 'auto') {
    return systemPrefersDark.value ? 'dark' : 'light';
  }
  return themeMode.value;
});

const applyTheme = (theme) => {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.theme = theme;
};

const persistThemeMode = (mode) => {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(THEME_STORAGE_KEY, mode);
  } catch {
    // ignore
  }
};

const setThemeMode = (mode) => {
  const allowed = themeOptions.map((opt) => opt.value);
  const resolved = allowed.includes(mode) ? mode : 'auto';
  themeMode.value = resolved;
  persistThemeMode(resolved);
};

let colorSchemeMedia = null;
const handleSystemThemeChange = (event) => {
  systemPrefersDark.value = !!event.matches;
};

const initTheme = () => {
  if (typeof window === 'undefined') {
    systemPrefersDark.value = true;
    return;
  }
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored && themeOptions.some((opt) => opt.value === stored)) {
      themeMode.value = stored;
    }
  } catch {
    // ignore
  }
  colorSchemeMedia = window.matchMedia('(prefers-color-scheme: dark)');
  if (colorSchemeMedia) {
    systemPrefersDark.value = colorSchemeMedia.matches;
    colorSchemeMedia.addEventListener('change', handleSystemThemeChange);
  }
};

if (typeof window !== 'undefined') {
  initTheme();
}

watch(
  resolvedTheme,
  (theme) => {
    applyTheme(theme);
  },
  { immediate: true },
);

export function useTheme() {
  return {
    themeMode,
    resolvedTheme,
    themeOptions,
    setThemeMode,
  };
}
