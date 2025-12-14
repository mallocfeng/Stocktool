<script setup>
import { ref, computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { login } from '../lib/useAuth';
import { useTheme } from '../lib/useTheme';

const router = useRouter();
const route = useRoute();
const form = ref({ username: '', password: '' });
const submitting = ref(false);
const error = ref('');
const registrationNotice = computed(() => route.query.registered === '1');
const disableAuth = import.meta.env.VITE_DISABLE_AUTH === 'true';
const { themeMode, themeOptions, setThemeMode } = useTheme();

if (disableAuth) {
  router.replace('/');
}

const handleSubmit = async () => {
  if (!form.value.username.trim() || !form.value.password) {
    error.value = '用户名/密码不能为空';
    return;
  }
  submitting.value = true;
  error.value = '';
  try {
    if (!disableAuth) {
      await login({
        username: form.value.username.trim(),
        password: form.value.password,
      });
    }
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/';
    router.replace(redirect);
  } catch (exc) {
    if (exc.response?.status === 401) {
      form.value.password = '';
    }
    error.value = exc.response?.data?.detail || exc.message || '登录失败';
  } finally {
    submitting.value = false;
  }
};
</script>

<template>
  <div class="auth-page">
    <div class="auth-header">
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
    <section class="login-card">
      <h1>StockTool 内部系统</h1>
      <p class="panel-subtitle">请使用内部账号登录以访问回测与管理功能</p>
      <form class="login-form" @submit.prevent="handleSubmit">
        <label>
          用户名
          <input v-model="form.username" type="text" placeholder="请输入用户名" autocomplete="username" />
        </label>
        <label>
          密码
          <input v-model="form.password" type="password" placeholder="请输入密码" autocomplete="current-password" />
        </label>
        <button class="primary" type="submit" :disabled="submitting">{{ submitting ? '登录中…' : '登录' }}</button>
        <p v-if="error" class="form-error">{{ error }}</p>
        <p v-else-if="registrationNotice" class="form-success">注册成功，请使用新账号登录</p>
      </form>
      <p class="login-hint">
        没有账号？<router-link to="/register">立即注册</router-link>。
      </p>
    </section>
  </div>
</template>
