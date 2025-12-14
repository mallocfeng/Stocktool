<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';

const router = useRouter();
const form = ref({ username: '', password: '', confirmPassword: '' });
const submitting = ref(false);
const error = ref('');
const info = ref('');

const handleSubmit = async () => {
  error.value = '';
  info.value = '';
  if (!form.value.username.trim()) {
    error.value = '用户名不能为空';
    return;
  }
  if (!form.value.password) {
    error.value = '密码不能为空';
    return;
  }
  if (form.value.password !== form.value.confirmPassword) {
    error.value = '两次密码输入不一致';
    return;
  }
  submitting.value = true;
  try {
    await axios.post('/register', {
      username: form.value.username.trim(),
      password: form.value.password,
    });
    info.value = '注册成功，正在跳转登录…';
    setTimeout(() => {
      router.replace({ path: '/login', query: { registered: 1 } });
    }, 600);
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '注册失败';
  } finally {
    submitting.value = false;
  }
};
</script>

<template>
  <div class="auth-page">
    <section class="login-card">
      <h1>StockTool 用户注册</h1>
      <p class="panel-subtitle">创建内部账号即可登录回测与管理界面</p>
      <form class="login-form" @submit.prevent="handleSubmit">
        <label>
          用户名
          <input v-model="form.username" type="text" placeholder="请输入用户名" autocomplete="username" />
        </label>
        <label>
          密码
          <input v-model="form.password" type="password" placeholder="请输入密码" autocomplete="new-password" />
        </label>
        <label>
          确认密码
          <input v-model="form.confirmPassword" type="password" placeholder="再次输入密码" autocomplete="new-password" />
        </label>
        <button class="primary" type="submit" :disabled="submitting">{{ submitting ? '注册中…' : '注册账号' }}</button>
        <p v-if="error" class="form-error">{{ error }}</p>
        <p v-if="info" class="form-success">{{ info }}</p>
      </form>
      <p class="login-hint">
        已有账号？<router-link to="/login">去登录</router-link>
      </p>
    </section>
  </div>
</template>
