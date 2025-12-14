<script setup>
import { reactive, ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { useAuth } from '../lib/useAuth';

const router = useRouter();
const users = ref([]);
const loadingUsers = ref(false);
const error = ref('');
const info = ref('');
const form = reactive({
  username: '',
  password: '',
  role: 'user',
  is_active: true,
});
const roleSelection = reactive({});
const tempDisableInputs = reactive({});
const saving = ref(false);
const auth = useAuth();
const currentUser = auth.currentUser;

const parseIso = (value) => {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString();
};

const syncRoleSelection = () => {
  Object.keys(roleSelection).forEach((key) => {
    delete roleSelection[key];
  });
  users.value.forEach((user) => {
    roleSelection[user.id] = user.role;
  });
};

const fetchUsers = async () => {
  loadingUsers.value = true;
  error.value = '';
  try {
    const res = await axios.get('/admin/users');
    users.value = res.data;
    syncRoleSelection();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '无法加载用户列表';
  } finally {
    loadingUsers.value = false;
  }
};

const handleLogout = async () => {
  await auth.logout();
  router.replace('/login');
};

const goHome = () => {
  router.push('/');
};

onMounted(fetchUsers);

const handleCreateUser = async () => {
  if (!form.username.trim() || !form.password) {
    error.value = '请填写用户名和密码';
    return;
  }
  saving.value = true;
  error.value = '';
  info.value = '';
  try {
    await axios.post('/admin/users', {
      username: form.username.trim(),
      password: form.password,
      role: form.role,
      is_active: form.is_active,
    });
    form.username = '';
    form.password = '';
    form.role = 'user';
    form.is_active = true;
    info.value = '用户创建成功';
    await fetchUsers();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '创建失败';
  } finally {
    saving.value = false;
  }
};

const handleDisable = async (userId) => {
  error.value = '';
  info.value = '';
  try {
    await axios.post(`/admin/users/${userId}/disable`);
    info.value = '已永久禁用用户';
    await fetchUsers();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '操作失败';
  }
};

const handleEnable = async (userId) => {
  error.value = '';
  info.value = '';
  try {
    await axios.post(`/admin/users/${userId}/enable`);
    info.value = '已重新启用用户';
    await fetchUsers();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '操作失败';
  }
};

const handleTemporary = async (userId) => {
  error.value = '';
  info.value = '';
  const value = parseIso(tempDisableInputs[userId]);
  if (!value) {
    error.value = '请选择有效的临时禁用结束时间';
    return;
  }
  try {
    await axios.post(`/admin/users/${userId}/disable-temporary`, {
      disabled_until: value,
    });
    info.value = '用户已设置临时禁用';
    tempDisableInputs[userId] = '';
    await fetchUsers();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '操作失败';
  }
};

const formatStatus = (user) => {
  if (!user.is_active) {
    return '已永久禁用';
  }
  if (user.disabled_until) {
    const until = new Date(user.disabled_until);
    if (until > new Date()) {
      return `暂时禁用至 ${until.toLocaleString()}`;
    }
  }
  return '正常';
};

const isSelf = (user) => user.id === currentUser.value?.id;
const handleRoleChange = async (user) => {
  if (!user) return;
  const targetRole = roleSelection[user.id];
  if (targetRole === user.role) return;
  error.value = '';
  info.value = '';
  try {
    await axios.put(`/admin/users/${user.id}`, { role: targetRole });
    info.value = '角色已更新';
    await fetchUsers();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '角色更新失败';
    roleSelection[user.id] = user.role;
  }
};

const handleResetPassword = async (user) => {
  error.value = '';
  info.value = '';
  const raw = window.prompt('为用户设置新密码（至少 6 位）：', '');
  if (!raw) {
    return;
  }
  if (raw.length < 6) {
    error.value = '密码至少需要 6 个字符';
    return;
  }
  try {
    await axios.post(`/admin/users/${user.id}/reset-password`, { password: raw });
    info.value = '密码已重置';
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '密码重置失败';
  }
};

const handleDeleteUser = async (user) => {
  if (!user || !window.confirm(`确定删除 ${user.username} 吗？此操作无法撤销`)) {
    return;
  }
  error.value = '';
  info.value = '';
  try {
    await axios.delete(`/admin/users/${user.id}`);
    info.value = '用户已删除';
    await fetchUsers();
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '删除失败';
  }
};
</script>

<template>
  <div class="admin-page">
    <section class="card admin-card">
      <header class="panel-header">
        <div>
          <h2>管理员 · 用户管理</h2>
          <p class="panel-subtitle">创建用户、修改角色与禁用状态。禁止操作自己以避免失去权限。</p>
        </div>
        <div class="admin-header-actions">
          <button class="secondary" type="button" @click="goHome">返回主页</button>
          <button class="secondary" type="button" @click="handleLogout">登出</button>
        </div>
      </header>
      <div class="admin-grid">
        <form class="admin-form" @submit.prevent="handleCreateUser">
          <h3>创建新用户</h3>
          <label>
            用户名
            <input v-model="form.username" type="text" placeholder="例如：alice" />
          </label>
          <label>
            密码
            <input v-model="form.password" type="password" placeholder="初始密码" />
          </label>
          <label>
            角色
            <select v-model="form.role">
              <option value="user">普通用户</option>
              <option value="admin">管理员</option>
            </select>
          </label>
          <label>
            <input type="checkbox" v-model="form.is_active" /> 立即启用
          </label>
          <button class="primary" type="submit" :disabled="saving">{{ saving ? '提交中…' : '创建用户' }}</button>
        </form>
        <div class="admin-table-wrapper">
          <div class="admin-actions">
            <span v-if="loadingUsers">加载用户列表中…</span>
            <span v-else>共 {{ users.length }} 条记录</span>
          </div>
          <p v-if="error" class="form-error">{{ error }}</p>
          <p v-if="info" class="form-success">{{ info }}</p>
          <table class="admin-table">
            <thead>
              <tr>
                <th>用户名</th>
                <th>角色</th>
                <th>状态</th>
                <th>禁用/恢复</th>
                <th>重置密码</th>
                <th>创建时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="user in users" :key="user.id">
                <td>{{ user.username }}</td>
                <td>
                  <select
                    v-model="roleSelection[user.id]"
                    @change="handleRoleChange(user)"
                    :disabled="isSelf(user)"
                  >
                    <option value="user">普通用户</option>
                    <option value="admin">管理员</option>
                  </select>
                </td>
                <td>{{ formatStatus(user) }}</td>
                <td>{{ user.disabled_until ? new Date(user.disabled_until).toLocaleString() : '—' }}</td>
                <td>
                  <button class="secondary" type="button" @click="handleResetPassword(user)">重置密码</button>
                </td>
                <td>{{ new Date(user.created_at).toLocaleString() }}</td>
                <td class="actions-cell">
                  <button class="secondary" type="button" @click="handleEnable(user.id)" :disabled="user.is_active">启用</button>
                  <button
                    class="secondary danger"
                    type="button"
                    @click="handleDisable(user.id)"
                    :disabled="!user.is_active || isSelf(user)"
                  >
                    永久禁用
                  </button>
                  <div class="temp-disable">
                    <input v-model="tempDisableInputs[user.id]" type="datetime-local" />
                    <button class="secondary" type="button" @click="handleTemporary(user.id)">设置</button>
                  </div>
                  <button
                    class="secondary danger"
                    type="button"
                    @click="handleDeleteUser(user)"
                    :disabled="isSelf(user)"
                  >
                    删除
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
</template>
