import { ref, computed } from 'vue';
import axios from 'axios';

const disableAuth = import.meta.env.VITE_DISABLE_AUTH !== 'false';
const fallbackUser = {
  id: 0,
  username: import.meta.env.VITE_BYPASS_USERNAME || 'admin',
  role: import.meta.env.VITE_BYPASS_ROLE || 'admin',
  is_active: true,
  disabled_until: null,
  created_at: new Date().toISOString(),
};

const currentUser = ref(null);
let pendingLoad = null;

const loadCurrentUser = async () => {
  if (disableAuth) {
    currentUser.value = fallbackUser;
    return;
  }
  try {
    const res = await axios.get('/me');
    currentUser.value = res.data;
  } catch (err) {
    currentUser.value = null;
  }
};

export function ensureUserLoaded() {
  if (disableAuth) {
    currentUser.value = fallbackUser;
    return Promise.resolve();
  }
  if (!pendingLoad) {
    pendingLoad = loadCurrentUser().finally(() => {
      pendingLoad = null;
    });
  }
  return pendingLoad;
}

export async function login(payload) {
  if (disableAuth) {
    currentUser.value = fallbackUser;
    return;
  }
  await axios.post('/login', payload);
  await loadCurrentUser();
}

export async function logout() {
  try {
    await axios.post('/logout');
  } finally {
    currentUser.value = null;
  }
}

export const isAdmin = computed(() => currentUser.value?.role === 'admin');

export function useAuth() {
  return {
    currentUser,
    isAdmin,
    login,
    logout,
  };
}

export { currentUser };
