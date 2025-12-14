import { ref, computed } from 'vue';
import axios from 'axios';

const currentUser = ref(null);
let pendingLoad = null;

const loadCurrentUser = async () => {
  try {
    const res = await axios.get('/me');
    currentUser.value = res.data;
  } catch (err) {
    currentUser.value = null;
  }
};

export function ensureUserLoaded() {
  if (!pendingLoad) {
    pendingLoad = loadCurrentUser().finally(() => {
      pendingLoad = null;
    });
  }
  return pendingLoad;
}

export async function login(payload) {
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
