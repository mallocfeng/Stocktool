import { createRouter, createWebHistory } from 'vue-router';
import BacktestView from '../views/BacktestView.vue';
import LoginView from '../views/LoginView.vue';
import AdminView from '../views/AdminView.vue';
import { ensureUserLoaded, currentUser, isAdmin } from '../lib/useAuth';

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'Login',
      component: LoginView,
      meta: { public: true },
    },
    {
      path: '/register',
      name: 'Register',
      component: () => import('../views/RegisterView.vue'),
      meta: { public: true },
    },
    {
      path: '/admin',
      name: 'Admin',
      component: AdminView,
      meta: { requiresAuth: true, requiresAdmin: true },
    },
    {
      path: '/',
      name: 'Dashboard',
      component: BacktestView,
      meta: { requiresAuth: true },
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/',
    },
  ],
});

router.beforeEach(async (to) => {
  if (to.meta.public) {
    return true;
  }
  await ensureUserLoaded();
  if (!currentUser.value) {
    return { path: '/login', query: { redirect: to.fullPath } };
  }
  if (to.meta.requiresAdmin && !isAdmin.value) {
    return { path: '/' };
  }
  return true;
});

export default router;
