import { createApp } from 'vue'
import axios from 'axios'
import './style.css'
import App from './App.vue'
import router from './router'

const apiBase = import.meta.env.VITE_API_BASE?.replace(/\/$/, '') || '/api'
axios.defaults.baseURL = apiBase
axios.defaults.withCredentials = true

createApp(App).use(router).mount('#app')
