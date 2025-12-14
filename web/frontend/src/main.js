import { createApp } from 'vue'
import axios from 'axios'
import './style.css'
import App from './App.vue'
import router from './router'

const envApiBase = import.meta.env.VITE_API_BASE?.replace(/\/$/, '')
const runtimeApiBase = window.__STOCKTOOL_RUNTIME_API_BASE?.replace(/\/$/, '')
const apiBase = envApiBase || runtimeApiBase || ''
axios.defaults.baseURL = apiBase
axios.defaults.withCredentials = true

createApp(App).use(router).mount('#app')
