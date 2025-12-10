import { createApp } from 'vue'
import axios from 'axios'
import './style.css'
import App from './App.vue'

const apiBase = import.meta.env.VITE_API_BASE?.replace(/\/$/, '') || 'http://127.0.0.1:8000'
axios.defaults.baseURL = apiBase

createApp(App).mount('#app')
