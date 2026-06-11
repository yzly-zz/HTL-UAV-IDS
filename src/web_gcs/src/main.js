import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import App from './App.vue'
import './assets/styles/main.css'

const app = createApp(App)

app.use(createPinia())
app.use(ElementPlus)

document.documentElement.classList.add('dark')

app.mount('#app')
