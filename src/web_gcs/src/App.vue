<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-left">
        <h1 class="text-2xl font-bold text-tech-blue glow-text">
          🛸 空地协同边缘 IDS 态势感知系统
        </h1>
        <p class="text-sm text-gray-500 ml-4">
          Projection-based Heterogeneous Transfer for UAV Intrusion Detection
        </p>
      </div>
      <div class="header-right flex items-center gap-4">
        <el-tag :type="connectionTagType" effect="dark" size="large">
          <el-icon class="mr-1">
            <component :is="connectionIcon" />
          </el-icon>
          {{ connectionText }}
        </el-tag>
        <el-button @click="resetData" type="warning" size="large">
          重置数据
        </el-button>
      </div>
    </header>
    
    <GridLayout />
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useSituationStore } from './stores/situation'
import { storeToRefs } from 'pinia'
import { useWebSocket } from './composables/useWebSocket'
import GridLayout from './components/GridLayout.vue'
import { Connection, Close } from '@element-plus/icons-vue'

const store = useSituationStore()
const { connectionStatus } = storeToRefs(store)
const { connect } = useWebSocket()

onMounted(() => {
  connect()
})

const connectionTagType = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return 'success'
    case 'error': return 'danger'
    default: return 'warning'
  }
})

const connectionIcon = computed(() => {
  return connectionStatus.value === 'connected' ? Connection : Close
})

const connectionText = computed(() => {
  switch (connectionStatus.value) {
    case 'connected': return '已连接'
    case 'error': return '连接错误'
    default: return '未连接'
  }
})

const resetData = () => {
  store.resetData()
}
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #0a0e17 0%, #111827 100%);
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: rgba(17, 24, 39, 0.8);
  border-bottom: 1px solid #1f2937;
  backdrop-filter: blur(10px);
}

.header-left {
  display: flex;
  align-items: center;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.glow-text {
  text-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
}
</style>
