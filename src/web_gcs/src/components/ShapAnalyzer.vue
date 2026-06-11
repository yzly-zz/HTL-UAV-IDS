<template>
  <div class="tech-card h-full flex flex-col overflow-hidden">
    <div class="tech-card-header flex-shrink-0">
      <h3 class="text-lg font-bold text-tech-yellow flex items-center gap-2">
        <el-icon>
          <DataAnalysis />
        </el-icon>
        SHAP-Lite 特征归因分析
      </h3>
    </div>

    <div class="p-4 flex-1 min-h-0 flex gap-4 overflow-hidden">
      <div class="flex flex-col overflow-hidden" style="width: 180px; flex-shrink: 0;">
        <div class="text-xs text-gray-400 mb-2 flex-shrink-0">攻击事件列表</div>
        <div class="flex-1 overflow-y-auto scrollbar-thin space-y-2 overflow-x-hidden pr-2">
          <div v-for="log in attackLogs" :key="log.id" @click="selectLog(log)"
            class="p-3 rounded-lg border cursor-pointer transition-all flex-shrink-0"
            :class="selectedThreat?.id === log.id ? 'bg-red-900/30 border-red-600' : 'bg-gray-800/50 border-gray-700 hover:border-red-500'">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-red-400 font-mono">{{ log.time }}</span>
              <el-tag type="danger" size="mini">攻击</el-tag>
            </div>
            <div class="text-sm text-gray-300 truncate">{{ log.latency.toFixed(2) }} ms</div>
          </div>
          <div v-if="attackLogs.length === 0" class="text-center text-gray-500 py-8">
            暂无攻击事件
          </div>
        </div>
      </div>

      <div class="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        <div class="text-xs text-gray-400 mb-2 flex-shrink-0">特征贡献度分析</div>

        <div class="flex-1 min-h-0 relative">
          <div v-if="!selectedThreat" class="absolute inset-0 flex items-center justify-center text-gray-500 z-10 bg-gray-900/80">
            <div class="text-center">
              <el-icon :size="48">
                <DataLine />
              </el-icon>
              <div class="mt-2">点击左侧事件查看 SHAP 分析</div>
            </div>
          </div>
          <div ref="waterfallRef" class="h-full w-full"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useSituationStore } from '../stores/situation'
import { storeToRefs } from 'pinia'
import { DataAnalysis, DataLine } from '@element-plus/icons-vue'

const store = useSituationStore()
const { attackLogs, selectedThreat, shapData } = storeToRefs(store)

const waterfallRef = ref(null)
let waterfallChart = null

const selectLog = (log) => {
  store.selectThreat(log)
}

const selectFirstLog = () => {
  if (attackLogs.value.length > 0) {
    store.selectThreat(attackLogs.value[0])
  }
}

const initWaterfallChart = () => {
  if (!waterfallRef.value) return

  waterfallChart = echarts.init(waterfallRef.value, 'dark')
  updateWaterfallChart()
}

const updateWaterfallChart = () => {
  if (!waterfallRef.value) return
  
  if (!waterfallChart) {
    waterfallChart = echarts.init(waterfallRef.value, 'dark')
  }

  if (shapData.value.length === 0) {
    waterfallChart.clear()
    return
  }

  const data = shapData.value
  const features = data.map(d => d.feature)
  const values = data.map(d => d.value)

  waterfallChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(17, 24, 39, 0.95)',
      borderColor: '#1f2937',
      textStyle: { color: '#fff' },
      formatter: (params) => {
        const item = params[0]
        const dataItem = shapData.value[item.dataIndex]
        return `
          <div style="font-family: monospace; font-size: 12px;">
            <div style="color: #9ca3af;">${item.name}</div>
            <div style="margin: 8px 0;">
              <div>原始贡献: <span style="color: #60a5fa;">${dataItem.rawValue.toFixed(2)}</span></div>
              <div>方向影响: <span style="color: ${dataItem.direction === 'negative' ? '#ef4444' : '#10b981'}">
                ${dataItem.direction === 'negative' ? '🔴 增加威胁' : '🟢 降低威胁'}
              </span></div>
            </div>
          </div>
        `
      }
    },
    grid: {
      left: '15%',
      right: '5%',
      bottom: '3%',
      top: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#1f2937' } },
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af' }
    },
    yAxis: {
      type: 'category',
      data: features,
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af', fontSize: 11 }
    },
    series: [{
      type: 'bar',
      data: values.map(v => ({
        value: v,
        itemStyle: {
          color: v > 0
            ? new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: '#ef4444' },
              { offset: 1, color: '#f87171' }
            ])
            : new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: '#10b981' },
              { offset: 1, color: '#34d399' }
            ])
        }
      })),
      barWidth: '60%',
      label: {
        show: true,
        position: 'right',
        formatter: '{c}',
        color: '#9ca3af',
        fontSize: 10
      }
    }]
  })
}

watch(shapData, () => {
  updateWaterfallChart()
}, { deep: true })

watch(attackLogs, () => {
  selectFirstLog()
})

onMounted(() => {
  initWaterfallChart()
  selectFirstLog()

  window.addEventListener('resize', () => {
    waterfallChart?.resize()
  })
})

onUnmounted(() => {
  waterfallChart?.dispose()
})
</script>
