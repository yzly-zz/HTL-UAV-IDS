<template>
  <div class="tech-card h-full flex flex-col overflow-hidden">
    <div class="tech-card-header flex-shrink-0">
      <h3 class="text-lg font-bold text-tech-red flex items-center gap-2">
        <el-icon>
          <CircleClose />
        </el-icon>
        异常流量威胁概率
      </h3>
      <div class="flex items-center gap-2">
        <div class="w-3 h-3 rounded-full bg-red-500 animate-pulse"></div>
        <span class="text-xs text-gray-500">实时监控</span>
      </div>
    </div>

    <div class="p-4 flex-1 min-h-0 overflow-hidden">
      <div ref="chartRef" class="w-full h-full"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useSituationStore } from '../stores/situation'
import { storeToRefs } from 'pinia'
import { CircleClose } from '@element-plus/icons-vue'

const store = useSituationStore()
const { threatData } = storeToRefs(store)

const chartRef = ref(null)
let chartInstance = null

const initChart = () => {
  if (!chartRef.value) return

  chartInstance = echarts.init(chartRef.value, 'dark')

  chartInstance.on('click', (params) => {
    if (threatData.value[params.dataIndex]?.raw?.is_attack) {
      store.selectThreat({
        id: Date.now(),
        time: threatData.value[params.dataIndex].time,
        latency: threatData.value[params.dataIndex].raw.latency_ms,
        explanation: threatData.value[params.dataIndex].raw.explanation
      })
    }
  })

  chartInstance.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(17, 24, 39, 0.95)',
      borderColor: '#1f2937',
      textStyle: { color: '#fff' },
      formatter: (params) => {
        const data = params[0]
        const isAttack = threatData.value[data.dataIndex]?.value === 100
        return `
          <div style="font-family: monospace;">
            <div style="color: #9ca3af;">${data.name}</div>
            <div style="color: ${isAttack ? '#ef4444' : '#10b981'}; font-size: 18px; font-weight: bold;">
              ${isAttack ? '⚠️ 威胁检测' : '✓ 正常'}
            </div>
            ${isAttack ? '<div style="color: #ef4444; font-size: 12px; margin-top: 4px;">点击查看详情</div>' : ''}
          </div>
        `
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: [],
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af', fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      max: 100,
      name: '威胁指数',
      nameTextStyle: { color: '#9ca3af' },
      splitLine: { lineStyle: { color: '#1f2937', type: 'dashed' } },
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { color: '#9ca3af' },
      markLine: {
        silent: true,
        lineStyle: { color: '#ef4444', type: 'dashed', width: 2 },
        data: [{ yAxis: 50, name: '告警阈值' }]
      }
    },
    series: [{
      name: '威胁概率',
      type: 'line',
      step: 'end',
      smooth: true,
      symbol: 'circle',
      symbolSize: 8,
      lineStyle: { width: 2, color: '#ef4444' },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(239, 68, 68, 0.5)' },
          { offset: 0.5, color: 'rgba(239, 68, 68, 0.2)' },
          { offset: 1, color: 'rgba(239, 68, 68, 0.05)' }
        ])
      },
      itemStyle: {
        color: (params) => {
          return threatData.value[params.dataIndex]?.value === 100 ? '#ef4444' : '#10b981'
        }
      },
      data: []
    }]
  })
}

const updateChart = () => {
  if (!chartInstance) return

  chartInstance.setOption({
    xAxis: {
      data: threatData.value.map(d => d.time)
    },
    series: [{
      data: threatData.value.map(d => d.value)
    }]
  })
}

watch(threatData, updateChart, { deep: true })

onMounted(() => {
  initChart()
  window.addEventListener('resize', () => chartInstance?.resize())
})

onUnmounted(() => {
  chartInstance?.dispose()
})
</script>
