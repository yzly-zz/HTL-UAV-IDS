<template>
  <div class="tech-card h-full flex flex-col overflow-hidden">
    <div class="tech-card-header flex-shrink-0">
      <h3 class="text-lg font-bold text-tech-cyan flex items-center gap-2">
        <el-icon>
          <Warning />
        </el-icon>
        实时告警总览
      </h3>
      <el-tag type="danger" size="small" effect="dark">
        {{ stats.attackCount }} 攻击
      </el-tag>
    </div>

    <div class="p-4 flex-1 min-h-0 overflow-hidden">
      <el-row :gutter="16" class="h-full">
        <el-col :span="8" class="h-full flex flex-col justify-center">
          <div ref="pieChartRef" class="w-full h-48"></div>
          <div class="text-center mt-2">
            <div class="text-3xl font-bold text-red-400">{{ stats.attackCount }}</div>
            <div class="text-xs text-gray-500">攻击事件</div>
          </div>
        </el-col>

        <el-col :span="16">
          <div class="grid grid-cols-2 gap-3">
            <div class="bg-gray-800/50 p-3 rounded-lg border border-gray-700">
              <div class="text-xs text-gray-400">正常运行</div>
              <div class="text-2xl font-bold text-green-400">{{ stats.normalCount }}</div>
            </div>
            <div class="bg-gray-800/50 p-3 rounded-lg border border-gray-700">
              <div class="text-xs text-gray-400">攻击占比</div>
              <div class="text-2xl font-bold" :class="attackRateClass">
                {{ stats.attackRate }}%
              </div>
            </div>
            <div class="bg-gray-800/50 p-3 rounded-lg border border-gray-700">
              <div class="text-xs text-gray-400">平均延迟</div>
              <div class="text-2xl font-bold text-blue-400">{{ stats.avgLatency.toFixed(2) }} ms</div>
            </div>
            <div class="bg-gray-800/50 p-3 rounded-lg border border-gray-700">
              <div class="text-xs text-gray-400">系统运行</div>
              <div class="text-2xl font-bold text-cyan-400">{{ formattedUptime }}</div>
            </div>
          </div>

          <div ref="trendChartRef" class="w-full h-24 mt-3"></div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useSituationStore } from '../stores/situation'
import { storeToRefs } from 'pinia'
import { Warning } from '@element-plus/icons-vue'

const store = useSituationStore()
const { stats, attackLogs } = storeToRefs(store)

const pieChartRef = ref(null)
const trendChartRef = ref(null)
let pieChart = null
let trendChart = null

const attackRateClass = computed(() => {
  const rate = parseFloat(stats.value.attackRate)
  if (rate > 20) return 'text-red-400'
  if (rate > 10) return 'text-yellow-400'
  return 'text-green-400'
})

const formattedUptime = computed(() => {
  const seconds = stats.value.uptime
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
})

const initCharts = () => {
  if (pieChartRef.value) {
    pieChart = echarts.init(pieChartRef.value, 'dark')
    updatePieChart()
  }

  if (trendChartRef.value) {
    trendChart = echarts.init(trendChartRef.value, 'dark')
    updateTrendChart()
  }
}

const updatePieChart = () => {
  if (!pieChart) return
  const pieData = [
    {
      value: stats.value.normalCount,
      name: '正常',
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#10b981' },
          { offset: 1, color: '#059669' }
        ])
      }
    }
  ]
  if (stats.value.attackCount > 0) {
    pieData.push({
      value: stats.value.attackCount,
      name: '攻击',
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#ef4444' },
          { offset: 1, color: '#dc2626' }
        ]),
        shadowBlur: 15,
        shadowColor: 'rgba(239, 68, 68, 0.5)'
      }
    })
  }

  pieChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(17, 24, 39, 0.95)',
      borderColor: '#1f2937',
      textStyle: { color: '#fff' },
      formatter: '{b}: {c} ({d}%)'
    },
    series: [{
      type: 'pie',
      radius: ['50%', '70%'],
      center: ['50%', '50%'],
      avoidLabelOverlap: true,
      label: {
        show: true,
        position: 'outside',
        formatter: '{b}\n{c}',
        color: '#9ca3af',
        fontSize: 11,
        lineHeight: 14
      },
      emphasis: {
        scale: true,
        scaleSize: 5,
        itemStyle: {
          shadowBlur: 20,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      },
      itemStyle: {
        borderRadius: 6,
        borderColor: '#1f2937',
        borderWidth: 2
      },
      data: pieData
    }]
  })
}

const updateTrendChart = () => {
  if (!trendChart) return

  const recentLogs = attackLogs.value.slice(0, 20).reverse()

  trendChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(17, 24, 39, 0.95)',
      borderColor: '#1f2937',
      textStyle: { color: '#fff' },
      formatter: (params) => {
        const data = recentLogs[params.dataIndex]
        return `
          <div style="font-family: monospace; font-size: 12px;">
            <div style="color: #9ca3af;">${data.time}</div>
            <div style="color: #ef4444; font-size: 16px; font-weight: bold; margin-top: 4px;">
              ${data.latency.toFixed(2)} ms
            </div>
          </div>
        `
      }
    },
    grid: {
      left: '3%',
      right: '3%',
      bottom: '15%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: recentLogs.map(l => l.time),
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: {
        color: '#6b7280',
        fontSize: 10,
        rotate: 0,
        interval: 'auto'
      },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      show: false
    },
    series: [{
      type: 'bar',
      data: recentLogs.map((l, index) => ({
        value: l.latency,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#ef4444' },
            { offset: 1, color: '#dc2626' }
          ]),
          borderRadius: [4, 4, 0, 0]
        }
      })),
      barWidth: '50%',
      barGap: '30%'
    }]
  })
}

watch(stats, () => {
  updatePieChart()
  updateTrendChart()
}, { deep: true, immediate: true })

onMounted(() => {
  initCharts()
  window.addEventListener('resize', () => {
    pieChart?.resize()
    trendChart?.resize()
  })
})

onUnmounted(() => {
  pieChart?.dispose()
  trendChart?.dispose()
})
</script>
