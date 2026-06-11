<template>
  <div class="tech-card h-full flex flex-col overflow-hidden">
    <div class="tech-card-header flex-shrink-0">
      <h3 class="text-lg font-bold text-tech-blue flex items-center gap-2">
        <el-icon>
          <Clock />
        </el-icon>
        边缘节点延迟监控
      </h3>
      <el-tag :type="currentStatus.type" size="small">
        {{ currentStatus.text }}
      </el-tag>
    </div>

    <div class="p-4 flex-1 min-h-0 flex flex-col overflow-hidden">
      <div class="grid grid-cols-4 gap-3 mb-4 flex-shrink-0">
        <StatCard title="平均延迟" :value="stats.avgLatency.toFixed(2)" unit="ms" trend="down"
          :good="stats.avgLatency < 10" />
        <StatCard title="最大延迟" :value="stats.maxLatency.toFixed(2)" unit="ms" :alert="stats.maxLatency > 20" />
        <StatCard title="总包数" :value="stats.totalPackets" unit="包" />
        <StatCard title="异常率" :value="stats.attackRate" unit="%" :alert="parseFloat(stats.attackRate) > 10" />
      </div>

      <div class="flex-1 min-h-0">
        <div ref="chartRef" class="w-full h-full"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { useSituationStore } from '../stores/situation'
import { storeToRefs } from 'pinia'
import StatCard from './StatCard.vue'
import { Clock } from '@element-plus/icons-vue'

const store = useSituationStore()
const { delayData, stats, delayThreshold } = storeToRefs(store)

const chartRef = ref(null)
let chartInstance = null

const currentStatus = computed(() => {
  if (stats.value.avgLatency > 15) {
    return { type: 'danger', text: '延迟过高' }
  } else if (stats.value.avgLatency > 10) {
    return { type: 'warning', text: '延迟偏高' }
  }
  return { type: 'success', text: '运行正常' }
})

const initChart = () => {
  if (!chartRef.value) return

  chartInstance = echarts.init(chartRef.value, 'dark')

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(17, 24, 39, 0.95)',
      borderColor: '#1f2937',
      textStyle: { color: '#fff' },
      formatter: (params) => {
        const data = params[0]
        const isAttack = delayData.value[data.dataIndex]?.isAttack
        return `
          <div style="font-family: monospace;">
            <div style="color: #9ca3af; margin-bottom: 4px;">${data.name}</div>
            <div style="color: ${isAttack ? '#ef4444' : '#10b981'}; font-size: 16px; font-weight: bold;">
              ${data.value.toFixed(2)} ms
            </div>
            ${isAttack ? '<div style="color: #ef4444; font-size: 12px;">⚠️ 异常</div>' : ''}
          </div>
        `
      }
    },
    grid: {
      left: '8%',
      right: '4%',
      bottom: '15%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: [],
      axisLine: { lineStyle: { color: '#374151' } },
      axisLabel: { 
        color: '#6b7280', 
        fontSize: 10,
        interval: 'auto',
        rotate: 0,
        margin: 8
      },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      name: '延迟 (ms)',
      nameTextStyle: { 
        color: '#9ca3af',
        fontSize: 11,
        padding: [0, 0, 0, 0]
      },
      nameLocation: 'middle',
      nameGap: 20,
      splitLine: { lineStyle: { color: '#1f2937', type: 'dashed' } },
      axisLine: { show: false },
      axisLabel: { 
        color: '#9ca3af',
        fontSize: 10,
        margin: 5
      }
    },
    series: [{
      name: '推理延迟',
      type: 'line',
      smooth: true,
      symbol: 'circle',
      symbolSize: 7,
      lineStyle: { 
        width: 3,
        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
          { offset: 0, color: '#10b981' },
          { offset: 1, color: '#34d399' }
        ])
      },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(16, 185, 129, 0.4)' },
          { offset: 0.5, color: 'rgba(16, 185, 129, 0.15)' },
          { offset: 1, color: 'rgba(16, 185, 129, 0.02)' }
        ])
      },
      data: [],
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { 
          color: '#f59e0b', 
          type: 'dashed', 
          width: 2,
          shadowBlur: 10,
          shadowColor: 'rgba(245, 158, 11, 0.5)'
        },
        data: [{ yAxis: delayThreshold.value, name: '基准线' }],
        label: {
          formatter: '基准 {c}ms',
          color: '#f59e0b',
          fontSize: 11,
          position: 'end'
        }
      },
      itemStyle: {
        color: (params) => {
          const isAttack = delayData.value[params.dataIndex]?.isAttack
          if (isAttack) {
            return new echarts.graphic.RadialGradient(0.5, 0.5, 0.5, [
              { offset: 0, color: '#f87171' },
              { offset: 1, color: '#ef4444' }
            ])
          } else {
            return new echarts.graphic.RadialGradient(0.5, 0.5, 0.5, [
              { offset: 0, color: '#34d399' },
              { offset: 1, color: '#10b981' }
            ])
          }
        },
        shadowBlur: 8,
        shadowColor: (params) => {
          const isAttack = delayData.value[params.dataIndex]?.isAttack
          return isAttack ? 'rgba(239, 68, 68, 0.6)' : 'rgba(16, 185, 129, 0.6)'
        }
      },
      emphasis: {
        focus: 'series',
        lineStyle: {
          width: 4,
          shadowBlur: 15,
          shadowColor: 'rgba(16, 185, 129, 0.8)'
        }
      }
    }]
  }

  chartInstance.setOption(option)
}

const updateChart = () => {
  if (!chartInstance) return

  chartInstance.setOption({
    xAxis: {
      data: delayData.value.map(d => d.time)
    },
    series: [{
      data: delayData.value.map(d => d.value)
    }]
  })
}

watch(delayData, updateChart, { deep: true })

onMounted(() => {
  initChart()
  window.addEventListener('resize', () => chartInstance?.resize())
})

onUnmounted(() => {
  chartInstance?.dispose()
})
</script>
