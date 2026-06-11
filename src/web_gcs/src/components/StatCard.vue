<template>
  <div 
    class="stat-card p-3 rounded-lg border"
    :class="cardClass"
  >
    <div class="text-xs text-gray-400 mb-1">{{ title }}</div>
    <div class="flex items-baseline gap-1">
      <span class="text-2xl font-bold" :class="valueClass">
        {{ value }}
      </span>
      <span class="text-xs text-gray-500">{{ unit }}</span>
    </div>
    <div v-if="trend" class="mt-1">
      <el-icon v-if="trend === 'up'" class="text-red-500"><Top /></el-icon>
      <el-icon v-else-if="trend === 'down'" class="text-green-500"><Bottom /></el-icon>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { Top, Bottom } from '@element-plus/icons-vue'

const props = defineProps({
  title: String,
  value: [String, Number],
  unit: { type: String, default: '' },
  trend: String,
  alert: Boolean,
  good: Boolean
})

const cardClass = computed(() => {
  if (props.alert) return 'bg-red-900/20 border-red-800'
  if (props.good) return 'bg-green-900/20 border-green-800'
  return 'bg-gray-800/50 border-gray-700'
})

const valueClass = computed(() => {
  if (props.alert) return 'text-red-400'
  if (props.good) return 'text-green-400'
  return 'text-white'
})
</script>
