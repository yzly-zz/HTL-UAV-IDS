<template>
  <div class="grid-container">
    <div class="grid-row">
      <div class="grid-cell">
        <DelayMonitor />
      </div>
      <div class="grid-cell">
        <AlertOverview />
      </div>
    </div>
    <div class="grid-row">
      <div class="grid-cell">
        <ThreatMonitor />
      </div>
      <div class="grid-cell">
        <ShapAnalyzer />
      </div>
    </div>
  </div>
</template>

<script setup>
import DelayMonitor from './DelayMonitor.vue'
import AlertOverview from './AlertOverview.vue'
import ThreatMonitor from './ThreatMonitor.vue'
import ShapAnalyzer from './ShapAnalyzer.vue'
</script>

<style scoped>
.grid-container {
  height: 100vh;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow: hidden;
}

.grid-row {
  flex: 1;
  min-height: 0;
  display: flex;
  gap: 16px;
  overflow: hidden;
}

.grid-cell {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}
</style>
