import { ref, onUnmounted } from 'vue'
import { useSituationStore } from '../stores/situation'

export function useSimulation() {
  const store = useSituationStore()
  let simulationInterval = null
  let uptimeInterval = null
  
  const startSimulation = () => {
    if (store.simulationActive) return
    
    store.startSimulation()
    
    simulationInterval = setInterval(() => {
      const data = generateMockData()
      store.processRealtimeData(data)
    }, 500)
    
    uptimeInterval = setInterval(() => {
      store.stats.uptime++
    }, 1000)
  }
  
  const stopSimulation = () => {
    if (simulationInterval) {
      clearInterval(simulationInterval)
      simulationInterval = null
    }
    if (uptimeInterval) {
      clearInterval(uptimeInterval)
      uptimeInterval = null
    }
    store.stopSimulation()
  }
  
  const generateMockData = () => {
    const isAttack = Math.random() > 0.85
    const baseLatency = isAttack ? 15 + Math.random() * 20 : 2 + Math.random() * 5
    
    const featureNames = [
      'flow_duration', 'packet_count', 'byte_count',
      'src_port', 'dst_port', 'tcp_flags',
      'packet_rate', 'byte_rate', 'packet_size_avg',
      'inter_arrival_time', 'syn_count', 'ack_count',
      'fin_count', 'rst_count', 'urg_count'
    ]
    
    let explanation = isAttack 
      ? `[DDoS Detection] High-frequency traffic pattern detected.\n`
        + `特征贡献分析:\n`
        + `- packet_rate: ${(Math.random() * 100).toFixed(2)} (极异常)\n`
        + `- byte_rate: ${(Math.random() * 100).toFixed(2)} (极异常)\n`
        + `- inter_arrival_time: ${(Math.random() * 100).toFixed(2)} (极异常)`
      : 'Normal traffic flow, no anomalies detected.'
    
    return {
      timestamp: Date.now() / 1000,
      latency_ms: parseFloat(baseLatency.toFixed(2)),
      is_attack: isAttack,
      explanation,
      features: featureNames.map(name => ({
        name,
        importance: Math.random() * 100,
        contribution: Math.random() > 0.5 ? 1 : -1
      }))
    }
  }
  
  onUnmounted(() => {
    stopSimulation()
  })
  
  return {
    startSimulation,
    stopSimulation
  }
}
