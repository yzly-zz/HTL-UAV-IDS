import { defineStore } from "pinia";

export const useSituationStore = defineStore("situation", {
  state: () => ({
    connectionStatus: "disconnected",
    delayData: [],
    delayThreshold: 10,
    stats: {
      totalPackets: 0,
      attackCount: 0,
      normalCount: 0,
      attackRate: 0,
      avgLatency: 0,
      maxLatency: 0,
      uptime: 0,
    },
    threatData: [],
    selectedThreat: null,
    shapData: [],
    shapTrends: [],
    attackLogs: [],
    simulationActive: false,
    startTime: null,
    uptimeTimer: null,
  }),

  actions: {
    setConnectionStatus(status) {
      this.connectionStatus = status;
    },

    processRealtimeData(data) {
      const now = new Date(data.timestamp * 1000);
      const timeStr = now.toLocaleTimeString("zh-CN", { hour12: false });

      this.stats.totalPackets++;
      if (data.is_attack) {
        this.stats.attackCount++;
        this.processAttackEvent(data, timeStr);
      } else {
        this.stats.normalCount++;
      }
      this.stats.attackRate = (
        (this.stats.attackCount / this.stats.totalPackets) *
        100
      ).toFixed(2);
      this.stats.avgLatency = (this.stats.avgLatency + data.latency_ms) / 2;
      this.stats.maxLatency = Math.max(this.stats.maxLatency, data.latency_ms);

      this.delayData.push({
        time: timeStr,
        value: data.latency_ms,
        isAttack: data.is_attack,
      });
      if (this.delayData.length > 60) {
        this.delayData.shift();
      }

      this.threatData.push({
        time: timeStr,
        value: data.is_attack ? 100 : 10,
        raw: data,
      });
      if (this.threatData.length > 60) {
        this.threatData.shift();
      }
    },

    processAttackEvent(data, timeStr) {
      const attackEvent = {
        id: Date.now(),
        time: timeStr,
        latency: data.latency_ms,
        explanation: data.explanation,
        features: this.parseRealFeatures(data.explanation),
      };

      this.attackLogs.unshift(attackEvent);
      if (this.attackLogs.length > 50) {
        this.attackLogs.pop();
      }

      this.generateShapData(attackEvent);
      this.selectThreat(attackEvent);
    },

    generateMockFeatures() {
      const featureNames = [
        "flow_duration",
        "packet_count",
        "byte_count",
        "src_port",
        "dst_port",
        "tcp_flags",
        "packet_rate",
        "byte_rate",
        "packet_size_avg",
        "inter_arrival_time",
        "syn_count",
        "ack_count",
      ];

      return featureNames
        .map((name) => ({
          name,
          importance: Math.random() * 100,
          contribution: Math.random() > 0.5 ? 1 : -1,
        }))
        .sort((a, b) => b.importance - a.importance);
    },

    generateShapData(event) {
      const features = event.features;

      this.shapData = features.slice(0, 10).map((f) => ({
        feature: f.name,
        value: f.contribution * f.importance,
        rawValue: f.importance,
        direction: f.contribution > 0 ? "negative" : "positive",
      }));

      features.forEach((f) => {
        if (!this.shapTrends[f.name]) {
          this.shapTrends[f.name] = [];
        }

        const existingIndex = this.shapTrends[f.name].findIndex(
          (t) => t.time === event.time,
        );
        if (existingIndex >= 0) {
          this.shapTrends[f.name][existingIndex] = {
            time: event.time,
            value: f.importance,
          };
        } else {
          this.shapTrends[f.name].push({
            time: event.time,
            value: f.importance,
          });
        }
      });
    },
    parseRealFeatures(explanationText) {
      // 解析 SHAP-Lite 返回的格式: "1. feature_name -> 威胁贡献比: XX.X%"
        const lines = explanationText.split('\n');
        const features = [];
        for (const line of lines) {
            const match = line.match(/\d+\.\s+(\S+)\s+->\s+威胁贡献比:\s+([\d.]+)%/);
            if (match) {
                features.push({
                    name: match[1],
                    importance: parseFloat(match[2]),
                    contribution: 1  // 攻击类特征贡献为正
                });
            }
        }
        return features.length > 0 ? features : this.generateMockFeatures();
    },
    selectThreat(event) {
      this.selectedThreat = event;
      this.generateShapData(event);
    },

    startSimulation() {
      this.simulationActive = true;
      this.connectionStatus = "connected";
      this.startTime = Date.now();
      this.stats.uptime = 0;

      if (this.uptimeTimer) {
        clearInterval(this.uptimeTimer);
      }

    },

    stopSimulation() {
      this.simulationActive = false;
      this.connectionStatus = "disconnected";

      if (this.uptimeTimer) {
        clearInterval(this.uptimeTimer);
        this.uptimeTimer = null;
      }
    },

    resetData() {
      if (this.uptimeTimer) {
        clearInterval(this.uptimeTimer);
        this.uptimeTimer = null;
      }

      this.delayData = [];
      this.threatData = [];
      this.attackLogs = [];
      this.shapData = [];
      this.shapTrends = {};
      this.selectedThreat = null;
      this.startTime = Date.now();
      this.simulationActive = false;
      this.stats = {
        totalPackets: 0,
        attackCount: 0,
        normalCount: 0,
        attackRate: 0,
        avgLatency: 0,
        maxLatency: 0,
        uptime: 0,
      };

    },
  },
});
