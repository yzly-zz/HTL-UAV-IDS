import { onUnmounted } from "vue";
import { useSituationStore } from "../stores/situation";

export function useWebSocket() {
  const store = useSituationStore();
  let ws = null;
  let reconnectTimer = null;

  const connect = () => {
    const wsUrl = `ws://${window.location.hostname}:${window.location.port || 8000}/ws/traffic`

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        store.setConnectionStatus("connected");
        // 不自动启动模拟——等待 real_time.py 发送真实数据
        store.startTime = Date.now();
        console.log("WebSocket connected");
      };

      ws.onclose = () => {
        store.setConnectionStatus("disconnected");
        console.log("WebSocket disconnected, reconnecting...");
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        store.setConnectionStatus("error");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          store.processRealtimeData(data);
        } catch (e) {
          console.error("Data parse error:", e);
        }
      };
    } catch (e) {
      console.error("WebSocket connection failed:", e);
      store.setConnectionStatus("error");
    }
  };

  const disconnect = () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      ws.close();
      ws = null;
    }
  };

  onUnmounted(() => {
    disconnect();
  });

  return {
    connect,
    disconnect,
  };
}
