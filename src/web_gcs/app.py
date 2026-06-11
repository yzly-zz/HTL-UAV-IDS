"""
功能描述：
无人机地面站 (GCS) 态势感知数据网关。
学术价值：基于 FastAPI 异步协程 (Asyncio) 构建。
它解耦了边缘计算节点（无人机）和监控展示终端（大屏）。
无人机仅需将推断结果 POST 到此网关，网关通过 WebSocket 广播给所有连入的监控者，
具备极高的并发吞吐能力，契合大型无人机集群协同防御的背景设定。
"""
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

app = FastAPI(title="ProjBridge Ground Control Station")

# 将 web_gcs 目录当作静态资源根目录，方便前端文件访问
base_dir = os.path.dirname(__file__)  # 指向 src/web_gcs
index_path = os.path.join(base_dir, "index.html")

# 如果前端还有静态子目录（例如 js/css），你可以把它们放到 src/web_gcs/static 并挂载如下：
static_dir = os.path.join(base_dir, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 根路由直接返回 index.html，使浏览器访问 http://127.0.0.1:8000/ 时加载大屏
@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok", "active_connections": len(manager.active_connections)}

# 允许跨域请求 (CORS)，方便前端单独作为一个本地 HTML 运行
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 维护活跃的 WebSocket 客户端列表
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"新监控大屏连入。当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logging.info(f"监控大屏断开。当前连接数: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead_connections.append(connection)
        for conn in dead_connections:
            self.active_connections.remove(conn)
            logging.warning(f"已移除死连接。当前连接数: {len(self.active_connections)}")


manager = ConnectionManager()


# 定义边缘节点上传的数据结构规范
class AlertPayload(BaseModel):
    timestamp: float
    latency_ms: float
    is_attack: bool
    explanation: str


@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    """前端大屏建立 WebSocket 监听的端点"""
    await manager.connect(websocket)
    try:
        while True:
            # 维持长连接，等待客户端心跳（如有）
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/report_traffic")
async def report_traffic(payload: AlertPayload):
    data_json = payload.model_dump_json()
    await manager.broadcast(data_json)

    if payload.is_attack:
        logging.warning(f"接收到高危告警！延迟: {payload.latency_ms:.2f}ms")
        # 持久化到日志文件
        with open("logs/alerts.jsonl", "a", encoding="utf-8") as f:
            f.write(data_json + "\n")

    return {"status": "success", "msg": "Payload broadcasted to GCS"}


if __name__ == "__main__":
    import uvicorn

    # 启动命令: python app.py
    # 服务默认运行在 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")