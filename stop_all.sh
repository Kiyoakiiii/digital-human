#!/bin/bash
# 停止前端和后端服务

# 设置项目路径 - 修改为你的实际项目路径
PROJECT_DIR="/home/zentek/project"

# 停止后端服务
if [ -f "$PROJECT_DIR/backend.pid" ]; then
    BACKEND_PID=$(cat "$PROJECT_DIR/backend.pid")
    echo "停止后端服务 (PID: $BACKEND_PID)..."
    kill -15 $BACKEND_PID 2>/dev/null || true
    rm "$PROJECT_DIR/backend.pid"
else
    echo "后端PID文件不存在，尝试通过进程查找..."
    pkill -f "uvicorn app:app" || true
fi

# 停止前端服务
if [ -f "$PROJECT_DIR/frontend.pid" ]; then
    FRONTEND_PID=$(cat "$PROJECT_DIR/frontend.pid")
    echo "停止前端服务 (PID: $FRONTEND_PID)..."
    kill -15 $FRONTEND_PID 2>/dev/null || true
    rm "$PROJECT_DIR/frontend.pid"
else
    echo "前端PID文件不存在，尝试通过进程查找..."
    pkill -f "react-scripts start" || true
fi

echo "所有服务已停止"
