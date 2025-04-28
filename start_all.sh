#!/bin/bash
# 在SSH环境中同时启动前端和后端服务
source ~/anaconda3/etc/profile.d/conda.sh  # 确保 conda 命令可以使用
conda activate cosyvoice2
# 设置项目路径 - 修改为你的实际项目路径
PROJECT_DIR="/home/zentek/project"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
TEMP_DIR="$PROJECT_DIR/temp"

# 确保临时目录存在
if [ ! -d "$TEMP_DIR" ]; then
    mkdir -p "$TEMP_DIR"
    echo "创建临时目录: $TEMP_DIR"
fi

# 启动后端服务作为后台进程
echo "启动后端服务..."
cd "$BACKEND_DIR"
nohup python -m uvicorn app:app --host 0.0.0.0 --port 5000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "后端服务已启动，PID: $BACKEND_PID"
sleep 3  # 等待后端启动

# 启动前端服务作为后台进程
echo "启动前端服务..."
cd "$FRONTEND_DIR"
# 设置React开发服务器使用HTTPS
export HTTPS=true
export SSL_CRT_FILE="/home/zentek/project/ssl/172.16.10.158.pem"
export SSL_KEY_FILE="/home/zentek/project/ssl/172.16.10.158-key.pem"
export HOST=0.0.0.0
nohup yarn start > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "前端服务已启动，PID: $FRONTEND_PID"

# 保存PID以便后续关闭
echo "$BACKEND_PID" > "$PROJECT_DIR/backend.pid"
echo "$FRONTEND_PID" > "$PROJECT_DIR/frontend.pid"

echo "服务已启动:"
echo "- 后端: http://localhost:5000"
echo "- 前端: https://172.16.10.158:3000"
echo "- 日志: $BACKEND_DIR/backend.log 和 $FRONTEND_DIR/frontend.log"
echo "要停止服务，请运行 ./stop_all.sh"
