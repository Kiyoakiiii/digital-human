#!/bin/bash
# 优化版启动脚本 - 在SSH环境中同时启动前端和后端服务

source ~/anaconda3/etc/profile.d/conda.sh  # 确保 conda 命令可以使用
conda activate cosyvoice2

# 设置项目路径 - 修改为你的实际项目路径
PROJECT_DIR="/home/zentek/project"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
TEMP_DIR="$PROJECT_DIR/temp"
LOGS_DIR="$PROJECT_DIR/logs"

# 确保目录存在
for DIR in "$TEMP_DIR" "$LOGS_DIR"; do
    if [ ! -d "$DIR" ]; then
        mkdir -p "$DIR"
        echo "创建目录: $DIR"
    fi
done

# 清理旧的临时文件
echo "清理临时文件..."
find "$TEMP_DIR" -type f -mtime +1 -delete

# 优化GPU设置 (如果有NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "设置GPU优化..."
    # 允许TensorRT优化和半精度
    export TF_ENABLE_AUTO_MIXED_PRECISION=1
    export TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT=1
fi

# 提高Node.js性能
export NODE_OPTIONS="--max-old-space-size=4096"

# 启动后端服务作为后台进程
echo "启动后端服务..."
cd "$BACKEND_DIR"
nohup python -m uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2 > "$LOGS_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "后端服务已启动，PID: $BACKEND_PID"
sleep 3  # 等待后端启动

# 启动前端服务作为后台进程
echo "启动前端服务..."
cd "$FRONTEND_DIR"
nohup yarn start > "$LOGS_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "前端服务已启动，PID: $FRONTEND_PID"

# 保存PID以便后续关闭
echo "$BACKEND_PID" > "$PROJECT_DIR/backend.pid"
echo "$FRONTEND_PID" > "$PROJECT_DIR/frontend.pid"

echo "服务已启动:"
echo "- 后端: http://localhost:5000"
echo "- 前端: https://172.16.10.158:3000"
echo "- 日志: $LOGS_DIR/backend.log 和 $LOGS_DIR/frontend.log"
echo "要停止服务，请运行 ./stop_all.sh"
