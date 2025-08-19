#!/bin/bash

# A股智能交易决策平台 - 全栈启动脚本
# 同时启动FastAPI后端服务和React前端应用

echo "🚀 启动A股智能交易决策平台全栈服务..."
echo "======================================================"

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 检查前端依赖
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 安装前端依赖..."
    cd frontend && npm install && cd ..
fi

# 激活Python虚拟环境
echo "🐍 激活Python虚拟环境..."
source "$PROJECT_ROOT/venv/bin/activate"

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动后端服务 (后台运行)
echo "🔧 启动FastAPI后端服务 (端口8000)..."
cd src/api
nohup python main.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 5

# 检查后端是否启动成功
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 后端服务启动成功"
else
    echo "❌ 后端服务启动失败"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# 启动前端服务
echo "⚛️  启动React前端应用 (端口3000)..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 全栈服务启动完成！"
echo "======================================================"
echo "📊 后端API服务:     http://localhost:8000"
echo "📖 API文档:         http://localhost:8000/docs"
echo "🔧 系统状态:        http://localhost:8000/health"
echo "⚛️  前端应用:        http://localhost:3000"
echo "📝 WebSocket:       ws://localhost:8000/ws/realtime"
echo "======================================================"
echo ""
echo "💡 按 Ctrl+C 停止所有服务"
echo ""

# 创建日志目录
mkdir -p logs

# 等待用户中断
trap 'echo ""; echo "🛑 正在停止服务..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# 保持脚本运行
wait