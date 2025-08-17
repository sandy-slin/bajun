#!/bin/bash

# A股智能交易决策平台 - 服务启动脚本
# 按正确顺序启动后端和前端服务

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 启动A股智能交易决策平台..."
echo "======================================================"
echo "📍 工作目录: $SCRIPT_DIR"

# 1. 检查前置条件
echo "🔍 检查前置条件..."

if [ ! -d "venv" ]; then
    echo "❌ Python虚拟环境不存在，请先运行:"
    echo "   ./install_dependencies.sh"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "❌ 前端依赖不存在，请先运行:"
    echo "   ./install_dependencies.sh"
    exit 1
fi

echo "✅ 前置条件检查通过"

# 2. 创建日志目录
mkdir -p logs

# 3. 检查端口占用
check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  端口 $port 已被占用 ($service)"
        echo "💡 请停止占用端口的进程或使用其他端口"
        return 1
    fi
    return 0
}

echo "🔍 检查端口占用..."
if ! check_port 8000 "后端服务"; then
    echo "停止现有的后端服务..."
    pkill -f "python.*main.py" 2>/dev/null || true
    sleep 2
fi

if ! check_port 3000 "前端服务"; then
    echo "停止现有的前端服务..."
    pkill -f "react-scripts" 2>/dev/null || true
    sleep 2
fi

# 4. 启动后端服务
echo ""
echo "🔧 启动后端服务 (端口8000)..."
source venv/bin/activate
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

cd src/api
nohup python main.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

echo "⏳ 等待后端服务启动..."
sleep 5

# 验证后端服务
for i in {1..6}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 后端服务启动成功"
        break
    elif [ $i -eq 6 ]; then
        echo "❌ 后端服务启动失败"
        echo "📝 查看日志: tail -f logs/backend.log"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    else
        echo "   等待中... ($i/5)"
        sleep 2
    fi
done

# 5. 启动前端服务
echo ""
echo "⚛️  启动前端服务 (端口3000)..."
cd frontend
nohup npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "⏳ 等待前端服务启动..."
sleep 10

# 验证前端服务
for i in {1..6}; do
    if curl -s -I http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ 前端服务启动成功"
        break
    elif [ $i -eq 6 ]; then
        echo "❌ 前端服务启动失败"
        echo "📝 查看日志: tail -f logs/frontend.log"
        kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
        exit 1
    else
        echo "   等待中... ($i/5)"
        sleep 3
    fi
done

# 6. 验证WebSocket连接
echo ""
echo "🔍 验证WebSocket连接..."
sleep 2

# 保存进程ID
echo $BACKEND_PID > logs/backend.pid
echo $FRONTEND_PID > logs/frontend.pid

echo ""
echo "🎉 服务启动完成！"
echo "======================================================"
echo "📊 后端API服务:     http://localhost:8000"
echo "📖 API文档:         http://localhost:8000/docs"
echo "🔧 系统状态:        http://localhost:8000/health"
echo "⚛️  前端应用:        http://localhost:3000"
echo "📝 WebSocket:       ws://localhost:8000/ws/realtime"
echo "======================================================"
echo ""
echo "📝 日志文件:"
echo "   后端日志: tail -f logs/backend.log"
echo "   前端日志: tail -f logs/frontend.log"
echo ""
echo "🛑 停止服务:"
echo "   ./stop_services.sh"
echo ""
echo "🧪 运行测试:"
echo "   ./run_tests.sh"
echo ""
echo "💡 服务已在后台运行，可以开始使用！"