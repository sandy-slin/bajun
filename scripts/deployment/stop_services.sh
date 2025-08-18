#!/bin/bash

# A股智能交易决策平台 - 服务停止脚本

echo "🛑 停止A股智能交易决策平台服务..."

# 获取脚本所在目录的绝对路径，然后切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 1. 从PID文件停止服务
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo "停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        sleep 2
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo "强制停止后端服务..."
            kill -9 $BACKEND_PID 2>/dev/null || true
        fi
    fi
    rm -f logs/backend.pid
fi

if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        sleep 2
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            echo "强制停止前端服务..."
            kill -9 $FRONTEND_PID 2>/dev/null || true
        fi
    fi
    rm -f logs/frontend.pid
fi

# 2. 通过进程名停止残留服务
echo "清理残留进程..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "react-scripts" 2>/dev/null || true
pkill -f "node.*start" 2>/dev/null || true

sleep 2

# 3. 验证停止结果
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口8000仍被占用"
else
    echo "✅ 后端服务已停止"
fi

if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口3000仍被占用"
else
    echo "✅ 前端服务已停止"
fi

echo "🎉 服务停止完成"