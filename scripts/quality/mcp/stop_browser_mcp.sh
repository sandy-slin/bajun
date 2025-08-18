#!/bin/bash

echo "🛑 停止Browser MCP服务器..."

# 停止端口3001上的服务
if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null; then
    kill $(lsof -t -i:3001)
    echo "✅ Browser MCP服务器已停止"
else
    echo "ℹ️ Browser MCP服务器未运行"
fi
