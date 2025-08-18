#!/bin/bash

# Browser MCP服务启动脚本

# 检查端口是否被占用
if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null; then
    echo "⚠️ 端口3001已被占用，尝试停止现有服务..."
    kill $(lsof -t -i:3001) 2>/dev/null || true
    sleep 2
fi

echo "🚀 启动Browser MCP服务器..."
echo "端口: 3001"
echo "访问地址: http://localhost:3001"

# 启动Browser MCP服务器
if command -v browser-mcp-server >/dev/null 2>&1; then
    browser-mcp-server --port 3001 --host localhost
else
    echo "❌ browser-mcp-server 未安装"
    echo "请运行: npm install -g browser-mcp-server"
    exit 1
fi
