#!/bin/bash

echo "🔧 配置Claude Code MCP连接..."

# 方法1: Browser MCP HTTP连接
echo "配置Browser MCP..."
if claude mcp add browser-automation \
    --transport http \
    http://localhost:3001/mcp \
    --name "Browser Automation for Frontend Testing"; then
    echo "✅ Browser MCP配置成功"
else
    echo "⚠️ Browser MCP配置可能失败，请手动配置"
    echo "手动命令: claude mcp add browser-automation --transport http http://localhost:3001/mcp"
fi

# 显示当前MCP配置
echo ""
echo "📋 当前MCP服务器列表:"
claude mcp list || echo "无法获取MCP列表"

echo ""
echo "✅ Claude Code MCP配置完成"
