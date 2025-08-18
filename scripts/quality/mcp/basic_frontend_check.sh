#!/bin/bash

# 基础前端MCP检查脚本

echo "🔍 启动基础前端MCP检查..."

# 确保服务运行
echo "📡 检查后端服务..."
if ! curl -f -s http://localhost:8000/health > /dev/null; then
    echo "❌ 后端服务未运行，请先启动: scripts/deployment/start_services.sh"
    exit 1
fi

echo "🌐 检查前端服务..."
if ! curl -f -s http://localhost:3000 > /dev/null; then
    echo "❌ 前端服务未运行，请先启动前端服务"
    exit 1
fi

echo "✅ 基础服务检查通过"

# 启动Browser MCP服务（如果需要）
if ! lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null; then
    echo "🚀 启动Browser MCP服务..."
    scripts/quality/mcp/start_browser_mcp.sh &
    MCP_PID=$!
    sleep 5
    
    # 注册清理函数
    trap "kill $MCP_PID 2>/dev/null" EXIT
fi

echo "🎯 执行前端页面检查..."

# 创建检查报告目录
mkdir -p logs/mcp_checks

# 基础页面访问测试
PAGES=("http://localhost:3000" "http://localhost:3000/dashboard" "http://localhost:3000/sector-analysis")
RESULTS_FILE="logs/mcp_checks/basic_check_$(date +%Y%m%d_%H%M%S).json"

echo "{" > "$RESULTS_FILE"
echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$RESULTS_FILE"
echo "  \"results\": [" >> "$RESULTS_FILE"

for i in "${!PAGES[@]}"; do
    PAGE="${PAGES[$i]}"
    echo "📄 检查页面: $PAGE"
    
    # 基础HTTP检查
    if curl -f -s "$PAGE" > /dev/null; then
        STATUS="PASS"
        MESSAGE="页面可访问"
    else
        STATUS="FAIL"
        MESSAGE="页面无法访问"
    fi
    
    echo "    {" >> "$RESULTS_FILE"
    echo "      \"url\": \"$PAGE\"," >> "$RESULTS_FILE"
    echo "      \"status\": \"$STATUS\"," >> "$RESULTS_FILE"
    echo "      \"message\": \"$MESSAGE\"," >> "$RESULTS_FILE"
    echo "      \"checked_at\": \"$(date -Iseconds)\"" >> "$RESULTS_FILE"
    
    if [ $i -lt $((${#PAGES[@]} - 1)) ]; then
        echo "    }," >> "$RESULTS_FILE"
    else
        echo "    }" >> "$RESULTS_FILE"
    fi
done

echo "  ]" >> "$RESULTS_FILE"
echo "}" >> "$RESULTS_FILE"

echo "📊 检查完成，报告保存到: $RESULTS_FILE"

# 显示简要结果
echo ""
echo "📋 检查结果摘要:"
cat "$RESULTS_FILE" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for result in data['results']:
        status_icon = '✅' if result['status'] == 'PASS' else '❌'
        print(f'   {status_icon} {result[\"url\"]}: {result[\"message\"]}')
except:
    print('   无法解析检查结果')
"

echo "✅ 基础前端MCP检查完成"
