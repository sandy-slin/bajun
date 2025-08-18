#!/bin/bash

# 集成的Chrome MCP前端检查脚本
# 与现有提交前检查流程无缝集成

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${CYAN}🌐 Chrome MCP前端运行效果检查${NC}"
echo "========================================================================"

# 创建日志目录
mkdir -p logs/mcp_checks

# 检查MCP是否可用
MCP_AVAILABLE=false
MCP_MODE="disabled"

# 检查Browser MCP服务器是否可用
if command -v browser-mcp-server >/dev/null 2>&1; then
    MCP_MODE="browser_mcp"
    MCP_AVAILABLE=true
    echo -e "${GREEN}✅ Browser MCP Server 可用${NC}"
elif command -v claude >/dev/null 2>&1 && claude mcp list | grep -q "browser"; then
    MCP_MODE="claude_mcp"
    MCP_AVAILABLE=true
    echo -e "${GREEN}✅ Claude MCP 连接可用${NC}"
else
    echo -e "${YELLOW}⚠️ MCP服务不可用，使用基础HTTP检查模式${NC}"
fi

# Phase 1: 基础服务检查
echo ""
echo -e "${BLUE}Phase 1: 基础服务状态检查${NC}"
echo "----------------------------------------"

BACKEND_OK=false
FRONTEND_OK=false

# 检查后端服务
echo "📡 检查后端服务 (http://localhost:8000)..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ 后端服务运行正常${NC}"
    BACKEND_OK=true
else
    echo -e "${RED}❌ 后端服务未运行${NC}"
fi

# 检查前端服务
echo "🌐 检查前端服务 (http://localhost:3000)..."
if curl -f -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ 前端服务运行正常${NC}"
    FRONTEND_OK=true
else
    echo -e "${RED}❌ 前端服务未运行${NC}"
fi

# 如果基础服务不可用，提供启动建议
if [ "$BACKEND_OK" = false ] || [ "$FRONTEND_OK" = false ]; then
    echo ""
    echo -e "${YELLOW}💡 服务启动建议:${NC}"
    if [ "$BACKEND_OK" = false ]; then
        echo "   后端: scripts/deployment/start_services.sh"
    fi
    if [ "$FRONTEND_OK" = false ]; then
        echo "   前端: cd frontend && npm start"
    fi
    echo ""
fi

# Phase 2: MCP服务启动（如果可用）
MCP_STARTED=false
MCP_PID=""

if [ "$MCP_AVAILABLE" = true ] && [ "$MCP_MODE" = "browser_mcp" ]; then
    echo ""
    echo -e "${BLUE}Phase 2: Browser MCP服务管理${NC}"
    echo "----------------------------------------"
    
    # 检查MCP服务是否已运行
    if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Browser MCP服务已运行 (端口3001)${NC}"
    else
        echo "🚀 启动Browser MCP服务..."
        if scripts/quality/mcp/start_browser_mcp.sh > logs/mcp_startup.log 2>&1 &
        then
            MCP_PID=$!
            MCP_STARTED=true
            sleep 3  # 等待服务启动
            
            if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null 2>&1; then
                echo -e "${GREEN}✅ Browser MCP服务启动成功${NC}"
            else
                echo -e "${RED}❌ Browser MCP服务启动失败${NC}"
                MCP_AVAILABLE=false
            fi
        else
            echo -e "${RED}❌ 无法启动Browser MCP服务${NC}"
            MCP_AVAILABLE=false
        fi
    fi
    
    # 注册清理函数
    if [ "$MCP_STARTED" = true ]; then
        trap 'echo "🛑 清理MCP服务..."; kill $MCP_PID 2>/dev/null; scripts/quality/mcp/stop_browser_mcp.sh >/dev/null 2>&1' EXIT
    fi
fi

# Phase 3: 前端运行效果检查
echo ""
echo -e "${BLUE}Phase 3: 前端运行效果检查${NC}"
echo "----------------------------------------"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="logs/mcp_checks/integrated_check_${TIMESTAMP}.json"

if [ "$MCP_AVAILABLE" = true ] && [ "$FRONTEND_OK" = true ]; then
    echo "🔍 执行高级MCP前端检查..."
    
    # 使用Python MCP检查器
    if python scripts/quality/mcp/chrome_frontend_checker.py > logs/mcp_checks/checker_output.log 2>&1; then
        CHECKER_EXIT_CODE=0
    else
        CHECKER_EXIT_CODE=$?
    fi
    
    # 查找最新的检查报告
    LATEST_REPORT=$(ls -t logs/mcp_checks/frontend_check_report_*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST_REPORT" ] && [ -f "$LATEST_REPORT" ]; then
        # 复制到标准位置
        cp "$LATEST_REPORT" "$REPORT_FILE"
        
        # 解析检查结果
        SUCCESS_RATE=$(python -c "
import json
try:
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)
    print(report['summary']['success_rate'].replace('%', ''))
except:
    print('0')
")
        
        FAILED_COUNT=$(python -c "
import json
try:
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)
    print(report['summary']['failed'])
except:
    print('1')
")
        
        echo ""
        echo -e "${CYAN}📊 MCP检查结果摘要:${NC}"
        python -c "
import json
try:
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)
    summary = report['summary']
    print(f'   总检查项: {summary[\"total_checks\"]}')
    print(f'   通过: {summary[\"passed\"]}')
    print(f'   失败: {summary[\"failed\"]}')
    print(f'   成功率: {summary[\"success_rate\"]}')
    
    if summary['failed'] > 0:
        print('')
        print('❌ 发现的问题:')
        for result in report['results']:
            if result['status'] == 'FAIL':
                print(f'   📄 {result[\"page\"]}: {len(result[\"issues\"])}个问题')
                for issue in result['issues'][:2]:
                    print(f'      - {issue}')
    
    print('')
    print('💡 修复建议:')
    for rec in report['recommendations']:
        print(f'   - {rec}')
        
except Exception as e:
    print(f'   解析报告失败: {e}')
"
        
        # 决定检查结果
        if [ "$FAILED_COUNT" -eq 0 ]; then
            echo -e "${GREEN}✅ Chrome MCP前端检查通过${NC}"
            CHECK_RESULT="PASS"
        else
            echo -e "${RED}❌ Chrome MCP前端检查发现问题${NC}"
            CHECK_RESULT="FAIL"
        fi
        
    else
        echo -e "${RED}❌ 无法生成MCP检查报告${NC}"
        CHECK_RESULT="FAIL"
    fi
    
elif [ "$FRONTEND_OK" = true ]; then
    echo "🔍 执行基础HTTP前端检查..."
    
    # 基础HTTP检查模式
    PAGES=(
        "http://localhost:3000|主页"
        "http://localhost:3000/dashboard|仪表板"
        "http://localhost:3000/sectors|板块分析"
        "http://localhost:3000/stock-recommendation|股票推荐"
    )
    
    BASIC_RESULTS=""
    BASIC_FAILED=0
    
    for page_info in "${PAGES[@]}"; do
        IFS='|' read -r URL NAME <<< "$page_info"
        echo "📄 检查 $NAME ($URL)..."
        
        if curl -f -s "$URL" > /dev/null; then
            echo -e "   ${GREEN}✅ 可访问${NC}"
            BASIC_RESULTS="${BASIC_RESULTS}${NAME}: 可访问\n"
        else
            echo -e "   ${RED}❌ 无法访问${NC}"
            BASIC_RESULTS="${BASIC_RESULTS}${NAME}: 无法访问\n"
            ((BASIC_FAILED++))
        fi
    done
    
    # 生成基础检查报告
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "mode": "basic_http_check",
  "summary": {
    "total_checks": ${#PAGES[@]},
    "passed": $((${#PAGES[@]} - BASIC_FAILED)),
    "failed": $BASIC_FAILED,
    "success_rate": "$(( (${#PAGES[@]} - BASIC_FAILED) * 100 / ${#PAGES[@]} ))%"
  },
  "results": "basic_http_check",
  "recommendations": [
    "安装Chrome MCP以获得更详细的检查结果",
    "运行 scripts/setup/setup_chrome_mcp.sh 进行安装"
  ]
}
EOF
    
    echo ""
    echo -e "${CYAN}📊 基础检查结果:${NC}"
    echo -e "$BASIC_RESULTS"
    
    if [ "$BASIC_FAILED" -eq 0 ]; then
        echo -e "${GREEN}✅ 基础前端检查通过${NC}"
        CHECK_RESULT="PASS"
    else
        echo -e "${RED}❌ 基础前端检查发现问题${NC}"
        CHECK_RESULT="FAIL"
    fi
    
else
    echo -e "${RED}❌ 前端服务未运行，无法执行检查${NC}"
    
    # 生成错误报告
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "mode": "service_unavailable",
  "summary": {
    "total_checks": 0,
    "passed": 0,
    "failed": 1,
    "success_rate": "0%"
  },
  "error": "前端服务未运行",
  "recommendations": [
    "启动前端服务: cd frontend && npm start",
    "确保后端服务运行: scripts/deployment/start_services.sh"
  ]
}
EOF
    
    CHECK_RESULT="FAIL"
fi

# Phase 4: 结果输出和建议
echo ""
echo -e "${PURPLE}📋 Chrome MCP前端检查完成${NC}"
echo "========================================================================"

echo -e "${CYAN}📁 生成的文件:${NC}"
echo "   检查报告: $REPORT_FILE"
if [ -f "logs/mcp_checks/checker_output.log" ]; then
    echo "   详细日志: logs/mcp_checks/checker_output.log"
fi

# 如果有截图，显示路径
if [ -d "logs/screenshots" ] && [ "$(ls -A logs/screenshots 2>/dev/null)" ]; then
    LATEST_SCREENSHOT=$(ls -t logs/screenshots/*.png 2>/dev/null | head -1)
    if [ -n "$LATEST_SCREENSHOT" ]; then
        echo "   最新截图: $LATEST_SCREENSHOT"
    fi
fi

echo ""
echo -e "${CYAN}🚀 使用Claude MCP的高级功能:${NC}"
echo '   claude "使用MCP检查前端应用运行状态并截图"'
echo '   claude "模拟用户从主页到仪表板的完整流程测试"'
echo '   claude "检查前端应用是否有JavaScript错误"'

echo ""
if [ "$CHECK_RESULT" = "PASS" ]; then
    echo -e "${GREEN}✅ 前端运行效果检查通过${NC}"
    exit 0
else
    echo -e "${RED}❌ 前端运行效果检查失败${NC}"
    echo "📝 详细信息请查看: $REPORT_FILE"
    exit 1
fi