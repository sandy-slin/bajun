#!/bin/bash

# Chrome MCP能力演示脚本
# 展示Claude如何通过MCP检查前端运行效果

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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${PURPLE}🚀 Chrome MCP能力演示${NC}"
echo "========================================================================"
echo "📍 项目根目录: $PROJECT_ROOT"
echo "🎯 演示Claude如何通过MCP检查前端运行效果"
echo ""

# 检查先决条件
echo -e "${CYAN}Phase 1: 环境检查${NC}"
echo "========================================================================"

# 检查服务状态
echo "📡 检查后端服务..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ 后端服务运行正常${NC}"
    BACKEND_OK=true
else
    echo -e "${RED}❌ 后端服务未运行${NC}"
    BACKEND_OK=false
fi

echo "🌐 检查前端服务..."
if curl -f -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ 前端服务运行正常${NC}"
    FRONTEND_OK=true
else
    echo -e "${RED}❌ 前端服务未运行${NC}"
    FRONTEND_OK=false
fi

# 如果服务未运行，提供启动建议
if [ "$BACKEND_OK" = false ] || [ "$FRONTEND_OK" = false ]; then
    echo ""
    echo -e "${YELLOW}💡 请先启动必要的服务:${NC}"
    if [ "$BACKEND_OK" = false ]; then
        echo "   后端: scripts/deployment/start_services.sh"
    fi
    if [ "$FRONTEND_OK" = false ]; then
        echo "   前端: cd frontend && npm start"
    fi
    echo ""
    echo "服务启动后，重新运行此演示脚本"
    exit 1
fi

# 检查MCP配置
echo ""
echo -e "${CYAN}Phase 2: MCP配置检查${NC}"
echo "========================================================================"

MCP_AVAILABLE=false

# 检查Chrome MCP配置文件
if [ -f "configs/mcp/frontend_check_config.json" ]; then
    echo -e "${GREEN}✅ MCP配置文件存在${NC}"
    MCP_CONFIG_OK=true
else
    echo -e "${RED}❌ MCP配置文件不存在${NC}"
    MCP_CONFIG_OK=false
fi

# 检查MCP脚本
if [ -x "scripts/quality/mcp/integrated_frontend_mcp_check.sh" ]; then
    echo -e "${GREEN}✅ MCP检查脚本可执行${NC}"
    MCP_SCRIPT_OK=true
else
    echo -e "${RED}❌ MCP检查脚本不可用${NC}"
    MCP_SCRIPT_OK=false
fi

# 检查Browser MCP服务器
if command -v browser-mcp-server >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Browser MCP Server 已安装${NC}"
    MCP_SERVER_OK=true
else
    echo -e "${YELLOW}⚠️ Browser MCP Server 未安装${NC}"
    MCP_SERVER_OK=false
fi

# 检查Claude Code
if command -v claude >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Claude Code 可用${NC}"
    CLAUDE_OK=true
else
    echo -e "${RED}❌ Claude Code 未安装${NC}"
    CLAUDE_OK=false
fi

if [ "$MCP_CONFIG_OK" = true ] && [ "$MCP_SCRIPT_OK" = true ]; then
    MCP_AVAILABLE=true
fi

echo ""
echo -e "${CYAN}Phase 3: MCP功能演示${NC}"
echo "========================================================================"

if [ "$MCP_AVAILABLE" = true ]; then
    echo -e "${GREEN}🎯 执行Chrome MCP前端检查演示...${NC}"
    echo ""
    
    # 运行MCP检查
    echo "🔍 运行集成的Chrome MCP检查..."
    if scripts/quality/mcp/integrated_frontend_mcp_check.sh; then
        echo -e "${GREEN}✅ MCP检查完成${NC}"
        
        # 查找最新的检查报告
        LATEST_REPORT=$(ls -t logs/mcp_checks/integrated_check_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_REPORT" ] && [ -f "$LATEST_REPORT" ]; then
            echo ""
            echo -e "${CYAN}📊 检查结果详情:${NC}"
            python -c "
import json
try:
    with open('$LATEST_REPORT', 'r') as f:
        report = json.load(f)
    
    print('📋 检查摘要:')
    summary = report['summary']
    print(f'   - 总检查项: {summary[\"total_checks\"]}')
    print(f'   - 通过: {summary[\"passed\"]}')
    print(f'   - 失败: {summary[\"failed\"]}')
    print(f'   - 成功率: {summary[\"success_rate\"]}')
    print(f'   - 检查模式: {report.get(\"mode\", \"未知\")}')
    
    if 'results' in report and isinstance(report['results'], list):
        print('')
        print('📄 页面检查结果:')
        for result in report['results']:
            status_icon = '✅' if result.get('status') == 'PASS' else '❌'
            print(f'   {status_icon} {result.get(\"page\", \"未知页面\")}: {result.get(\"url\", \"\")}')
            if result.get('issues'):
                for issue in result['issues'][:2]:
                    print(f'      - {issue}')
    
    if 'recommendations' in report:
        print('')
        print('💡 系统建议:')
        for rec in report['recommendations']:
            print(f'   - {rec}')
            
except Exception as e:
    print(f'解析报告失败: {e}')
"
        else
            echo -e "${YELLOW}⚠️ 检查报告文件未找到${NC}"
        fi
    else
        echo -e "${RED}❌ MCP检查执行失败${NC}"
    fi
    
else
    echo -e "${YELLOW}⚠️ MCP功能不完整，运行基础演示${NC}"
    echo ""
    
    # 基础HTTP检查演示
    echo "🔍 执行基础HTTP检查演示..."
    
    PAGES=(
        "http://localhost:3000|主页"
        "http://localhost:3000/dashboard|仪表板"
        "http://localhost:3000/sector-analysis|板块分析"
    )
    
    for page_info in "${PAGES[@]}"; do
        IFS='|' read -r URL NAME <<< "$page_info"
        echo "📄 检查 $NAME..."
        
        if curl -f -s "$URL" > /dev/null; then
            echo -e "   ${GREEN}✅ 页面可访问${NC}"
        else
            echo -e "   ${RED}❌ 页面无法访问${NC}"
        fi
        
        sleep 1
    done
fi

echo ""
echo -e "${CYAN}Phase 4: Claude MCP使用指南${NC}"
echo "========================================================================"

echo -e "${BLUE}🚀 使用Claude进行前端检查的命令示例:${NC}"
echo ""

if [ "$CLAUDE_OK" = true ]; then
    echo "1. ${YELLOW}基础前端状态检查:${NC}"
    echo '   claude "检查前端应用的运行状态，确认所有页面都能正常加载"'
    echo ""
    
    echo "2. ${YELLOW}页面截图和视觉检查:${NC}"
    echo '   claude "截图前端应用的主要页面，检查是否有布局或渲染问题"'
    echo ""
    
    echo "3. ${YELLOW}用户流程测试:${NC}"
    echo '   claude "模拟用户从主页进入仪表板的完整流程，检查每个步骤"'
    echo ""
    
    echo "4. ${YELLOW}错误检测:${NC}"
    echo '   claude "检查前端应用是否有JavaScript错误或控制台警告"'
    echo ""
    
    echo "5. ${YELLOW}性能监控:${NC}"
    echo '   claude "测试前端页面的加载性能，报告加载时间"'
    echo ""
    
    echo "6. ${YELLOW}API连接测试:${NC}"
    echo '   claude "验证前端应用与后端API的连接是否正常"'
    
else
    echo -e "${RED}❌ Claude Code未安装，无法使用MCP功能${NC}"
    echo "安装命令: npm install -g @anthropic-ai/claude-code"
fi

echo ""
echo -e "${CYAN}Phase 5: 完整安装指南${NC}"
echo "========================================================================"

if [ "$MCP_AVAILABLE" = false ]; then
    echo -e "${YELLOW}💡 完整安装Chrome MCP功能:${NC}"
    echo ""
    echo "1. 运行安装脚本:"
    echo "   scripts/setup/setup_chrome_mcp.sh"
    echo ""
    echo "2. 安装Chrome扩展 (Browser MCP):"
    echo "   https://chromewebstore.google.com/detail/browser-mcp-automate-your/bjfgambnhccakkhmkepdoekmckoijdlc"
    echo ""
    echo "3. 配置Claude Code MCP连接:"
    echo "   scripts/setup/configure_claude_mcp.sh"
    echo ""
    echo "4. 测试MCP功能:"
    echo "   scripts/quality/mcp/basic_frontend_check.sh"
    echo ""
else
    echo -e "${GREEN}✅ Chrome MCP功能已可用${NC}"
    echo ""
    echo "快速测试命令:"
    echo "   scripts/quality/mcp/integrated_frontend_mcp_check.sh"
    echo ""
fi

echo -e "${CYAN}📁 相关文件:${NC}"
echo "   - 配置文件: configs/mcp/frontend_check_config.json"
echo "   - 检查脚本: scripts/quality/mcp/integrated_frontend_mcp_check.sh"
echo "   - 检查器: scripts/quality/mcp/chrome_frontend_checker.py"
echo "   - 安装脚本: scripts/setup/setup_chrome_mcp.sh"

# 查找并显示最新的检查报告
LATEST_REPORT=$(ls -t logs/mcp_checks/*.json 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    echo "   - 最新报告: $LATEST_REPORT"
fi

# 查找并显示最新的截图
LATEST_SCREENSHOT=$(ls -t logs/screenshots/*.png 2>/dev/null | head -1)
if [ -n "$LATEST_SCREENSHOT" ]; then
    echo "   - 最新截图: $LATEST_SCREENSHOT"
fi

echo ""
echo -e "${PURPLE}🎉 Chrome MCP能力演示完成！${NC}"
echo ""

if [ "$MCP_AVAILABLE" = true ] && [ "$CLAUDE_OK" = true ]; then
    echo -e "${GREEN}✨ 现在您可以使用Claude通过MCP直接检查前端运行效果了！${NC}"
    echo ""
    echo "试试这个命令:"
    echo -e "${BLUE}claude \"使用MCP检查前端应用，并截图显示当前状态\"${NC}"
else
    echo -e "${YELLOW}💡 完成安装后，您将能够使用Claude进行高级前端自动化检查${NC}"
fi

echo ""