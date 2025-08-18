#!/bin/bash

# Chrome MCP集成安装脚本
# 为Claude Code提供Chrome浏览器自动化能力

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

echo -e "${PURPLE}🚀 Chrome MCP集成安装向导${NC}"
echo "========================================================================"
echo "📍 项目根目录: $PROJECT_ROOT"
echo "🎯 目标: 为Claude Code集成Chrome浏览器自动化检查能力"
echo ""

# 创建必要的目录
mkdir -p configs/mcp
mkdir -p logs/screenshots
mkdir -p scripts/quality/mcp

echo -e "${CYAN}Phase 1: 依赖检查和安装${NC}"
echo "========================================================================"

# 检查Node.js
if ! command -v node >/dev/null 2>&1; then
    echo -e "${RED}❌ Node.js 未安装${NC}"
    echo "请先安装Node.js: https://nodejs.org/"
    exit 1
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js $NODE_VERSION${NC}"
fi

# 检查npm
if ! command -v npm >/dev/null 2>&1; then
    echo -e "${RED}❌ npm 未安装${NC}"
    exit 1
else
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✅ npm $NPM_VERSION${NC}"
fi

# 检查Claude Code
if ! command -v claude >/dev/null 2>&1; then
    echo -e "${RED}❌ Claude Code 未安装${NC}"
    echo "请先安装Claude Code: npm install -g @anthropic-ai/claude-code"
    exit 1
else
    echo -e "${GREEN}✅ Claude Code 已安装${NC}"
fi

echo ""
echo -e "${CYAN}Phase 2: MCP服务器安装选择${NC}"
echo "========================================================================"

echo "请选择要安装的Chrome MCP方案:"
echo "1) Browser MCP (推荐) - 专为AI自动化设计"
echo "2) Puppeteer MCP - 强大的浏览器自动化"
echo "3) 两者都安装"
echo "4) 跳过自动安装，仅创建配置文件"

read -p "请输入选择 (1-4): " choice

case $choice in
    1|3)
        echo -e "${YELLOW}📦 安装Browser MCP...${NC}"
        
        # 检查Browser MCP是否可用
        if npm list -g browser-mcp-server >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Browser MCP Server 已安装${NC}"
        else
            echo "安装Browser MCP Server..."
            if npm install -g browser-mcp-server; then
                echo -e "${GREEN}✅ Browser MCP Server 安装成功${NC}"
            else
                echo -e "${RED}❌ Browser MCP Server 安装失败${NC}"
                echo "请手动安装: npm install -g browser-mcp-server"
            fi
        fi
        
        echo -e "${YELLOW}📝 Browser MCP配置信息:${NC}"
        echo "   Chrome扩展: https://chromewebstore.google.com/detail/browser-mcp-automate-your/bjfgambnhccakkhmkepdoekmckoijdlc"
        echo "   本地服务端口: 3001"
        ;;
esac

case $choice in
    2|3)
        echo -e "${YELLOW}📦 安装Puppeteer MCP...${NC}"
        
        # 检查Puppeteer是否已安装
        if npm list puppeteer >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Puppeteer 已安装${NC}"
        else
            echo "安装Puppeteer..."
            if npm install puppeteer; then
                echo -e "${GREEN}✅ Puppeteer 安装成功${NC}"
            else
                echo -e "${RED}❌ Puppeteer 安装失败${NC}"
                echo "请手动安装: npm install puppeteer"
            fi
        fi
        ;;
esac

echo ""
echo -e "${CYAN}Phase 3: 配置文件创建${NC}"
echo "========================================================================"

# 创建Browser MCP配置
echo -e "${YELLOW}📝 创建Browser MCP配置...${NC}"
cat > configs/mcp/browser_mcp_config.json << 'EOF'
{
  "server_type": "browser_mcp",
  "connection": {
    "transport": "http",
    "host": "localhost",
    "port": 3001,
    "endpoint": "/mcp"
  },
  "capabilities": [
    "navigate",
    "screenshot",
    "click",
    "type",
    "evaluate_js",
    "get_console_logs",
    "monitor_network",
    "check_performance"
  ],
  "security": {
    "allowed_domains": [
      "localhost:3000",
      "localhost:8000",
      "127.0.0.1:3000",
      "127.0.0.1:8000"
    ],
    "restrict_file_access": true,
    "timeout_ms": 30000
  }
}
EOF

# 创建前端检查配置
echo -e "${YELLOW}📝 创建前端检查配置...${NC}"
cat > configs/mcp/frontend_check_config.json << 'EOF'
{
  "check_points": {
    "homepage": {
      "url": "http://localhost:3000",
      "name": "主页",
      "checks": [
        "page_loads",
        "react_renders", 
        "no_js_errors",
        "responsive_design",
        "basic_navigation"
      ],
      "expected_elements": [
        "#root",
        "header",
        "nav"
      ],
      "performance_thresholds": {
        "load_time_ms": 3000,
        "first_paint_ms": 1500
      }
    },
    "dashboard": {
      "url": "http://localhost:3000/dashboard",
      "name": "仪表板",
      "checks": [
        "page_loads",
        "data_loads",
        "charts_render",
        "websocket_connects",
        "real_time_updates"
      ],
      "expected_elements": [
        ".dashboard-container",
        ".market-overview",
        ".performance-metrics"
      ],
      "api_endpoints": [
        "http://localhost:8000/health",
        "http://localhost:8000/api/v1/sectors/"
      ]
    },
    "sector_analysis": {
      "url": "http://localhost:3000/sector-analysis",
      "name": "板块分析",
      "checks": [
        "page_loads",
        "api_data_loads",
        "table_renders",
        "filtering_works",
        "export_functions"
      ],
      "expected_elements": [
        ".sector-analysis-container",
        "table",
        ".filter-controls"
      ]
    },
    "stock_recommendation": {
      "url": "http://localhost:3000/stock-recommendation",
      "name": "股票推荐",
      "checks": [
        "page_loads",
        "recommendation_data_loads",
        "interactive_elements",
        "data_visualization"
      ],
      "expected_elements": [
        ".stock-recommendation-container",
        ".recommendation-card",
        ".stock-list"
      ]
    }
  },
  "test_scenarios": [
    {
      "name": "complete_user_workflow",
      "description": "完整用户工作流测试",
      "steps": [
        {
          "action": "navigate",
          "target": "http://localhost:3000",
          "description": "访问主页"
        },
        {
          "action": "wait_for_element",
          "selector": "#root",
          "timeout": 5000
        },
        {
          "action": "click",
          "selector": "a[href='/dashboard']",
          "description": "进入仪表板"
        },
        {
          "action": "wait_for_element",
          "selector": ".dashboard-container",
          "timeout": 10000
        },
        {
          "action": "check_no_errors",
          "description": "确认无JavaScript错误"
        },
        {
          "action": "navigate",
          "target": "http://localhost:3000/sector-analysis",
          "description": "进入板块分析"
        },
        {
          "action": "wait_for_element",
          "selector": "table",
          "timeout": 10000
        },
        {
          "action": "screenshot",
          "path": "logs/screenshots/user_workflow_final.png"
        }
      ]
    }
  ],
  "global_settings": {
    "screenshot_on_error": true,
    "max_wait_time": 30000,
    "retry_count": 2,
    "headless": false,
    "viewport": {
      "width": 1920,
      "height": 1080
    }
  }
}
EOF

# 创建MCP启动脚本
echo -e "${YELLOW}📝 创建MCP服务启动脚本...${NC}"
cat > scripts/quality/mcp/start_browser_mcp.sh << 'EOF'
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
EOF

chmod +x scripts/quality/mcp/start_browser_mcp.sh

# 创建MCP停止脚本
cat > scripts/quality/mcp/stop_browser_mcp.sh << 'EOF'
#!/bin/bash

echo "🛑 停止Browser MCP服务器..."

# 停止端口3001上的服务
if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null; then
    kill $(lsof -t -i:3001)
    echo "✅ Browser MCP服务器已停止"
else
    echo "ℹ️ Browser MCP服务器未运行"
fi
EOF

chmod +x scripts/quality/mcp/stop_browser_mcp.sh

echo ""
echo -e "${CYAN}Phase 4: Claude Code MCP配置${NC}"
echo "========================================================================"

# 检查是否已配置MCP
echo -e "${YELLOW}🔧 配置Claude Code MCP连接...${NC}"

# 创建Claude Code配置脚本
cat > scripts/setup/configure_claude_mcp.sh << 'EOF'
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
EOF

chmod +x scripts/setup/configure_claude_mcp.sh

echo ""
echo -e "${CYAN}Phase 5: 基础前端检查脚本${NC}"
echo "========================================================================"

# 创建简单的前端检查脚本
echo -e "${YELLOW}📝 创建前端检查脚本...${NC}"
cat > scripts/quality/mcp/basic_frontend_check.sh << 'EOF'
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
EOF

chmod +x scripts/quality/mcp/basic_frontend_check.sh

echo ""
echo -e "${GREEN}🎉 Chrome MCP集成安装完成！${NC}"
echo "========================================================================"

echo -e "${BLUE}📋 后续步骤:${NC}"
echo ""
echo "1. ${YELLOW}安装Chrome扩展${NC} (如果选择了Browser MCP):"
echo "   访问: https://chromewebstore.google.com/detail/browser-mcp-automate-your/bjfgambnhccakkhmkepdoekmckoijdlc"
echo ""
echo "2. ${YELLOW}配置Claude Code MCP连接${NC}:"
echo "   运行: scripts/setup/configure_claude_mcp.sh"
echo ""
echo "3. ${YELLOW}测试基础功能${NC}:"
echo "   运行: scripts/quality/mcp/basic_frontend_check.sh"
echo ""
echo "4. ${YELLOW}启动MCP服务${NC}:"
echo "   启动: scripts/quality/mcp/start_browser_mcp.sh"
echo "   停止: scripts/quality/mcp/stop_browser_mcp.sh"
echo ""

echo -e "${CYAN}📁 创建的文件:${NC}"
echo "   - configs/mcp/browser_mcp_config.json"
echo "   - configs/mcp/frontend_check_config.json"
echo "   - scripts/quality/mcp/start_browser_mcp.sh"
echo "   - scripts/quality/mcp/stop_browser_mcp.sh"
echo "   - scripts/quality/mcp/basic_frontend_check.sh"
echo "   - scripts/setup/configure_claude_mcp.sh"
echo ""

echo -e "${GREEN}🚀 现在可以使用Claude进行前端自动化检查了！${NC}"
echo ""
echo "示例命令:"
echo '   claude "使用MCP检查前端应用的运行状态"'
echo '   claude "截图当前前端应用并检查是否有渲染问题"'
echo '   claude "自动测试从主页到仪表板的用户流程"'

echo ""
echo -e "${PURPLE}✨ 安装完成！${NC}"