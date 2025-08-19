#!/bin/bash

# 统一前端UI检查脚本
# 集成TypeScript错误检测、UI质量检查、页面渲染验证

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

echo -e "${PURPLE}🚀 启动统一前端UI检查系统${NC}"
echo "========================================================================"
echo "📍 项目根目录: $PROJECT_ROOT"
echo "🎯 检查范围: TypeScript编译, UI质量, 页面渲染, 可访问性"
echo "🔧 检查工具: TypeScript Compiler + Python分析器 + BeautifulSoup"

# 创建日志目录
mkdir -p logs

# 检查前置条件
echo ""
echo -e "${CYAN}Phase 0: 前置条件检查${NC}"
echo "========================================================================"

# 检查基本目录结构

if [ ! -d "frontend" ]; then
    echo -e "${RED}❌ 前端目录不存在${NC}"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo -e "${RED}❌ package.json不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 基础环境检查通过${NC}"

# 激活虚拟环境
source "$PROJECT_ROOT/venv/bin/activate"

# 检查Python依赖
echo -e "${YELLOW}📦 检查Python依赖...${NC}"
pip list | grep -E "(requests|beautifulsoup4)" > /dev/null 2>&1 || {
    echo "安装页面检查依赖..."
    pip install requests beautifulsoup4 > logs/ui_deps_install.log 2>&1
}

# Phase 1: TypeScript编译错误检查
echo ""
echo -e "${CYAN}Phase 1: TypeScript编译错误检测${NC}"
echo "========================================================================"

echo -e "${YELLOW}🔧 运行TypeScript错误分析器...${NC}"
python scripts/quality/typescript_error_analyzer.py > logs/typescript_analysis_output.log 2>&1
TS_ANALYSIS_RESULT=$?

if [ $TS_ANALYSIS_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ TypeScript编译检查通过 - 无错误${NC}"
    TS_STATUS="PASS"
    TS_ERROR_COUNT=0
else
    echo -e "${RED}❌ 发现TypeScript编译错误${NC}"
    TS_STATUS="FAIL"
    
    # 从JSON报告中提取错误信息
    if [ -f "logs/typescript_error_analysis.json" ]; then
        TS_ERROR_COUNT=$(python -c "
import json
try:
    with open('logs/typescript_error_analysis.json', 'r') as f:
        data = json.load(f)
    print(data['total_errors'])
except:
    print('0')
")
        
        TS_WARNING_COUNT=$(python -c "
import json
try:
    with open('logs/typescript_error_analysis.json', 'r') as f:
        data = json.load(f)
    print(data['total_warnings'])
except:
    print('0')
")
        
        ESTIMATED_FIX_TIME=$(python -c "
import json
try:
    with open('logs/typescript_error_analysis.json', 'r') as f:
        data = json.load(f)
    print(data['estimated_fix_time'])
except:
    print('未知')
")
        
        echo "   📊 错误统计:"
        echo "      总错误数: $TS_ERROR_COUNT"
        echo "      总警告数: $TS_WARNING_COUNT" 
        echo "      预估修复时间: $ESTIMATED_FIX_TIME"
        
        # 显示错误类型分布
        echo ""
        echo -e "${YELLOW}📋 错误类型分布:${NC}"
        python -c "
import json
try:
    with open('logs/typescript_error_analysis.json', 'r') as f:
        data = json.load(f)
    for error_type, count in data['errors_by_type'].items():
        print(f'      {error_type}: {count}个')
except:
    print('      无法读取错误分布')
"
        
        # 显示修复优先级
        echo ""
        echo -e "${YELLOW}🔧 修复建议:${NC}"
        python -c "
import json
try:
    with open('logs/typescript_error_analysis.json', 'r') as f:
        data = json.load(f)
    for priority in data['fix_priority']:
        print(f'      {priority}')
    
    # 显示前3个具体错误
    print('\\n   🎯 具体错误示例:')
    for i, error in enumerate(data['all_errors'][:3]):
        print(f'      {i+1}. {error[\"file_path\"]}:{error[\"line\"]} - {error[\"error_code\"]}')
        if error['fix_suggestions']:
            print(f'         💡 {error[\"fix_suggestions\"][0]}')
except:
    print('      无法读取修复建议')
"
    else
        TS_ERROR_COUNT="未知"
    fi
fi

# Phase 2: 前端服务和页面渲染检查
echo ""
echo -e "${CYAN}Phase 2: 前端服务和页面渲染检查${NC}"
echo "========================================================================"

# 检查前端服务状态
echo -e "${YELLOW}🌐 检查前端服务状态...${NC}"
SERVICES_STARTED=false

if ! curl -s -I http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ 前端服务未运行，尝试启动...${NC}"
    
    # 启动服务
    scripts/deployment/start_services.sh > logs/ui_check_startup.log 2>&1 &
    STARTUP_PID=$!
    
    # 等待服务启动
    echo -e "${YELLOW}⏳ 等待服务启动 (最多60秒)...${NC}"
    for i in {1..12}; do
        if curl -s -I http://localhost:3000 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ 前端服务启动成功${NC}"
            SERVICES_STARTED=true
            break
        elif [ $i -eq 12 ]; then
            echo -e "${RED}❌ 前端服务启动失败${NC}"
            RUNTIME_STATUS="FAIL"
        else
            echo "   等待中... ($i/12)"
            sleep 5
        fi
    done
else
    echo -e "${GREEN}✅ 前端服务已运行${NC}"
fi

# 页面渲染检查
if curl -s -I http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}🔍 检查页面渲染状态...${NC}"
    
    # 获取页面内容
    PAGE_CONTENT=$(curl -s http://localhost:3000 2>/dev/null || echo "")
    
    if [ -n "$PAGE_CONTENT" ]; then
        # 检查编译错误页面
        if echo "$PAGE_CONTENT" | grep -q "Compiled with problems:"; then
            echo -e "${RED}❌ 页面显示编译错误${NC}"
            RUNTIME_STATUS="FAIL"
            
            # 尝试提取错误信息
            echo -e "${YELLOW}🔍 页面编译错误摘要:${NC}"
            echo "$PAGE_CONTENT" | grep -A 3 -B 1 "ERROR in" | head -10 | sed 's/^/      /'
            
        elif echo "$PAGE_CONTENT" | grep -q "<div id=\"root\"></div>" || [ ${#PAGE_CONTENT} -lt 1000 ]; then
            echo -e "${RED}❌ React应用未正确渲染${NC}"
            RUNTIME_STATUS="FAIL"
        else
            echo -e "${GREEN}✅ 页面正常渲染${NC}"
            RUNTIME_STATUS="PASS"
            
            # 基本页面质量检查
            echo -e "${YELLOW}📊 页面质量检查:${NC}"
            
            # 检查页面标题
            PAGE_TITLE=$(echo "$PAGE_CONTENT" | grep -o '<title>[^<]*' | sed 's/<title>//' || echo "")
            if [ -n "$PAGE_TITLE" ]; then
                echo "      页面标题: $PAGE_TITLE ✅"
            else
                echo "      页面标题: 缺失 ⚠️"
            fi
            
            # 检查基本元素
            ROOT_CONTENT=$(echo "$PAGE_CONTENT" | grep -o '<div id="root">[^<]*' | wc -c)
            echo "      Root内容长度: ${ROOT_CONTENT}字符"
            
            # 检查CSS链接
            CSS_COUNT=$(echo "$PAGE_CONTENT" | grep -c 'rel="stylesheet"' || echo "0")
            echo "      CSS文件数量: $CSS_COUNT"
            
            # 检查JavaScript
            JS_COUNT=$(echo "$PAGE_CONTENT" | grep -c '<script' || echo "0")
            echo "      JavaScript文件数量: $JS_COUNT"
        fi
    else
        echo -e "${RED}❌ 无法获取页面内容${NC}"
        RUNTIME_STATUS="FAIL"
    fi
else
    echo -e "${RED}❌ 前端服务不可访问${NC}"
    RUNTIME_STATUS="FAIL"
fi

# Phase 3: 生成综合报告
echo ""
echo -e "${CYAN}Phase 3: 生成综合检查报告${NC}"
echo "========================================================================"

# 计算整体状态
if [ "$TS_STATUS" = "FAIL" ]; then
    OVERALL_STATUS="FAIL"
    OVERALL_REASON="TypeScript编译错误"
elif [ "$RUNTIME_STATUS" = "FAIL" ]; then
    OVERALL_STATUS="FAIL" 
    OVERALL_REASON="页面渲染问题"
else
    OVERALL_STATUS="PASS"
    OVERALL_REASON="所有检查通过"
fi

# 创建综合报告
REPORT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > logs/unified_frontend_check_report.json << EOF
{
  "timestamp": "$REPORT_TIME",
  "overall_status": "$OVERALL_STATUS",
  "overall_reason": "$OVERALL_REASON",
  "checks": {
    "typescript_compilation": {
      "status": "$TS_STATUS",
      "error_count": ${TS_ERROR_COUNT:-0},
      "warning_count": ${TS_WARNING_COUNT:-0},
      "estimated_fix_time": "${ESTIMATED_FIX_TIME:-unknown}",
      "details_file": "logs/typescript_error_analysis.json"
    },
    "page_rendering": {
      "status": "$RUNTIME_STATUS",
      "page_accessible": $([ "$RUNTIME_STATUS" = "PASS" ] && echo "true" || echo "false"),
      "react_rendered": $([ "$RUNTIME_STATUS" = "PASS" ] && echo "true" || echo "false")
    },
    "service_management": {
      "services_started_by_script": $SERVICES_STARTED,
      "frontend_service_running": $(curl -s -I http://localhost:3000 > /dev/null 2>&1 && echo "true" || echo "false")
    }
  },
  "logs": {
    "typescript_analysis": "logs/typescript_analysis_output.log",
    "service_startup": "logs/ui_check_startup.log",
    "unified_check": "logs/unified_frontend_check_report.json"
  }
}
EOF

# 清理：停止服务（如果是脚本启动的）
if [ "$SERVICES_STARTED" = true ]; then
    echo -e "${YELLOW}🛑 停止测试服务...${NC}"
    scripts/deployment/stop_services.sh > /dev/null 2>&1
fi

# Phase 4: 最终报告和建议
echo ""
echo -e "${PURPLE}📋 统一前端UI检查完成${NC}"
echo "========================================================================"

echo -e "${BLUE}📊 检查结果摘要:${NC}"
echo "   整体状态: $OVERALL_STATUS"
echo "   原因: $OVERALL_REASON"
echo "   TypeScript编译: $TS_STATUS"
echo "   页面渲染: $RUNTIME_STATUS"

if [ "$TS_ERROR_COUNT" != "0" ] && [ "$TS_ERROR_COUNT" != "未知" ]; then
    echo ""
    echo -e "${RED}🚨 TypeScript编译问题需要修复:${NC}"
    echo "   错误数量: $TS_ERROR_COUNT"
    echo "   预估修复时间: ${ESTIMATED_FIX_TIME:-未知}"
    echo "   详细分析: cat logs/typescript_error_analysis.json"
fi

echo ""
echo -e "${BLUE}📝 详细报告文件:${NC}"
echo "   综合报告: logs/unified_frontend_check_report.json"
echo "   TypeScript分析: logs/typescript_error_analysis.json"
echo "   服务启动日志: logs/ui_check_startup.log"

# 提供修复建议
if [ "$OVERALL_STATUS" != "PASS" ]; then
    echo ""
    echo -e "${YELLOW}🔧 修复建议:${NC}"
    
    if [ "$TS_STATUS" = "FAIL" ]; then
        echo "   1. 修复TypeScript编译错误 (高优先级)"
        echo "      → 运行: python scripts/quality/typescript_error_analyzer.py"
        echo "      → 查看详细建议和修复方案"
        echo "      → 使用VSCode TypeScript检查获得实时反馈"
    fi
    
    if [ "$RUNTIME_STATUS" = "FAIL" ]; then
        echo "   2. 修复页面渲染问题 (高优先级)"
        echo "      → 检查浏览器控制台错误信息"
        echo "      → 确保React组件正确导入和渲染"
        echo "      → 检查CSS和JavaScript资源加载"
    fi
    
    echo ""
    echo -e "${CYAN}💡 快速修复流程:${NC}"
    echo "   1. 运行 TypeScript 分析器获取具体错误和修复建议"
    echo "   2. 使用IDE的自动修复功能处理简单错误"
    echo "   3. 重新运行此检查脚本验证修复效果"
fi

# 返回状态码
if [ "$OVERALL_STATUS" = "PASS" ]; then
    echo -e "${GREEN}🎉 所有前端检查通过！${NC}"
    exit 0
else
    echo -e "${RED}❌ 前端检查失败，请根据建议修复问题后重新运行${NC}"
    exit 1
fi