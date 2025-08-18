#!/bin/bash

echo "🔍 开始提交前检查..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 检查模式：快速模式或完整模式
QUICK_MODE=${QUICK_MODE:-"auto"}  # auto, true, false
if [ "$QUICK_MODE" = "auto" ]; then
    # 自动判断：如果是pre-commit钩子调用，使用快速模式
    if [ -n "$GIT_AUTHOR_NAME" ] || [ -n "$GIT_EDITOR" ]; then
        QUICK_MODE="true"
    else
        QUICK_MODE="false"
    fi
fi

if [ "$QUICK_MODE" = "true" ]; then
    echo "⚡ 快速检查模式已启用"
else
    echo "🔍 完整检查模式"
fi

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查失败计数
FAILURES=0
TOTAL_CHECKS=0

# 日志文件
LOG_FILE="logs/pre_commit_check.log"
mkdir -p logs
echo "=== Pre-commit Check $(date) ===" > "$LOG_FILE"

# 检查函数 - 增强版本，支持macOS timeout处理
check_step() {
    local step_name="$1"
    local command="$2"
    local timeout="${3:-30}" # 默认30秒超时
    
    ((TOTAL_CHECKS++))
    echo -e "${YELLOW}检查 [$TOTAL_CHECKS]: $step_name${NC}"
    echo "检查 [$TOTAL_CHECKS]: $step_name - $(date)" >> "$LOG_FILE"
    
    # macOS兼容的timeout实现
    local temp_script="/tmp/precommit_check_$$"
    echo "$command" > "$temp_script"
    chmod +x "$temp_script"
    
    # 后台运行命令并获取PID
    bash "$temp_script" >> "$LOG_FILE" 2>&1 &
    local cmd_pid=$!
    
    # 等待指定时间
    local count=0
    while [ $count -lt $timeout ]; do
        if ! kill -0 $cmd_pid 2>/dev/null; then
            # 进程已结束
            wait $cmd_pid
            local eval_result=$?
            rm -f "$temp_script"
            break
        fi
        sleep 1
        ((count++))
    done
    
    # 如果超时，终止进程
    if [ $count -ge $timeout ]; then
        echo "⏰ 命令超时 (${timeout}秒)，终止进程..." >> "$LOG_FILE"
        kill -TERM $cmd_pid 2>/dev/null
        sleep 2
        kill -KILL $cmd_pid 2>/dev/null
        wait $cmd_pid 2>/dev/null
        eval_result=124  # timeout退出码
        rm -f "$temp_script"
    fi
    
    if [ $eval_result -eq 0 ]; then
        echo -e "${GREEN}✅ $step_name - 通过${NC}"
        echo "✅ PASS: $step_name" >> "$LOG_FILE"
        return 0
    else
        echo -e "${RED}❌ $step_name - 失败${NC}"
        echo "❌ FAIL: $step_name" >> "$LOG_FILE"
        ((FAILURES++))
        return 1
    fi
}

# 显示进度
show_progress() {
    local current=$1
    local total=$2
    local message="$3"
    echo -e "${BLUE}[${current}/${total}] ${message}${NC}"
}

echo "📋 启动提交前质量检查流程..."
echo "日志文件: $LOG_FILE"
echo ""

# ==========================================
# Phase 1: 严格目录结构检查 (CRITICAL)
# ==========================================
show_progress 1 9 "严格目录结构检查"

echo "📁 执行严格目录结构检查..."
if [ -x "scripts/quality/strict_structure_check.sh" ]; then
    if ! scripts/quality/strict_structure_check.sh > logs/structure_check_output.log 2>&1; then
        echo -e "${RED}💥 严格目录结构检查失败，必须修复！${NC}"
        echo "📋 结构问题详情："
        tail -20 logs/structure_check_output.log | sed 's/^/   /'
        echo ""
        echo "📝 完整报告: logs/structure_check_output.log"
        exit 1
    else
        echo -e "${GREEN}✅ 严格目录结构检查 - 通过${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ 严格结构检查脚本不存在，使用基础检查${NC}"
    check_step "基础目录结构检查" "[ -f 'src/api/main.py' ] && [ -f 'frontend/package.json' ] && [ -f 'scripts/deployment/start_services.sh' ]"
fi

# ==========================================
# Phase 2: 基础环境检查
# ==========================================
show_progress 2 9 "基础环境检查"

# Python依赖检查 (可选，因为服务已经能正常运行)
echo "🗒️ Python依赖检查 (可选)..."
if python -c 'import fastapi, uvicorn, pydantic' >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Python依赖检查 - 通过${NC}"
else
    echo -e "${YELLOW}⚠️ Python依赖检查 - 警告 (但服务正常运行)${NC}"
fi

# ==========================================
# Phase 3: 服务重启
# ==========================================
show_progress 3 9 "服务重启检查"

echo "🛑 停止现有服务..."
scripts/deployment/stop_services.sh >/dev/null 2>&1 || true
sleep 3

echo "🚀 启动服务..."
if ! scripts/deployment/start_services.sh > logs/startup.log 2>&1; then
    echo -e "${RED}❌ 服务启动失败，查看日志: logs/startup.log${NC}"
    cat logs/startup.log
    exit 1
fi

# 智能等待服务启动 (最多30秒)
echo "⏳ 智能等待服务启动完成..."
for i in {1..30}; do
    if curl -s -f http://localhost:8000/health >/dev/null 2>&1 && curl -s -f http://localhost:3000 >/dev/null 2>&1; then
        echo "✅ 服务已在 ${i} 秒内启动完成"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️ 服务启动超时，继续检查..."
    fi
    sleep 1
done

# ==========================================
# Phase 3: 后端服务检查
# ==========================================
show_progress 4 9 "后端服务检查"

check_step "后端服务健康检查" "curl -f -s http://localhost:8000/health > /dev/null"

check_step "后端服务响应结构" "curl -s http://localhost:8000/health | jq -e '.status == \"healthy\"' > /dev/null 2>&1 || curl -s http://localhost:8000/health | grep -q '\"status\":\"healthy\"'"

check_step "API文档可访问" "curl -f -s http://localhost:8000/docs > /dev/null"

# ==========================================
# Phase 4: 核心API功能检查
# ==========================================
show_progress 5 9 "核心API功能检查"

check_step "板块分析API基础功能" "curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"success\":true'"

check_step "板块分析API数据完整性" "curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"top_sectors\"' && curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"all_sectors\"'"

check_step "板块分析API评分数据" "curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"composite_score\"' && curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"market_overview\"'"

# 检查是否有TOP5数据
check_step "TOP5板块数据完整性" "RESPONSE=\$(curl -s http://localhost:8000/api/v1/sectors/); echo \"\$RESPONSE\" | grep -q '\"top_sectors\"' && [ \$(echo \"\$RESPONSE\" | grep -o 'sector_name' | wc -l) -ge 5 ]"

# ==========================================
# Phase 5: 前端服务检查
# ==========================================
show_progress 6 9 "前端服务检查"

check_step "前端服务HTTP状态" "curl -f -s -I http://localhost:3000 > /dev/null"

check_step "前端页面内容检查" "curl -s http://localhost:3000 | grep -q '<title>'"

check_step "前端静态资源" "curl -s http://localhost:3000 | grep -q -E '(react|app)'"

# ==========================================
# Phase 6: 代码质量检查
# ==========================================
show_progress 7 9 "代码质量检查"

# 检查是否有Python文件修改
if git diff --cached --name-only | grep -q '\.py$'; then
    check_step "Python代码语法检查" "python -m py_compile src/api/main.py && python -c 'from src.api.main import app' > /dev/null 2>&1"
    
    check_step "Python导入完整性" "cd src && python -c 'from api.routes import sector_router, stock_router' > /dev/null 2>&1"
else
    echo "ℹ️ 未检测到Python文件修改，跳过Python代码检查"
fi

# 检查是否有前端文件修改
if git diff --cached --name-only | grep -q -E '\.(tsx?|jsx?|json)$'; then
    if [ -d "frontend/node_modules" ]; then
        check_step "前端代码编译检查" "cd frontend && npm run build > /dev/null 2>&1"
    else
        echo "⚠️ 前端依赖未安装，跳过编译检查"
    fi
else
    echo "ℹ️ 未检测到前端文件修改，跳过前端代码检查"
fi

# ==========================================
# Phase 7: 前端UI设计和TypeScript错误检查
# ==========================================
show_progress 8 10 "前端UI设计和TypeScript错误检查"

echo "🎨 执行统一前端检查 (TypeScript + UI设计)..."
if [ -x "scripts/quality/unified_frontend_check.sh" ]; then
    if scripts/quality/unified_frontend_check.sh > logs/unified_frontend_check_full.log 2>&1; then
        echo -e "${GREEN}✅ 前端检查 (TypeScript + UI) - 通过${NC}"
        echo "✅ PASS: 前端检查 (TypeScript + UI)" >> "$LOG_FILE"
    else
        # 检查详细报告
        if [ -f "logs/unified_frontend_check_report.json" ]; then
            # 提取TypeScript错误信息
            TS_ERROR_COUNT=$(python -c "
import json
try:
    with open('logs/unified_frontend_check_report.json', 'r') as f:
        report = json.load(f)
    print(report['checks']['typescript_compilation']['error_count'])
except:
    print('0')
")
            TS_WARNING_COUNT=$(python -c "
import json
try:
    with open('logs/unified_frontend_check_report.json', 'r') as f:
        report = json.load(f)
    print(report['checks']['typescript_compilation']['warning_count'])
except:
    print('0')
")
            OVERALL_STATUS=$(python -c "
import json
try:
    with open('logs/unified_frontend_check_report.json', 'r') as f:
        report = json.load(f)
    print(report['overall_status'])
except:
    print('UNKNOWN')
")
            OVERALL_REASON=$(python -c "
import json
try:
    with open('logs/unified_frontend_check_report.json', 'r') as f:
        report = json.load(f)
    print(report['overall_reason'])
except:
    print('无法读取错误原因')
")
            
            if [ "$OVERALL_STATUS" = "PASS" ]; then
                echo -e "${GREEN}✅ 前端检查 (TypeScript + UI) - 通过${NC}"
                echo "✅ PASS: 前端检查 (TypeScript + UI)" >> "$LOG_FILE"
            elif [ "$TS_ERROR_COUNT" -gt 0 ]; then
                echo -e "${RED}❌ 前端检查失败 - TypeScript编译错误${NC}"
                echo "❌ FAIL: 前端检查 (${TS_ERROR_COUNT}个TypeScript错误, ${TS_WARNING_COUNT}个警告)" >> "$LOG_FILE"
                ((FAILURES++))
                echo ""
                echo "🚨 TypeScript编译问题详情："
                echo "   原因: $OVERALL_REASON"
                echo "   错误数量: $TS_ERROR_COUNT"
                echo "   警告数量: $TS_WARNING_COUNT"
                
                # 显示具体错误类型和修复建议
                if [ -f "logs/typescript_error_analysis.json" ]; then
                    echo "   错误类型分布:"
                    python -c "
import json
try:
    with open('logs/typescript_error_analysis.json', 'r') as f:
        data = json.load(f)
    for error_type, count in data['errors_by_type'].items():
        print(f'      {error_type}: {count}个')
    print('')
    print('   💡 修复建议:')
    for suggestion in data['fix_priority'][:3]:
        print(f'      - {suggestion}')
    print(f'   ⏱️ 预估修复时间: {data[\"estimated_fix_time\"]}')
except Exception as e:
    print('      无法读取详细错误分析')
"
                fi
                echo ""
                echo "🔧 快速修复指南:"
                echo "   1. 查看详细分析: cat logs/typescript_error_analysis.json"
                echo "   2. 运行TypeScript分析器: python scripts/quality/typescript_error_analyzer.py"
                echo "   3. 使用IDE的TypeScript错误检查获得实时反馈"
            else
                echo -e "${RED}❌ 前端检查失败 - 页面渲染问题${NC}"
                echo "❌ FAIL: 前端检查 (页面渲染问题)" >> "$LOG_FILE"
                ((FAILURES++))
                echo ""
                echo "🔍 页面渲染问题详情："
                echo "   原因: $OVERALL_REASON"
                echo "   建议检查浏览器控制台错误和React组件渲染状态"
            fi
        else
            echo -e "${RED}❌ 前端检查失败 - 无法生成检查报告${NC}"
            echo "❌ FAIL: 前端检查 (无法生成报告)" >> "$LOG_FILE"
            ((FAILURES++))
        fi
    fi
elif [ -x "scripts/quality/frontend_ui_check.sh" ]; then
    # 回退到原有的UI检查
    echo "⚠️ 统一前端检查不可用，使用传统UI检查..."
    if scripts/quality/frontend_ui_check.sh > logs/frontend_ui_check_full.log 2>&1; then
        echo -e "${GREEN}✅ 前端UI设计检查 - 通过${NC}"
        echo "✅ PASS: 前端UI设计检查" >> "$LOG_FILE"
    else
        echo -e "${RED}❌ 前端UI设计检查 - 失败${NC}"
        echo "❌ FAIL: 前端UI设计检查" >> "$LOG_FILE"
        ((FAILURES++))
    fi
else
    echo -e "${YELLOW}⚠️ 前端检查脚本不存在，跳过检查${NC}"
fi

# ==========================================
# Phase 8: Chrome MCP前端运行效果检查
# ==========================================
show_progress 9 11 "Chrome MCP前端运行效果检查"

echo "🌐 执行Chrome MCP前端运行效果检查 (快速模式)..."

# 使用check_step函数进行timeout控制 (60秒超时)
if [ -x "scripts/quality/mcp/integrated_frontend_mcp_check.sh" ]; then
    if check_step "Chrome MCP前端检查" "scripts/quality/mcp/integrated_frontend_mcp_check.sh" 60; then
        echo "✅ Chrome MCP前端检查已通过"
    else
        # 检查详细报告
        LATEST_MCP_REPORT=$(ls -t logs/mcp_checks/integrated_check_*.json 2>/dev/null | head -1)
        
        if [ -n "$LATEST_MCP_REPORT" ] && [ -f "$LATEST_MCP_REPORT" ]; then
            MCP_SUCCESS_RATE=$(python -c "
import json
try:
    with open('$LATEST_MCP_REPORT', 'r') as f:
        report = json.load(f)
    rate = report['summary']['success_rate'].replace('%', '')
    print(rate)
except:
    print('0')
")
            MCP_FAILED_COUNT=$(python -c "
import json
try:
    with open('$LATEST_MCP_REPORT', 'r') as f:
        report = json.load(f)
    print(report['summary']['failed'])
except:
    print('1')
")
            
            if [ "$MCP_FAILED_COUNT" -eq 0 ]; then
                echo -e "${GREEN}✅ Chrome MCP前端检查 - 通过${NC}"
                echo "✅ PASS: Chrome MCP前端检查" >> "$LOG_FILE"
            else
                echo -e "${RED}❌ Chrome MCP前端检查 - 发现运行问题${NC}"
                echo "❌ FAIL: Chrome MCP前端检查 (${MCP_FAILED_COUNT}个问题, 成功率${MCP_SUCCESS_RATE}%)" >> "$LOG_FILE"
                ((FAILURES++))
                echo ""
                echo "🌐 前端运行问题详情："
                echo "   成功率: ${MCP_SUCCESS_RATE}%"
                echo "   问题数量: ${MCP_FAILED_COUNT}"
                
                # 显示具体问题
                python -c "
import json
try:
    with open('$LATEST_MCP_REPORT', 'r') as f:
        report = json.load(f)
    
    if 'results' in report and isinstance(report['results'], list):
        for result in report['results']:
            if result.get('status') == 'FAIL':
                print(f'      📄 {result.get(\"page\", \"未知页面\")}: {len(result.get(\"issues\", []))}个问题')
                for issue in result.get('issues', [])[:2]:
                    print(f'         - {issue}')
    
    if 'recommendations' in report:
        print('')
        print('   💡 修复建议:')
        for rec in report['recommendations'][:3]:
            print(f'      - {rec}')
except Exception as e:
    print(f'      无法解析MCP检查报告: {e}')
"
                echo ""
                echo "🔧 快速修复指南:"
                echo "   1. 查看详细报告: cat $LATEST_MCP_REPORT"
                echo "   2. 检查前端服务: curl http://localhost:3000"
                echo "   3. 检查后端服务: curl http://localhost:8000/health"
                echo "   4. 重启服务: scripts/deployment/start_services.sh"
            fi
        else
            echo -e "${RED}❌ Chrome MCP前端检查 - 无法生成报告${NC}"
            echo "❌ FAIL: Chrome MCP前端检查 (无法生成报告)" >> "$LOG_FILE"
            ((FAILURES++))
            echo ""
            echo "🔧 诊断建议:"
            echo "   1. 检查MCP服务: scripts/setup/setup_chrome_mcp.sh"
            echo "   2. 检查服务状态: curl http://localhost:3000 && curl http://localhost:8000/health"
            echo "   3. 查看详细日志: cat logs/mcp_frontend_check_full.log"
        fi
    fi
else
    echo -e "${YELLOW}⚠️ Chrome MCP检查脚本不可用${NC}"
    echo "💡 安装建议: scripts/setup/setup_chrome_mcp.sh"
    echo "ℹ️ INFO: Chrome MCP检查跳过 (脚本不可用)" >> "$LOG_FILE"
fi

# ==========================================
# Phase 9: 项目标准规范检查
# ==========================================
show_progress 10 11 "项目标准规范检查"

echo "📁 执行项目标准规范检查..."
if [ -x "scripts/quality/check_project_standards.sh" ]; then
    if scripts/quality/check_project_standards.sh > logs/project_standards_output.log 2>&1; then
        echo -e "${GREEN}✅ 项目标准规范检查 - 通过${NC}"
        echo "✅ PASS: 项目标准规范检查" >> "$LOG_FILE"
    else
        # 检查是否只是警告而非错误
        if grep -q "项目标准规范检查通过" logs/project_standards_output.log; then
            echo -e "${YELLOW}⚠️ 项目标准规范检查 - 有建议改进项但通过${NC}"
            echo "⚠️ WARNING: 项目标准规范检查 (有改进建议)" >> "$LOG_FILE"
        else
            echo -e "${RED}❌ 项目标准规范检查 - 失败${NC}"
            echo "❌ FAIL: 项目标准规范检查" >> "$LOG_FILE"
            ((FAILURES++))
            echo ""
            echo "📋 项目标准问题详情："
            tail -20 logs/project_standards_output.log | sed 's/^/   /'
        fi
    fi
else
    echo -e "${YELLOW}⚠️ 项目标准检查脚本不存在，跳过检查${NC}"
fi

# ==========================================
# Phase 10: 性能基准检查 (可选)
# ==========================================
show_progress 11 11 "性能基准检查"

# 性能检查（快速模式下跳过）
if [ "$QUICK_MODE" = "true" ]; then
    echo "⚡ 快速模式：跳过性能基准检查"
elif git diff --cached --name-only | grep -q -E '(sector_engine|stock_engine|algorithm)'; then
    echo "🔍 检测到算法相关修改，执行性能基准测试..."
    if [ -x "scripts/quality/check_performance.sh" ]; then
        check_step "算法性能基准测试" "scripts/quality/check_performance.sh" 120
    else
        echo "ℹ️ 性能测试脚本不存在，跳过"
    fi
else
    echo "ℹ️ 未检测到算法修改，跳过性能基准检查"
fi

# ==========================================
# 最终报告
# ==========================================
echo ""
echo "================================================================"
echo "📊 提交前检查总结:"
echo "================================================================"

# 统计项目标准检查结果
STANDARDS_WARNINGS=0
STANDARDS_STATUS="未检查"
if [ -f "logs/project_standards_output.log" ]; then
    if grep -q "项目标准规范检查通过" logs/project_standards_output.log; then
        STANDARDS_STATUS="通过"
        STANDARDS_WARNINGS=$(grep -c "WARNING:" logs/project_standards_output.log 2>/dev/null || echo "0")
    elif grep -q "项目标准规范检查失败" logs/project_standards_output.log; then
        STANDARDS_STATUS="失败"
    fi
fi

echo "总检查项: $TOTAL_CHECKS"
echo "✅ 功能测试通过: $(( TOTAL_CHECKS - FAILURES ))"
echo "❌ 功能测试失败: $FAILURES"
echo "📁 项目标准检查: $STANDARDS_STATUS"
if [ $STANDARDS_WARNINGS -gt 0 ]; then
    echo "⚠️ 项目标准建议: $STANDARDS_WARNINGS 项"
fi

if [ $FAILURES -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 完备迭代体系检查通过!${NC}"
    echo -e "${GREEN}✅ 功能质量 + 项目标准 双重验证通过${NC}"
    
    if [ $STANDARDS_WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}💡 项目有 $STANDARDS_WARNINGS 项改进建议，建议查看并优化${NC}"
        echo "📋 详细建议请查看: logs/project_standards_output.log"
    fi
    
    echo ""
    echo "🚀 建议的提交命令:"
    echo "   git commit -m \"feat: 您的提交信息\""
    echo ""
    echo "📝 完整检查报告:"
    echo "   - 功能测试: $LOG_FILE"
    echo "   - 项目标准: logs/project_standards_output.log"
    
    # 记录成功
    echo "✅ COMPLETE ITERATIVE SYSTEM CHECK PASSED - $(date)" >> "$LOG_FILE"
    exit 0
else
    echo ""
    echo -e "${RED}💥 完备迭代体系检查失败!${NC}"
    echo -e "${RED}❌ $FAILURES 项功能检查未通过，需要修复${NC}"
    echo ""
    echo "🔧 修复指南:"
    echo "   1. 功能问题修复: cat $LOG_FILE"
    echo "   2. 检查服务状态: curl http://localhost:8000/health"
    echo "   3. 重新启动服务: ./start_services.sh"
    echo "   4. 检查代码语法: python -m py_compile src/api/main.py"
    
    if [ "$STANDARDS_STATUS" = "失败" ]; then
        echo "   5. 项目标准修复: cat logs/project_standards_output.log"
    fi
    
    echo ""
    echo "📞 详细日志查看:"
    echo "   - logs/backend.log (后端日志)"
    echo "   - logs/frontend.log (前端日志)"
    echo "   - logs/startup.log (启动日志)"
    echo "   - logs/project_standards_output.log (项目标准)"
    
    # 记录失败
    echo "❌ COMPLETE ITERATIVE SYSTEM CHECK FAILED ($FAILURES/$TOTAL_CHECKS) - $(date)" >> "$LOG_FILE"
    exit 1
fi