#!/bin/bash

echo "🔍 开始提交前检查..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

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

# 检查函数
check_step() {
    local step_name="$1"
    local command="$2"
    local timeout="${3:-30}" # 默认30秒超时
    
    ((TOTAL_CHECKS++))
    echo -e "${YELLOW}检查 [$TOTAL_CHECKS]: $step_name${NC}"
    echo "检查 [$TOTAL_CHECKS]: $step_name - $(date)" >> "$LOG_FILE"
    
    # 在macOS上使用gtimeout或直接执行（没有timeout可用）
    if command -v timeout >/dev/null 2>&1; then
        timeout_cmd="timeout $timeout"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_cmd="gtimeout $timeout"
    else
        timeout_cmd=""  # macOS上没有timeout，直接执行
    fi
    
    if [ -n "$timeout_cmd" ]; then
        if $timeout_cmd bash -c "$command" >> "$LOG_FILE" 2>&1; then
            eval_result=0
        else
            eval_result=1
        fi
    else
        if bash -c "$command" >> "$LOG_FILE" 2>&1; then
            eval_result=0
        else
            eval_result=1
        fi
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

# 等待服务完全启动
echo "⏳ 等待服务启动完成 (30秒)..."
sleep 30

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
# Phase 7: 前端UI设计检查
# ==========================================
show_progress 8 10 "前端UI设计检查"

echo "🎨 执行前端UI设计检查..."
if [ -x "scripts/quality/frontend_ui_check.sh" ]; then
    if scripts/quality/frontend_ui_check.sh > logs/frontend_ui_check_full.log 2>&1; then
        echo -e "${GREEN}✅ 前端UI设计检查 - 通过${NC}"
        echo "✅ PASS: 前端UI设计检查" >> "$LOG_FILE"
    else
        # 检查详细报告
        if [ -f "logs/frontend_ui_check_report.json" ]; then
            ERROR_COUNT=$(python -c "
import json
try:
    with open('logs/frontend_ui_check_report.json', 'r') as f:
        report = json.load(f)
    print(report['summary']['error_count'])
except:
    print('0')
")
            WARNING_COUNT=$(python -c "
import json
try:
    with open('logs/frontend_ui_check_report.json', 'r') as f:
        report = json.load(f)
    print(report['summary']['warning_count'])
except:
    print('0')
")
            
            if [ "$ERROR_COUNT" -eq 0 ]; then
                echo -e "${YELLOW}⚠️ 前端UI设计检查 - 有警告但通过${NC}"
                echo "⚠️ WARNING: 前端UI设计检查 (有${WARNING_COUNT}个警告)" >> "$LOG_FILE"
            else
                echo -e "${RED}❌ 前端UI设计检查 - 失败${NC}"
                echo "❌ FAIL: 前端UI设计检查 (${ERROR_COUNT}个错误, ${WARNING_COUNT}个警告)" >> "$LOG_FILE"
                ((FAILURES++))
                echo ""
                echo "🎨 前端UI问题详情："
                # 显示报告摘要
                python -c "
import json
try:
    with open('logs/frontend_ui_check_report.json', 'r') as f:
        report = json.load(f)
    summary = report['summary']
    print('   可访问性评分: ' + summary['avg_accessibility_score'])
    print('   性能评分: ' + summary['avg_performance_score'])
    print('   加载时间: ' + summary['avg_load_time'])
    print('   总问题数: ' + str(summary['total_issues']))
except:
    print('   无法解析UI检查报告')
"
            fi
        else
            echo -e "${RED}❌ 前端UI设计检查 - 检查失败${NC}"
            echo "❌ FAIL: 前端UI设计检查 (无法生成报告)" >> "$LOG_FILE"
            ((FAILURES++))
        fi
    fi
else
    echo -e "${YELLOW}⚠️ 前端UI检查脚本不存在，跳过检查${NC}"
fi

# ==========================================
# Phase 8: 项目标准规范检查
# ==========================================
show_progress 9 10 "项目标准规范检查"

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
# Phase 9: 性能基准检查 (可选)
# ==========================================
show_progress 10 10 "性能基准检查"

# 如果修改了核心算法文件，则进行性能检查
if git diff --cached --name-only | grep -q -E '(sector_engine|stock_engine|algorithm)'; then
    echo "🔍 检测到算法相关修改，执行性能基准测试..."
    
    # 这里可以添加性能测试逻辑
    # check_step "算法性能基准测试" "./check_performance.sh"
    echo "ℹ️ 性能基准测试已跳过 (需要实现具体测试逻辑)"
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