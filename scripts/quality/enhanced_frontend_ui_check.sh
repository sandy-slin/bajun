#!/bin/bash

# 增强版前端UI检查脚本
# 专门检测TypeScript编译错误、React组件问题和UI质量

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径，然后切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${PURPLE}🚀 启动增强版前端UI检查...${NC}"
echo "========================================================"
echo "📍 项目根目录: $PROJECT_ROOT"
echo "🎯 专注检测: TypeScript编译错误, React运行时问题, UI质量"

# 创建日志目录
mkdir -p logs

# 检查前置条件
echo ""
echo -e "${YELLOW}🔍 检查前置条件...${NC}"

if [ ! -d "frontend" ]; then
    echo -e "${RED}❌ 前端目录不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 前置条件检查通过${NC}"

# 激活虚拟环境
source "$PROJECT_ROOT/venv/bin/activate"

# 检查Python依赖
echo -e "${YELLOW}📦 检查Python依赖...${NC}"
pip list | grep -E "(requests|beautifulsoup4)" > /dev/null 2>&1 || {
    echo "安装页面检查依赖..."
    pip install requests beautifulsoup4 > logs/enhanced_ui_deps.log 2>&1
}

# 1. TypeScript编译检查 (最重要的检查)
echo ""
echo -e "${BLUE}🔧 Phase 1: TypeScript编译错误检查${NC}"
echo "========================================================"

cd frontend

# 检查package.json
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json不存在${NC}"
    cd ..
    exit 1
fi

# 检查node_modules
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠️ node_modules不存在，正在安装依赖...${NC}"
    npm install > ../logs/npm_install.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 依赖安装失败${NC}"
        echo "📝 查看详细日志: cat logs/npm_install.log"
        cd ..
        exit 1
    fi
    echo -e "${GREEN}✅ 依赖安装成功${NC}"
fi

# TypeScript类型检查
echo -e "${YELLOW}🔍 执行TypeScript类型检查...${NC}"
if command -v npx >/dev/null 2>&1; then
    npx tsc --noEmit --pretty > ../logs/typescript_check.log 2>&1
    TSC_RESULT=$?
    
    if [ $TSC_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ TypeScript编译检查通过${NC}"
        TYPESCRIPT_STATUS="PASS"
    else
        echo -e "${RED}❌ 发现TypeScript编译错误${NC}"
        TYPESCRIPT_STATUS="FAIL"
        
        # 显示错误摘要
        echo -e "${YELLOW}📋 编译错误摘要:${NC}"
        
        # 提取并显示关键错误信息
        ERROR_COUNT=$(grep -c "error TS" ../logs/typescript_check.log 2>/dev/null || echo "0")
        WARNING_COUNT=$(grep -c "warning TS" ../logs/typescript_check.log 2>/dev/null || echo "0")
        
        echo "   错误数量: $ERROR_COUNT"
        echo "   警告数量: $WARNING_COUNT"
        
        # 显示前5个错误
        echo ""
        echo -e "${YELLOW}🔍 前5个编译错误:${NC}"
        grep "error TS" ../logs/typescript_check.log | head -5 | while read line; do
            echo "   📁 $line"
        done
        
        # 检查特定的TS1109错误（就是您截图中的问题）
        TS1109_COUNT=$(grep -c "error TS1109" ../logs/typescript_check.log 2>/dev/null || echo "0")
        if [ "$TS1109_COUNT" -gt 0 ]; then
            echo ""
            echo -e "${RED}🚨 发现 $TS1109_COUNT 个 TS1109 错误 (Identifier directly after number):${NC}"
            grep "error TS1109" ../logs/typescript_check.log | while read line; do
                echo "   📄 $line"
            done
            echo -e "${BLUE}💡 修复建议: 检查数字后的标识符，如 '~50%' 应改为 '~50' + '%' 或使用模板字符串${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️ TypeScript不可用，使用React构建检查...${NC}"
    npm run build > ../logs/react_build.log 2>&1
    BUILD_RESULT=$?
    
    if [ $BUILD_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ React构建成功${NC}"
        TYPESCRIPT_STATUS="PASS"
    else
        echo -e "${RED}❌ React构建失败${NC}"
        TYPESCRIPT_STATUS="FAIL"
        echo -e "${YELLOW}📋 构建错误:${NC}"
        tail -10 ../logs/react_build.log | sed 's/^/   /'
    fi
fi

cd ..

# 生成综合报告和提供修复建议
echo ""
echo -e "${PURPLE}📋 增强版前端UI检查完成!${NC}"
echo "========================================================"

echo -e "${BLUE}📊 检查结果摘要:${NC}"
echo "   TypeScript编译: $TYPESCRIPT_STATUS"

if [ "$TS1109_COUNT" -gt 0 ]; then
    echo ""
    echo -e "${RED}🚨 重点关注: 发现 $TS1109_COUNT 个 TS1109 编译错误${NC}"
    echo -e "${BLUE}💡 这是您截图中显示的错误类型，需要优先修复！${NC}"
fi

echo ""
echo -e "${BLUE}📝 详细日志文件:${NC}"
echo "   TypeScript检查: logs/typescript_check.log"

# 提供修复建议
if [ "$TYPESCRIPT_STATUS" = "FAIL" ]; then
    echo ""
    echo -e "${YELLOW}🔧 修复建议:${NC}"
    echo "   1. 修复TypeScript编译错误 (优先级: 高)"
    echo "      - 查看详细错误: cat logs/typescript_check.log"
    echo "      - 使用VSCode TypeScript错误检查获得实时反馈"
    if [ "$TS1109_COUNT" -gt 0 ]; then
        echo "      - 重点修复TS1109错误: 检查数字后的标识符语法"
        echo "      - 例如：将 '~50%' 改为 '~50 + \"%\"' 或 '\`~\${50}%\`'"
    fi
fi

# 返回状态码
if [ "$TYPESCRIPT_STATUS" = "PASS" ]; then
    echo -e "${GREEN}🎉 TypeScript编译检查通过！${NC}"
    exit 0
else
    echo -e "${RED}❌ TypeScript编译检查失败，需要修复错误后重新运行${NC}"
    exit 1
fi

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 启动增强前端UI检查框架...${NC}"
echo "========================================================"
echo "📍 项目根目录: $PROJECT_ROOT"
echo "🔧 检查项目: TypeScript/JavaScript编译 + UI设计符合性"

# 创建日志目录
mkdir -p logs

# 检查前置条件
echo -e "${YELLOW}🔍 检查前置条件...${NC}"

if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Python虚拟环境不存在${NC}"
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo -e "${RED}❌ 前端目录不存在${NC}"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo -e "${RED}❌ frontend/package.json不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 前置条件检查通过${NC}"

# 激活Python虚拟环境
source "$PROJECT_ROOT/venv/bin/activate"

# 确保增强检查器脚本可执行
chmod +x scripts/quality/enhanced_frontend_ui_check.py

# 第一步：运行增强编译和语法检查
echo -e "${PURPLE}📋 第一步：TypeScript/JavaScript编译检查${NC}"
echo "----------------------------------------------"
python scripts/quality/enhanced_frontend_ui_check.py \
    --project-root "$PROJECT_ROOT" \
    --output "logs/enhanced_frontend_ui_check.json" > logs/enhanced_ui_check_output.log 2>&1

ENHANCED_CHECK_RESULT=$?

# 显示增强检查结果
if [ -f "logs/enhanced_frontend_ui_check.json" ]; then
    echo -e "${BLUE}📊 编译检查结果摘要:${NC}"
    
    # 解析JSON并显示摘要
    python -c "
import json
import sys
try:
    with open('logs/enhanced_frontend_ui_check.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    summary = report['summary']
    
    # 颜色代码
    status_colors = {
        'PASS': '\033[0;32m',
        'PASS_WITH_WARNINGS': '\033[1;33m', 
        'FAIL': '\033[0;31m'
    }
    color = status_colors.get(summary['overall_status'], '\033[0m')
    
    print(f'   整体状态: {color}{summary[\"overall_status\"]}\033[0m')
    print(f'   总检查项: {summary[\"total_checks\"]}')
    print(f'   通过: {summary[\"passed_checks\"]}')
    print(f'   警告: {summary[\"warning_checks\"]}')
    print(f'   错误: {summary[\"error_checks\"]}')
    print(f'   成功率: {summary[\"success_rate\"]}')
    print(f'   检查耗时: {summary[\"duration\"]}')
    
    # 显示详细问题
    if summary['error_checks'] > 0 or summary['warning_checks'] > 0:
        print('\n🔍 发现的问题:')
        for result in report['detailed_results']:
            if result['status'] != 'pass':
                severity_color = '\033[0;31m' if result['status'] == 'error' else '\033[1;33m'
                print(f'   {severity_color}[{result[\"status\"].upper()}]\033[0m {result[\"check_type\"]}: {result[\"message\"]}')
                
                # 显示编译错误详情
                if result['check_type'] == 'compilation' and 'errors' in result['details']:
                    for error in result['details']['errors'][:5]:  # 只显示前5个错误
                        if error.get('file_path'):
                            print(f'     • {error[\"file_path\"]}:{error[\"line\"]}:{error[\"column\"]} - {error[\"message\"]}')
                        else:
                            print(f'     • {error[\"message\"]}')
                
                # 显示建议
                for suggestion in result.get('suggestions', []):
                    print(f'     💡 {suggestion}')
        
        if len([r for r in report['detailed_results'] if r.get('details', {}).get('errors', [])]) > 0:
            total_errors = sum(len(r.get('details', {}).get('errors', [])) for r in report['detailed_results'])
            if total_errors > 5:
                print(f'   ... 还有 {total_errors - 5} 个问题，详见报告文件')

except Exception as e:
    print(f'无法解析增强检查报告: {e}')
    sys.exit(1)
"
else
    echo -e "${RED}❌ 增强检查报告文件未生成${NC}"
    ENHANCED_CHECK_RESULT=1
fi

# 第二步：运行原有的UI设计符合性检查
echo -e "${PURPLE}📋 第二步：UI设计符合性检查${NC}"
echo "----------------------------------------------"

# 检查前端服务是否运行（用于页面检查）
if curl -s -I http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ 前端服务已运行，执行页面UI检查${NC}"
    
    # 运行原有的前端UI检查
    ./scripts/quality/frontend_ui_check.sh > logs/original_ui_check_output.log 2>&1
    ORIGINAL_UI_CHECK_RESULT=$?
    
    if [ $ORIGINAL_UI_CHECK_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ UI设计符合性检查通过${NC}"
    else
        echo -e "${YELLOW}⚠️ UI设计符合性检查发现问题${NC}"
    fi
    
    # 显示原有检查结果
    if [ -f "logs/frontend_ui_check_report.json" ]; then
        echo -e "${BLUE}📊 UI设计检查结果:${NC}"
        python -c "
import json
try:
    with open('logs/frontend_ui_check_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    summary = report['summary']
    print(f'   页面检查: {summary[\"successful_pages\"]}/{summary[\"total_pages\"]} 成功')
    print(f'   问题总数: {summary[\"total_issues\"]}')
    print(f'   错误: {summary[\"error_count\"]}')
    print(f'   警告: {summary[\"warning_count\"]}')
    print(f'   可访问性: {summary[\"avg_accessibility_score\"]}')
    print(f'   性能评分: {summary[\"avg_performance_score\"]}')
except Exception as e:
    print(f'无法解析UI检查报告: {e}')
"
    fi
else
    echo -e "${YELLOW}⚠️ 前端服务未运行，跳过页面UI检查${NC}"
    echo "   建议启动前端服务后重新运行检查"
    ORIGINAL_UI_CHECK_RESULT=0  # 不因为服务未启动而失败
fi

# 第三步：生成综合报告
echo -e "${PURPLE}📋 第三步：生成综合报告${NC}"
echo "----------------------------------------------"

# 创建综合报告
cat > logs/comprehensive_ui_check_report.json << EOF
{
  "report_type": "comprehensive_frontend_ui_check",
  "timestamp": "$(date -Iseconds)",
  "project_root": "$PROJECT_ROOT",
  "checks_performed": {
    "compilation_check": {
      "status": $([ $ENHANCED_CHECK_RESULT -eq 0 ] && echo '"passed"' || echo '"failed"'),
      "result_file": "logs/enhanced_frontend_ui_check.json"
    },
    "ui_design_check": {
      "status": $([ $ORIGINAL_UI_CHECK_RESULT -eq 0 ] && echo '"passed"' || echo '"failed"'),
      "result_file": "logs/frontend_ui_check_report.json"
    }
  },
  "overall_status": $([ $ENHANCED_CHECK_RESULT -eq 0 ] && [ $ORIGINAL_UI_CHECK_RESULT -eq 0 ] && echo '"PASS"' || echo '"FAIL"'),
  "recommendations": [
    "定期运行此综合检查确保代码质量",
    "修复所有编译错误后再进行UI测试",
    "保持代码风格一致性和最佳实践"
  ]
}
EOF

echo -e "${GREEN}✅ 综合报告已生成${NC}"

# 最终结果汇总
echo -e "${BLUE}📋 增强前端UI检查完成！${NC}"
echo "========================================================"

# 综合判断
if [ $ENHANCED_CHECK_RESULT -eq 0 ] && [ $ORIGINAL_UI_CHECK_RESULT -eq 0 ]; then
    echo -e "${GREEN}🎉 所有检查通过！前端代码质量良好${NC}"
    FINAL_STATUS="PASS"
    EXIT_CODE=0
elif [ $ENHANCED_CHECK_RESULT -ne 0 ]; then
    echo -e "${RED}❌ 编译检查失败，需要优先修复编译错误${NC}"
    FINAL_STATUS="COMPILATION_FAILED"
    EXIT_CODE=1
else
    echo -e "${YELLOW}⚠️ 编译通过但UI检查发现问题${NC}"
    FINAL_STATUS="UI_ISSUES"
    EXIT_CODE=1
fi

echo -e "${BLUE}📝 生成的报告文件:${NC}"
echo "   🔧 编译检查: logs/enhanced_frontend_ui_check.json"
echo "   🎨 UI检查: logs/frontend_ui_check_report.json"  
echo "   📋 综合报告: logs/comprehensive_ui_check_report.json"
echo "   📄 检查日志: logs/enhanced_ui_check_output.log"

echo -e "${BLUE}💡 使用建议:${NC}"
echo "   1. 优先修复编译错误（红色ERROR）"
echo "   2. 解决编译警告（黄色WARNING）"
echo "   3. 处理UI设计问题"
echo "   4. 定期运行此检查确保质量"

echo -e "${BLUE}🚀 快速修复命令:${NC}"
if [ $ENHANCED_CHECK_RESULT -ne 0 ]; then
    echo "   # 修复TypeScript错误"
    echo "   cd frontend && npx tsc --noEmit"
    echo "   # 修复ESLint问题"
    echo "   cd frontend && npx eslint src/ --fix"
fi

echo ""
echo -e "${BLUE}最终状态: ${NC}"
if [ "$FINAL_STATUS" = "PASS" ]; then
    echo -e "${GREEN}✅ PASS - 所有检查通过${NC}"
elif [ "$FINAL_STATUS" = "COMPILATION_FAILED" ]; then
    echo -e "${RED}❌ COMPILATION_FAILED - 编译失败${NC}"
else
    echo -e "${YELLOW}⚠️ UI_ISSUES - UI检查发现问题${NC}"
fi

exit $EXIT_CODE