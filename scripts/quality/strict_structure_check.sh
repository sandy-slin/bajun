#!/bin/bash

echo "🏗️ 严格目录结构检查..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查失败计数
CRITICAL_FAILURES=0
WARNING_COUNT=0
TOTAL_CHECKS=0

# 日志文件
LOG_FILE="logs/structure_check.log"
mkdir -p logs
echo "=== Strict Structure Check $(date) ===" > "$LOG_FILE"

# 检查函数
check_structure() {
    local check_name="$1"
    local check_type="$2"  # "critical" 或 "warning"
    local condition="$3"
    
    ((TOTAL_CHECKS++))
    echo -e "${YELLOW}检查 [$TOTAL_CHECKS]: $check_name${NC}"
    echo "检查 [$TOTAL_CHECKS]: $check_name - $(date)" >> "$LOG_FILE"
    
    if eval "$condition" >> "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}✅ $check_name - 通过${NC}"
        echo "✅ PASS: $check_name" >> "$LOG_FILE"
        return 0
    else
        if [ "$check_type" = "critical" ]; then
            echo -e "${RED}💥 $check_name - 失败 (CRITICAL)${NC}"
            echo "💥 CRITICAL FAIL: $check_name" >> "$LOG_FILE"
            ((CRITICAL_FAILURES++))
        else
            echo -e "${YELLOW}⚠️ $check_name - 警告 (WARNING)${NC}"
            echo "⚠️ WARNING: $check_name" >> "$LOG_FILE"
            ((WARNING_COUNT++))
        fi
        return 1
    fi
}

echo "📋 启动严格目录结构检查流程..."
echo "日志文件: $LOG_FILE"
echo ""

# ==========================================
# Phase 1: 核心目录结构检查 (CRITICAL)
# ==========================================
echo -e "${BLUE}[1/6] 核心目录结构检查 (CRITICAL)${NC}"

check_structure "源代码目录 src/" "critical" "[ -d 'src' ]"
check_structure "前端目录 frontend/" "critical" "[ -d 'frontend' ]"
check_structure "测试目录 tests/" "critical" "[ -d 'tests' ]"
check_structure "文档目录 docs/" "critical" "[ -d 'docs' ]"
check_structure "脚本目录 scripts/" "critical" "[ -d 'scripts' ]"
check_structure "日志目录 logs/" "critical" "[ -d 'logs' ]"

# ==========================================
# Phase 2: 标准子目录检查 (CRITICAL)
# ==========================================
echo -e "${BLUE}[2/6] 标准子目录检查 (CRITICAL)${NC}"

check_structure "API路由目录 src/api/routes/" "critical" "[ -d 'src/api/routes' ]"
check_structure "核心引擎目录 src/core/" "critical" "[ -d 'src/core' ]"
check_structure "数据处理目录 src/data/" "critical" "[ -d 'src/data' ]"
check_structure "分析模块目录 src/analysis/" "critical" "[ -d 'src/analysis' ]"
check_structure "脚本质量目录 scripts/quality/" "critical" "[ -d 'scripts/quality' ]"
check_structure "脚本部署目录 scripts/deployment/" "critical" "[ -d 'scripts/deployment' ]"

# ==========================================
# Phase 3: 必需文件存在检查 (CRITICAL)
# ==========================================
echo -e "${BLUE}[3/6] 必需文件存在检查 (CRITICAL)${NC}"

check_structure "项目说明 README.md" "critical" "[ -f 'README.md' ]"
check_structure "开发指南 CLAUDE.md" "critical" "[ -f 'CLAUDE.md' ]"
check_structure "Git忽略 .gitignore" "critical" "[ -f '.gitignore' ]"
check_structure "Python依赖 requirements.txt" "critical" "[ -f 'requirements.txt' ]"
check_structure "后端入口 src/api/main.py" "critical" "[ -f 'src/api/main.py' ]"
check_structure "前端配置 frontend/package.json" "critical" "[ -f 'frontend/package.json' ]"

# ==========================================
# Phase 4: 根目录文件数量检查 (CRITICAL)
# ==========================================
echo -e "${BLUE}[4/6] 根目录文件数量检查 (CRITICAL)${NC}"

ROOT_FILE_COUNT=$(find . -maxdepth 1 -type f | wc -l)
check_structure "根目录文件数量 ≤ 10个 (当前: $ROOT_FILE_COUNT)" "critical" "[ $ROOT_FILE_COUNT -le 10 ]"

# 检查根目录是否有不应该存在的文件类型
INVALID_PY_FILES=$(find . -maxdepth 1 -name "*.py" | wc -l)
check_structure "根目录无Python文件 (当前: $INVALID_PY_FILES)" "critical" "[ $INVALID_PY_FILES -eq 0 ]"

INVALID_SH_FILES=$(find . -maxdepth 1 -name "*.sh" | wc -l)
check_structure "根目录无脚本文件 (当前: $INVALID_SH_FILES)" "critical" "[ $INVALID_SH_FILES -eq 0 ]"

INVALID_TEST_FILES=$(find . -maxdepth 1 -name "*test*.py" -o -name "*Test*.py" | wc -l)
check_structure "根目录无测试文件 (当前: $INVALID_TEST_FILES)" "critical" "[ $INVALID_TEST_FILES -eq 0 ]"

# ==========================================
# Phase 5: 文件位置合规性检查 (CRITICAL)
# ==========================================
echo -e "${BLUE}[5/6] 文件位置合规性检查 (CRITICAL)${NC}"

# 检查脚本文件是否在正确位置
SCRIPTS_IN_ROOT=$(find . -maxdepth 1 -name "*.sh" | wc -l)
check_structure "所有脚本在 scripts/ 目录" "critical" "[ $SCRIPTS_IN_ROOT -eq 0 ]"

# 检查测试文件是否在正确位置
TESTS_OUTSIDE=$(find src -name "*test*.py" -o -name "*Test*.py" 2>/dev/null | wc -l)
check_structure "测试文件在 tests/ 目录" "critical" "[ $TESTS_OUTSIDE -eq 0 ]"

# 检查配置文件位置
CONFIGS_IN_SRC=$(find src -name "*.yaml" -o -name "*.yml" -o -name "*.json" 2>/dev/null | grep -v __pycache__ | wc -l)
check_structure "配置文件在 configs/ 目录" "warning" "[ $CONFIGS_IN_SRC -eq 0 ]"

# ==========================================
# Phase 6: 命名规范检查 (WARNING)
# ==========================================
echo -e "${BLUE}[6/6] 命名规范检查 (WARNING)${NC}"

# 检查目录命名规范 (小写+下划线)
INVALID_DIR_NAMES=$(find . -type d -name "*[A-Z]*" | grep -v node_modules | grep -v __pycache__ | grep -v .git | wc -l)
check_structure "目录命名规范 (小写+下划线)" "warning" "[ $INVALID_DIR_NAMES -eq 0 ]"

# 检查Python文件命名规范
INVALID_PY_NAMES=$(find src -name "*.py" | grep -E "[A-Z]" | wc -l)
check_structure "Python文件命名规范 (小写+下划线)" "warning" "[ $INVALID_PY_NAMES -eq 0 ]"

# 检查是否有重复的README文件
MULTIPLE_READMES=$(find . -name "README*" | wc -l)
check_structure "避免重复README文件 (当前: $MULTIPLE_READMES)" "warning" "[ $MULTIPLE_READMES -le 2 ]"

# ==========================================
# 智能修复建议
# ==========================================
echo ""
echo "🤖 智能修复建议:"

SUGGESTIONS=()

if [ $CRITICAL_FAILURES -gt 0 ]; then
    SUGGESTIONS+=("🔥 有 $CRITICAL_FAILURES 个关键结构问题，必须立即修复")
fi

if [ $ROOT_FILE_COUNT -gt 10 ]; then
    SUGGESTIONS+=("📁 根目录文件过多 ($ROOT_FILE_COUNT 个)，建议移动到子目录")
fi

if [ $INVALID_SH_FILES -gt 0 ]; then
    SUGGESTIONS+=("📜 发现 $INVALID_SH_FILES 个脚本文件在根目录，请移动到 scripts/ 目录")
fi

if [ $INVALID_PY_FILES -gt 0 ]; then
    SUGGESTIONS+=("🐍 发现 $INVALID_PY_FILES 个Python文件在根目录，请移动到 src/ 或 tests/ 目录")
fi

if [ $WARNING_COUNT -gt 3 ]; then
    SUGGESTIONS+=("⚠️ 有 $WARNING_COUNT 个命名或组织建议，建议逐步改进")
fi

# 显示建议
if [ ${#SUGGESTIONS[@]} -gt 0 ]; then
    for suggestion in "${SUGGESTIONS[@]}"; do
        echo "   $suggestion"
    done
else
    echo "   🎉 目录结构完全符合标准，无需改进"
fi

# ==========================================
# 最终报告
# ==========================================
echo ""
echo "================================================================"
echo "📊 严格目录结构检查总结:"
echo "================================================================"

TOTAL_PASSED=$(( TOTAL_CHECKS - CRITICAL_FAILURES - WARNING_COUNT ))

echo "检查项目总数: $TOTAL_CHECKS"
echo "✅ 通过: $TOTAL_PASSED"
echo "💥 关键失败: $CRITICAL_FAILURES"
echo "⚠️ 警告: $WARNING_COUNT"
echo ""

if [ $CRITICAL_FAILURES -eq 0 ]; then
    echo -e "${GREEN}🎉 严格目录结构检查通过!${NC}"
    echo -e "${GREEN}✅ 所有关键结构要求都已满足${NC}"
    
    if [ $WARNING_COUNT -gt 0 ]; then
        echo -e "${YELLOW}💡 建议处理 $WARNING_COUNT 个命名和组织优化项${NC}"
    fi
    
    echo "✅ STRICT STRUCTURE CHECK PASSED - $(date)" >> "$LOG_FILE"
    exit 0
else
    echo -e "${RED}💥 严格目录结构检查失败!${NC}"
    echo -e "${RED}❌ 有 $CRITICAL_FAILURES 个关键结构问题未解决${NC}"
    echo ""
    echo "🔧 必须修复的问题:"
    grep "CRITICAL FAIL" "$LOG_FILE" | sed 's/^/   - /'
    echo ""
    echo "📝 详细检查结果请查看: $LOG_FILE"
    
    echo "❌ STRICT STRUCTURE CHECK FAILED - $(date)" >> "$LOG_FILE"
    exit 1
fi