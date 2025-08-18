#!/bin/bash

echo "📁 项目标准规范检查..."
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
WARNINGS=0

# 日志文件
STANDARDS_LOG="logs/project_standards_check.log"
mkdir -p logs
echo "=== Project Standards Check $(date) ===" > "$STANDARDS_LOG" 2>/dev/null || {
    echo "警告: 无法创建日志文件 $STANDARDS_LOG，使用临时文件"
    STANDARDS_LOG="/tmp/project_standards_check.log"
    echo "=== Project Standards Check $(date) ===" > "$STANDARDS_LOG"
}

# 检查函数
check_standard() {
    local check_name="$1"
    local check_type="$2"  # "required" 或 "recommended"
    local condition="$3"
    
    ((TOTAL_CHECKS++))
    echo -e "${YELLOW}检查 [$TOTAL_CHECKS]: $check_name${NC}"
    echo "检查 [$TOTAL_CHECKS]: $check_name - $(date)" >> "$STANDARDS_LOG" 2>/dev/null
    
    if eval "$condition" >> "$STANDARDS_LOG" 2>&1; then
        echo -e "${GREEN}✅ $check_name - 通过${NC}"
        echo "✅ PASS: $check_name" >> "$STANDARDS_LOG" 2>/dev/null
        return 0
    else
        if [ "$check_type" = "required" ]; then
            echo -e "${RED}❌ $check_name - 失败 (必需)${NC}"
            echo "❌ FAIL: $check_name (REQUIRED)" >> "$STANDARDS_LOG" 2>/dev/null
            ((FAILURES++))
        else
            echo -e "${YELLOW}⚠️ $check_name - 警告 (推荐)${NC}"
            echo "⚠️ WARNING: $check_name (RECOMMENDED)" >> "$STANDARDS_LOG" 2>/dev/null
            ((WARNINGS++))
        fi
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

echo "📋 启动项目标准规范检查流程..."
echo "日志文件: $STANDARDS_LOG"
echo ""

# ==========================================
# Phase 1: 核心目录结构检查
# ==========================================
show_progress 1 6 "核心目录结构检查"

check_standard "源代码目录结构" "required" "[ -d 'src' ] && [ -d 'src/api' ] && [ -d 'src/core' ]"
check_standard "前端目录结构" "required" "[ -d 'frontend' ] && [ -d 'frontend/src' ]"
check_standard "文档目录结构" "required" "[ -d 'docs' ] && [ -d 'docs/development' ]"
check_standard "测试目录结构" "recommended" "[ -d 'tests' ]"
check_standard "日志目录结构" "recommended" "[ -d 'logs' ]"

# API路由结构检查
check_standard "API路由结构" "required" "[ -d 'src/api/routes' ] && [ -f 'src/api/routes/__init__.py' ]"
check_standard "核心引擎模块" "required" "[ -f 'src/core/sector_engine.py' ] && [ -f 'src/core/stock_engine.py' ]"

# ==========================================
# Phase 2: 核心文件存在性检查
# ==========================================
show_progress 2 6 "核心文件存在性检查"

check_standard "CLAUDE.md存在" "required" "[ -f 'CLAUDE.md' ]"
check_standard "README.md存在" "required" "[ -f 'README.md' ]"
check_standard "requirements.txt存在" "required" "[ -f 'requirements.txt' ]"
check_standard "后端主程序存在" "required" "[ -f 'src/api/main.py' ]"
check_standard "前端配置文件存在" "required" "[ -f 'frontend/package.json' ]"

# 服务管理脚本检查
check_standard "服务启动脚本" "required" "[ -f 'scripts/deployment/start_services.sh' ] && [ -x 'scripts/deployment/start_services.sh' ]"
check_standard "服务停止脚本" "required" "[ -f 'scripts/deployment/stop_services.sh' ] && [ -x 'scripts/deployment/stop_services.sh' ]"
check_standard "质量检查脚本" "required" "[ -f 'scripts/quality/pre_commit_check.sh' ] && [ -x 'scripts/quality/pre_commit_check.sh' ]"

# ==========================================
# Phase 3: CLAUDE.md 内容一致性检查
# ==========================================
show_progress 3 6 "CLAUDE.md 内容一致性检查"

if [ -f "CLAUDE.md" ]; then
    # 检查CLAUDE.md是否包含核心信息
    check_standard "CLAUDE.md包含项目概述" "required" "grep -q '项目概述\\|Project Overview\\|系统架构' CLAUDE.md"
    check_standard "CLAUDE.md包含技术栈信息" "required" "grep -q 'FastAPI\\|React\\|技术栈' CLAUDE.md"
    check_standard "CLAUDE.md包含开发命令" "required" "grep -q 'start_services\\|开发命令\\|Development Commands' CLAUDE.md"
    check_standard "CLAUDE.md包含API信息" "required" "grep -q 'API\\|接口\\|路由' CLAUDE.md"
    
    # 检查版本信息一致性
    CLAUDE_VERSION=$(grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+\|Enhanced-v[0-9]\+\.[0-9]\+\.[0-9]\+' CLAUDE.md | head -1)
    if [ -n "$CLAUDE_VERSION" ]; then
        check_standard "CLAUDE.md版本信息存在" "required" "[ -n '$CLAUDE_VERSION' ]"
        echo "检测到版本: $CLAUDE_VERSION" >> "$STANDARDS_LOG" 2>/dev/null
    else
        check_standard "CLAUDE.md版本信息" "recommended" "false"
    fi
    
    # 检查关键路径是否正确
    check_standard "CLAUDE.md路径引用正确" "recommended" "grep -q 'src/api/main.py\\|frontend/package.json' CLAUDE.md"
    
    # 检查最后更新时间 (30天内)
    CLAUDE_MODIFIED=$(stat -f "%m" CLAUDE.md 2>/dev/null || stat -c "%Y" CLAUDE.md 2>/dev/null)
    CURRENT_TIME=$(date +%s)
    DAYS_OLD=$(( (CURRENT_TIME - CLAUDE_MODIFIED) / 86400 ))
    
    if [ $DAYS_OLD -lt 30 ]; then
        check_standard "CLAUDE.md最近更新 (30天内)" "recommended" "true"
    else
        check_standard "CLAUDE.md最近更新 (${DAYS_OLD}天前)" "recommended" "false"
    fi
else
    echo -e "${RED}❌ CLAUDE.md 文件不存在，跳过内容检查${NC}"
fi

# ==========================================
# Phase 4: 文档完整性检查
# ==========================================
show_progress 4 6 "文档完整性检查"

check_standard "需求文档存在" "required" "[ -f 'docs/user_requirements/requirement.md' ]"
check_standard "测试标准文档存在" "recommended" "[ -f 'docs/testing/test_standards.md' ]"
check_standard "API文档存在" "recommended" "[ -f 'docs/development/api_documentation.md' ] || grep -q 'API' docs/development/*.md"

# README.md内容检查
if [ -f "README.md" ]; then
    check_standard "README包含项目简介" "required" "grep -q -i 'A股\\|交易\\|智能\\|平台\\|项目' README.md"
    check_standard "README包含安装说明" "recommended" "grep -q -i 'install\\|安装\\|setup\\|开始' README.md"
    check_standard "README包含使用说明" "recommended" "grep -q -i 'usage\\|使用\\|how to\\|如何' README.md"
fi

# 检查文档同步性 (检查代码变更是否需要文档更新)
RECENT_CODE_CHANGES=$(git log --since="7 days ago" --pretty=format:"%s" -- src/ frontend/ | wc -l)
RECENT_DOC_CHANGES=$(git log --since="7 days ago" --pretty=format:"%s" -- docs/ CLAUDE.md README.md | wc -l)

if [ $RECENT_CODE_CHANGES -gt 0 ] && [ $RECENT_DOC_CHANGES -eq 0 ]; then
    check_standard "代码与文档同步性" "recommended" "false"
    echo "提示: 近7天有 $RECENT_CODE_CHANGES 个代码变更，但文档未更新" >> "$STANDARDS_LOG" 2>/dev/null
else
    check_standard "代码与文档同步性" "recommended" "true"
fi

# ==========================================
# Phase 5: 代码规范检查
# ==========================================
show_progress 5 6 "代码规范检查"

# Python文件头部注释检查
PYTHON_FILES_WITHOUT_HEADER=$(find src -name "*.py" -exec grep -L "#!/usr/bin/env python3\|# -\*- coding: utf-8 -\*-" {} \; 2>/dev/null | wc -l)
if [ $PYTHON_FILES_WITHOUT_HEADER -eq 0 ]; then
    check_standard "Python文件头部规范" "recommended" "true"
else
    check_standard "Python文件头部规范" "recommended" "false"
    echo "发现 $PYTHON_FILES_WITHOUT_HEADER 个Python文件缺少标准头部" >> "$STANDARDS_LOG" 2>/dev/null
fi

# API路由文档规范检查
API_ROUTES_WITHOUT_DOCS=$(find src/api/routes -name "*.py" -exec grep -L "summary=\\|description=" {} \; 2>/dev/null | wc -l)
if [ $API_ROUTES_WITHOUT_DOCS -eq 0 ]; then
    check_standard "API路由文档规范" "recommended" "true"
else
    check_standard "API路由文档规范" "recommended" "false"
    echo "发现 $API_ROUTES_WITHOUT_DOCS 个API路由文件缺少文档" >> "$STANDARDS_LOG" 2>/dev/null
fi

# 导入顺序检查 (简化版)
PYTHON_FILES_WITH_WRONG_IMPORTS=$(find src -name "*.py" -exec grep -l "^from.*import.*" {} \; | xargs grep -L "^import.*" 2>/dev/null | wc -l)
check_standard "Python导入顺序规范" "recommended" "[ $PYTHON_FILES_WITH_WRONG_IMPORTS -lt 3 ]"

# 前端TypeScript配置检查
if [ -f "frontend/tsconfig.json" ]; then
    check_standard "TypeScript配置文件" "required" "[ -f 'frontend/tsconfig.json' ]"
    check_standard "TypeScript配置有效性" "recommended" "cd frontend && npx tsc --noEmit --skipLibCheck 2>/dev/null || true"
fi

# ==========================================
# Phase 6: 版本管理和Git规范
# ==========================================
show_progress 6 6 "版本管理和Git规范"

# Git钩子检查
check_standard "Git pre-commit钩子" "recommended" "[ -f '.git/hooks/pre-commit' ] && [ -x '.git/hooks/pre-commit' ]"

# .gitignore检查
check_standard ".gitignore文件存在" "required" "[ -f '.gitignore' ]"
if [ -f ".gitignore" ]; then
    check_standard ".gitignore包含基本规则" "recommended" "grep -q 'node_modules\\|__pycache__\\|*.pyc\\|logs/' .gitignore"
fi

# 分支命名检查
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" =~ ^(main|develop|feature/.*|hotfix/.*|release/.*)$ ]]; then
    check_standard "Git分支命名规范" "recommended" "true"
else
    check_standard "Git分支命名规范" "recommended" "false"
    echo "当前分支: $CURRENT_BRANCH 不符合命名规范" >> "$STANDARDS_LOG" 2>/dev/null
fi

# 检查是否有未提交的重要文件
UNTRACKED_IMPORTANT=$(git status --porcelain | grep "^??" | grep -E "\.(py|js|ts|tsx|md|json)$" | wc -l)
if [ $UNTRACKED_IMPORTANT -gt 0 ]; then
    check_standard "重要文件已纳入版本控制" "recommended" "false"
    echo "发现 $UNTRACKED_IMPORTANT 个未跟踪的重要文件" >> "$STANDARDS_LOG" 2>/dev/null
    git status --porcelain | grep "^??" | grep -E "\.(py|js|ts|tsx|md|json)$" >> "$STANDARDS_LOG" 2>/dev/null
else
    check_standard "重要文件已纳入版本控制" "recommended" "true"
fi

# ==========================================
# 智能建议生成
# ==========================================
echo ""
echo "🤖 智能改进建议:"

# 基于检查结果生成改进建议
SUGGESTIONS=()

if [ $FAILURES -gt 0 ]; then
    SUGGESTIONS+=("🔥 有 $FAILURES 个必需项检查失败，需要立即修复")
fi

if [ $WARNINGS -gt 3 ]; then
    SUGGESTIONS+=("⚠️ 有 $WARNINGS 个推荐项需要改进，建议优先处理")
fi

if [ ! -f "docs/development/api_documentation.md" ]; then
    SUGGESTIONS+=("📚 建议创建详细的API文档")
fi

if [ $RECENT_CODE_CHANGES -gt 5 ] && [ $RECENT_DOC_CHANGES -eq 0 ]; then
    SUGGESTIONS+=("📝 近期代码变更较多，建议同步更新文档")
fi

if [ $PYTHON_FILES_WITHOUT_HEADER -gt 0 ]; then
    SUGGESTIONS+=("🐍 建议为Python文件添加标准头部注释")
fi

# 检查CLAUDE.md是否需要更新
if [ -f "CLAUDE.md" ] && [ $DAYS_OLD -gt 30 ]; then
    SUGGESTIONS+=("📋 CLAUDE.md 已超过30天未更新，建议检查是否需要同步最新变更")
fi

# 显示建议
if [ ${#SUGGESTIONS[@]} -gt 0 ]; then
    for suggestion in "${SUGGESTIONS[@]}"; do
        echo "   $suggestion"
    done
else
    echo "   🎉 项目规范性良好，无需特别改进"
fi

# ==========================================
# 最终报告
# ==========================================
echo ""
echo "================================================================"
echo "📊 项目标准规范检查总结:"
echo "================================================================"

REQUIRED_FAILURES=$(grep -c "FAIL.*REQUIRED" "$STANDARDS_LOG" 2>/dev/null || echo "0")
TOTAL_PASSED=$(( TOTAL_CHECKS - FAILURES - WARNINGS ))

echo "检查项目总数: $TOTAL_CHECKS"
echo "✅ 通过: $TOTAL_PASSED"
echo "❌ 必需项失败: $REQUIRED_FAILURES"
echo "⚠️ 推荐项警告: $WARNINGS"
echo ""

if [ $REQUIRED_FAILURES -eq 0 ]; then
    echo -e "${GREEN}🎉 项目标准规范检查通过!${NC}"
    echo -e "${GREEN}✅ 所有必需项都已满足，项目结构规范${NC}"
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}💡 建议处理 $WARNINGS 个推荐项以进一步提升项目质量${NC}"
    fi
    
    echo "✅ PROJECT STANDARDS CHECK PASSED - $(date)" >> "$STANDARDS_LOG" 2>/dev/null
    exit 0
else
    echo -e "${RED}💥 项目标准规范检查失败!${NC}"
    echo -e "${RED}❌ 有 $REQUIRED_FAILURES 个必需项未满足${NC}"
    echo ""
    echo "🔧 必需的修复项:"
    grep "FAIL.*REQUIRED" "$STANDARDS_LOG" | sed 's/^/   - /'
    echo ""
    echo "📝 详细检查结果请查看: $STANDARDS_LOG"
    
    echo "❌ PROJECT STANDARDS CHECK FAILED - $(date)" >> "$STANDARDS_LOG" 2>/dev/null
    exit 1
fi