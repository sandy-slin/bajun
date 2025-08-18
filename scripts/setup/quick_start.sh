#!/bin/bash

echo "🚀 A股智能交易决策平台 - 快速启动"
echo "================================================================"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}选择操作:${NC}"
echo "1. 📦 安装依赖"
echo "2. 🚀 启动服务"
echo "3. 🛑 停止服务"
echo "4. 🧪 运行测试"
echo "5. 🔍 质量检查"
echo "6. 📊 查看状态"
echo "7. 📖 查看文档"

read -p "请输入选项 (1-7): " choice

case $choice in
    1)
        echo -e "${YELLOW}正在安装依赖...${NC}"
        ./scripts/setup/install_dependencies.sh
        ;;
    2)
        echo -e "${YELLOW}正在启动服务...${NC}"
        ./scripts/deployment/start_services.sh
        ;;
    3)
        echo -e "${YELLOW}正在停止服务...${NC}"
        ./scripts/deployment/stop_services.sh
        ;;
    4)
        echo -e "${YELLOW}正在运行测试...${NC}"
        ./scripts/testing/run_tests.sh
        ;;
    5)
        echo -e "${YELLOW}正在执行质量检查...${NC}"
        ./scripts/quality/pre_commit_check.sh
        ;;
    6)
        echo -e "${YELLOW}正在查看系统状态...${NC}"
        echo ""
        echo "🌐 服务状态:"
        curl -s http://localhost:8000/health 2>/dev/null && echo "✅ 后端服务运行正常" || echo "❌ 后端服务未运行"
        curl -s -I http://localhost:3000 2>/dev/null >/dev/null && echo "✅ 前端服务运行正常" || echo "❌ 前端服务未运行"
        echo ""
        echo "📂 访问地址:"
        echo "   前端应用: http://localhost:3000"
        echo "   API文档: http://localhost:8000/docs"
        echo "   系统状态: http://localhost:8000/health"
        ;;
    7)
        echo -e "${YELLOW}查看项目文档...${NC}"
        echo ""
        echo "📖 主要文档:"
        echo "   项目说明: README.md"
        echo "   开发指南: CLAUDE.md"
        echo "   快速启动: docs/development/QUICK_START.md"
        echo "   目录结构: docs/development/DIRECTORY_STRUCTURE_STANDARD.md"
        echo "   测试标准: docs/testing/test_standards.md"
        ;;
    *)
        echo "❌ 无效选项，请重新运行脚本"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}操作完成！${NC}"