#!/bin/bash

# A股智能交易决策平台 - 依赖安装脚本
# 确保所有依赖都正确安装

set -e  # 遇到错误立即退出

echo "🔧 安装A股智能交易决策平台依赖..."
echo "======================================================"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📍 工作目录: $SCRIPT_DIR"

# 1. 检查Python虚拟环境
if [ ! -d "venv" ]; then
    echo "🐍 创建Python虚拟环境..."
    python3 -m venv venv
    echo "✅ Python虚拟环境创建完成"
else
    echo "✅ Python虚拟环境已存在"
fi

# 2. 激活虚拟环境并安装Python依赖
echo "📦 安装Python依赖..."
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装核心依赖
echo "安装FastAPI核心依赖..."
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 websockets==15.0.1

echo "安装数据处理依赖..."
pip install pandas==2.1.4 numpy>=1.24.0

echo "安装其他必要依赖..."
pip install requests==2.31.0 python-dotenv==1.0.0

echo "✅ Python依赖安装完成"

# 3. 检查并安装前端依赖
if [ ! -d "frontend" ]; then
    echo "❌ 错误: frontend目录不存在"
    exit 1
fi

cd frontend

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 错误: Node.js未安装，请先安装Node.js 16+"
    echo "💡 访问 https://nodejs.org/ 下载安装"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ 错误: Node.js版本过低 (当前: $(node -v), 需要: 16+)"
    exit 1
fi

echo "✅ Node.js版本检查通过: $(node -v)"

# 安装前端依赖
if [ ! -d "node_modules" ] || [ ! -f "package-lock.json" ]; then
    echo "📦 安装前端依赖..."
    npm install
    echo "✅ 前端依赖安装完成"
else
    echo "✅ 前端依赖已存在"
fi

cd ..

# 4. 创建必要的目录
echo "📁 创建必要目录..."
mkdir -p logs
mkdir -p cache
mkdir -p reports
echo "✅ 目录创建完成"

# 5. 验证安装
echo ""
echo "🔍 验证安装..."
source venv/bin/activate

echo "检查Python模块..."
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')" || echo "❌ FastAPI导入失败"
python -c "import uvicorn; print('✅ Uvicorn')" || echo "❌ Uvicorn导入失败"
python -c "import websockets; print('✅ WebSockets')" || echo "❌ WebSockets导入失败"
python -c "import pandas; print(f'✅ Pandas {pandas.__version__}')" || echo "❌ Pandas导入失败"

echo ""
echo "🎉 依赖安装完成！"
echo "======================================================"
echo "接下来可以运行:"
echo "  ./start_services.sh  # 启动服务"
echo "  ./run_tests.sh       # 运行测试"
echo "======================================================"