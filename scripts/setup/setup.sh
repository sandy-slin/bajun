#!/bin/bash
# A股基本信息获取系统 - 环境初始化脚本

set -e  # 遇到错误时停止执行

echo "🚀 开始初始化A股基本信息获取系统..."

# 检查Python版本
echo "📋 检查Python版本..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3，请先安装Python 3.7+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ 找到Python版本: $PYTHON_VERSION"

# 创建虚拟环境
echo "📦 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source "$PROJECT_ROOT/venv/bin/activate"

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装项目依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p cache
mkdir -p logs

# 设置环境变量
echo "🔧 配置环境变量..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# DeepSeek API配置
DEEPSEEK_API_KEY=sk-f4affcb7b78243f5a138e7c9bdbbd6ee

# 缓存配置
CACHE_DIR=cache
CACHE_EXPIRY_DAYS=7

# 日志配置
LOG_LEVEL=INFO
EOF
    echo "✅ .env 文件已创建"
else
    echo "⚠️  .env 文件已存在，跳过创建"
fi

# 创建启动脚本
echo "📝 创建启动脚本..."
cat > run.sh << EOF
#!/bin/bash
source "$PROJECT_ROOT/venv/bin/activate"
python src/main.py \$@
EOF

chmod +x run.sh
echo "✅ 启动脚本 run.sh 已创建"

# 运行测试验证
echo "🧪 运行基础测试..."
if python -m pytest tests/ --tb=short -q; then
    echo "✅ 测试通过"
else
    echo "⚠️  测试失败，请检查代码"
fi

echo "🎉 初始化完成！"
echo ""
echo "使用方法："
echo "  ./run.sh --help          # 查看帮助"
echo "  ./run.sh -s 000001       # 分析特定股票"
echo "  ./run.sh -v              # 详细输出模式"
echo ""
echo "测试命令："
echo "  source "$PROJECT_ROOT/venv/bin/activate""
echo "  python -m pytest tests/ -v --cov=src"