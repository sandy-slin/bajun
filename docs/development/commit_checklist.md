# 提交前检查清单 (Pre-Commit Checklist)

## 📋 概述

为确保每次提交的代码质量和系统稳定性，所有代码提交前必须通过以下检查清单。**只有在所有检查项通过后才能执行git commit**。

## ✅ 必要检查项 (Mandatory Checks)

### 🔧 1. 基础功能验证

#### 1.1 服务启动检查
```bash
# 检查项目根目录
pwd  # 应该在 /path/to/bajun

# 停止现有服务
./stop_services.sh

# 启动服务
./start_services.sh

# 验证启动成功 (必须等待30秒以上)
sleep 30
```

#### 1.2 后端服务健康检查
```bash
# 健康检查API (必须返回200状态)
curl -f http://localhost:8000/health || exit 1

# 检查关键字段
curl -s http://localhost:8000/health | grep -q '"status":"healthy"' || exit 1
curl -s http://localhost:8000/health | grep -q '"services"' || exit 1
```

#### 1.3 前端服务检查
```bash
# 前端服务状态检查
curl -f -I http://localhost:3000 || exit 1

# 检查是否返回HTML页面
curl -s http://localhost:3000 | grep -q '<title>' || exit 1
```

### 📊 2. 核心API功能验证

#### 2.1 板块分析API
```bash
# 板块分析接口测试
RESPONSE=$(curl -s http://localhost:8000/api/v1/sectors/)

# 检查响应结构
echo "$RESPONSE" | grep -q '"success":true' || exit 1
echo "$RESPONSE" | grep -q '"top_sectors"' || exit 1
echo "$RESPONSE" | grep -q '"all_sectors"' || exit 1
echo "$RESPONSE" | grep -q '"market_overview"' || exit 1

# 检查数据完整性
echo "$RESPONSE" | grep -q '"total_sectors_analyzed"' || exit 1
echo "$RESPONSE" | grep -q '"composite_score"' || exit 1
```

#### 2.2 股票推荐API (如果有变更)
```bash
# 测试股票推荐接口 (POST请求)
curl -s -X POST http://localhost:8000/api/v1/stocks/recommend-from-top-sectors \
  -H "Content-Type: application/json" \
  -d '{"top_sectors":[{"sector_name":"银行","composite_score":85}],"stocks_per_sector":3}' \
  | grep -q '"success":true' || exit 1
```

#### 2.3 其他核心API
```bash
# 股票列表API
curl -s http://localhost:8000/api/v1/stocks/ | grep -q '"success":true' || exit 1

# 投资组合API
curl -s http://localhost:8000/api/v1/portfolio/analysis \
  -H "Content-Type: application/json" \
  -d '{"holdings":[]}' \
  | grep -q '"success":true' || exit 1
```

### 🎯 3. 性能基准验证

#### 3.1 算法性能测试
```bash
# 运行标准性能测试 (必须与v1.3.0基准对比)
python src/main.py --optimized-analysis --opt-analysis-months 2 --opt-prediction-days 5

# 检查Enhanced Momentum准确率不能低于66.5%
# 这个需要在测试脚本中实现具体的数值检查
```

#### 3.2 API响应时间检查
```bash
# 板块分析响应时间 (应该<2秒)
time curl -s http://localhost:8000/api/v1/sectors/ > /dev/null

# 健康检查响应时间 (应该<200ms)
time curl -s http://localhost:8000/health > /dev/null
```

### 🔍 4. 代码质量检查

#### 4.1 Python代码检查 (如果修改了Python文件)
```bash
# 语法检查
python -m py_compile src/api/main.py
python -m py_compile src/api/routes/*.py

# 导入检查
python -c "from src.api.main import app; print('Import OK')"
```

#### 4.2 前端代码检查 (如果修改了前端文件)
```bash
cd frontend

# TypeScript编译检查
npm run build

# 回到项目根目录
cd ..
```

## 🚀 自动化检查脚本

### 创建自动化检查脚本

```bash
# 创建自动化检查脚本
cat > pre_commit_check.sh << 'EOF'
#!/bin/bash

echo "🔍 开始提交前检查..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查失败计数
FAILURES=0

# 检查函数
check_step() {
    local step_name="$1"
    local command="$2"
    
    echo -e "${YELLOW}检查: $step_name${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ $step_name - 通过${NC}"
        return 0
    else
        echo -e "${RED}❌ $step_name - 失败${NC}"
        ((FAILURES++))
        return 1
    fi
}

# 1. 基础环境检查
check_step "项目目录检查" "[ -f 'src/api/main.py' ] && [ -f 'frontend/package.json' ]"

# 2. 停止现有服务
echo "🛑 停止现有服务..."
./stop_services.sh 2>/dev/null || true
sleep 5

# 3. 启动服务
echo "🚀 启动服务..."
if ! ./start_services.sh; then
    echo -e "${RED}❌ 服务启动失败${NC}"
    exit 1
fi

# 等待服务完全启动
echo "⏳ 等待服务启动..."
sleep 30

# 4. 核心功能检查
check_step "后端健康检查" "curl -f -s http://localhost:8000/health > /dev/null"
check_step "前端服务检查" "curl -f -s -I http://localhost:3000 > /dev/null"
check_step "板块分析API" "curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"success\":true'"
check_step "API响应格式" "curl -s http://localhost:8000/api/v1/sectors/ | grep -q '\"top_sectors\"'"

# 5. 代码语法检查
if [ -n "$(find src -name '*.py' -newer .git/COMMIT_EDITMSG 2>/dev/null)" ]; then
    check_step "Python语法检查" "python -c 'from src.api.main import app'"
fi

# 6. 总结
echo ""
echo "📊 检查总结:"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}🎉 所有检查通过! 可以安全提交代码。${NC}"
    exit 0
else
    echo -e "${RED}💥 检查失败: $FAILURES 个项目${NC}"
    echo -e "${RED}请修复所有问题后再提交代码!${NC}"
    exit 1
fi
EOF

chmod +x pre_commit_check.sh
```

## 📈 性能回退防护

### 性能基准管理

```bash
# 创建性能基准脚本
cat > check_performance.sh << 'EOF'
#!/bin/bash

echo "📈 性能基准检查..."

# 运行标准性能测试
RESULT=$(python src/main.py --optimized-analysis --opt-analysis-months 2 --opt-prediction-days 5 2>/dev/null | tail -10)

# 提取Enhanced Momentum准确率 (需要根据实际输出格式调整)
ACCURACY=$(echo "$RESULT" | grep -o 'Enhanced Momentum.*[0-9.]*%' | grep -o '[0-9.]*' | head -1)

if [ -z "$ACCURACY" ]; then
    echo "⚠️ 无法获取性能数据"
    exit 1
fi

# 基准值检查 (66.5%)
BASELINE=66.5
if (( $(echo "$ACCURACY >= $BASELINE" | bc -l) )); then
    echo "✅ 性能检查通过: Enhanced Momentum = $ACCURACY% (>= $BASELINE%)"
    exit 0
else
    echo "❌ 性能回退: Enhanced Momentum = $ACCURACY% (< $BASELINE%)"
    exit 1
fi
EOF

chmod +x check_performance.sh
```

## 🔄 Git钩子集成

### 创建Git pre-commit钩子

```bash
# 创建Git pre-commit钩子
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

echo "🔍 Git pre-commit 钩子启动..."

# 运行检查脚本
if ./pre_commit_check.sh; then
    echo "✅ 提交前检查通过"
    exit 0
else
    echo "❌ 提交前检查失败，阻止提交"
    echo "请运行 ./pre_commit_check.sh 查看详细错误信息"
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

## 📝 使用方式

### 手动检查
```bash
# 开发完成后，提交前运行
./pre_commit_check.sh

# 如果所有检查通过，再执行提交
git add .
git commit -m "your commit message"
```

### 自动检查 (推荐)
安装了Git钩子后，每次`git commit`都会自动运行检查：

```bash
git add .
git commit -m "your commit message"
# Git钩子会自动运行检查，只有通过才允许提交
```

## 🎯 检查项目标

### 功能完整性
- ✅ 所有核心API正常响应
- ✅ 前后端服务正常启动
- ✅ 关键业务逻辑无破坏

### 性能保证
- ✅ 算法性能不低于基准值
- ✅ API响应时间在预期范围内
- ✅ 系统稳定性验证

### 代码质量
- ✅ 语法正确性
- ✅ 依赖完整性
- ✅ 构建成功

## 🚨 紧急情况处理

如果需要紧急修复跳过检查：
```bash
# 临时禁用钩子
git commit --no-verify -m "hotfix: emergency fix"

# 修复后立即补充完整检查
./pre_commit_check.sh
```

## 📊 检查报告

每次检查会生成日志文件：
- `logs/pre_commit_check.log` - 详细检查日志
- `logs/performance_check.log` - 性能测试结果

定期审查这些日志，持续优化检查流程。