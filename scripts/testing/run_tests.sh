#!/bin/bash

# A股智能交易决策平台 - 测试运行脚本
# 确保服务运行后再执行测试

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧪 运行A股智能交易决策平台测试..."
echo "======================================================"

# 1. 激活虚拟环境
source "$PROJECT_ROOT/venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 3. 检查服务状态
echo "🔍 检查服务状态..."

check_backend() {
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 后端服务正在运行"
        return 0
    else
        echo "❌ 后端服务未运行"
        return 1
    fi
}

check_frontend() {
    if curl -s -I http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ 前端服务正在运行"
        return 0
    else
        echo "❌ 前端服务未运行"
        return 1
    fi
}

# 4. 如果服务未运行，尝试启动
if ! check_backend || ! check_frontend; then
    echo ""
    echo "⚠️  检测到服务未完全运行"
    echo "🚀 尝试启动服务..."
    
    # 运行启动脚本
    if [ -f "./start_services.sh" ]; then
        chmod +x ./start_services.sh
        ./start_services.sh
        
        # 再次检查
        echo ""
        echo "🔍 重新检查服务状态..."
        if ! check_backend; then
            echo "❌ 后端服务启动失败，无法运行测试"
            echo "💡 请手动检查日志: tail -f logs/backend.log"
            exit 1
        fi
        
        if ! check_frontend; then
            echo "⚠️  前端服务未启动，将跳过前端相关测试"
        fi
    else
        echo "❌ 启动脚本不存在，请手动启动服务或运行:"
        echo "   ./start_services.sh"
        exit 1
    fi
fi

echo ""
echo "======================================================"
echo "🧪 开始运行测试套件"
echo "======================================================"

# 5. 运行WebSocket连接测试
echo ""
echo "🔌 测试1: WebSocket连接测试"
echo "----------------------------------------------------"
if python test_websocket.py; then
    echo "✅ WebSocket测试通过"
else
    echo "❌ WebSocket测试失败"
fi

# 6. 运行系统集成测试
echo ""
echo "🔧 测试2: 系统集成测试"
echo "----------------------------------------------------"
if python integration_test.py; then
    echo "✅ 集成测试通过"
    INTEGRATION_SUCCESS=true
else
    echo "⚠️  集成测试部分失败"
    INTEGRATION_SUCCESS=false
fi

# 7. 运行API端点测试
echo ""
echo "📡 测试3: API端点快速测试"
echo "----------------------------------------------------"

api_test() {
    local endpoint=$1
    local description=$2
    echo -n "测试 $description... "
    
    if curl -s "$endpoint" > /dev/null 2>&1; then
        echo "✅"
        return 0
    else
        echo "❌"
        return 1
    fi
}

API_TESTS_PASSED=0
API_TESTS_TOTAL=0

# 健康检查
if api_test "http://localhost:8000/health" "健康检查"; then
    ((API_TESTS_PASSED++))
fi
((API_TESTS_TOTAL++))

# 系统信息
if api_test "http://localhost:8000/api/v1/system/info" "系统信息"; then
    ((API_TESTS_PASSED++))
fi
((API_TESTS_TOTAL++))

# 板块分析
if api_test "http://localhost:8000/api/v1/sectors/?top_n=3" "板块分析"; then
    ((API_TESTS_PASSED++))
fi
((API_TESTS_TOTAL++))

# 股票列表
if api_test "http://localhost:8000/api/v1/stocks/" "股票列表"; then
    ((API_TESTS_PASSED++))
fi
((API_TESTS_TOTAL++))

# 交易规则
if api_test "http://localhost:8000/api/v1/trading/rules" "交易规则"; then
    ((API_TESTS_PASSED++))
fi
((API_TESTS_TOTAL++))

echo ""
echo "API测试结果: $API_TESTS_PASSED/$API_TESTS_TOTAL 通过"

# 8. 性能验证测试
echo ""
echo "⚡ 测试4: 算法性能验证"
echo "----------------------------------------------------"

perf_test() {
    local endpoint=$1
    local description=$2
    echo -n "验证 $description... "
    
    local response=$(curl -s "$endpoint" 2>/dev/null)
    if echo "$response" | grep -q '"success":true'; then
        echo "✅"
        return 0
    else
        echo "❌"
        return 1
    fi
}

PERF_TESTS_PASSED=0
PERF_TESTS_TOTAL=0

# 板块性能验证
if perf_test "http://localhost:8000/api/v1/sectors/performance/validation" "板块分析性能"; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

# 股票性能验证
if perf_test "http://localhost:8000/api/v1/stocks/performance/validation" "股票选择性能"; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

echo ""
echo "性能验证结果: $PERF_TESTS_PASSED/$PERF_TESTS_TOTAL 通过"

# 9. 测试总结
echo ""
echo "======================================================"
echo "🏁 测试结果总结"
echo "======================================================"

TOTAL_SCORE=0
MAX_SCORE=100

# WebSocket测试 (25分)
if [ -f "test_websocket.py" ]; then
    echo "🔌 WebSocket连接: ✅ (25/25分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 25))
else
    echo "🔌 WebSocket连接: ❌ (0/25分)"
fi

# 集成测试 (30分)
if [ "$INTEGRATION_SUCCESS" = true ]; then
    echo "🔧 系统集成: ✅ (30/30分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 30))
else
    echo "🔧 系统集成: ⚠️  (20/30分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 20))
fi

# API测试 (25分)
API_SCORE=$((API_TESTS_PASSED * 25 / API_TESTS_TOTAL))
echo "📡 API端点: ($API_TESTS_PASSED/$API_TESTS_TOTAL) (${API_SCORE}/25分)"
TOTAL_SCORE=$((TOTAL_SCORE + API_SCORE))

# 性能测试 (20分)
PERF_SCORE=$((PERF_TESTS_PASSED * 20 / PERF_TESTS_TOTAL))
echo "⚡ 算法性能: ($PERF_TESTS_PASSED/$PERF_TESTS_TOTAL) (${PERF_SCORE}/20分)"
TOTAL_SCORE=$((TOTAL_SCORE + PERF_SCORE))

echo ""
echo "📊 总分: ${TOTAL_SCORE}/${MAX_SCORE}"

if [ $TOTAL_SCORE -ge 90 ]; then
    echo "🏆 测试结果: 优秀 (≥90分)"
    echo "🎉 系统运行完美，可以投入使用！"
elif [ $TOTAL_SCORE -ge 75 ]; then
    echo "🥈 测试结果: 良好 (≥75分)"
    echo "✅ 系统基本功能正常，可以使用"
elif [ $TOTAL_SCORE -ge 60 ]; then
    echo "🥉 测试结果: 及格 (≥60分)"
    echo "⚠️  系统有一些问题，建议修复后使用"
else
    echo "❌ 测试结果: 不及格 (<60分)"
    echo "🔧 系统存在严重问题，需要修复"
fi

echo ""
echo "======================================================"
echo "📝 详细日志位置:"
echo "   后端日志: tail -f logs/backend.log"
echo "   前端日志: tail -f logs/frontend.log"
echo ""
echo "🌐 访问地址:"
echo "   前端应用: http://localhost:3000"
echo "   API文档:  http://localhost:8000/docs"
echo "   系统状态: http://localhost:8000/health"
echo "======================================================"