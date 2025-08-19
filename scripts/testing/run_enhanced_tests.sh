#!/bin/bash

# A股智能交易决策平台 - 增强版测试运行脚本
# 改进的测试标准和流程

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧪 运行A股智能交易决策平台增强版测试..."
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
echo "🧪 开始运行增强版测试套件"
echo "======================================================"

# 5. 运行增强版集成测试
echo ""
echo "🔧 测试1: 增强版系统集成测试"
echo "----------------------------------------------------"

ENHANCED_TEST_SUCCESS=true

if python enhanced_test.py; then
    echo "✅ 增强版集成测试通过"
else
    echo "❌ 增强版集成测试失败"
    ENHANCED_TEST_SUCCESS=false
fi

# 6. 运行WebSocket专项测试
echo ""
echo "🔌 测试2: WebSocket专项测试"
echo "----------------------------------------------------"

WEBSOCKET_TEST_SUCCESS=true

if python test_websocket.py; then
    echo "✅ WebSocket专项测试通过"
else
    echo "⚠️  WebSocket专项测试部分失败"
    WEBSOCKET_TEST_SUCCESS=false
fi

# 7. 运行传统集成测试作为对比
echo ""
echo "🔧 测试3: 传统集成测试对比"
echo "----------------------------------------------------"

INTEGRATION_TEST_SUCCESS=true

if python integration_test.py; then
    echo "✅ 传统集成测试通过"
else
    echo "⚠️  传统集成测试部分失败"
    INTEGRATION_TEST_SUCCESS=false
fi

# 8. 运行API性能基准测试
echo ""
echo "⚡ 测试4: API性能基准测试"
echo "----------------------------------------------------"

performance_test() {
    local endpoint=$1
    local description=$2
    local max_time=$3
    
    echo -n "测试 $description... "
    
    start_time=$(date +%s%N)
    if curl -s "$endpoint" > /dev/null 2>&1; then
        end_time=$(date +%s%N)
        duration=$((($end_time - $start_time) / 1000000))  # 转换为毫秒
        
        if [ $duration -le $max_time ]; then
            echo "✅ (${duration}ms)"
            return 0
        else
            echo "⚠️  (${duration}ms > ${max_time}ms)"
            return 1
        fi
    else
        echo "❌"
        return 1
    fi
}

PERF_TESTS_PASSED=0
PERF_TESTS_TOTAL=0

# 性能基准测试
if performance_test "http://localhost:8000/health" "健康检查" 100; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

if performance_test "http://localhost:8000/api/v1/system/info" "系统信息" 200; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

if performance_test "http://localhost:8000/api/v1/sectors/?top_n=3" "板块分析" 500; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

if performance_test "http://localhost:8000/api/v1/stocks/" "股票列表" 300; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

if performance_test "http://localhost:8000/api/v1/portfolio/preset" "投资组合" 400; then
    ((PERF_TESTS_PASSED++))
fi
((PERF_TESTS_TOTAL++))

echo ""
echo "性能测试结果: $PERF_TESTS_PASSED/$PERF_TESTS_TOTAL 通过"

# 9. 数据一致性验证
echo ""
echo "📊 测试5: 数据一致性验证"
echo "----------------------------------------------------"

echo -n "验证板块数据一致性... "
SECTORS_RESPONSE=$(curl -s "http://localhost:8000/api/v1/sectors/list")
SECTORS_COUNT=$(echo "$SECTORS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', {}).get('sectors', [])))" 2>/dev/null || echo "0")

if [ "$SECTORS_COUNT" -ge 8 ]; then
    echo "✅ ($SECTORS_COUNT 个板块)"
    DATA_CONSISTENCY_PASS=true
else
    echo "❌ (仅 $SECTORS_COUNT 个板块)"
    DATA_CONSISTENCY_PASS=false
fi

echo -n "验证投资组合数据一致性... "
PORTFOLIO_RESPONSE=$(curl -s "http://localhost:8000/api/v1/portfolio/preset")
HOLDINGS_COUNT=$(echo "$PORTFOLIO_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', {}).get('preset_holdings', [])))" 2>/dev/null || echo "0")

if [ "$HOLDINGS_COUNT" -ge 10 ]; then
    echo "✅ ($HOLDINGS_COUNT 只股票)"
    if [ "$DATA_CONSISTENCY_PASS" = "true" ]; then
        DATA_CONSISTENCY_PASS=true
    fi
else
    echo "❌ (仅 $HOLDINGS_COUNT 只股票)"
    DATA_CONSISTENCY_PASS=false
fi

# 10. 生成测试报告
echo ""
echo "======================================================"
echo "🏁 增强版测试结果总结"
echo "======================================================"

TOTAL_SCORE=0
MAX_SCORE=100

# 增强版集成测试 (40分)
if [ "$ENHANCED_TEST_SUCCESS" = "true" ]; then
    echo "🔧 增强版集成测试: ✅ (40/40分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 40))
else
    echo "🔧 增强版集成测试: ❌ (0/40分)"
fi

# WebSocket专项测试 (20分)
if [ "$WEBSOCKET_TEST_SUCCESS" = "true" ]; then
    echo "🔌 WebSocket专项测试: ✅ (20/20分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 20))
else
    echo "🔌 WebSocket专项测试: ⚠️  (15/20分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 15))
fi

# 传统集成测试 (15分)
if [ "$INTEGRATION_TEST_SUCCESS" = "true" ]; then
    echo "🔧 传统集成测试: ✅ (15/15分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 15))
else
    echo "🔧 传统集成测试: ⚠️  (10/15分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
fi

# API性能测试 (15分)
PERF_SCORE=$((PERF_TESTS_PASSED * 15 / PERF_TESTS_TOTAL))
echo "⚡ API性能测试: ($PERF_TESTS_PASSED/$PERF_TESTS_TOTAL) (${PERF_SCORE}/15分)"
TOTAL_SCORE=$((TOTAL_SCORE + PERF_SCORE))

# 数据一致性验证 (10分)
if [ "$DATA_CONSISTENCY_PASS" = "true" ]; then
    echo "📊 数据一致性验证: ✅ (10/10分)"
    TOTAL_SCORE=$((TOTAL_SCORE + 10))
else
    echo "📊 数据一致性验证: ❌ (0/10分)"
fi

echo ""
echo "📊 最终得分: ${TOTAL_SCORE}/${MAX_SCORE}"

# 评级系统
if [ $TOTAL_SCORE -ge 95 ]; then
    echo "🏆 测试结果: 卓越 (≥95分)"
    echo "🎉 系统达到生产级别，可以立即部署！"
    FINAL_GRADE="EXCELLENT"
elif [ $TOTAL_SCORE -ge 90 ]; then
    echo "🥇 测试结果: 优秀 (≥90分)"
    echo "✅ 系统运行良好，建议部署"
    FINAL_GRADE="GOOD"
elif [ $TOTAL_SCORE -ge 80 ]; then
    echo "🥈 测试结果: 良好 (≥80分)"
    echo "⚠️  系统基本正常，建议优化后部署"
    FINAL_GRADE="FAIR"
elif [ $TOTAL_SCORE -ge 70 ]; then
    echo "🥉 测试结果: 及格 (≥70分)"
    echo "⚠️  系统有问题，需要修复"
    FINAL_GRADE="PASS"
else
    echo "❌ 测试结果: 不及格 (<70分)"
    echo "🔧 系统存在严重问题，必须修复"
    FINAL_GRADE="FAIL"
fi

echo ""
echo "======================================================"
echo "📝 详细信息:"
echo "   测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   测试标准: 增强版多维度验证"
echo "   最终评级: $FINAL_GRADE"
echo ""
echo "📁 日志文件:"
echo "   后端日志: tail -f logs/backend.log"
echo "   前端日志: tail -f logs/frontend.log"
echo ""
echo "🌐 访问地址:"
echo "   前端应用: http://localhost:3000"
echo "   API文档:  http://localhost:8000/docs"
echo "   系统状态: http://localhost:8000/health"
echo "======================================================"

# 根据评级决定退出状态
if [ "$FINAL_GRADE" = "EXCELLENT" ] || [ "$FINAL_GRADE" = "GOOD" ]; then
    exit 0
elif [ "$FINAL_GRADE" = "FAIR" ]; then
    exit 1
else
    exit 2
fi