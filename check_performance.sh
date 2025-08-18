#!/bin/bash

echo "📈 算法性能基准检查..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 日志文件
PERF_LOG="logs/performance_check.log"
mkdir -p logs
echo "=== Performance Check $(date) ===" > "$PERF_LOG"

# 基准值配置
BASELINE_ENHANCED_MOMENTUM=66.5
BASELINE_SECTOR_ACCURACY=64.0
BASELINE_STOCK_WIN_RATE=40.0

echo "🎯 性能基准配置:"
echo "   Enhanced Momentum准确率基准: ${BASELINE_ENHANCED_MOMENTUM}%"
echo "   板块预测准确率基准: ${BASELINE_SECTOR_ACCURACY}%"
echo "   股票选择胜率基准: ${BASELINE_STOCK_WIN_RATE}%"
echo ""

# 运行标准性能测试
echo "🧪 执行标准性能测试..."
echo "命令: python src/main.py --optimized-analysis --opt-analysis-months 2 --opt-prediction-days 5"

# 检查测试脚本是否存在
if [ ! -f "src/main.py" ]; then
    echo -e "${RED}❌ 测试脚本 src/main.py 不存在${NC}"
    exit 1
fi

# 执行性能测试 (设置超时)
TEMP_RESULT_FILE="logs/temp_performance_result.txt"

echo "⏳ 运行性能测试 (最多等待5分钟)..."
if timeout 300 python src/main.py --optimized-analysis --opt-analysis-months 2 --opt-prediction-days 5 > "$TEMP_RESULT_FILE" 2>&1; then
    echo "✅ 性能测试执行完成"
else
    echo -e "${RED}❌ 性能测试执行失败或超时${NC}"
    echo "查看错误详情:"
    cat "$TEMP_RESULT_FILE"
    exit 1
fi

# 分析测试结果
echo ""
echo "📊 分析测试结果..."

# 记录原始结果到日志
cat "$TEMP_RESULT_FILE" >> "$PERF_LOG"

# 提取关键性能指标 (根据实际输出格式调整)
ENHANCED_MOMENTUM=$(grep -o 'Enhanced Momentum.*[0-9.]*%' "$TEMP_RESULT_FILE" | grep -o '[0-9.]*' | head -1)
SECTOR_ACCURACY=$(grep -o '板块预测准确率.*[0-9.]*%' "$TEMP_RESULT_FILE" | grep -o '[0-9.]*' | head -1)
STOCK_WIN_RATE=$(grep -o '股票选择胜率.*[0-9.]*%' "$TEMP_RESULT_FILE" | grep -o '[0-9.]*' | head -1)

# 如果无法解析，尝试其他格式
if [ -z "$ENHANCED_MOMENTUM" ]; then
    ENHANCED_MOMENTUM=$(grep -o 'accuracy.*[0-9.]*' "$TEMP_RESULT_FILE" | grep -o '[0-9.]*' | tail -1)
fi

echo "🔍 提取的性能指标:"
echo "   Enhanced Momentum: ${ENHANCED_MOMENTUM:-'未检测到'}%"
echo "   板块预测准确率: ${SECTOR_ACCURACY:-'未检测到'}%"
echo "   股票选择胜率: ${STOCK_WIN_RATE:-'未检测到'}%"

# 性能检查结果
PERFORMANCE_PASS=true
PERFORMANCE_ISSUES=()

# 检查Enhanced Momentum准确率
if [ -n "$ENHANCED_MOMENTUM" ]; then
    if (( $(echo "$ENHANCED_MOMENTUM >= $BASELINE_ENHANCED_MOMENTUM" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${GREEN}✅ Enhanced Momentum性能检查通过: $ENHANCED_MOMENTUM% (>= $BASELINE_ENHANCED_MOMENTUM%)${NC}"
    else
        echo -e "${RED}❌ Enhanced Momentum性能回退: $ENHANCED_MOMENTUM% (< $BASELINE_ENHANCED_MOMENTUM%)${NC}"
        PERFORMANCE_PASS=false
        PERFORMANCE_ISSUES+=("Enhanced Momentum准确率低于基准")
    fi
else
    echo -e "${YELLOW}⚠️ 无法获取Enhanced Momentum性能数据${NC}"
    PERFORMANCE_ISSUES+=("无法获取Enhanced Momentum数据")
fi

# 检查板块预测准确率
if [ -n "$SECTOR_ACCURACY" ]; then
    if (( $(echo "$SECTOR_ACCURACY >= $BASELINE_SECTOR_ACCURACY" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${GREEN}✅ 板块预测性能检查通过: $SECTOR_ACCURACY% (>= $BASELINE_SECTOR_ACCURACY%)${NC}"
    else
        echo -e "${RED}❌ 板块预测性能回退: $SECTOR_ACCURACY% (< $BASELINE_SECTOR_ACCURACY%)${NC}"
        PERFORMANCE_PASS=false
        PERFORMANCE_ISSUES+=("板块预测准确率低于基准")
    fi
fi

# 检查股票选择胜率
if [ -n "$STOCK_WIN_RATE" ]; then
    if (( $(echo "$STOCK_WIN_RATE >= $BASELINE_STOCK_WIN_RATE" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${GREEN}✅ 股票选择性能检查通过: $STOCK_WIN_RATE% (>= $BASELINE_STOCK_WIN_RATE%)${NC}"
    else
        echo -e "${RED}❌ 股票选择性能回退: $STOCK_WIN_RATE% (< $BASELINE_STOCK_WIN_RATE%)${NC}"
        PERFORMANCE_PASS=false
        PERFORMANCE_ISSUES+=("股票选择胜率低于基准")
    fi
fi

# 清理临时文件
rm -f "$TEMP_RESULT_FILE"

echo ""
echo "================================================================"

# 最终结果
if [ "$PERFORMANCE_PASS" = true ] && [ ${#PERFORMANCE_ISSUES[@]} -eq 0 ]; then
    echo -e "${GREEN}🎉 性能基准检查通过!${NC}"
    echo -e "${GREEN}✅ 所有算法性能指标均达到或超过基准值${NC}"
    echo "✅ PERFORMANCE CHECK PASSED - $(date)" >> "$PERF_LOG"
    exit 0
else
    echo -e "${RED}💥 性能基准检查失败!${NC}"
    echo -e "${RED}❌ 检测到性能回退，不建议提交此版本${NC}"
    echo ""
    echo "🔍 发现的问题:"
    for issue in "${PERFORMANCE_ISSUES[@]}"; do
        echo "   - $issue"
    done
    echo ""
    echo "💡 建议的修复步骤:"
    echo "   1. 检查算法参数是否正确"
    echo "   2. 验证数据源是否正常"
    echo "   3. 对比上一个stable版本的差异"
    echo "   4. 考虑回滚有问题的算法修改"
    echo ""
    echo "📝 详细测试结果请查看: $PERF_LOG"
    
    echo "❌ PERFORMANCE CHECK FAILED - $(date)" >> "$PERF_LOG"
    for issue in "${PERFORMANCE_ISSUES[@]}"; do
        echo "   ISSUE: $issue" >> "$PERF_LOG"
    done
    
    exit 1
fi