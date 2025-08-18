#!/bin/bash

# 测试TS1109错误检测逻辑

LOG_FILE="logs/manual_typescript_check.log"

echo "🔍 测试TS1109错误检测逻辑..."
echo "=========================================="

# 检查是否有TS1109错误的不同方法
echo "方法1: 搜索 'error TS1109'"
ERROR_COUNT1=$(grep -c "error TS1109" $LOG_FILE 2>/dev/null || echo "0")
echo "   结果: $ERROR_COUNT1"

echo "方法2: 搜索 'TS1109'"
ERROR_COUNT2=$(grep -c "TS1109" $LOG_FILE 2>/dev/null || echo "0")
echo "   结果: $ERROR_COUNT2"

echo "方法3: 搜索包含TS1109的行"
echo "   详细内容:"
grep "TS1109" $LOG_FILE 2>/dev/null | head -3 | sed 's/^/      /'

echo ""
echo "📊 检测结果总结:"
if [ "$ERROR_COUNT2" -gt 0 ]; then
    echo "   ✅ 成功检测到 $ERROR_COUNT2 个TS1109错误"
    echo "   🎯 这就是您截图中显示的错误类型！"
    echo ""
    echo "💡 修复建议:"
    echo "   - 将 '~50%' 改为 '~50' + '%'"
    echo "   - 或使用模板字符串: \`~\${50}%\`"
    echo "   - 或使用普通字符串: '~50%'"
else
    echo "   ❌ 未检测到TS1109错误"
fi