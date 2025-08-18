# UI设计检查报告

## 检查摘要
- **总检查项**: 17
- **通过**: 4
- **失败**: 13
- **成功率**: 23.5%

### 问题分布
- **错误**: 7
- **警告**: 6
- **信息**: 0

## 详细结果

### ERRORS

- **bad_stock_card**: stock_code 格式不正确，应为6位数字
- **bad_stock_card**: 验证规则失败: if price_change > 0:
  color = colors.success
elif price_change < 0:
  color = colors.error
else:
  color = colors.text_secondary

- **trading_panel**: 股票数量必须是100的整数倍
- **trading_panel**: 价格超出当日涨跌停限制
- **trading_panel**: 账户余额不足
- **page**: 页面缺少必需组件: portfolio_summary
- **page**: 页面缺少必需组件: quick_actions

### WARNINGS

- **good_stock_card**: 颜色 border 不符合设计系统
  - 建议: 使用设计系统中的颜色: colors.border
- **good_stock_card**: 验证规则失败: data_timestamp <= 5_minutes_ago
- **bad_stock_card**: 验证规则失败: data_timestamp <= 5_minutes_ago
- **page**: 缺少响应式断点: 576px
- **page**: 缺少响应式断点: 992px
- **page**: 缺少响应式断点: 1200px


---
生成时间: 2025-08-18T10:36:08.844328
