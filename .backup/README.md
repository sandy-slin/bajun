# Backup 目录说明

## 概述
此目录存放项目重构过程中被移除的历史文件，包括过时的代码、文档和测试结果。

## 目录结构

### 历史测试文件
- `backtest_summary_*.txt` - 历史回测摘要报告
- `systematic_backtest_report_*.json` - 系统性回测详细报告
- `systematic_backtest_validator.py` - 旧版本的系统验证器
- `test_stage3_optimization.py` - 阶段3优化测试脚本
- `sector_predictions_*.json` - 历史板块预测结果

### 过时代码模块
- `src/analysis/` - 与新三步骤架构不符的分析模块
  - `advanced_signal_filter.py` - 高级信号过滤器
  - `ensemble_prediction_engine.py` - 集成预测引擎
  - `optimized_data_analyzer.py` - 优化数据分析器
  - 等等...

### 历史文档
- `docs/accuracy_improvement_plan.md` - 旧准确率改进计划
- `docs/real_data_accuracy_plan.md` - 旧真实数据计划
- `docs/user_input/analyze_stocks.md` - 过时股票分析文档
- `docs/user_input/bajun.md` - 特定八骏分析文档

### 历史报告
- `reports/analysis/` - 大量历史分析报告
- `reports/backtest/` - 回测报告
- `reports/screening/` - 筛选报告
- `reports/validation/` - 验证报告

### 训练和模型文件
- `catboost_info/` - CatBoost训练信息
- `models/` - 历史模型文件
- `logs/` - 历史日志文件

## 保留原因
这些文件虽然不符合新的架构要求，但可能包含有价值的算法思路或实现细节，保留用于：
1. 参考历史实现方法
2. 对比新旧版本性能差异
3. 恢复特定功能（如需要）
4. 学习和研究目的

## 注意事项
- 备份目录中的代码不应在新系统中使用
- 如需引用历史实现，应重新设计以符合新架构规范
- 定期清理过于陈旧的备份文件

---
**备份时间**: 2025-08-17
**备份原因**: 项目重构，建立新的三步骤交易决策系统架构