# 测试标准文档

## 概述
本文档定义A股三步骤交易决策系统的测试规则、测试方法和具体测试命令。所有功能变更涉及测试变更时，必须更新此文档。

## 测试规则体系

### 核心测试原则
1. **历史数据验证**：基于(T, T+1~T+5)时间窗口进行回测验证
2. **统一测试标准**：保持测试方法、指标和命令的一致性
3. **版本性能保障**：新版本各项指标不得低于前序版本
4. **多时间点验证**：版本发布前必须通过≥5组T值的综合验证

### 测试级别定义

#### 小迭代测试（Quick Test）
- **用途**：开发过程中的快速功能验证
- **测试数据**：1组T值
- **执行频率**：每次代码修改后
- **通过标准**：功能正常运行，无异常错误

#### 版本测试（Release Test）
- **用途**：版本发布前的性能评估
- **测试数据**：≥5组T值
- **执行频率**：版本提交前必须执行
- **通过标准**：所有指标不低于前序版本基准

#### 基准更新测试（Baseline Update）
- **用途**：测试方法变更后重新建立性能基准
- **测试数据**：所有历史代表性T值
- **执行频率**：测试框架变更时
- **通过标准**：成功建立新的基准指标

## 测试指标体系

### 智能投资建议系统测试指标

#### 第一步：板块分析测试指标

**核心指标**
- **TOP5准确率**：预测的TOP5板块在T+1~T+5期间的实际表现准确率
- **排名相关性**：预测排名与实际涨跌幅排名的Spearman相关系数
- **稳定性指标**：不同T值下预测准确率的标准差
- **超额收益率**：TOP5板块相对市场基准的超额收益

**计算公式**
```
TOP5准确率 = (实际涨幅为正的预测TOP5板块数量) / 5
排名相关性 = Spearman(预测排名, 实际涨跌幅排名)
稳定性 = std(各T值下的TOP5准确率)
超额收益率 = TOP5板块平均收益率 - 市场基准收益率
```

#### 第二步：股票筛选测试指标

**核心指标**
- **平均收益率**：选中股票在T+1~T+5期间的平均收益率
- **胜率**：收益为正的股票占比
- **夏普比率**：风险调整后收益
- **板块内排名准确性**：选中股票在板块内的排名准确度

**计算公式**
```
平均收益率 = sum(各股票收益率) / 股票总数
胜率 = 盈利股票数量 / 股票总数
夏普比率 = (年化收益率 - 3%) / 年化波动率
板块内排名准确性 = 选中股票在板块内实际排名的准确度
```

#### 第三步：持仓分析与建议测试指标

**核心指标**
- **调仓建议准确性**：买入/卖出/持有建议的后续表现准确率
- **风险预警有效性**：风险提示的准确率和及时性
- **组合优化效果**：调仓后组合表现相比调仓前的改善程度
- **个性化适配度**：建议与用户实际风险承受能力的匹配度

**计算公式**
```
调仓建议准确性 = 正确建议数量 / 总建议数量
风险预警有效性 = 准确预警次数 / 总预警次数
组合优化效果 = (调仓后收益率 - 调仓前收益率) / 调仓前收益率
```

### 反人性交易助手测试指标

#### 情绪控制模块测试指标

**核心指标**
- **冲动交易阻止率**：成功阻止的冲动交易占比
- **情绪识别准确率**：正确识别用户情绪状态的准确率
- **纪律遵守率**：用户遵守系统建议的比例
- **长期行为改善**：用户交易行为的月度/季度改善趋势

#### 纪律执行模块测试指标

**核心指标**
- **规则执行率**：自动执行交易规则的成功率
- **风险控制效果**：实际最大回撤与设定目标的对比
- **止损执行准确性**：止损触发的及时性和准确性
- **仓位控制有效性**：仓位控制规则的遵守情况

### 系统整体测试指标

#### 持续进化能力测试

**核心指标**
- **模型性能趋势**：各指标随时间的改善趋势
- **参数自适应效果**：系统参数调整对性能的改善程度
- **用户满意度**：用户对系统建议的接受度和满意度
- **系统稳定性**：不同市场环境下的性能稳定性

**测试方法**
- **滚动窗口测试**：使用6个月滚动窗口测试模型稳定性
- **压力测试**：极端市场条件下的系统表现
- **A/B测试**：不同策略版本的效果对比
- **长期跟踪**：至少12个月的长期效果跟踪

### 整体流水线指标

#### 端到端指标
- **整体收益率**：完整三步骤流水线产生的最终投资组合收益率
- **年化收益率**：按年化计算的投资组合收益率
- **收益风险比**：年化收益率 / 年化波动率
- **基准超额收益**：相对于沪深300指数的超额收益

## 标准测试时间点

### 代表性T值选择
覆盖不同市场环境和季节特征的时间点：

```python
STANDARD_T_VALUES = [
    "20240615",  # Q2季度末，中报预期期
    "20240715",  # 夏季中期，传统淡季前
    "20240815",  # 传统淡季，低波动期
    "20240915",  # 秋季开始，三季报预期期
    "20241015",  # Q3季度末，年报预期期
    "20241115",  # 冬季开始，年底行情前
    "20241215",  # 年底行情期，机构调仓期
]
```

### T值选择原则
1. **时间分布均匀**：覆盖全年不同月份
2. **市场环境多样**：包含牛市、熊市、震荡市
3. **财报周期覆盖**：涵盖财报发布前后期
4. **节假日避开**：避开长假和特殊事件影响

## 测试命令标准

### 环境准备
```bash
# 激活虚拟环境
source venv/bin/activate

# 检查测试环境
python tests/framework/test_env_check.py
```

### 小迭代测试
```bash
# 第一步板块预测快速测试
python tests/step1_tests/sector_prediction_test.py --mode quick --t-value 20240815

# 第二步股票筛选快速测试
python tests/step2_tests/stock_selection_test.py --mode quick --t-value 20240815

# 第三步跟踪池管理快速测试  
python tests/step3_tests/tracking_management_test.py --mode quick --t-value 20240815

# 完整快速测试
python tests/framework/test_orchestrator.py --mode quick --t-value 20240815
```

### 版本测试
```bash
# 第一步版本测试（5组T值）
python tests/step1_tests/sector_prediction_test.py --mode release --t-count 5

# 第二步版本测试（5组T值）
python tests/step2_tests/stock_selection_test.py --mode release --t-count 5

# 第三步版本测试（5组T值）
python tests/step3_tests/tracking_management_test.py --mode release --t-count 5

# 端到端版本测试
python tests/integration_tests/end_to_end_test.py --mode release --t-count 5

# 完整版本测试
python tests/framework/test_orchestrator.py --mode release --t-count 5
```

### 基准更新测试
```bash
# 更新第一步基准
python tests/step1_tests/sector_prediction_test.py --mode baseline --update-baseline

# 更新第二步基准
python tests/step2_tests/stock_selection_test.py --mode baseline --update-baseline

# 更新第三步基准  
python tests/step3_tests/tracking_management_test.py --mode baseline --update-baseline

# 更新全部基准
python tests/framework/test_orchestrator.py --mode baseline --update-all
```

### 性能对比测试
```bash
# 与前一版本性能对比
python tests/framework/performance_comparator.py --compare-with previous

# 与指定版本性能对比
python tests/framework/performance_comparator.py --compare-with v1.2.0

# 生成性能趋势报告
python tests/framework/performance_comparator.py --trend-report --versions 5
```

## 提交标准和检查清单

### 提交前必检项
1. ✅ **代码质量检查**：通过linting和代码规范检查
2. ✅ **单元测试**：所有单元测试通过
3. ✅ **版本测试**：完整版本测试通过（≥5组T值）
4. ✅ **性能基准**：各项指标不低于前序版本
5. ✅ **测试报告**：自动生成详细测试报告
6. ✅ **异常确认**：如有性能下降，需人工确认和说明

### 提交流程
```bash
# 1. 运行完整版本测试
python tests/framework/test_orchestrator.py --mode release --t-count 5

# 2. 生成性能对比报告
python tests/framework/performance_comparator.py --compare-with previous

# 3. 检查测试结果
python tests/framework/test_result_checker.py --validate-for-commit

# 4. 如果通过，执行提交
git add .
git commit -m "feat: 版本更新，通过性能测试"
```

### 性能回退处理
如果新版本性能低于前序版本：
1. **自动阻止提交**：测试框架自动阻止git提交
2. **人工确认机制**：特殊情况下可通过`--force-commit-with-reason`强制提交
3. **回退说明要求**：必须提供详细的性能下降原因和改进计划
4. **追踪修复**：在下一版本中必须修复性能问题

## 测试报告规范

### 报告文件命名
```
reports/test_results/
├── quick_test_YYYYMMDD_HHMMSS.json        # 快速测试报告
├── release_test_YYYYMMDD_HHMMSS.json      # 版本测试报告
├── baseline_update_YYYYMMDD_HHMMSS.json   # 基准更新报告
└── performance_comparison_YYYYMMDD_HHMMSS.json  # 性能对比报告
```

### 报告内容标准
1. **测试元信息**：测试时间、版本号、测试类型、T值列表
2. **分步骤结果**：三个步骤的详细测试结果和指标
3. **整体性能**：端到端流水线的综合性能指标
4. **性能对比**：与前序版本的详细对比分析
5. **异常记录**：测试过程中的异常和警告信息

## 文档维护规则

### 更新触发条件
1. **新增测试指标**：添加新的性能衡量指标
2. **修改测试方法**：改变测试数据、计算公式或验证逻辑
3. **调整测试命令**：修改标准测试命令或参数
4. **变更提交标准**：调整版本发布的性能要求

### 强制文档同步机制
**重大更新后必须同步以下文档**：
1. **CLAUDE.md** - 测试相关架构和命令接口
2. **requirement.md** - 验证策略和成功标准
3. **test_standards.md** - 测试指标和验证方法
4. **README.md** - 用户测试说明

### 同步检查清单
```markdown
□ 测试指标定义与实际实现一致
□ 测试命令可以正常执行
□ 性能基准数据已更新
□ 文档间交叉引用正确
□ 版本号信息同步
```

### 文档同步要求
1. **代码提交前**：相关测试文档必须先行更新
2. **版本发布前**：确保文档与实际测试框架一致
3. **强制验证**：每次重大更新必须通过文档一致性检查

### 版本控制
- 本文档跟随代码版本进行版本控制
- 重大测试标准变更需要独立的文档版本号
- 历史测试标准需要保留，便于回溯对比
- 每次更新需要在变更日志中记录具体修改内容

---

**文档版本**：v1.0.0  
**创建时间**：2025-08-17  
**最后更新**：2025-08-17  
**下次审核**：2025-09-17