# UI Design Validation Framework

基于YAML配置的UI设计符合性检查框架，专为交易平台UI组件设计验证而构建。

## 🎯 框架特点

- **YAML配置驱动**: 使用可读性强的YAML文件定义检查规则
- **多层次检查**: 支持组件级、页面级和业务规则检查
- **灵活扩展**: 易于添加新的组件类型和检查规则
- **详细报告**: 提供JSON、Markdown等多种格式的检查报告
- **设计系统集成**: 内置设计token解析，确保设计系统一致性

## 📁 文件结构

```
ui_checker/
├── ui_validation_schema.yaml    # 核心验证架构和设计系统定义
├── component_check_rules.yaml   # 具体组件检查规则
├── ui_checker.py               # 检查引擎核心代码
├── example_usage.py            # 使用示例和演示
└── README.md                   # 说明文档（本文件）
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install pyyaml
```

### 2. 基本使用

```python
from ui_checker import UIChecker, ComponentData

# 初始化检查器
checker = UIChecker(
    schema_file="ui_validation_schema.yaml",
    rules_file="component_check_rules.yaml"
)

# 创建组件数据
stock_card = ComponentData(
    name="my_stock_card",
    type="stock_card",
    props={
        "stock_code": "000001",
        "stock_name": "平安银行",
        "current_price": 15.50,
        "price_change": 0.32,
        "change_percentage": 2.11
    },
    styles={
        "width": "250px",
        "height": "120px",
        "color": "#52c41a"
    }
)

# 运行检查
checker.run_checks([stock_card])

# 生成报告
report = checker.generate_report("json")
print(report)
```

### 3. 运行完整示例

```bash
cd ui_checker
python example_usage.py
```

## 📋 支持的检查类型

### 🏗️ 结构检查
- **必需元素验证**: 检查组件是否包含所有必需的属性和元素
- **数据格式验证**: 验证数据格式（如股票代码6位数字格式）
- **组件层次结构**: 检查组件嵌套关系是否正确

### 🎨 视觉检查
- **尺寸要求**: 验证最小/最大宽高、纵横比等
- **颜色系统**: 检查是否使用设计系统中定义的颜色
- **字体规范**: 验证字体大小、字重是否符合设计系统
- **间距规范**: 检查padding、margin是否使用设计token

### ♿ 可访问性检查
- **对比度验证**: 确保文字与背景的对比度≥4.5:1
- **焦点指示器**: 检查是否有清晰的焦点状态
- **语义化标签**: 验证是否有合适的ARIA标签和语义化元素
- **键盘导航**: 检查键盘可访问性

### 🔍 业务规则检查
- **交易逻辑**: 验证价格涨跌颜色逻辑是否正确
- **数据时效性**: 检查数据是否在有效时间范围内
- **风险控制**: 验证交易金额、仓位控制等业务约束
- **用户体验**: 检查加载状态、错误处理等

## 🏷️ 支持的组件类型

### 交易平台专用组件
- **stock_card**: 股票卡片组件
- **portfolio_summary**: 投资组合摘要
- **trading_panel**: 交易操作面板

### 通用UI组件
- **button**: 按钮组件
- **input**: 表单输入组件
- **modal**: 模态对话框
- **data_table**: 数据表格

## ⚙️ 配置说明

### 设计系统配置 (ui_validation_schema.yaml)

```yaml
design_system:
  colors:
    primary: "#1890ff"
    success: "#52c41a"
    error: "#f5222d"
  
  typography:
    font_sizes:
      sm: "14px"
      md: "16px"
      lg: "18px"
  
  spacing:
    sm: "8px"
    md: "16px"
    lg: "24px"
```

### 组件规则配置 (component_check_rules.yaml)

```yaml
trading_platform_components:
  stock_card:
    structure_requirements:
      required_elements:
        - stock_code:
            type: "text"
            format: "6位数字"
        - current_price:
            type: "number"
            precision: 2
    
    visual_requirements:
      dimensions:
        min_width: "200px"
        min_height: "120px"
      
      colors:
        price_up: "colors.success"
        price_down: "colors.error"
    
    validation_rules:
      - name: "price_color_logic"
        rule: "price_change > 0 ? color == colors.success"
        severity: "error"
```

## 📊 报告格式

### JSON报告
```json
{
  "summary": {
    "total_checks": 15,
    "passed_checks": 12,
    "failed_checks": 3,
    "success_rate": "80.0%"
  },
  "results_by_severity": {
    "errors": [...],
    "warnings": [...],
    "infos": [...]
  }
}
```

### Markdown报告
```markdown
# UI设计检查报告

## 检查摘要
- **总检查项**: 15
- **通过**: 12
- **失败**: 3
- **成功率**: 80.0%

### ERRORS
- **stock_card**: 股票代码格式不正确，应为6位数字
```

## 🔧 扩展开发

### 添加新组件类型

1. 在 `component_check_rules.yaml` 中添加组件规则：

```yaml
my_custom_components:
  new_component:
    component_name: "NewComponent"
    structure_requirements:
      required_props:
        - prop1
        - prop2
    validation_rules:
      - name: "custom_rule"
        rule: "prop1 > 0"
        severity: "error"
```

2. 在代码中使用：

```python
new_comp = ComponentData(
    name="test",
    type="new_component", 
    props={"prop1": 5, "prop2": "value"}
)
```

### 添加自定义检查逻辑

在 `ui_checker.py` 的相应检查方法中添加特殊处理逻辑：

```python
def _check_validation(self, component, rules):
    # 添加自定义检查逻辑
    if rule_name == "my_custom_rule":
        # 自定义检查实现
        passed = my_custom_check(component)
```

## 🎯 使用场景

### 1. 开发阶段
- 在组件开发完成后运行检查，确保符合设计规范
- 集成到开发工作流，自动化设计一致性验证

### 2. 代码审查
- 在Pull Request中自动运行UI检查
- 提供客观的设计符合性评估

### 3. 设计系统维护
- 定期检查现有组件是否符合最新设计系统
- 识别需要更新的组件和页面

### 4. 质量保证
- 在发布前进行全面的UI质量检查
- 确保用户体验的一致性和可访问性

## 📈 性能考虑

- **大规模检查**: 框架支持批量检查多个组件
- **并行处理**: 可以并行检查不同类型的规则
- **缓存机制**: 设计token解析结果可以缓存
- **增量检查**: 支持只检查变更的组件

## 🐛 故障排除

### 常见问题

1. **YAML解析错误**
   - 检查YAML文件缩进是否正确
   - 确保特殊字符正确转义

2. **组件类型未找到**
   - 确认组件类型名称与规则文件中的定义一致
   - 检查是否在正确的分类下定义组件规则

3. **设计token解析失败**
   - 确认token路径正确（如 `colors.primary`）
   - 检查设计系统配置是否完整

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/new-component`)
3. 提交更改 (`git commit -am 'Add new component rules'`)
4. 推送分支 (`git push origin feature/new-component`)
5. 创建Pull Request

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🙏 致谢

- 感谢所有贡献者的支持
- 特别感谢交易平台设计团队提供的设计规范
- 基于现代前端最佳实践构建