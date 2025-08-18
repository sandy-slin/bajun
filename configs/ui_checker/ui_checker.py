#!/usr/bin/env python3
"""
UI Design Validation Framework
基于YAML配置的UI设计符合性检查引擎
"""

import yaml
import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CheckResult:
    """检查结果数据类"""
    name: str
    passed: bool
    severity: str
    message: str
    component: str = ""
    rule: str = ""
    expected: Any = None
    actual: Any = None
    suggestion: str = ""

@dataclass  
class ComponentData:
    """组件数据结构"""
    name: str
    type: str
    props: Dict[str, Any] = field(default_factory=dict)
    styles: Dict[str, Any] = field(default_factory=dict)
    children: List['ComponentData'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class UIChecker:
    """UI设计验证检查器"""
    
    def __init__(self, schema_file: str, rules_file: str):
        """初始化检查器"""
        self.schema = self._load_yaml(schema_file)
        self.rules = self._load_yaml(rules_file)
        self.design_system = self.schema.get('design_system', {})
        self.results: List[CheckResult] = []
        
    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise
            
    def _resolve_design_token(self, value: str) -> Any:
        """解析设计系统token (如 colors.primary)"""
        if not isinstance(value, str) or not value.startswith(('colors.', 'typography.', 'spacing.')):
            return value
            
        parts = value.split('.')
        current = self.design_system
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            logger.warning(f"设计token未找到: {value}")
            return value
            
    def _evaluate_rule(self, rule: str, context: Dict[str, Any]) -> bool:
        """评估规则表达式"""
        try:
            # 简单的规则评估，实际项目中可以使用更复杂的表达式引擎
            if '>=' in rule:
                left, right = rule.split('>=')
                return float(context.get(left.strip(), 0)) >= float(right.strip())
            elif '<=' in rule:
                left, right = rule.split('<=')
                return float(context.get(left.strip(), 0)) <= float(right.strip())
            elif '>' in rule:
                left, right = rule.split('>')
                return float(context.get(left.strip(), 0)) > float(right.strip())
            elif '<' in rule:
                left, right = rule.split('<')
                return float(context.get(left.strip(), 0)) < float(right.strip())
            elif '==' in rule:
                left, right = rule.split('==')
                return str(context.get(left.strip(), '')).strip() == right.strip()
            elif 'in' in rule:
                left, right = rule.split(' in ')
                return context.get(left.strip()) in eval(right.strip())
            else:
                # 对于更复杂的规则，可以使用eval（注意安全性）
                return eval(rule, {"__builtins__": {}}, context)
        except Exception as e:
            logger.warning(f"规则评估失败: {rule}, 错误: {e}")
            return False
            
    def check_component(self, component: ComponentData) -> List[CheckResult]:
        """检查单个组件"""
        results = []
        component_type = component.type.lower()
        
        # 查找对应的组件规则
        component_rules = None
        for category in ['trading_platform_components', 'generic_components']:
            if category in self.rules:
                for comp_name, comp_rules in self.rules[category].items():
                    if comp_name.lower() == component_type or comp_rules.get('component_name', '').lower() == component_type:
                        component_rules = comp_rules
                        break
                if component_rules:
                    break
                    
        if not component_rules:
            logger.warning(f"未找到组件 {component_type} 的检查规则")
            return results
            
        # 检查结构要求
        if 'structure_requirements' in component_rules:
            results.extend(self._check_structure(component, component_rules['structure_requirements']))
            
        # 检查视觉要求
        if 'visual_requirements' in component_rules:
            results.extend(self._check_visual(component, component_rules['visual_requirements']))
            
        # 检查可访问性要求
        if 'accessibility_requirements' in component_rules:
            results.extend(self._check_accessibility(component, component_rules['accessibility_requirements']))
            
        # 检查验证规则
        if 'validation_rules' in component_rules:
            results.extend(self._check_validation(component, component_rules['validation_rules']))
            
        return results
        
    def _check_structure(self, component: ComponentData, requirements: Dict[str, Any]) -> List[CheckResult]:
        """检查组件结构要求"""
        results = []
        
        # 检查必需元素
        if 'required_elements' in requirements:
            for element_name, element_config in requirements['required_elements'].items():
                if element_name not in component.props:
                    results.append(CheckResult(
                        name=f"missing_required_element_{element_name}",
                        passed=False,
                        severity="error",
                        message=f"缺少必需元素: {element_name}",
                        component=component.name,
                        rule="structure_requirements.required_elements"
                    ))
                else:
                    # 检查元素格式
                    if 'format' in element_config:
                        value = component.props[element_name]
                        format_rule = element_config['format']
                        
                        if format_rule == "6位数字":
                            if not (isinstance(value, str) and len(value) == 6 and value.isdigit()):
                                results.append(CheckResult(
                                    name=f"invalid_format_{element_name}",
                                    passed=False,
                                    severity="error",
                                    message=f"{element_name} 格式不正确，应为6位数字",
                                    component=component.name,
                                    rule="structure_requirements.format",
                                    expected="6位数字",
                                    actual=value
                                ))
                                
        # 检查必需属性
        if 'required_props' in requirements:
            for prop_name in requirements['required_props']:
                if prop_name not in component.props:
                    results.append(CheckResult(
                        name=f"missing_required_prop_{prop_name}",
                        passed=False,
                        severity="error", 
                        message=f"缺少必需属性: {prop_name}",
                        component=component.name,
                        rule="structure_requirements.required_props"
                    ))
                    
        return results
        
    def _check_visual(self, component: ComponentData, requirements: Dict[str, Any]) -> List[CheckResult]:
        """检查视觉要求"""
        results = []
        
        # 检查尺寸要求
        if 'dimensions' in requirements:
            dimensions = requirements['dimensions']
            styles = component.styles
            
            for dim_name, expected_value in dimensions.items():
                if dim_name in styles:
                    actual_value = styles[dim_name]
                    expected_resolved = self._resolve_design_token(expected_value)
                    
                    # 简化的尺寸检查逻辑
                    if dim_name.startswith('min_') and actual_value < expected_resolved:
                        results.append(CheckResult(
                            name=f"dimension_violation_{dim_name}",
                            passed=False,
                            severity="warning",
                            message=f"{dim_name} 不满足最小值要求",
                            component=component.name,
                            rule="visual_requirements.dimensions",
                            expected=expected_resolved,
                            actual=actual_value
                        ))
                        
        # 检查颜色使用
        if 'colors' in requirements:
            colors = requirements['colors']
            for color_name, expected_color in colors.items():
                if color_name in component.styles:
                    actual_color = component.styles[color_name]
                    expected_resolved = self._resolve_design_token(expected_color)
                    
                    if actual_color != expected_resolved:
                        results.append(CheckResult(
                            name=f"color_mismatch_{color_name}",
                            passed=False,
                            severity="warning",
                            message=f"颜色 {color_name} 不符合设计系统",
                            component=component.name,
                            rule="visual_requirements.colors",
                            expected=expected_resolved,
                            actual=actual_color,
                            suggestion=f"使用设计系统中的颜色: {expected_color}"
                        ))
                        
        return results
        
    def _check_accessibility(self, component: ComponentData, requirements: List[Dict[str, Any]]) -> List[CheckResult]:
        """检查可访问性要求"""
        results = []
        
        for requirement in requirements:
            name = requirement.get('name', '')
            rule = requirement.get('rule', '')
            description = requirement.get('description', '')
            severity = requirement.get('severity', 'warning')
            
            # 简化的可访问性检查
            passed = True
            message = ""
            
            if rule == "has_focus_indicator":
                if 'focus' not in component.styles:
                    passed = False
                    message = "缺少焦点指示器样式"
                    
            elif rule == "has_associated_label":
                if 'aria-label' not in component.props and 'label' not in component.props:
                    passed = False
                    message = "缺少关联标签或aria-label"
                    
            elif rule == "contrast_ratio >= 4.5":
                # 这里需要实际的对比度计算，暂时简化处理
                if component.styles.get('color') == component.styles.get('background_color'):
                    passed = False
                    message = "文字与背景颜色相同，对比度不足"
                    
            results.append(CheckResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message or description,
                component=component.name,
                rule="accessibility_requirements"
            ))
            
        return results
        
    def _check_validation(self, component: ComponentData, rules: List[Dict[str, Any]]) -> List[CheckResult]:
        """检查验证规则"""
        results = []
        
        for rule_config in rules:
            name = rule_config.get('name', '')
            rule = rule_config.get('rule', '')
            severity = rule_config.get('severity', 'warning')
            message = rule_config.get('message', '')
            
            # 创建上下文用于规则评估
            context = {
                **component.props,
                **component.styles,
                **component.metadata
            }
            
            # 特殊规则处理
            passed = True
            if name == "price_color_logic":
                price_change = context.get('price_change', 0)
                color = context.get('color', '')
                
                if price_change > 0 and color != self._resolve_design_token('colors.success'):
                    passed = False
                elif price_change < 0 and color != self._resolve_design_token('colors.error'):
                    passed = False
                elif price_change == 0 and color != self._resolve_design_token('colors.text_secondary'):
                    passed = False
            else:
                # 通用规则评估
                passed = self._evaluate_rule(rule, context)
                
            results.append(CheckResult(
                name=name,
                passed=passed,
                severity=severity,
                message=message or f"验证规则失败: {rule}",
                component=component.name,
                rule="validation_rules"
            ))
            
        return results
        
    def check_page(self, page_data: Dict[str, Any]) -> List[CheckResult]:
        """检查页面级别规则"""
        results = []
        page_type = page_data.get('type', '')
        
        # 查找页面级别规则
        page_rules = self.rules.get('page_level_checks', {}).get(page_type, {})
        
        if not page_rules:
            logger.warning(f"未找到页面类型 {page_type} 的检查规则")
            return results
            
        # 检查布局要求
        if 'layout_requirements' in page_rules:
            layout_req = page_rules['layout_requirements']
            
            if 'responsive_breakpoints' in layout_req:
                expected_breakpoints = layout_req['responsive_breakpoints']
                actual_breakpoints = page_data.get('breakpoints', [])
                
                for bp in expected_breakpoints:
                    if bp not in actual_breakpoints:
                        results.append(CheckResult(
                            name=f"missing_breakpoint_{bp}",
                            passed=False,
                            severity="warning",
                            message=f"缺少响应式断点: {bp}",
                            component="page",
                            rule="layout_requirements.responsive_breakpoints"
                        ))
                        
        # 检查内容要求
        if 'content_requirements' in page_rules:
            for requirement in page_rules['content_requirements']:
                if isinstance(requirement, dict):
                    for component_name, requirement_type in requirement.items():
                        if requirement_type == "required":
                            if component_name not in page_data.get('components', []):
                                results.append(CheckResult(
                                    name=f"missing_required_component_{component_name}",
                                    passed=False,
                                    severity="error",
                                    message=f"页面缺少必需组件: {component_name}",
                                    component="page",
                                    rule="content_requirements"
                                ))
                                
        return results
        
    def generate_report(self, output_format: str = "json") -> Union[str, Dict[str, Any]]:
        """生成检查报告"""
        # 统计结果
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        infos = [r for r in self.results if not r.passed and r.severity == "info"]
        
        report_data = {
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "error_count": len(errors),
                "warning_count": len(warnings),
                "info_count": len(infos),
                "success_rate": f"{(passed_checks/total_checks*100):.1f}%" if total_checks > 0 else "0%"
            },
            "results_by_severity": {
                "errors": [self._result_to_dict(r) for r in errors],
                "warnings": [self._result_to_dict(r) for r in warnings],
                "infos": [self._result_to_dict(r) for r in infos]
            },
            "results_by_component": self._group_results_by_component(),
            "timestamp": datetime.now().isoformat()
        }
        
        if output_format == "json":
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        elif output_format == "markdown":
            return self._generate_markdown_report(report_data)
        else:
            return report_data
            
    def _result_to_dict(self, result: CheckResult) -> Dict[str, Any]:
        """将检查结果转换为字典"""
        return {
            "name": result.name,
            "passed": result.passed,
            "severity": result.severity,
            "message": result.message,
            "component": result.component,
            "rule": result.rule,
            "expected": result.expected,
            "actual": result.actual,
            "suggestion": result.suggestion
        }
        
    def _group_results_by_component(self) -> Dict[str, List[Dict[str, Any]]]:
        """按组件分组结果"""
        grouped = {}
        for result in self.results:
            component = result.component or "unknown"
            if component not in grouped:
                grouped[component] = []
            grouped[component].append(self._result_to_dict(result))
        return grouped
        
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        summary = report_data["summary"]
        
        md = f"""# UI设计检查报告

## 检查摘要
- **总检查项**: {summary['total_checks']}
- **通过**: {summary['passed_checks']}
- **失败**: {summary['failed_checks']}
- **成功率**: {summary['success_rate']}

### 问题分布
- **错误**: {summary['error_count']}
- **警告**: {summary['warning_count']}
- **信息**: {summary['info_count']}

## 详细结果

"""
        
        # 按严重程度显示问题
        for severity, results in report_data["results_by_severity"].items():
            if results:
                md += f"### {severity.upper()}\n\n"
                for result in results:
                    md += f"- **{result['component']}**: {result['message']}\n"
                    if result['suggestion']:
                        md += f"  - 建议: {result['suggestion']}\n"
                md += "\n"
                
        md += f"\n---\n生成时间: {report_data['timestamp']}\n"
        return md
        
    def run_checks(self, components: List[ComponentData], page_data: Optional[Dict[str, Any]] = None) -> None:
        """运行所有检查"""
        self.results = []
        
        # 检查组件
        for component in components:
            component_results = self.check_component(component)
            self.results.extend(component_results)
            
        # 检查页面
        if page_data:
            page_results = self.check_page(page_data)
            self.results.extend(page_results)
            
        logger.info(f"检查完成，共进行了 {len(self.results)} 项检查")


def main():
    """示例用法"""
    try:
        # 初始化检查器
        checker = UIChecker(
            schema_file="ui_validation_schema.yaml",
            rules_file="component_check_rules.yaml"
        )
        
        # 示例组件数据
        stock_card = ComponentData(
            name="stock_card_example",
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
                "color": "#52c41a",  # 应该是绿色表示上涨
                "background_color": "#ffffff"
            }
        )
        
        # 示例页面数据
        page_data = {
            "type": "dashboard_page",
            "components": ["portfolio_summary", "market_overview", "quick_actions"],
            "breakpoints": ["576px", "768px", "992px"]
        }
        
        # 运行检查
        checker.run_checks([stock_card], page_data)
        
        # 生成报告
        print("=== JSON报告 ===")
        json_report = checker.generate_report("json")
        print(json_report)
        
        print("\n=== Markdown报告 ===")
        md_report = checker.generate_report("markdown")
        print(md_report)
        
    except Exception as e:
        logger.error(f"检查过程中出现错误: {e}")


if __name__ == "__main__":
    main()