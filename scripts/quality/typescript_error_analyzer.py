#!/usr/bin/env python3
"""
TypeScript错误分析器
通用的TypeScript编译错误检测、分类和修复建议系统
"""

import re
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class TypeScriptError:
    """TypeScript错误数据结构"""
    file_path: str
    line: int
    column: int
    error_code: str
    message: str
    severity: str
    source_line: str = ""
    fix_suggestions: List[str] = None
    auto_fixable: bool = False
    
    def __post_init__(self):
        if self.fix_suggestions is None:
            self.fix_suggestions = []

@dataclass
class ErrorAnalysisResult:
    """错误分析结果"""
    total_errors: int
    total_warnings: int
    errors_by_type: Dict[str, int]
    errors_by_file: Dict[str, int]
    all_errors: List[TypeScriptError]
    fix_priority: List[str]
    estimated_fix_time: str
    success_rate: float

class TypeScriptErrorAnalyzer:
    """TypeScript错误分析器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.frontend_dir = self.project_root / "frontend"
        
        # 错误代码到修复建议的映射
        self.error_fixes = {
            # 语法错误
            "TS1109": {
                "description": "表达式预期错误 - 通常是数字后直接跟标识符",
                "fixes": [
                    "检查数字和标识符之间是否缺少运算符",
                    "使用模板字符串: `~${value}%` 而不是 ~value%",
                    "使用字符串拼接: value + '%' 而不是 value%",
                    "检查是否缺少分号或逗号"
                ],
                "auto_fixable": False,
                "priority": "high"
            },
            "TS1005": {
                "description": "';' 预期错误",
                "fixes": [
                    "在语句末尾添加分号",
                    "检查前一行是否缺少分号",
                    "验证括号是否正确匹配"
                ],
                "auto_fixable": True,
                "priority": "high"
            },
            "TS1003": {
                "description": "标识符预期错误",
                "fixes": [
                    "检查变量名是否有效",
                    "确保没有使用保留关键字作为变量名",
                    "检查对象属性声明语法"
                ],
                "auto_fixable": False,
                "priority": "high"
            },
            
            # 类型错误
            "TS2322": {
                "description": "类型不匹配错误",
                "fixes": [
                    "检查赋值的类型是否匹配声明的类型",
                    "使用类型断言: value as Type",
                    "更新类型定义以匹配实际值",
                    "使用联合类型: Type1 | Type2"
                ],
                "auto_fixable": False,
                "priority": "medium"
            },
            "TS2304": {
                "description": "找不到名称错误",
                "fixes": [
                    "检查变量名拼写是否正确",
                    "添加相应的 import 语句",
                    "检查变量是否在正确的作用域中定义",
                    "确认第三方库是否正确安装"
                ],
                "auto_fixable": False,
                "priority": "medium"
            },
            "TS2339": {
                "description": "属性不存在错误",
                "fixes": [
                    "检查属性名拼写是否正确",
                    "更新接口或类型定义",
                    "使用可选链操作符: obj?.property",
                    "添加属性到类型定义中"
                ],
                "auto_fixable": False,
                "priority": "medium"
            },
            "TS2307": {
                "description": "找不到模块错误",
                "fixes": [
                    "检查模块路径是否正确",
                    "安装缺失的依赖包: npm install package-name",
                    "检查文件扩展名是否正确",
                    "更新tsconfig.json的路径映射"
                ],
                "auto_fixable": False,
                "priority": "high"
            },
            "TS2345": {
                "description": "参数类型错误",
                "fixes": [
                    "检查函数调用的参数类型",
                    "更新函数参数类型定义",
                    "使用类型转换或断言",
                    "检查参数数量是否正确"
                ],
                "auto_fixable": False,
                "priority": "medium"
            },
            "TS2554": {
                "description": "参数数量错误",
                "fixes": [
                    "检查函数调用的参数数量",
                    "添加缺失的参数",
                    "使用可选参数: param?",
                    "检查函数重载定义"
                ],
                "auto_fixable": False,
                "priority": "medium"
            },
            "TS2571": {
                "description": "对象字面量错误",
                "fixes": [
                    "检查对象属性初始化语法",
                    "确保所有必需属性都已定义",
                    "使用正确的对象字面量语法",
                    "检查属性名是否有效"
                ],
                "auto_fixable": False,
                "priority": "medium"
            },
            
            # React/JSX 特定错误
            "TS2746": {
                "description": "JSX children 类型错误",
                "fixes": [
                    "将多个children包装在Fragment中: <></>",
                    "使用数组语法包装children",
                    "检查组件的children类型定义",
                    "确保JSX元素正确嵌套"
                ],
                "auto_fixable": True,
                "priority": "medium"
            },
            "TS2769": {
                "description": "JSX属性错误",
                "fixes": [
                    "检查组件属性类型定义",
                    "移除不存在的属性",
                    "更新组件Props接口",
                    "使用正确的属性名和类型"
                ],
                "auto_fixable": False,
                "priority": "medium"
            }
        }
        
    def run_typescript_check(self) -> str:
        """运行TypeScript类型检查"""
        if not self.frontend_dir.exists():
            raise FileNotFoundError(f"前端目录不存在: {self.frontend_dir}")
            
        try:
            result = subprocess.run(
                ["npx", "tsc", "--noEmit"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            output = result.stdout + result.stderr
            print(f"调试信息: TypeScript输出长度 {len(output)} 字符")
            if output.strip():
                print(f"调试信息: 前200字符: {output[:200]}")
            return output
        except subprocess.TimeoutExpired:
            raise TimeoutError("TypeScript检查超时")
        except FileNotFoundError:
            raise FileNotFoundError("TypeScript编译器不可用")
            
    def parse_typescript_errors(self, output: str) -> List[TypeScriptError]:
        """解析TypeScript错误输出"""
        errors = []
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 匹配标准TypeScript错误格式
            # 例如: src/pages/Component.tsx(381,36): error TS1109: Expression expected.
            error_pattern = r'(.+\.tsx?)\((\d+),(\d+)\):\s*(error|warning)\s*TS(\d+):\s*(.+)'
            match = re.match(error_pattern, line)
            
            if match:
                file_path, line_num, col_num, severity, error_code, message = match.groups()
                
                # 尝试获取源代码行
                source_line = ""
                if i + 2 < len(lines) and lines[i + 2].strip():
                    source_line = lines[i + 2].strip()
                
                # 生成修复建议
                fix_suggestions = self._generate_fix_suggestions(f"TS{error_code}", message, file_path)
                
                errors.append(TypeScriptError(
                    file_path=file_path,
                    line=int(line_num),
                    column=int(col_num),
                    error_code=f"TS{error_code}",
                    message=message,
                    severity=severity,
                    source_line=source_line,
                    fix_suggestions=fix_suggestions,
                    auto_fixable=self._is_auto_fixable(f"TS{error_code}")
                ))
                
        return errors
        
    def _generate_fix_suggestions(self, error_code: str, message: str, file_path: str) -> List[str]:
        """基于错误代码和消息生成修复建议"""
        suggestions = []
        
        # 获取预定义的修复建议
        if error_code in self.error_fixes:
            error_info = self.error_fixes[error_code]
            suggestions.extend(error_info["fixes"])
        
        # 基于错误消息生成特定建议
        message_lower = message.lower()
        
        if "cannot find module" in message_lower:
            if "'" in message:
                module_name = message.split("'")[1]
                suggestions.append(f"运行: npm install {module_name}")
            
        elif "property" in message_lower and "does not exist" in message_lower:
            if "'" in message:
                prop_name = message.split("'")[1]
                suggestions.append(f"检查属性 '{prop_name}' 是否拼写正确")
                suggestions.append(f"在接口中添加属性: {prop_name}?: any")
                
        elif "type" in message_lower and "is not assignable to type" in message_lower:
            suggestions.append("使用类型断言: (value as TargetType)")
            suggestions.append("更新变量的类型声明")
            
        elif "expression expected" in message_lower:
            suggestions.append("检查语法错误，可能缺少运算符或分号")
            
        # 如果没有特定建议，提供通用建议
        if not suggestions:
            suggestions = [
                "查看TypeScript官方文档了解该错误",
                "使用IDE的TypeScript错误检查功能",
                "检查相关代码的语法和类型"
            ]
            
        return suggestions
        
    def _is_auto_fixable(self, error_code: str) -> bool:
        """判断错误是否可以自动修复"""
        if error_code in self.error_fixes:
            return self.error_fixes[error_code].get("auto_fixable", False)
        return False
        
    def analyze_errors(self, errors: List[TypeScriptError]) -> ErrorAnalysisResult:
        """分析错误并生成报告"""
        total_errors = len([e for e in errors if e.severity == "error"])
        total_warnings = len([e for e in errors if e.severity == "warning"])
        
        # 按错误类型分组
        errors_by_type = {}
        for error in errors:
            errors_by_type[error.error_code] = errors_by_type.get(error.error_code, 0) + 1
            
        # 按文件分组
        errors_by_file = {}
        for error in errors:
            errors_by_file[error.file_path] = errors_by_file.get(error.file_path, 0) + 1
            
        # 生成修复优先级
        fix_priority = self._generate_fix_priority(errors)
        
        # 估算修复时间
        estimated_fix_time = self._estimate_fix_time(errors)
        
        # 计算成功率（基于可自动修复的错误比例）
        auto_fixable_count = len([e for e in errors if e.auto_fixable])
        success_rate = auto_fixable_count / max(len(errors), 1) * 100
        
        return ErrorAnalysisResult(
            total_errors=total_errors,
            total_warnings=total_warnings,
            errors_by_type=errors_by_type,
            errors_by_file=errors_by_file,
            all_errors=errors,
            fix_priority=fix_priority,
            estimated_fix_time=estimated_fix_time,
            success_rate=success_rate
        )
        
    def _generate_fix_priority(self, errors: List[TypeScriptError]) -> List[str]:
        """生成修复优先级建议"""
        priority_order = []
        
        # 高优先级：语法错误和模块错误
        high_priority = [e for e in errors if e.error_code in ["TS1109", "TS1005", "TS1003", "TS2307"]]
        if high_priority:
            priority_order.append("1. 优先修复语法错误和模块导入问题")
            
        # 中优先级：类型错误
        type_errors = [e for e in errors if e.error_code in ["TS2322", "TS2304", "TS2339", "TS2345", "TS2554"]]
        if type_errors:
            priority_order.append("2. 修复类型不匹配和属性错误")
            
        # 低优先级：JSX和React特定错误
        jsx_errors = [e for e in errors if e.error_code in ["TS2746", "TS2769"]]
        if jsx_errors:
            priority_order.append("3. 修复JSX和React组件错误")
            
        return priority_order
        
    def _estimate_fix_time(self, errors: List[TypeScriptError]) -> str:
        """估算修复时间"""
        total_errors = len(errors)
        auto_fixable = len([e for e in errors if e.auto_fixable])
        
        if total_errors == 0:
            return "0分钟"
        elif total_errors <= 5:
            return "10-30分钟"
        elif total_errors <= 15:
            return "30-60分钟"
        elif total_errors <= 30:
            return "1-2小时"
        else:
            return "2小时以上"
            
    def generate_report(self, analysis: ErrorAnalysisResult, output_format: str = "json") -> str:
        """生成分析报告"""
        if output_format == "json":
            return json.dumps(asdict(analysis), indent=2, ensure_ascii=False)
        elif output_format == "text":
            return self._generate_text_report(analysis)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
            
    def _generate_text_report(self, analysis: ErrorAnalysisResult) -> str:
        """生成文本格式报告"""
        report = []
        report.append("🔍 TypeScript错误分析报告")
        report.append("=" * 50)
        report.append(f"总错误数: {analysis.total_errors}")
        report.append(f"总警告数: {analysis.total_warnings}")
        report.append(f"预估修复时间: {analysis.estimated_fix_time}")
        report.append(f"自动修复成功率: {analysis.success_rate:.1f}%")
        
        if analysis.errors_by_type:
            report.append("\n📊 错误类型分布:")
            for error_type, count in sorted(analysis.errors_by_type.items(), key=lambda x: x[1], reverse=True):
                description = self.error_fixes.get(error_type, {}).get("description", "未知错误")
                report.append(f"  {error_type}: {count}个 - {description}")
                
        if analysis.errors_by_file:
            report.append("\n📁 错误文件分布:")
            for file_path, count in sorted(analysis.errors_by_file.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {file_path}: {count}个错误")
                
        if analysis.fix_priority:
            report.append("\n🔧 修复优先级建议:")
            for priority in analysis.fix_priority:
                report.append(f"  {priority}")
                
        if analysis.all_errors:
            report.append("\n📋 详细错误列表:")
            for error in analysis.all_errors[:10]:  # 只显示前10个错误
                report.append(f"\n  📁 {error.file_path}:{error.line}:{error.column}")
                report.append(f"  ❌ {error.error_code}: {error.message}")
                if error.fix_suggestions:
                    report.append("  💡 修复建议:")
                    for suggestion in error.fix_suggestions[:3]:  # 只显示前3个建议
                        report.append(f"    - {suggestion}")
                        
        return "\n".join(report)

def main():
    """主函数"""
    print("🔍 启动TypeScript错误分析器...")
    
    analyzer = TypeScriptErrorAnalyzer()
    
    try:
        # 运行TypeScript检查
        print("🔧 执行TypeScript类型检查...")
        output = analyzer.run_typescript_check()
        
        # 解析错误
        print("📊 解析错误信息...")
        errors = analyzer.parse_typescript_errors(output)
        
        # 分析错误
        print("🎯 分析错误并生成建议...")
        analysis = analyzer.analyze_errors(errors)
        
        # 生成报告
        text_report = analyzer.generate_report(analysis, "text")
        json_report = analyzer.generate_report(analysis, "json")
        
        # 输出结果
        print("\n" + text_report)
        
        # 保存详细报告
        report_file = analyzer.project_root / "logs" / "typescript_error_analysis.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(json_report)
            
        print(f"\n📝 详细报告已保存至: {report_file}")
        
        # 返回状态码
        return 0 if analysis.total_errors == 0 else 1
        
    except Exception as e:
        print(f"❌ 错误分析失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)