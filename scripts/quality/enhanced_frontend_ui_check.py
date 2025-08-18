#!/usr/bin/env python3
"""
Enhanced Frontend UI Check Framework
增强的前端UI检查框架，包含TypeScript/JavaScript编译错误检测
"""

import os
import subprocess
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class CompilationError:
    """编译错误数据类"""
    file_path: str
    line: int
    column: int
    error_code: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    source_line: str = ""

@dataclass
class UICheckResult:
    """UI检查结果"""
    check_type: str
    status: str  # 'pass', 'warning', 'error'
    message: str
    details: Dict[str, Any]
    suggestions: List[str]

class EnhancedFrontendUIChecker:
    """增强的前端UI检查器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.frontend_dir = self.project_root / "frontend"
        self.results: List[UICheckResult] = []
        
    def check_compilation(self) -> UICheckResult:
        """检查TypeScript/JavaScript编译错误"""
        print("🔍 检查TypeScript/JavaScript编译...")
        
        errors = []
        
        # 首先检查前端目录是否存在
        if not self.frontend_dir.exists():
            return UICheckResult(
                check_type="compilation",
                status="error",
                message="前端目录不存在",
                details={"frontend_dir": str(self.frontend_dir)},
                suggestions=["创建frontend目录并初始化React项目"]
            )
            
        # 检查package.json
        package_json = self.frontend_dir / "package.json"
        if not package_json.exists():
            return UICheckResult(
                check_type="compilation",
                status="error", 
                message="package.json不存在",
                details={},
                suggestions=["运行 npm init 或 create-react-app 初始化项目"]
            )
            
        # 检查node_modules
        node_modules = self.frontend_dir / "node_modules"
        if not node_modules.exists():
            print("⚠️ node_modules不存在，尝试安装依赖...")
            try:
                install_result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.frontend_dir,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if install_result.returncode != 0:
                    return UICheckResult(
                        check_type="compilation",
                        status="error",
                        message="依赖安装失败",
                        details={"error": install_result.stderr},
                        suggestions=["检查网络连接，手动运行 npm install"]
                    )
            except subprocess.TimeoutExpired:
                return UICheckResult(
                    check_type="compilation",
                    status="error",
                    message="依赖安装超时",
                    details={},
                    suggestions=["检查网络连接，使用 npm install --timeout=60000"]
                )
                
        # 进行TypeScript类型检查
        print("🔧 执行TypeScript编译检查...")
        try:
            # 首先尝试TypeScript编译检查
            tsc_result = subprocess.run(
                ["npx", "tsc", "--noEmit", "--pretty"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if tsc_result.returncode == 0:
                print("✅ TypeScript编译检查通过")
                return UICheckResult(
                    check_type="compilation",
                    status="pass",
                    message="TypeScript编译检查通过",
                    details={},
                    suggestions=[]
                )
            else:
                # 解析TypeScript错误
                compilation_errors = self._parse_typescript_errors(tsc_result.stdout + tsc_result.stderr)
                
                error_count = len([e for e in compilation_errors if e.severity == "error"])
                warning_count = len([e for e in compilation_errors if e.severity == "warning"])
                
                status = "error" if error_count > 0 else "warning"
                message = f"发现 {error_count} 个编译错误, {warning_count} 个警告"
                
                suggestions = self._generate_fix_suggestions(compilation_errors)
                
                return UICheckResult(
                    check_type="compilation",
                    status=status,
                    message=message,
                    details={
                        "errors": [asdict(e) for e in compilation_errors],
                        "error_count": error_count,
                        "warning_count": warning_count
                    },
                    suggestions=suggestions
                )
                
        except FileNotFoundError:
            # 如果没有TypeScript，尝试React构建检查
            print("⚠️ TypeScript不可用，使用React构建检查...")
            return self._check_react_build()
        except subprocess.TimeoutExpired:
            return UICheckResult(
                check_type="compilation",
                status="error",
                message="TypeScript检查超时",
                details={},
                suggestions=["检查项目大小，考虑增加内存或使用增量编译"]
            )
            
    def _parse_typescript_errors(self, error_output: str) -> List[CompilationError]:
        """解析TypeScript错误输出"""
        errors = []
        lines = error_output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 匹配TypeScript错误格式: file(line,col): error TS####: message
            # 示例: src/pages/SectorAnalysisEnhanced.tsx(381,36): error TS1109: Expression expected.
            ts_error_pattern = r'(.+\.tsx?)\((\d+),(\d+)\):\s*(error|warning)\s*TS(\d+):\s*(.+)'
            match = re.match(ts_error_pattern, line)
            
            if match:
                file_path, line_num, col_num, severity, error_code, message = match.groups()
                
                errors.append(CompilationError(
                    file_path=file_path,
                    line=int(line_num),
                    column=int(col_num),
                    error_code=f"TS{error_code}",
                    message=message,
                    severity=severity,
                    source_line=""
                ))
            else:
                # 处理其他格式的编译错误
                if any(keyword in line.lower() for keyword in ["error", "failed", "cannot"]):
                    errors.append(CompilationError(
                        file_path="",
                        line=0,
                        column=0,
                        error_code="COMPILE_ERROR",
                        message=line,
                        severity="error",
                        source_line=""
                    ))
                    
        return errors
        
    def _generate_fix_suggestions(self, errors: List[CompilationError]) -> List[str]:
        """基于编译错误生成修复建议"""
        suggestions = []
        error_types = {}
        
        # 统计错误类型
        for error in errors:
            error_types[error.error_code] = error_types.get(error.error_code, 0) + 1
            
        # 基于常见错误类型生成建议
        for error_code, count in error_types.items():
            if error_code == "TS1109":
                suggestions.append(f"修复 {count} 个 TS1109 错误: 检查数字后是否缺少运算符或分号。例如：'~50%' 应改为 '~50 + \"%\"' 或 '`~${50}%`'")
            elif error_code == "TS2304":
                suggestions.append(f"修复 {count} 个 TS2304 错误: 检查变量名拼写或添加相应的 import 语句")
            elif error_code == "TS2322":
                suggestions.append(f"修复 {count} 个 TS2322 错误: 检查类型匹配，使用类型断言 'as Type' 或更新类型定义")
            elif error_code == "TS2339":
                suggestions.append(f"修复 {count} 个 TS2339 错误: 检查属性名是否正确或对象类型定义")
            elif error_code == "TS2307":
                suggestions.append(f"修复 {count} 个 TS2307 错误: 检查模块路径或安装缺失的依赖包")
            else:
                suggestions.append(f"修复 {count} 个 {error_code} 错误: 查看TypeScript官方文档了解详细解决方案")
                
        # 添加通用建议
        if errors:
            suggestions.extend([
                "运行 'npm run build' 查看完整的编译错误信息",
                "使用 VSCode 的 TypeScript 错误检查功能获得实时反馈",
                "考虑暂时使用 // @ts-ignore 注释跳过类型检查，但要尽快修复"
            ])
            
        return suggestions
        
    def _check_react_build(self) -> UICheckResult:
        """检查React应用构建"""
        print("🔧 执行React应用构建检查...")
        try:
            build_result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if build_result.returncode == 0:
                return UICheckResult(
                    check_type="compilation",
                    status="pass",
                    message="React应用构建成功",
                    details={},
                    suggestions=[]
                )
            else:
                return UICheckResult(
                    check_type="compilation",
                    status="error",
                    message="React应用构建失败",
                    details={"error": build_result.stderr},
                    suggestions=[
                        "检查控制台错误信息",
                        "修复JavaScript/TypeScript语法错误",
                        "确保所有依赖正确安装"
                    ]
                )
        except subprocess.TimeoutExpired:
            return UICheckResult(
                check_type="compilation",
                status="error",
                message="React构建超时",
                details={},
                suggestions=["优化构建配置或增加构建超时时间"]
            )
        warnings = []
        original_dir = os.getcwd()
        
        try:
            # 检查是否存在前端目录
            if not self.frontend_dir.exists():
                return UICheckResult(
                    check_type="compilation",
                    status="error",
                    message="前端目录不存在",
                    details={"frontend_dir": str(self.frontend_dir), "project_root": str(self.project_root)},
                    suggestions=["确保前端项目已正确初始化"]
                )
            
            # 检查package.json是否存在
            package_json_path = self.frontend_dir / "package.json"
            if not package_json_path.exists():
                return UICheckResult(
                    check_type="compilation",
                    status="error", 
                    message="package.json不存在",
                    details={"expected_path": str(package_json_path)},
                    suggestions=["运行 npm init 初始化项目"]
                )
            
            # 切换到前端目录
            os.chdir(self.frontend_dir)
            print(f"   切换到前端目录: {self.frontend_dir}")
            
            # 运行TypeScript编译检查
            print("   运行 tsc --noEmit 检查TypeScript语法...")
            try:
                result = subprocess.run(
                    ["npx", "tsc", "--noEmit", "--skipLibCheck"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    tsc_errors = self._parse_tsc_output(result.stderr)
                    errors.extend(tsc_errors)
                    print(f"   发现 {len(tsc_errors)} 个TypeScript错误")
                else:
                    print("   ✅ TypeScript编译检查通过")
                    
            except subprocess.TimeoutExpired:
                errors.append(CompilationError(
                    file_path="",
                    line=0,
                    column=0,
                    error_code="TS_TIMEOUT",
                    message="TypeScript编译检查超时",
                    severity="error"
                ))
            except FileNotFoundError:
                print("   ⚠️ TypeScript编译器未找到，跳过TypeScript检查")
            
            # 运行ESLint检查
            print("   运行 ESLint 检查JavaScript/TypeScript语法...")
            try:
                result = subprocess.run(
                    ["npx", "eslint", "src/", "--format", "json", "--ext", ".js,.jsx,.ts,.tsx"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.stdout:
                    eslint_results = json.loads(result.stdout)
                    eslint_errors = self._parse_eslint_output(eslint_results)
                    errors.extend(eslint_errors)
                    print(f"   发现 {len(eslint_errors)} 个ESLint问题")
                else:
                    print("   ✅ ESLint检查通过")
                    
            except subprocess.TimeoutExpired:
                warnings.append(CompilationError(
                    file_path="",
                    line=0,
                    column=0,
                    error_code="ESLINT_TIMEOUT",
                    message="ESLint检查超时",
                    severity="warning"
                ))
            except (FileNotFoundError, json.JSONDecodeError):
                print("   ⚠️ ESLint未配置或输出格式错误，跳过ESLint检查")
            
            # 运行npm run build来检查编译问题
            print("   运行 npm run build 检查构建...")
            try:
                result = subprocess.run(
                    ["npm", "run", "build"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env={**os.environ, "CI": "true"}  # 避免交互式输出
                )
                
                if result.returncode != 0:
                    build_errors = self._parse_build_output(result.stdout + result.stderr)
                    errors.extend(build_errors)
                    print(f"   发现 {len(build_errors)} 个构建错误")
                else:
                    print("   ✅ 构建检查通过")
                    
            except subprocess.TimeoutExpired:
                errors.append(CompilationError(
                    file_path="",
                    line=0,
                    column=0,
                    error_code="BUILD_TIMEOUT",
                    message="构建过程超时",
                    severity="error"
                ))
                
        except Exception as e:
            errors.append(CompilationError(
                file_path="",
                line=0,
                column=0,
                error_code="UNKNOWN_ERROR",
                message=f"编译检查过程中发生错误: {str(e)}",
                severity="error"
            ))
        finally:
            # 切换回原始目录
            os.chdir(original_dir)
        
        # 汇总结果
        total_errors = len([e for e in errors if e.severity == "error"])
        total_warnings = len([e for e in errors if e.severity == "warning"])
        
        if total_errors > 0:
            status = "error"
            message = f"发现 {total_errors} 个编译错误，{total_warnings} 个警告"
        elif total_warnings > 0:
            status = "warning"
            message = f"发现 {total_warnings} 个编译警告"
        else:
            status = "pass"
            message = "编译检查通过"
        
        suggestions = []
        if total_errors > 0:
            suggestions.extend([
                "修复所有编译错误后重新检查",
                "检查TypeScript配置文件（tsconfig.json）",
                "确保所有依赖项已正确安装"
            ])
        
        return UICheckResult(
            check_type="compilation",
            status=status,
            message=message,
            details={
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "errors": [asdict(e) for e in errors],
                "frontend_dir": str(self.frontend_dir)
            },
            suggestions=suggestions
        )
    
    def _parse_tsc_output(self, output: str) -> List[CompilationError]:
        """解析TypeScript编译器输出"""
        errors = []
        
        # TypeScript错误格式: src/file.tsx(line,col): error TS2304: message
        pattern = r'(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+):\s+(.+)'
        
        for line in output.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                file_path, line_num, col_num, severity, error_code, message = match.groups()
                errors.append(CompilationError(
                    file_path=file_path,
                    line=int(line_num),
                    column=int(col_num),
                    error_code=error_code,
                    message=message,
                    severity=severity
                ))
        
        return errors
    
    def _parse_eslint_output(self, eslint_results: List[Dict]) -> List[CompilationError]:
        """解析ESLint输出"""
        errors = []
        
        for file_result in eslint_results:
            file_path = file_result.get('filePath', '')
            for message in file_result.get('messages', []):
                errors.append(CompilationError(
                    file_path=file_path,
                    line=message.get('line', 0),
                    column=message.get('column', 0),
                    error_code=message.get('ruleId', 'ESLINT'),
                    message=message.get('message', ''),
                    severity='error' if message.get('severity') == 2 else 'warning'
                ))
        
        return errors
    
    def _parse_build_output(self, output: str) -> List[CompilationError]:
        """解析构建输出错误"""
        errors = []
        
        lines = output.split('\n')
        current_error = None
        
        for line in lines:
            line = line.strip()
            
            # 检测编译错误开始
            if 'ERROR in' in line or 'Failed to compile' in line:
                # 解析文件路径和位置信息
                # 格式可能是: ERROR in ./src/file.tsx:line:col
                match = re.search(r'ERROR in (.+?):(\d+):(\d+)', line)
                if match:
                    file_path, line_num, col_num = match.groups()
                    current_error = CompilationError(
                        file_path=file_path,
                        line=int(line_num),
                        column=int(col_num),
                        error_code="BUILD_ERROR",
                        message="",
                        severity="error"
                    )
                else:
                    # 简单格式
                    current_error = CompilationError(
                        file_path="",
                        line=0,
                        column=0,
                        error_code="BUILD_ERROR",
                        message=line,
                        severity="error"
                    )
            
            # 收集错误消息
            elif current_error and line and not line.startswith('at '):
                if current_error.message:
                    current_error.message += " " + line
                else:
                    current_error.message = line
            
            # 错误结束
            elif current_error and (line == '' or 'webpack compiled' in line):
                if current_error.message:
                    errors.append(current_error)
                current_error = None
        
        # 添加最后一个错误
        if current_error and current_error.message:
            errors.append(current_error)
        
        return errors
    
    def check_syntax_specific_issues(self) -> UICheckResult:
        """检查特定的语法问题"""
        print("🔍 检查特定语法问题...")
        
        issues = []
        suggestions = []
        
        try:
            # 搜索常见的JavaScript/TypeScript语法问题
            source_files = list(self.frontend_dir.glob("src/**/*.{js,jsx,ts,tsx}"))
            
            for file_path in source_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # 检查常见语法问题
                    file_issues = self._check_file_syntax_issues(file_path, content)
                    issues.extend(file_issues)
                    
                except Exception as e:
                    issues.append({
                        "file": str(file_path),
                        "issue": f"无法读取文件: {e}",
                        "severity": "warning"
                    })
            
            if issues:
                error_count = len([i for i in issues if i.get('severity') == 'error'])
                warning_count = len([i for i in issues if i.get('severity') == 'warning'])
                
                if error_count > 0:
                    status = "error"
                    message = f"发现 {error_count} 个语法错误，{warning_count} 个语法警告"
                else:
                    status = "warning"
                    message = f"发现 {warning_count} 个语法警告"
                    
                suggestions.extend([
                    "检查并修复所有标识的语法问题",
                    "使用代码编辑器的语法高亮功能",
                    "配置自动代码格式化工具"
                ])
            else:
                status = "pass"
                message = "未发现特定语法问题"
            
        except Exception as e:
            status = "error"
            message = f"语法检查过程中发生错误: {e}"
            issues = [{"error": str(e)}]
        
        return UICheckResult(
            check_type="syntax_specific",
            status=status,
            message=message,
            details={
                "issues": issues,
                "files_checked": len(source_files) if 'source_files' in locals() else 0
            },
            suggestions=suggestions
        )
    
    def _check_file_syntax_issues(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """检查单个文件的语法问题"""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # 检查数字后直接跟标识符（如原错误）
            if re.search(r'\d+[a-zA-Z_]', line_stripped):
                issues.append({
                    "file": str(file_path),
                    "line": line_num,
                    "issue": "数字后直接跟标识符，可能的语法错误",
                    "content": line.strip(),
                    "severity": "error"
                })
            
            # 检查未闭合的标签
            if line_stripped.count('<') != line_stripped.count('>'):
                # 简单检查，实际需要更复杂的解析
                bracket_balance = line_stripped.count('<') - line_stripped.count('>')
                if abs(bracket_balance) > 2:  # 避免误报
                    issues.append({
                        "file": str(file_path),
                        "line": line_num,
                        "issue": "可能存在未闭合的HTML/JSX标签",
                        "content": line.strip(),
                        "severity": "warning"
                    })
            
            # 检查未转义的HTML字符
            if '<' in line_stripped and any(char in line_stripped for char in ['<50', '<30', '<100']):
                if '&lt;' not in line_stripped:
                    issues.append({
                        "file": str(file_path),
                        "line": line_num,
                        "issue": "HTML中应使用 &lt; 而不是 < 符号",
                        "content": line.strip(),
                        "severity": "warning"
                    })
            
            # 检查分号缺失（简单检查）
            if (line_stripped.endswith(')') or line_stripped.endswith('}')) and \
               not line_stripped.endswith('};') and \
               not line_stripped.endswith('),') and \
               'return' not in line_stripped and \
               'if' not in line_stripped and \
               'for' not in line_stripped:
                # 这是一个非常简化的检查
                pass  # 实际上需要更复杂的AST分析
            
        return issues
    
    def check_build_process(self) -> UICheckResult:
        """检查构建过程"""
        print("🔍 检查构建过程...")
        
        original_dir = os.getcwd()
        
        try:
            # 检查前端目录是否存在
            if not self.frontend_dir.exists():
                return UICheckResult(
                    check_type="build_process",
                    status="error",
                    message="前端目录不存在",
                    details={"frontend_dir": str(self.frontend_dir)},
                    suggestions=["确保前端项目已正确初始化"]
                )
            
            os.chdir(self.frontend_dir)
            print(f"   切换到前端目录进行构建: {self.frontend_dir}")
            
            # 检查依赖安装
            if not (self.frontend_dir / "node_modules").exists():
                return UICheckResult(
                    check_type="build_process",
                    status="error",
                    message="依赖未安装",
                    details={"node_modules": "不存在"},
                    suggestions=["运行 npm install 安装依赖"]
                )
            
            # 执行构建
            print("   执行构建测试...")
            start_time = time.time()
            
            result = subprocess.run(
                ["npm", "run", "build"],
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ, "CI": "true"}
            )
            
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                status = "pass"
                message = f"构建成功，耗时 {build_time:.1f}s"
                suggestions = []
                
                if build_time > 60:
                    status = "warning"
                    message += " (构建时间较长)"
                    suggestions.append("考虑优化构建配置以提高构建速度")
            else:
                status = "error"
                message = f"构建失败，耗时 {build_time:.1f}s"
                suggestions = [
                    "检查构建错误日志",
                    "确保所有依赖已正确安装",
                    "检查TypeScript和JavaScript语法"
                ]
            
            return UICheckResult(
                check_type="build_process",
                status=status,
                message=message,
                details={
                    "build_time": f"{build_time:.1f}s",
                    "return_code": result.returncode,
                    "stdout": result.stdout[-1000:] if result.stdout else "",  # 最后1000字符
                    "stderr": result.stderr[-1000:] if result.stderr else ""
                },
                suggestions=suggestions
            )
            
        except subprocess.TimeoutExpired:
            return UICheckResult(
                check_type="build_process",
                status="error",
                message="构建超时",
                details={"timeout": "180s"},
                suggestions=["检查构建配置，可能存在死循环或资源不足"]
            )
        except Exception as e:
            return UICheckResult(
                check_type="build_process",
                status="error",
                message=f"构建检查失败: {e}",
                details={"error": str(e)},
                suggestions=["检查项目配置和环境"]
            )
        finally:
            os.chdir(original_dir)
    
    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有检查"""
        print("🚀 启动增强前端UI检查...")
        
        start_time = datetime.now()
        
        # 执行各项检查
        checks = [
            self.check_compilation,
            self.check_syntax_specific_issues,
            self.check_build_process
        ]
        
        for check_func in checks:
            try:
                result = check_func()
                self.results.append(result)
            except Exception as e:
                error_result = UICheckResult(
                    check_type=check_func.__name__,
                    status="error",
                    message=f"检查过程中发生错误: {e}",
                    details={"error": str(e)},
                    suggestions=["检查系统环境和项目配置"]
                )
                self.results.append(error_result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 生成综合报告
        return self._generate_comprehensive_report(duration)
    
    def _generate_comprehensive_report(self, duration: float) -> Dict[str, Any]:
        """生成综合报告"""
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == "pass"])
        warning_checks = len([r for r in self.results if r.status == "warning"])
        error_checks = len([r for r in self.results if r.status == "error"])
        
        overall_status = "PASS"
        if error_checks > 0:
            overall_status = "FAIL"
        elif warning_checks > 0:
            overall_status = "PASS_WITH_WARNINGS"
        
        report = {
            "summary": {
                "overall_status": overall_status,
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "warning_checks": warning_checks,
                "error_checks": error_checks,
                "success_rate": f"{(passed_checks/total_checks*100):.1f}%" if total_checks > 0 else "0%",
                "duration": f"{duration:.1f}s",
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于检查结果生成建议
        error_results = [r for r in self.results if r.status == "error"]
        warning_results = [r for r in self.results if r.status == "warning"]
        
        if error_results:
            recommendations.append("优先修复所有编译错误，确保代码能够正确构建")
            
            compilation_errors = [r for r in error_results if r.check_type == "compilation"]
            if compilation_errors:
                recommendations.append("检查TypeScript配置和语法，确保代码符合类型规范")
        
        if warning_results:
            recommendations.append("解决编译警告，提高代码质量")
            
        if not error_results and not warning_results:
            recommendations.append("代码质量良好，建议定期运行此检查确保持续质量")
        
        return recommendations

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强前端UI检查框架")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--output", default="logs/enhanced_frontend_ui_check.json", help="输出报告文件")
    
    args = parser.parse_args()
    
    checker = EnhancedFrontendUIChecker(args.project_root)
    report = checker.run_all_checks()
    
    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    summary = report["summary"]
    print(f"\n📊 增强前端UI检查摘要:")
    print(f"   整体状态: {summary['overall_status']}")
    print(f"   总检查项: {summary['total_checks']}")
    print(f"   通过: {summary['passed_checks']}")
    print(f"   警告: {summary['warning_checks']}")
    print(f"   错误: {summary['error_checks']}")
    print(f"   成功率: {summary['success_rate']}")
    print(f"   检查耗时: {summary['duration']}")
    
    # 显示详细结果
    print(f"\n🔍 详细检查结果:")
    for result in checker.results:
        status_color = {
            'pass': '\033[0;32m',  # 绿色
            'warning': '\033[1;33m',  # 黄色
            'error': '\033[0;31m'  # 红色
        }.get(result.status, '\033[0m')
        
        print(f"   {status_color}[{result.status.upper()}]\033[0m {result.check_type}: {result.message}")
        
        if result.suggestions:
            for suggestion in result.suggestions:
                print(f"     💡 {suggestion}")
    
    print(f"\n📝 详细报告已保存至: {output_path}")
    
    # 返回状态码
    return 0 if summary['overall_status'] in ['PASS', 'PASS_WITH_WARNINGS'] else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)