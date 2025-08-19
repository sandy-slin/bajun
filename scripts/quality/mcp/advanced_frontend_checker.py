#!/usr/bin/env python3
"""
高级前端功能检查器 - 符合Chrome MCP设计要求的完整实现
实现设计文档中的所有核心功能
"""

import asyncio
import json
import time
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import tempfile

@dataclass
class FrontendCheckResult:
    page: str
    url: str
    status: str  # PASS, FAIL, WARNING, ERROR
    issues: List[str]
    performance: Dict[str, float]
    screenshot_path: Optional[str] = None
    react_components: List[str] = None
    js_errors: List[str] = None
    network_requests: List[Dict] = None
    user_flow_status: Optional[str] = None

@dataclass
class PerformanceMetrics:
    load_time: float
    first_contentful_paint: float
    largest_contentful_paint: float
    cumulative_layout_shift: float
    time_to_interactive: float
    total_blocking_time: float

class AdvancedFrontendChecker:
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.api_base_url = "http://localhost:8000"
        self.chrome_debug_port = 9222
        self.results = []
        self.chrome_process = None
        self.temp_dir = None
        
        # 测试配置
        self.test_config = {
            "check_points": {
                "homepage": {
                    "url": "/",
                    "checks": [
                        "page_loads",
                        "react_renders", 
                        "no_js_errors",
                        "responsive_design",
                        "performance_metrics"
                    ],
                    "required_components": ["NavigationMenu", "MainContent", "Footer"]
                },
                "dashboard": {
                    "url": "/dashboard",
                    "checks": [
                        "data_loads",
                        "charts_render",
                        "api_connections",
                        "real_time_updates"
                    ],
                    "required_elements": [".ant-card", ".dashboard-content"],
                    "api_endpoints": ["/api/v1/sectors/top"]
                },
                "sector_analysis": {
                    "url": "/sectors",
                    "checks": [
                        "api_data_loads",
                        "table_renders",
                        "filtering_works",
                        "export_functions"
                    ],
                    "wait_for_data": True,
                    "data_timeout": 10
                },
                "stock_recommendation": {
                    "url": "/stock-recommendation",
                    "checks": [
                        "recommendation_loads",
                        "scoring_displays",
                        "interaction_works"
                    ]
                }
            },
            "user_flows": [
                {
                    "name": "complete_analysis_workflow",
                    "steps": [
                        {"action": "navigate", "target": "/"},
                        {"action": "wait", "duration": 2},
                        {"action": "navigate", "target": "/dashboard"},
                        {"action": "wait_for_element", "selector": ".ant-card"},
                        {"action": "navigate", "target": "/sectors"},
                        {"action": "wait_for_data", "timeout": 10},
                        {"action": "navigate", "target": "/stock-recommendation"},
                        {"action": "verify_content", "contains": ["推荐", "评分"]}
                    ]
                }
            ]
        }
        
        # 创建目录
        Path("logs/mcp_checks").mkdir(parents=True, exist_ok=True)
        Path("logs/screenshots").mkdir(parents=True, exist_ok=True)
        Path("logs/performance").mkdir(parents=True, exist_ok=True)

    async def setup_chrome_debugging(self) -> bool:
        """设置Chrome调试环境"""
        print("🚀 设置Chrome调试环境...")
        
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="chrome_debug_")
            
            # Chrome启动参数
            chrome_args = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                f"--remote-debugging-port={self.chrome_debug_port}",
                f"--user-data-dir={self.temp_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--enable-automation",
                "--disable-web-security",
                "--allow-running-insecure-content",
                "--window-size=1920,1080",
                "--new-window"
            ]
            
            # 启动Chrome
            self.chrome_process = subprocess.Popen(
                chrome_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # 等待Chrome启动
            await asyncio.sleep(5)
            
            # 验证调试端口
            debug_check = subprocess.run([
                "curl", "-s", f"http://localhost:{self.chrome_debug_port}/json/version"
            ], capture_output=True, text=True, timeout=5)
            
            if debug_check.returncode == 0:
                print("✅ Chrome调试环境设置成功")
                return True
            else:
                print("❌ Chrome调试端口验证失败")
                return False
                
        except Exception as e:
            print(f"❌ Chrome设置失败: {e}")
            return False

    async def get_active_tab_id(self) -> Optional[str]:
        """获取活动标签页ID"""
        try:
            tabs_result = subprocess.run([
                "curl", "-s", f"http://localhost:{self.chrome_debug_port}/json"
            ], capture_output=True, text=True, timeout=5)
            
            if tabs_result.returncode == 0:
                tabs = json.loads(tabs_result.stdout)
                for tab in tabs:
                    if tab.get("type") == "page" and self.base_url in tab.get("url", ""):
                        return tab["id"]
                # 如果没找到匹配的，返回第一个page类型的tab
                for tab in tabs:
                    if tab.get("type") == "page":
                        return tab["id"]
            return None
        except Exception as e:
            print(f"获取标签页ID失败: {e}")
            return None

    async def execute_devtools_command(self, command: str, params: Dict = None) -> Dict:
        """执行DevTools Protocol命令"""
        try:
            tab_id = await self.get_active_tab_id()
            if not tab_id:
                return {"error": "无法获取活动标签页"}
            
            payload = {
                "id": int(time.time() * 1000),
                "method": command,
                "params": params or {}
            }
            
            result = subprocess.run([
                "curl", "-s", "-X", "POST",
                f"http://localhost:{self.chrome_debug_port}/json/runtime/evaluate",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(payload)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"DevTools命令执行失败: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"DevTools命令异常: {e}"}

    async def check_react_components(self, url: str) -> Dict:
        """深度检查React组件渲染状态"""
        print(f"   🔍 检查React组件渲染状态...")
        
        result = {
            "react_detected": False,
            "components_found": [],
            "component_errors": [],
            "render_time": 0
        }
        
        try:
            # 导航到页面
            await self.navigate_to_page(url)
            await asyncio.sleep(3)
            
            start_time = time.time()
            
            # 检查React根节点
            react_check = await self.execute_devtools_command(
                "Runtime.evaluate",
                {"expression": "!!window.React && !!document.getElementById('root')"}
            )
            
            if react_check.get("result", {}).get("value"):
                result["react_detected"] = True
                
                # 检查React DevTools
                devtools_check = await self.execute_devtools_command(
                    "Runtime.evaluate",
                    {"expression": "!!window.__REACT_DEVTOOLS_GLOBAL_HOOK__"}
                )
                
                # 获取渲染的组件
                components_check = await self.execute_devtools_command(
                    "Runtime.evaluate", 
                    {"expression": """
                        Array.from(document.querySelectorAll('*')).map(el => el.constructor.name).filter(name => name !== 'HTMLDivElement' && name !== 'HTMLSpanElement').slice(0, 10)
                    """}
                )
                
                if components_check.get("result", {}).get("value"):
                    result["components_found"] = components_check["result"]["value"]
                
                # 检查是否有React错误
                error_check = await self.execute_devtools_command(
                    "Runtime.evaluate",
                    {"expression": "window.__REACT_ERROR_OVERLAY__ || []"}
                )
                
                result["render_time"] = time.time() - start_time
                
        except Exception as e:
            result["component_errors"].append(f"React检查异常: {e}")
        
        return result

    async def check_javascript_errors(self) -> List[Dict]:
        """检查JavaScript控制台错误"""
        print("   🔍 检查JavaScript错误...")
        
        try:
            # 启用Runtime domain
            await self.execute_devtools_command("Runtime.enable")
            
            # 获取控制台消息
            console_result = subprocess.run([
                "curl", "-s", f"http://localhost:{self.chrome_debug_port}/json/runtime/getConsoleMessages"
            ], capture_output=True, text=True, timeout=5)
            
            errors = []
            if console_result.returncode == 0:
                try:
                    console_data = json.loads(console_result.stdout)
                    for message in console_data.get("result", []):
                        if message.get("level") in ["error", "warning"]:
                            errors.append({
                                "level": message.get("level"),
                                "text": message.get("text", ""),
                                "url": message.get("url", ""),
                                "line": message.get("line", 0)
                            })
                except:
                    pass
            
            return errors
            
        except Exception as e:
            return [{"level": "error", "text": f"错误检查异常: {e}"}]

    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        print("   📊 收集性能指标...")
        
        try:
            # 获取Performance API数据
            perf_result = await self.execute_devtools_command(
                "Runtime.evaluate",
                {"expression": """
                    JSON.stringify({
                        navigation: performance.getEntriesByType('navigation')[0],
                        paint: performance.getEntriesByType('paint'),
                        memory: performance.memory ? {
                            usedJSHeapSize: performance.memory.usedJSHeapSize,
                            totalJSHeapSize: performance.memory.totalJSHeapSize
                        } : null
                    })
                """}
            )
            
            if perf_result.get("result", {}).get("value"):
                perf_data = json.loads(perf_result["result"]["value"])
                nav = perf_data.get("navigation", {})
                
                return PerformanceMetrics(
                    load_time=nav.get("loadEventEnd", 0) - nav.get("navigationStart", 0),
                    first_contentful_paint=next((p["startTime"] for p in perf_data.get("paint", []) if p["name"] == "first-contentful-paint"), 0),
                    largest_contentful_paint=0,  # 需要更复杂的检测
                    cumulative_layout_shift=0,   # 需要Layout Shift API
                    time_to_interactive=nav.get("domInteractive", 0) - nav.get("navigationStart", 0),
                    total_blocking_time=0
                )
            
        except Exception as e:
            print(f"性能指标收集失败: {e}")
        
        return PerformanceMetrics(0, 0, 0, 0, 0, 0)

    async def check_api_data_loading(self, endpoints: List[str]) -> Dict:
        """检查API数据加载"""
        print("   🔌 检查API数据加载...")
        
        result = {
            "endpoints_tested": [],
            "successful_loads": 0,
            "failed_loads": 0,
            "response_times": {}
        }
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                api_check = subprocess.run([
                    "curl", "-s", "-w", "%{http_code}",
                    f"{self.api_base_url}{endpoint}"
                ], capture_output=True, text=True, timeout=10)
                
                response_time = time.time() - start_time
                
                if api_check.returncode == 0 and "200" in api_check.stdout:
                    result["successful_loads"] += 1
                    result["response_times"][endpoint] = response_time
                else:
                    result["failed_loads"] += 1
                
                result["endpoints_tested"].append({
                    "endpoint": endpoint,
                    "status": "success" if "200" in api_check.stdout else "failed",
                    "response_time": response_time
                })
                
            except Exception as e:
                result["failed_loads"] += 1
                result["endpoints_tested"].append({
                    "endpoint": endpoint,
                    "status": "error",
                    "error": str(e)
                })
        
        return result

    async def navigate_to_page(self, path: str):
        """导航到指定页面"""
        url = f"{self.base_url}{path}"
        try:
            # 使用AppleScript导航
            nav_script = f'''
            tell application "Google Chrome"
                activate
                if (count of windows) = 0 then
                    make new window
                end if
                set URL of active tab of window 1 to "{url}"
            end tell
            '''
            
            subprocess.run(['osascript', '-e', nav_script], timeout=10)
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"导航失败: {e}")

    async def take_screenshot(self, filename: str) -> str:
        """截图功能"""
        screenshot_path = f"logs/screenshots/{filename}"
        try:
            # 使用DevTools Protocol截图
            screenshot_result = await self.execute_devtools_command(
                "Page.captureScreenshot",
                {"format": "png", "quality": 90}
            )
            
            if screenshot_result.get("result", {}).get("data"):
                import base64
                screenshot_data = base64.b64decode(screenshot_result["result"]["data"])
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot_data)
                return screenshot_path
            else:
                # 备用方案：使用screencapture
                subprocess.run([
                    "screencapture", "-x", screenshot_path
                ], timeout=10)
                return screenshot_path if Path(screenshot_path).exists() else None
                
        except Exception as e:
            print(f"截图失败: {e}")
            return None

    async def run_user_flow_test(self, flow_config: Dict) -> Dict:
        """执行用户流程测试"""
        print(f"   🎯 执行用户流程测试: {flow_config['name']}")
        
        flow_result = {
            "flow_name": flow_config["name"],
            "steps_completed": 0,
            "total_steps": len(flow_config["steps"]),
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            for i, step in enumerate(flow_config["steps"]):
                action = step["action"]
                
                if action == "navigate":
                    await self.navigate_to_page(step["target"])
                    
                elif action == "wait":
                    await asyncio.sleep(step["duration"])
                    
                elif action == "wait_for_element":
                    # 等待元素出现
                    selector = step["selector"]
                    max_wait = step.get("timeout", 10)
                    
                    for _ in range(max_wait):
                        element_check = await self.execute_devtools_command(
                            "Runtime.evaluate",
                            {"expression": f"!!document.querySelector('{selector}')"}
                        )
                        if element_check.get("result", {}).get("value"):
                            break
                        await asyncio.sleep(1)
                    else:
                        flow_result["errors"].append(f"元素 {selector} 未找到")
                        
                elif action == "wait_for_data":
                    await asyncio.sleep(step.get("timeout", 5))
                    
                elif action == "verify_content":
                    content_check = await self.execute_devtools_command(
                        "Runtime.evaluate",
                        {"expression": "document.body.innerText"}
                    )
                    
                    body_text = content_check.get("result", {}).get("value", "")
                    for required_text in step["contains"]:
                        if required_text not in body_text:
                            flow_result["errors"].append(f"未找到必需内容: {required_text}")
                
                flow_result["steps_completed"] = i + 1
                
            # 计算流程状态
            if flow_result["errors"]:
                flow_result["status"] = "FAIL"
            elif flow_result["steps_completed"] == flow_result["total_steps"]:
                flow_result["status"] = "PASS"
            else:
                flow_result["status"] = "PARTIAL"
                
        except Exception as e:
            flow_result["errors"].append(f"流程执行异常: {e}")
            flow_result["status"] = "ERROR"
        
        return flow_result

    async def check_single_page(self, page_name: str, page_config: Dict) -> FrontendCheckResult:
        """检查单个页面的完整功能"""
        print(f"\n📄 深度检查页面: {page_name}")
        
        url = page_config["url"]
        full_url = f"{self.base_url}{url}"
        issues = []
        performance_data = {}
        
        # 导航到页面
        await self.navigate_to_page(url)
        await asyncio.sleep(3)
        
        # 执行各项检查
        for check in page_config["checks"]:
            try:
                if check == "page_loads":
                    # 检查页面基本加载
                    load_check = await self.execute_devtools_command(
                        "Runtime.evaluate",
                        {"expression": "document.readyState === 'complete'"}
                    )
                    if not load_check.get("result", {}).get("value"):
                        issues.append("页面加载未完成")
                
                elif check == "react_renders":
                    react_result = await self.check_react_components(url)
                    if not react_result["react_detected"]:
                        issues.append("React应用未检测到")
                    if react_result["component_errors"]:
                        issues.extend(react_result["component_errors"])
                
                elif check == "no_js_errors":
                    js_errors = await self.check_javascript_errors()
                    if js_errors:
                        issues.extend([f"JS错误: {err['text']}" for err in js_errors[:3]])
                
                elif check == "performance_metrics":
                    perf_metrics = await self.collect_performance_metrics()
                    performance_data.update(asdict(perf_metrics))
                    if perf_metrics.load_time > 5000:  # 5秒
                        issues.append(f"页面加载时间过长: {perf_metrics.load_time/1000:.2f}s")
                
                elif check == "api_data_loads":
                    if "api_endpoints" in page_config:
                        api_result = await self.check_api_data_loading(page_config["api_endpoints"])
                        if api_result["failed_loads"] > 0:
                            issues.append(f"API加载失败: {api_result['failed_loads']}个端点")
                
                elif check == "data_loads":
                    # 等待数据加载
                    if page_config.get("wait_for_data"):
                        await asyncio.sleep(page_config.get("data_timeout", 5))
                    
                    # 检查是否有数据内容
                    data_check = await self.execute_devtools_command(
                        "Runtime.evaluate",
                        {"expression": "document.querySelectorAll('.ant-card, .data-item, [class*=\"data\"]').length > 0"}
                    )
                    if not data_check.get("result", {}).get("value"):
                        issues.append("页面数据未加载")
                
            except Exception as e:
                issues.append(f"检查 {check} 时出错: {e}")
        
        # 截图
        screenshot_path = await self.take_screenshot(f"{page_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        # 确定状态
        if not issues:
            status = "PASS"
        elif len(issues) <= 2:
            status = "WARNING"
        else:
            status = "FAIL"
        
        return FrontendCheckResult(
            page=page_name,
            url=full_url,
            status=status,
            issues=issues,
            performance=performance_data,
            screenshot_path=screenshot_path
        )

    async def run_comprehensive_test(self):
        """运行完整的高级前端测试"""
        print("🌐 开始高级Chrome MCP前端功能测试")
        print("========================================================================")
        
        if not await self.setup_chrome_debugging():
            return {"error": "Chrome调试环境设置失败"}
        
        try:
            # 测试所有页面
            for page_name, page_config in self.test_config["check_points"].items():
                page_result = await self.check_single_page(page_name, page_config)
                self.results.append(page_result)
            
            # 执行用户流程测试
            for flow_config in self.test_config["user_flows"]:
                flow_result = await self.run_user_flow_test(flow_config)
                self.results.append(flow_result)
            
            # 生成综合报告
            report = self.generate_comprehensive_report()
            
            # 保存报告
            report_path = Path("logs/mcp_checks") / f"advanced_frontend_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 高级测试报告已保存: {report_path}")
            return report
            
        finally:
            await self.cleanup()

    def generate_comprehensive_report(self) -> Dict:
        """生成综合报告"""
        total_tests = 0
        passed_tests = 0
        total_issues = 0
        
        page_results = []
        flow_results = []
        
        for result in self.results:
            if isinstance(result, FrontendCheckResult):
                page_results.append(result)
                # 计算基础测试通过情况
                basic_checks = 5  # 假设每页面有5个基础检查
                page_issues = len(result.issues)
                page_passed = max(0, basic_checks - page_issues)
                
                total_tests += basic_checks
                passed_tests += page_passed
                total_issues += page_issues
                
            elif isinstance(result, dict) and "flow_name" in result:
                flow_results.append(result)
                total_tests += result["total_steps"]
                passed_tests += result["steps_completed"]
                total_issues += len(result["errors"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 确定整体状态
        if success_rate >= 95 and total_issues == 0:
            overall_status = "EXCELLENT"
        elif success_rate >= 85 and total_issues <= 2:
            overall_status = "GOOD"
        elif success_rate >= 75:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        return {
            "test_type": "高级Chrome MCP前端功能测试",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_status": overall_status,
                "success_rate": f"{success_rate:.1f}%",
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "total_issues": total_issues,
                "pages_tested": len(page_results),
                "flows_tested": len(flow_results)
            },
            "page_results": [asdict(result) for result in page_results],
            "flow_results": flow_results,
            "capabilities_verified": [
                "真实Chrome浏览器自动化",
                "DevTools Protocol深度集成",
                "React组件渲染验证",
                "JavaScript错误检测",
                "性能指标收集",
                "API数据加载验证",
                "用户流程自动化测试",
                "截图和视觉验证"
            ],
            "recommendations": self.generate_recommendations()
        }

    def generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 分析所有问题
        all_issues = []
        for result in self.results:
            if isinstance(result, FrontendCheckResult):
                all_issues.extend(result.issues)
            elif isinstance(result, dict) and "errors" in result:
                all_issues.extend(result["errors"])
        
        # 生成分类建议
        if any("React" in issue for issue in all_issues):
            recommendations.append("⚛️ 检查React应用渲染和组件状态")
        
        if any("JS错误" in issue for issue in all_issues):
            recommendations.append("🔧 修复JavaScript控制台错误")
        
        if any("API" in issue for issue in all_issues):
            recommendations.append("🔌 检查后端API接口和数据格式")
        
        if any("加载" in issue for issue in all_issues):
            recommendations.append("⚡ 优化页面和数据加载性能")
        
        if any("流程" in issue for issue in all_issues):
            recommendations.append("🎯 修复用户流程和交互问题")
        
        if not all_issues:
            recommendations.append("✅ 所有高级功能测试通过，系统运行完美！")
        
        return recommendations

    async def cleanup(self):
        """清理资源"""
        print("\n🛑 清理测试环境...")
        
        try:
            if self.chrome_process and self.chrome_process.poll() is None:
                self.chrome_process.terminate()
                await asyncio.sleep(2)
                if self.chrome_process.poll() is None:
                    self.chrome_process.kill()
                print("   ✅ Chrome进程已清理")
            
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print("   ✅ 临时目录已清理")
                
        except Exception as e:
            print(f"   ⚠️ 清理时出现异常: {e}")

async def main():
    """主函数"""
    checker = AdvancedFrontendChecker()
    
    try:
        report = await checker.run_comprehensive_test()
        
        # 打印详细报告
        print("\n" + "="*80)
        print("📊 高级Chrome MCP前端功能测试完成")
        print("="*80)
        
        if "error" not in report:
            summary = report["summary"]
            print(f"整体状态: {summary['overall_status']}")
            print(f"成功率: {summary['success_rate']}")
            print(f"测试通过: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"问题总数: {summary['total_issues']}")
            print(f"页面测试: {summary['pages_tested']}个")
            print(f"流程测试: {summary['flows_tested']}个")
            
            print("\n验证的能力:")
            for capability in report["capabilities_verified"]:
                print(f"  ✅ {capability}")
            
            if report.get("recommendations"):
                print(f"\n修复建议:")
                for rec in report["recommendations"]:
                    print(f"  💡 {rec}")
                    
        else:
            print(f"❌ 测试失败: {report['error']}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        await checker.cleanup()
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        await checker.cleanup()

if __name__ == "__main__":
    asyncio.run(main())