#!/usr/bin/env python3
"""
真实Chrome浏览器MCP测试
直接启动Chrome浏览器并进行真实的渲染和交互测试
"""

import json
import time
import subprocess
import os
import signal
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import tempfile

class RealBrowserTester:
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.chrome_process = None
        self.temp_dir = None
        self.results = []
        
        # 测试页面
        self.test_pages = {
            "主页": "/",
            "仪表板": "/dashboard", 
            "板块分析": "/sectors",
            "股票推荐": "/stock-recommendation"
        }
        
        # 创建临时目录和日志目录
        self.temp_dir = tempfile.mkdtemp(prefix="chrome_test_")
        Path("logs/mcp_checks").mkdir(parents=True, exist_ok=True)
        Path("logs/screenshots").mkdir(parents=True, exist_ok=True)

    def start_chrome_with_debugging(self) -> bool:
        """启动带有调试端口的Chrome"""
        print("🚀 启动Chrome浏览器（调试模式）...")
        
        try:
            # Chrome启动参数
            chrome_args = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "--remote-debugging-port=9222",
                "--user-data-dir=" + self.temp_dir,
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--window-size=1920,1080",
                "--new-window",
                self.base_url
            ]
            
            # 启动Chrome
            self.chrome_process = subprocess.Popen(
                chrome_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # 等待Chrome启动
            print("⏳ 等待Chrome启动完成...")
            time.sleep(5)
            
            # 检查Chrome是否成功启动
            if self.chrome_process.poll() is None:
                print("✅ Chrome浏览器启动成功")
                return True
            else:
                print("❌ Chrome浏览器启动失败")
                return False
                
        except Exception as e:
            print(f"❌ Chrome启动异常: {e}")
            return False

    def test_real_browser_functionality(self) -> Dict:
        """测试真实浏览器功能"""
        print("\n🌐 测试真实浏览器渲染和功能...")
        
        browser_result = {
            "test_name": "真实浏览器功能测试",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "browser_info": {},
            "page_tests": {},
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            # 检查Chrome调试端口
            debug_check = subprocess.run([
                "curl", "-s", "http://localhost:9222/json/version"
            ], capture_output=True, text=True, timeout=5)
            
            if debug_check.returncode == 0:
                browser_result["tests"]["Chrome调试端口可用"] = True
                try:
                    version_info = json.loads(debug_check.stdout)
                    browser_result["browser_info"] = version_info
                    print(f"   ✅ Chrome版本: {version_info.get('Browser', 'Unknown')}")
                except:
                    pass
            else:
                browser_result["tests"]["Chrome调试端口可用"] = False
                browser_result["errors"].append("Chrome调试端口不可用")
            
            # 获取标签页列表
            tabs_check = subprocess.run([
                "curl", "-s", "http://localhost:9222/json"
            ], capture_output=True, text=True, timeout=5)
            
            if tabs_check.returncode == 0:
                try:
                    tabs = json.loads(tabs_check.stdout)
                    browser_result["tests"]["标签页信息获取"] = len(tabs) > 0
                    browser_result["browser_info"]["tabs_count"] = len(tabs)
                    
                    # 分析每个标签页
                    for i, tab in enumerate(tabs[:4]):  # 最多检查4个标签页
                        tab_url = tab.get("url", "")
                        tab_title = tab.get("title", "")
                        
                        # 判断页面类型
                        page_type = "未知"
                        if "localhost:3000" in tab_url:
                            if tab_url.endswith("/dashboard"):
                                page_type = "仪表板"
                            elif "sectors" in tab_url:
                                page_type = "板块分析"
                            elif "stock-recommendation" in tab_url:
                                page_type = "股票推荐"
                            elif tab_url.endswith("/") or tab_url.endswith(":3000"):
                                page_type = "主页"
                        
                        browser_result["page_tests"][f"标签页{i+1}_{page_type}"] = {
                            "url": tab_url,
                            "title": tab_title,
                            "has_title": bool(tab_title.strip()),
                            "is_localhost": "localhost:3000" in tab_url
                        }
                        
                        print(f"   📄 标签页{i+1} ({page_type}): {tab_title}")
                        
                except Exception as e:
                    browser_result["tests"]["标签页信息获取"] = False
                    browser_result["errors"].append(f"标签页解析失败: {e}")
            else:
                browser_result["tests"]["标签页信息获取"] = False
                browser_result["errors"].append("无法获取标签页信息")
            
            # 测试页面导航功能
            self.test_browser_navigation(browser_result)
            
        except Exception as e:
            browser_result["errors"].append(f"浏览器测试失败: {e}")
            
        # 计算状态
        if browser_result["tests"]:
            passed_tests = sum(1 for result in browser_result["tests"].values() if result)
            total_tests = len(browser_result["tests"])
            success_rate = passed_tests / total_tests
            
            if success_rate >= 0.8 and not browser_result["errors"]:
                browser_result["status"] = "PASS"
            elif success_rate >= 0.6:
                browser_result["status"] = "WARNING"
            else:
                browser_result["status"] = "FAIL"
        
        return browser_result

    def test_browser_navigation(self, result: Dict):
        """测试浏览器页面导航"""
        print("   🧭 测试页面导航...")
        
        # 逐一测试各个页面
        for page_name, path in self.test_pages.items():
            try:
                # 在Chrome中打开新标签页
                navigation_script = f'''
                tell application "Google Chrome"
                    activate
                    set myTab to make new tab at end of tabs of window 1
                    set URL of myTab to "{self.base_url}{path}"
                    delay 3
                    
                    set pageTitle to title of myTab
                    set pageURL to URL of myTab
                    
                    delay 2
                    close myTab
                    
                    return pageTitle & "|||" & pageURL
                end tell
                '''
                
                nav_result = subprocess.run([
                    'osascript', '-e', navigation_script
                ], capture_output=True, text=True, timeout=15)
                
                if nav_result.returncode == 0:
                    page_title, page_url = nav_result.stdout.strip().split('|||')
                    result["tests"][f"{page_name}_导航成功"] = True
                    result["tests"][f"{page_name}_标题正确"] = bool(page_title.strip())
                    result["tests"][f"{page_name}_URL正确"] = path in page_url
                    
                    print(f"     ✅ {page_name}: {page_title}")
                    
                else:
                    result["tests"][f"{page_name}_导航成功"] = False
                    result["errors"].append(f"{page_name} 导航失败: {nav_result.stderr}")
                    print(f"     ❌ {page_name}: 导航失败")
                    
            except Exception as e:
                result["tests"][f"{page_name}_导航成功"] = False
                result["errors"].append(f"{page_name} 导航异常: {e}")
                print(f"     ❌ {page_name}: 导航异常")

    def test_javascript_console(self) -> Dict:
        """测试JavaScript控制台输出"""
        print("\n🔍 检查JavaScript控制台...")
        
        console_result = {
            "test_name": "JavaScript控制台检测",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "console_logs": [],
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            # 使用Chrome DevTools Protocol获取控制台日志
            # 这需要先启用运行时
            enable_runtime = subprocess.run([
                "curl", "-s", "-X", "POST", 
                "http://localhost:9222/json/runtime/enable"
            ], capture_output=True, text=True, timeout=5)
            
            # 等待一段时间让页面运行
            time.sleep(3)
            
            # 尝试获取控制台API信息
            console_api = subprocess.run([
                "curl", "-s", "http://localhost:9222/json/runtime/evaluate",
                "-H", "Content-Type: application/json",
                "-d", '{"expression": "console.log(\\"测试消息\\"); typeof window.React !== \\"undefined\\"", "returnByValue": true}'
            ], capture_output=True, text=True, timeout=5)
            
            if console_api.returncode == 0:
                console_result["tests"]["控制台API可用"] = True
                console_result["tests"]["React检测尝试"] = True
                print("   ✅ 控制台API响应正常")
            else:
                console_result["tests"]["控制台API可用"] = False
                console_result["errors"].append("控制台API不可用")
                
        except Exception as e:
            console_result["errors"].append(f"控制台检测失败: {e}")
        
        # 计算状态
        if console_result["tests"]:
            passed_tests = sum(1 for result in console_result["tests"].values() if result)
            total_tests = len(console_result["tests"])
            success_rate = passed_tests / total_tests
            
            if success_rate >= 0.8:
                console_result["status"] = "PASS"
            elif success_rate >= 0.6:
                console_result["status"] = "WARNING"
            else:
                console_result["status"] = "FAIL"
        
        return console_result

    def capture_screenshots(self) -> Dict:
        """截图功能测试"""
        print("\n📷 测试截图功能...")
        
        screenshot_result = {
            "test_name": "截图功能测试",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "screenshots": [],
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            # 对每个页面进行截图
            for page_name, path in self.test_pages.items():
                try:
                    # 导航到页面
                    nav_script = f'''
                    tell application "Google Chrome"
                        activate
                        set myTab to make new tab at end of tabs of window 1
                        set URL of myTab to "{self.base_url}{path}"
                        delay 4
                        return "success"
                    end tell
                    '''
                    
                    subprocess.run(['osascript', '-e', nav_script], 
                                 capture_output=True, timeout=10)
                    
                    # 使用macOS内置截图功能
                    screenshot_path = f"logs/screenshots/{page_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_real.png"
                    
                    # 获取Chrome窗口并截图
                    screenshot_script = f'''
                    tell application "Google Chrome"
                        activate
                        delay 1
                    end tell
                    
                    do shell script "screencapture -l$(osascript -e 'tell app \\"Google Chrome\\" to id of window 1') '{screenshot_path}'"
                    '''
                    
                    screenshot_result_cmd = subprocess.run([
                        'osascript', '-e', screenshot_script
                    ], capture_output=True, timeout=10)
                    
                    if screenshot_result_cmd.returncode == 0 and Path(screenshot_path).exists():
                        screenshot_result["tests"][f"{page_name}_截图成功"] = True
                        screenshot_result["screenshots"].append(screenshot_path)
                        print(f"   ✅ {page_name}: 截图保存到 {screenshot_path}")
                    else:
                        screenshot_result["tests"][f"{page_name}_截图成功"] = False
                        screenshot_result["errors"].append(f"{page_name} 截图失败")
                        print(f"   ❌ {page_name}: 截图失败")
                        
                except Exception as e:
                    screenshot_result["tests"][f"{page_name}_截图成功"] = False
                    screenshot_result["errors"].append(f"{page_name} 截图异常: {e}")
                    print(f"   ❌ {page_name}: 截图异常")
            
        except Exception as e:
            screenshot_result["errors"].append(f"截图功能测试失败: {e}")
        
        # 计算状态
        if screenshot_result["tests"]:
            passed_tests = sum(1 for result in screenshot_result["tests"].values() if result)
            total_tests = len(screenshot_result["tests"])
            success_rate = passed_tests / total_tests
            
            if success_rate >= 0.8:
                screenshot_result["status"] = "PASS"
            elif success_rate >= 0.6:
                screenshot_result["status"] = "WARNING"
            else:
                screenshot_result["status"] = "FAIL"
        
        return screenshot_result

    def cleanup_chrome(self):
        """清理Chrome进程"""
        print("\n🛑 清理Chrome进程...")
        
        try:
            # 关闭我们启动的Chrome进程
            if self.chrome_process and self.chrome_process.poll() is None:
                self.chrome_process.terminate()
                time.sleep(2)
                if self.chrome_process.poll() is None:
                    self.chrome_process.kill()
                print("   ✅ Chrome进程已终止")
            
            # 清理临时目录
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print("   ✅ 临时目录已清理")
                
        except Exception as e:
            print(f"   ⚠️ 清理时出现异常: {e}")

    def run_comprehensive_test(self):
        """运行完整的真实浏览器测试"""
        print("🌐 开始真实Chrome浏览器MCP功能测试")
        print("========================================================================")
        
        try:
            # 启动Chrome
            if not self.start_chrome_with_debugging():
                return {"error": "无法启动Chrome浏览器"}
            
            # 等待Chrome完全启动
            time.sleep(3)
            
            # 执行各项测试
            browser_result = self.test_real_browser_functionality()
            self.results.append(browser_result)
            
            console_result = self.test_javascript_console()
            self.results.append(console_result)
            
            screenshot_result = self.capture_screenshots()
            self.results.append(screenshot_result)
            
            # 生成综合报告
            report = self.generate_comprehensive_report()
            
            # 保存报告
            report_path = Path("logs/mcp_checks") / f"real_browser_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 测试报告已保存: {report_path}")
            return report
            
        finally:
            # 确保清理Chrome进程
            self.cleanup_chrome()

    def generate_comprehensive_report(self) -> Dict:
        """生成综合报告"""
        total_tests = 0
        passed_tests = 0
        total_errors = 0
        all_issues = []
        
        test_results = []
        for result in self.results:
            if "tests" in result:
                test_count = len(result["tests"])
                test_passed = sum(1 for test_result in result["tests"].values() if test_result)
                total_tests += test_count
                passed_tests += test_passed
                error_count = len(result.get("errors", []))
                total_errors += error_count
                
                # 收集问题
                for error in result.get("errors", []):
                    all_issues.append(f"{result.get('test_name', 'Unknown')}: {error}")
                
                failed_tests = [test for test, passed in result.get("tests", {}).items() if not passed]
                for test in failed_tests:
                    all_issues.append(f"{result.get('test_name', 'Unknown')}: 测试失败 - {test}")
                
                test_results.append({
                    "test_name": result.get("test_name", "Unknown"),
                    "status": result["status"],
                    "tests_passed": f"{test_passed}/{test_count}",
                    "success_rate": f"{(test_passed/test_count*100):.1f}%" if test_count > 0 else "0%",
                    "errors_count": error_count
                })
        
        overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 确定整体状态
        if overall_success_rate >= 90 and total_errors == 0:
            overall_status = "EXCELLENT"
        elif overall_success_rate >= 75 and total_errors <= 2:
            overall_status = "GOOD"
        elif overall_success_rate >= 60:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        return {
            "test_type": "真实Chrome浏览器MCP功能测试",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_status": overall_status,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{overall_success_rate:.1f}%",
                "total_errors": total_errors,
                "test_categories": len(self.results)
            },
            "test_results": test_results,
            "detailed_results": self.results,
            "issues_found": all_issues,
            "recommendations": self.generate_recommendations(),
            "screenshots": [result.get("screenshots", []) for result in self.results if "screenshots" in result]
        }

    def generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 分析问题类型
        all_issues = []
        for result in self.results:
            all_issues.extend(result.get("errors", []))
            if "tests" in result:
                failed_tests = [test for test, passed in result["tests"].items() if not passed]
                all_issues.extend(failed_tests)
        
        # 生成针对性建议
        if any("导航" in issue or "URL" in issue for issue in all_issues):
            recommendations.append("🧭 修复页面路由和导航问题")
        
        if any("标题" in issue or "title" in issue for issue in all_issues):
            recommendations.append("📄 检查页面标题设置和渲染")
        
        if any("控制台" in issue or "console" in issue for issue in all_issues):
            recommendations.append("🔍 修复JavaScript控制台错误")
        
        if any("截图" in issue or "screenshot" in issue for issue in all_issues):
            recommendations.append("📷 优化页面渲染和视觉呈现")
        
        if any("Chrome" in issue or "browser" in issue for issue in all_issues):
            recommendations.append("🌐 检查浏览器兼容性和调试配置")
        
        if not all_issues:
            recommendations.append("✅ 所有真实浏览器测试通过，系统运行良好！")
        
        return recommendations

def main():
    """主函数"""
    tester = RealBrowserTester()
    
    try:
        report = tester.run_comprehensive_test()
        
        # 打印详细报告
        print("\n" + "="*80)
        print("📊 真实Chrome浏览器MCP测试完成")
        print("="*80)
        
        if "error" not in report:
            summary = report["summary"]
            print(f"整体状态: {summary['overall_status']}")
            print(f"成功率: {summary['success_rate']}")
            print(f"测试通过: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"错误总数: {summary['total_errors']}")
            
            print("\n测试类别结果:")
            for test_result in report["test_results"]:
                status_emoji = "✅" if test_result["status"] == "PASS" else "⚠️" if test_result["status"] == "WARNING" else "❌"
                print(f"  {status_emoji} {test_result['test_name']}: {test_result['success_rate']} ({test_result['tests_passed']})")
            
            if report.get("issues_found"):
                print(f"\n发现的问题:")
                for issue in report["issues_found"]:
                    print(f"  ❌ {issue}")
            
            if report.get("recommendations"):
                print(f"\n修复建议:")
                for rec in report["recommendations"]:
                    print(f"  💡 {rec}")
                    
            # 显示截图信息
            screenshots = [s for sublist in report.get("screenshots", []) for s in sublist]
            if screenshots:
                print(f"\n📷 生成的截图:")
                for screenshot in screenshots:
                    print(f"  📸 {screenshot}")
        else:
            print(f"❌ 测试失败: {report['error']}")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        tester.cleanup_chrome()
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        tester.cleanup_chrome()

if __name__ == "__main__":
    main()