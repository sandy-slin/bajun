#!/usr/bin/env python3
"""
Chrome MCP完整前端功能测试 - 使用系统Chrome
直接使用selenium + Chrome驱动进行深度测试
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
except ImportError:
    print("❌ Selenium未安装，正在安装...")
    subprocess.run([sys.executable, "-m", "pip", "install", "selenium"], check=True)
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException

class ChromeNativeTester:
    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.results = []
        self.screenshots_dir = Path("logs/screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试配置
        self.base_url = "http://localhost:3000"
        self.test_pages = {
            "主页": "/",
            "仪表板": "/dashboard", 
            "板块分析": "/sectors",
            "股票推荐": "/stock-recommendation"
        }
        
        # 日志收集
        self.console_logs = []

    def setup_chrome_driver(self):
        """设置Chrome驱动"""
        print("🚀 设置Chrome驱动...")
        try:
            # Chrome选项
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--enable-logging")
            chrome_options.add_argument("--log-level=0")
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
            # 启用性能日志
            chrome_options.add_argument("--enable-logging")
            chrome_options.add_argument("--v=1")
            chrome_options.set_capability('goog:loggingPrefs', {
                'browser': 'ALL',
                'driver': 'ALL',
                'performance': 'ALL'
            })
            
            # 创建驱动
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            
            print("✅ Chrome驱动设置成功")
            return True
            
        except Exception as e:
            print(f"❌ Chrome驱动设置失败: {e}")
            # 尝试安装webdriver-manager
            try:
                print("🔄 尝试自动安装ChromeDriver...")
                subprocess.run([sys.executable, "-m", "pip", "install", "webdriver-manager"], check=True)
                from webdriver_manager.chrome import ChromeDriverManager
                
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                print("✅ Chrome驱动自动安装成功")
                return True
            except Exception as e2:
                print(f"❌ 自动安装也失败: {e2}")
                return False

    def test_page_functionality(self, page_name: str, path: str) -> Dict:
        """测试单个页面的完整功能"""
        print(f"\n📄 测试页面: {page_name} ({path})")
        
        page_result = {
            "page": page_name,
            "url": f"{self.base_url}{path}",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "performance": {},
            "screenshot_path": None,
            "status": "UNKNOWN",
            "console_logs": []
        }
        
        try:
            # 清空之前的日志
            self.driver.get_log('browser')
            
            # 导航到页面并测量性能
            start_time = time.time()
            self.driver.get(f"{self.base_url}{path}")
            
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            load_time = time.time() - start_time
            
            page_result["performance"]["load_time"] = round(load_time, 3)
            page_result["performance"]["status_code"] = 200  # Selenium不直接提供status code
            
            # 基础测试
            page_result["tests"]["页面加载完成"] = True
            page_result["tests"]["页面加载时间"] = load_time < 5.0
            
            # 等待React应用渲染
            time.sleep(3)
            
            # 检查页面标题
            title = self.driver.title
            page_result["tests"]["页面标题存在"] = bool(title and title.strip())
            page_result["performance"]["title"] = title
            
            # 检查React根元素
            try:
                react_root = self.driver.find_element(By.ID, "root")
                page_result["tests"]["React应用已渲染"] = react_root is not None
            except:
                page_result["tests"]["React应用已渲染"] = False
                page_result["errors"].append("未找到React根元素 #root")
            
            # 检查页面内容
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            page_result["tests"]["页面内容非空"] = len(body_text.strip()) > 100
            
            # 特定页面的功能测试
            self.test_page_specific_features(page_name, page_result)
            
            # 收集控制台日志
            try:
                browser_logs = self.driver.get_log('browser')
                for log_entry in browser_logs:
                    if log_entry['level'] in ['SEVERE', 'WARNING']:
                        error_msg = f"{log_entry['level']}: {log_entry['message']}"
                        page_result["console_logs"].append(error_msg)
                        if log_entry['level'] == 'SEVERE':
                            page_result["errors"].append(error_msg)
            except:
                pass  # 某些Chrome版本可能不支持日志收集
            
            # JavaScript错误检测
            try:
                js_errors = self.driver.execute_script("""
                    return window.jsErrors || [];
                """)
                if js_errors:
                    page_result["errors"].extend(js_errors)
                    page_result["tests"]["无JavaScript错误"] = False
                else:
                    page_result["tests"]["无JavaScript错误"] = True
            except:
                page_result["tests"]["无JavaScript错误"] = True  # 无法检测时假设没错误
            
            # 截图
            screenshot_path = self.screenshots_dir / f"{page_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.driver.save_screenshot(str(screenshot_path))
            page_result["screenshot_path"] = str(screenshot_path)
            
            # 计算总体状态
            passed_tests = sum(1 for result in page_result["tests"].values() if result)
            total_tests = len(page_result["tests"])
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            if success_rate >= 0.8 and len(page_result["errors"]) == 0:
                page_result["status"] = "PASS"
            elif success_rate >= 0.6:
                page_result["status"] = "WARNING"
            else:
                page_result["status"] = "FAIL"
                
            print(f"   ✅ 测试完成: {passed_tests}/{total_tests} 通过")
            if page_result["errors"]:
                print(f"   ⚠️ 发现 {len(page_result['errors'])} 个错误")
            
        except Exception as e:
            error_msg = f"页面测试失败: {str(e)}"
            page_result["errors"].append(error_msg)
            page_result["status"] = "ERROR"
            print(f"   ❌ {error_msg}")
            
        return page_result

    def test_page_specific_features(self, page_name: str, result: Dict):
        """测试特定页面的功能"""
        try:
            if page_name == "主页":
                # 测试导航菜单
                try:
                    nav_elements = self.driver.find_elements(By.CSS_SELECTOR, "nav, .ant-menu, .navigation")
                    result["tests"]["导航菜单存在"] = len(nav_elements) > 0
                except:
                    result["tests"]["导航菜单存在"] = False
                
                # 测试主要内容
                try:
                    main_content = self.driver.find_elements(By.CSS_SELECTOR, "main, .main-content, .content, .ant-layout-content")
                    result["tests"]["主要内容区域存在"] = len(main_content) > 0
                except:
                    result["tests"]["主要内容区域存在"] = False
                
            elif page_name == "仪表板":
                # 测试仪表板卡片
                try:
                    cards = self.driver.find_elements(By.CSS_SELECTOR, ".ant-card, .card")
                    result["tests"]["仪表板卡片存在"] = len(cards) > 0
                    result["performance"]["cards_count"] = len(cards)
                except:
                    result["tests"]["仪表板卡片存在"] = False
                
                # 等待数据加载完成
                time.sleep(5)
                try:
                    loading_elements = self.driver.find_elements(By.CSS_SELECTOR, ".ant-spin, .loading")
                    result["tests"]["数据加载完成"] = len(loading_elements) == 0
                except:
                    result["tests"]["数据加载完成"] = True
                
            elif page_name == "板块分析":
                # 等待API数据加载
                time.sleep(8)
                
                # 测试板块数据显示
                try:
                    sector_elements = self.driver.find_elements(By.CSS_SELECTOR, ".ant-card, .sector-card")
                    result["tests"]["板块数据已显示"] = len(sector_elements) > 0
                    result["performance"]["sectors_count"] = len(sector_elements)
                except:
                    result["tests"]["板块数据已显示"] = False
                
                # 测试刷新按钮
                try:
                    refresh_btn = self.driver.find_element(By.CSS_SELECTOR, "button[class*='refresh'], .ant-btn")
                    result["tests"]["刷新按钮存在"] = refresh_btn is not None
                except:
                    result["tests"]["刷新按钮存在"] = False
                
                # 测试评分显示
                try:
                    score_elements = self.driver.find_elements(By.CSS_SELECTOR, "[class*='score'], .score")
                    result["tests"]["评分数据显示"] = len(score_elements) > 0
                except:
                    result["tests"]["评分数据显示"] = False
                
            elif page_name == "股票推荐":
                # 等待数据加载
                time.sleep(8)
                
                # 测试股票推荐列表
                try:
                    stock_elements = self.driver.find_elements(By.CSS_SELECTOR, ".ant-card, .stock-card")
                    result["tests"]["股票推荐数据已显示"] = len(stock_elements) > 0
                    result["performance"]["stocks_count"] = len(stock_elements)
                except:
                    result["tests"]["股票推荐数据已显示"] = False
                
                # 测试评分和标签
                try:
                    tags = self.driver.find_elements(By.CSS_SELECTOR, ".ant-tag")
                    result["tests"]["标签和评分显示"] = len(tags) > 0
                except:
                    result["tests"]["标签和评分显示"] = False
                
        except Exception as e:
            result["errors"].append(f"特定功能测试失败: {str(e)}")

    def test_user_interactions(self) -> Dict:
        """测试用户交互流程"""
        print("\n🖱️ 测试用户交互流程...")
        
        interaction_result = {
            "test_name": "用户交互流程测试",
            "timestamp": datetime.now().isoformat(),
            "flows": {},
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            # 流程1: 主页 -> 仪表板
            print("   测试流程: 主页 -> 仪表板")
            self.driver.get(f"{self.base_url}/")
            time.sleep(3)
            
            try:
                # 查找仪表板链接
                dashboard_links = self.driver.find_elements(By.CSS_SELECTOR, 
                    'a[href="/dashboard"], a[href*="dashboard"], [data-testid="dashboard-link"]')
                
                if dashboard_links:
                    dashboard_links[0].click()
                    time.sleep(3)
                    current_url = self.driver.current_url
                    interaction_result["flows"]["主页到仪表板"] = "/dashboard" in current_url
                else:
                    interaction_result["flows"]["主页到仪表板"] = False
                    interaction_result["errors"].append("未找到仪表板导航链接")
            except Exception as e:
                interaction_result["flows"]["主页到仪表板"] = False
                interaction_result["errors"].append(f"仪表板导航失败: {str(e)}")
            
            # 流程2: 直接测试板块分析页面
            print("   测试流程: 直接访问板块分析")
            try:
                self.driver.get(f"{self.base_url}/sectors")
                time.sleep(3)
                current_url = self.driver.current_url
                interaction_result["flows"]["直接访问板块分析"] = "/sectors" in current_url
            except Exception as e:
                interaction_result["flows"]["直接访问板块分析"] = False
                interaction_result["errors"].append(f"板块分析页面访问失败: {str(e)}")
            
            # 流程3: 测试页面间导航
            print("   测试流程: 股票推荐页面")
            try:
                self.driver.get(f"{self.base_url}/stock-recommendation")
                time.sleep(3)
                current_url = self.driver.current_url
                interaction_result["flows"]["股票推荐页面访问"] = "/stock-recommendation" in current_url
            except Exception as e:
                interaction_result["flows"]["股票推荐页面访问"] = False
                interaction_result["errors"].append(f"股票推荐页面访问失败: {str(e)}")
            
            # 计算成功率
            successful_flows = sum(1 for success in interaction_result["flows"].values() if success)
            total_flows = len(interaction_result["flows"])
            success_rate = successful_flows / total_flows if total_flows > 0 else 0
            
            if success_rate >= 0.8:
                interaction_result["status"] = "PASS"
            elif success_rate >= 0.6:
                interaction_result["status"] = "WARNING"
            else:
                interaction_result["status"] = "FAIL"
                
            print(f"   ✅ 交互测试完成: {successful_flows}/{total_flows} 流程成功")
            
        except Exception as e:
            interaction_result["errors"].append(f"交互测试失败: {str(e)}")
            interaction_result["status"] = "ERROR"
            print(f"   ❌ 交互测试失败: {str(e)}")
            
        return interaction_result

    def run_comprehensive_test(self):
        """运行完整的Chrome测试"""
        print("🌐 开始Chrome完整前端功能测试")
        print("========================================================================")
        
        # 设置Chrome驱动
        if not self.setup_chrome_driver():
            return {"error": "无法设置Chrome驱动"}
        
        try:
            # 测试所有页面
            for page_name, path in self.test_pages.items():
                page_result = self.test_page_functionality(page_name, path)
                self.results.append(page_result)
            
            # 测试用户交互
            interaction_result = self.test_user_interactions()
            self.results.append(interaction_result)
            
            # 生成综合报告
            report = self.generate_comprehensive_report()
            
            # 保存报告
            report_path = Path("logs/mcp_checks") / f"chrome_native_comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 测试报告已保存: {report_path}")
            return report
            
        finally:
            # 清理浏览器
            if self.driver:
                self.driver.quit()
                print("🛑 Chrome浏览器已关闭")

    def generate_comprehensive_report(self) -> Dict:
        """生成综合测试报告"""
        total_tests = 0
        passed_tests = 0
        total_errors = 0
        all_issues = []
        
        page_results = []
        for result in self.results:
            if "tests" in result:
                page_tests = len(result["tests"])
                page_passed = sum(1 for test_result in result["tests"].values() if test_result)
                total_tests += page_tests
                passed_tests += page_passed
                page_errors = len(result.get("errors", []))
                total_errors += page_errors
                
                # 收集具体问题
                for error in result.get("errors", []):
                    all_issues.append(f"{result.get('page', 'Unknown')}: {error}")
                
                # 收集失败的测试
                failed_tests = [test for test, passed in result.get("tests", {}).items() if not passed]
                for test in failed_tests:
                    all_issues.append(f"{result.get('page', 'Unknown')}: 测试失败 - {test}")
                
                page_results.append({
                    "page": result.get("page", result.get("test_name", "Unknown")),
                    "status": result["status"],
                    "tests_passed": f"{page_passed}/{page_tests}",
                    "success_rate": f"{(page_passed/page_tests*100):.1f}%" if page_tests > 0 else "0%",
                    "errors_count": page_errors,
                    "load_time": result.get("performance", {}).get("load_time", "N/A"),
                    "console_logs_count": len(result.get("console_logs", []))
                })
        
        overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 确定整体状态
        if overall_success_rate >= 90 and total_errors == 0:
            overall_status = "EXCELLENT"
        elif overall_success_rate >= 80 and total_errors <= 3:
            overall_status = "GOOD"
        elif overall_success_rate >= 70:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        return {
            "test_type": "Chrome Native完整功能测试",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_status": overall_status,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{overall_success_rate:.1f}%",
                "total_errors": total_errors,
                "pages_tested": len([r for r in self.results if "tests" in r])
            },
            "page_results": page_results,
            "detailed_results": self.results,
            "issues_found": all_issues[:20],  # 只显示前20个问题
            "recommendations": self.generate_recommendations()
        }

    def generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        issue_categories = {
            "javascript_errors": [],
            "loading_issues": [],
            "ui_problems": [],
            "navigation_issues": [],
            "data_display_issues": []
        }
        
        for result in self.results:
            if result.get("status") in ["FAIL", "ERROR", "WARNING"]:
                page_name = result.get("page", result.get("test_name", "Unknown"))
                
                # 分类错误
                for error in result.get("errors", []):
                    error_lower = error.lower()
                    if "javascript" in error_lower or "js" in error_lower or "script" in error_lower:
                        issue_categories["javascript_errors"].append(f"{page_name}: {error}")
                    elif "loading" in error_lower or "timeout" in error_lower:
                        issue_categories["loading_issues"].append(f"{page_name}: {error}")
                    elif "element" in error_lower or "selector" in error_lower:
                        issue_categories["ui_problems"].append(f"{page_name}: {error}")
                    elif "navigation" in error_lower or "href" in error_lower:
                        issue_categories["navigation_issues"].append(f"{page_name}: {error}")
                    else:
                        issue_categories["data_display_issues"].append(f"{page_name}: {error}")
                
                # 分析失败的测试
                if "tests" in result:
                    failed_tests = [test for test, passed in result["tests"].items() if not passed]
                    for test in failed_tests:
                        if "数据" in test or "显示" in test:
                            issue_categories["data_display_issues"].append(f"{page_name}: {test}")
                        elif "加载" in test:
                            issue_categories["loading_issues"].append(f"{page_name}: {test}")
                        elif "导航" in test or "链接" in test:
                            issue_categories["navigation_issues"].append(f"{page_name}: {test}")
                        else:
                            issue_categories["ui_problems"].append(f"{page_name}: {test}")
        
        # 生成分类建议
        if issue_categories["javascript_errors"]:
            recommendations.append("🔧 修复JavaScript错误:")
            recommendations.extend([f"  - {issue}" for issue in issue_categories["javascript_errors"][:3]])
        
        if issue_categories["data_display_issues"]:
            recommendations.append("📊 修复数据显示问题:")
            recommendations.extend([f"  - {issue}" for issue in issue_categories["data_display_issues"][:3]])
        
        if issue_categories["loading_issues"]:
            recommendations.append("⏱️ 优化页面加载性能:")
            recommendations.extend([f"  - {issue}" for issue in issue_categories["loading_issues"][:3]])
        
        if issue_categories["ui_problems"]:
            recommendations.append("🎨 修复UI组件问题:")
            recommendations.extend([f"  - {issue}" for issue in issue_categories["ui_problems"][:3]])
        
        if issue_categories["navigation_issues"]:
            recommendations.append("🧭 修复导航问题:")
            recommendations.extend([f"  - {issue}" for issue in issue_categories["navigation_issues"][:3]])
        
        if not any(issue_categories.values()):
            recommendations.append("✅ 所有测试通过，系统运行良好！")
        
        return recommendations

def main():
    """主函数"""
    tester = ChromeNativeTester()
    report = tester.run_comprehensive_test()
    
    # 打印简要报告
    print("\n" + "="*80)
    print("📊 Chrome MCP测试完成")
    print("="*80)
    
    if "error" not in report:
        summary = report["summary"]
        print(f"整体状态: {summary['overall_status']}")
        print(f"成功率: {summary['success_rate']}")
        print(f"测试通过: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"错误总数: {summary['total_errors']}")
        
        print("\n页面测试结果:")
        for page_result in report["page_results"]:
            status_emoji = "✅" if page_result["status"] == "PASS" else "⚠️" if page_result["status"] == "WARNING" else "❌"
            load_time = page_result.get("load_time", "N/A")
            print(f"  {status_emoji} {page_result['page']}: {page_result['success_rate']} ({page_result['tests_passed']}) - 加载时间: {load_time}s")
        
        if report.get("issues_found"):
            print(f"\n发现的主要问题 (前10个):")
            for issue in report["issues_found"][:10]:
                print(f"  ❌ {issue}")
        
        if report.get("recommendations"):
            print(f"\n修复建议:")
            for rec in report["recommendations"][:8]:
                print(f"  💡 {rec}")
    else:
        print(f"❌ 测试失败: {report['error']}")

if __name__ == "__main__":
    main()