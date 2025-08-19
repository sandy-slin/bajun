#!/usr/bin/env python3
"""
Chrome MCP完整前端功能测试
不使用HTTP fallback，直接使用Puppeteer/Chrome进行深度测试
"""

import asyncio
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
    from pyppeteer import launch
    from pyppeteer.page import Page
    from pyppeteer.browser import Browser
except ImportError:
    print("❌ Pyppeteer未安装，正在安装...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyppeteer"], check=True)
    from pyppeteer import launch
    from pyppeteer.page import Page
    from pyppeteer.browser import Browser

class ChromeMCPTester:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.pages: Dict[str, Page] = {}
        self.results = []
        self.screenshots_dir = Path("logs/screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试配置
        self.base_url = "http://localhost:3000"
        self.test_pages = {
            "主页": "/",
            "仪表板": "/dashboard", 
            "板块分析": "/sectors",
            "股票推荐": "/stock-recommendation",
            "板块分析详情": "/sector-analysis"
        }
        
        # API测试配置
        self.api_base_url = "http://localhost:8000"
        self.api_endpoints = {
            "健康检查": "/health",
            "板块数据": "/api/v1/sectors/top",
            "API文档": "/docs"
        }

    async def setup_browser(self):
        """启动Chrome浏览器"""
        print("🚀 启动Chrome浏览器...")
        try:
            self.browser = await launch(
                headless=False,  # 使用可见模式，便于调试
                defaultViewport={'width': 1920, 'height': 1080},
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-web-security',
                    '--allow-running-insecure-content',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            print("✅ Chrome浏览器启动成功")
            return True
        except Exception as e:
            print(f"❌ Chrome浏览器启动失败: {e}")
            return False

    async def test_page_functionality(self, page_name: str, path: str) -> Dict:
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
            "status": "UNKNOWN"
        }
        
        try:
            # 创建新页面
            page = await self.browser.newPage()
            self.pages[page_name] = page
            
            # 监听控制台错误
            console_errors = []
            def handle_console(msg):
                if msg.type in ['error', 'warning']:
                    console_errors.append(f"{msg.type.upper()}: {msg.text}")
                    
            page.on('console', handle_console)
            
            # 监听网络错误
            network_errors = []
            def handle_response(response):
                if response.status >= 400:
                    network_errors.append(f"HTTP {response.status}: {response.url}")
                    
            page.on('response', handle_response)
            
            # 导航到页面并测量性能
            start_time = time.time()
            response = await page.goto(f"{self.base_url}{path}", {
                'waitUntil': 'networkidle2',
                'timeout': 30000
            })
            load_time = time.time() - start_time
            
            page_result["performance"]["load_time"] = round(load_time, 3)
            page_result["performance"]["status_code"] = response.status
            
            # 基础可访问性测试
            page_result["tests"]["页面可访问"] = response.status == 200
            page_result["tests"]["页面加载时间"] = load_time < 5.0
            
            # 等待React应用渲染
            await asyncio.sleep(2)
            
            # 检查页面标题
            title = await page.title()
            page_result["tests"]["页面标题存在"] = bool(title and title.strip())
            page_result["performance"]["title"] = title
            
            # 检查React根元素
            react_root = await page.querySelector('#root')
            page_result["tests"]["React应用已渲染"] = react_root is not None
            
            # 检查页面内容
            content = await page.content()
            page_result["tests"]["页面内容非空"] = len(content.strip()) > 1000
            
            # 特定页面的功能测试
            await self.test_page_specific_features(page, page_name, page_result)
            
            # 检查JavaScript错误
            if console_errors:
                page_result["errors"].extend(console_errors)
                page_result["tests"]["无JavaScript错误"] = False
            else:
                page_result["tests"]["无JavaScript错误"] = True
                
            # 检查网络错误
            if network_errors:
                page_result["errors"].extend(network_errors)
                page_result["tests"]["无网络错误"] = False
            else:
                page_result["tests"]["无网络错误"] = True
            
            # 截图
            screenshot_path = self.screenshots_dir / f"{page_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot({'path': str(screenshot_path), 'fullPage': True})
            page_result["screenshot_path"] = str(screenshot_path)
            
            # 计算总体状态
            passed_tests = sum(1 for result in page_result["tests"].values() if result)
            total_tests = len(page_result["tests"])
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            if success_rate >= 0.8 and not page_result["errors"]:
                page_result["status"] = "PASS"
            elif success_rate >= 0.6:
                page_result["status"] = "WARNING"
            else:
                page_result["status"] = "FAIL"
                
            print(f"   ✅ 测试完成: {passed_tests}/{total_tests} 通过")
            
        except Exception as e:
            error_msg = f"页面测试失败: {str(e)}"
            page_result["errors"].append(error_msg)
            page_result["status"] = "ERROR"
            print(f"   ❌ {error_msg}")
            
        finally:
            if page_name in self.pages:
                await self.pages[page_name].close()
                del self.pages[page_name]
                
        return page_result

    async def test_page_specific_features(self, page: Page, page_name: str, result: Dict):
        """测试特定页面的功能"""
        try:
            if page_name == "主页":
                # 测试主页导航链接
                nav_links = await page.querySelectorAll('nav a, .nav-link, [data-testid="nav-link"]')
                result["tests"]["导航链接存在"] = len(nav_links) > 0
                
                # 测试主要内容区域
                main_content = await page.querySelector('main, .main-content, .content')
                result["tests"]["主要内容区域存在"] = main_content is not None
                
            elif page_name == "仪表板":
                # 测试仪表板卡片
                cards = await page.querySelectorAll('.ant-card, .card, [class*="card"]')
                result["tests"]["仪表板卡片存在"] = len(cards) > 0
                
                # 测试数据加载指示器
                await asyncio.sleep(3)  # 等待数据加载
                loading = await page.querySelectorAll('.ant-spin, .loading, [class*="loading"]')
                result["tests"]["数据已加载"] = len(loading) == 0
                
            elif page_name == "板块分析":
                # 测试板块列表
                await asyncio.sleep(5)  # 等待API数据加载
                sectors = await page.querySelectorAll('.ant-card, .sector-card, [class*="sector"]')
                result["tests"]["板块数据已显示"] = len(sectors) > 0
                
                # 测试刷新按钮
                refresh_btn = await page.querySelector('button[class*="refresh"], .ant-btn')
                result["tests"]["刷新按钮存在"] = refresh_btn is not None
                
            elif page_name == "股票推荐":
                # 测试股票推荐列表
                await asyncio.sleep(5)  # 等待数据加载
                stocks = await page.querySelectorAll('.ant-card, .stock-card, [class*="stock"]')
                result["tests"]["股票推荐数据已显示"] = len(stocks) > 0
                
                # 测试评分显示
                scores = await page.querySelectorAll('[class*="score"], .score')
                result["tests"]["评分数据已显示"] = len(scores) > 0
                
        except Exception as e:
            result["errors"].append(f"特定功能测试失败: {str(e)}")

    async def test_user_interactions(self) -> Dict:
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
            page = await self.browser.newPage()
            
            # 流程1: 主页 -> 仪表板
            print("   测试流程: 主页 -> 仪表板")
            await page.goto(f"{self.base_url}/")
            await asyncio.sleep(2)
            
            # 查找导航链接
            dashboard_link = await page.querySelector('a[href="/dashboard"], a[href*="dashboard"]')
            if dashboard_link:
                await dashboard_link.click()
                await page.waitForNavigation({'waitUntil': 'networkidle2', 'timeout': 10000})
                current_url = page.url
                interaction_result["flows"]["主页到仪表板"] = "/dashboard" in current_url
            else:
                interaction_result["flows"]["主页到仪表板"] = False
                interaction_result["errors"].append("未找到仪表板导航链接")
            
            # 流程2: 仪表板 -> 板块分析
            print("   测试流程: 仪表板 -> 板块分析")
            sectors_link = await page.querySelector('a[href="/sectors"], a[href*="sector"]')
            if sectors_link:
                await sectors_link.click()
                await page.waitForNavigation({'waitUntil': 'networkidle2', 'timeout': 10000})
                current_url = page.url
                interaction_result["flows"]["仪表板到板块分析"] = "/sectors" in current_url or "/sector" in current_url
            else:
                interaction_result["flows"]["仪表板到板块分析"] = False
                interaction_result["errors"].append("未找到板块分析导航链接")
            
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
            
            await page.close()
            
        except Exception as e:
            interaction_result["errors"].append(f"交互测试失败: {str(e)}")
            interaction_result["status"] = "ERROR"
            print(f"   ❌ 交互测试失败: {str(e)}")
            
        return interaction_result

    async def run_comprehensive_test(self):
        """运行完整的Chrome MCP测试"""
        print("🌐 开始Chrome MCP完整前端功能测试")
        print("========================================================================")
        
        # 启动浏览器
        if not await self.setup_browser():
            return {"error": "无法启动Chrome浏览器"}
        
        try:
            # 测试所有页面
            for page_name, path in self.test_pages.items():
                page_result = await self.test_page_functionality(page_name, path)
                self.results.append(page_result)
            
            # 测试用户交互
            interaction_result = await self.test_user_interactions()
            self.results.append(interaction_result)
            
            # 生成综合报告
            report = self.generate_comprehensive_report()
            
            # 保存报告
            report_path = Path("logs/mcp_checks") / f"chrome_mcp_comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 测试报告已保存: {report_path}")
            return report
            
        finally:
            # 清理浏览器
            if self.browser:
                await self.browser.close()
                print("🛑 Chrome浏览器已关闭")

    def generate_comprehensive_report(self) -> Dict:
        """生成综合测试报告"""
        total_tests = 0
        passed_tests = 0
        total_errors = 0
        
        page_results = []
        for result in self.results:
            if "tests" in result:
                page_tests = len(result["tests"])
                page_passed = sum(1 for test_result in result["tests"].values() if test_result)
                total_tests += page_tests
                passed_tests += page_passed
                total_errors += len(result.get("errors", []))
                
                page_results.append({
                    "page": result.get("page", result.get("test_name", "Unknown")),
                    "status": result["status"],
                    "tests_passed": f"{page_passed}/{page_tests}",
                    "success_rate": f"{(page_passed/page_tests*100):.1f}%" if page_tests > 0 else "0%",
                    "errors_count": len(result.get("errors", [])),
                    "load_time": result.get("performance", {}).get("load_time", "N/A")
                })
        
        overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 确定整体状态
        if overall_success_rate >= 90 and total_errors == 0:
            overall_status = "EXCELLENT"
        elif overall_success_rate >= 80 and total_errors <= 2:
            overall_status = "GOOD"
        elif overall_success_rate >= 70:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        return {
            "test_type": "Chrome MCP完整功能测试",
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
            "recommendations": self.generate_recommendations()
        }

    def generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        for result in self.results:
            if result.get("status") in ["FAIL", "ERROR"]:
                page_name = result.get("page", result.get("test_name", "Unknown"))
                recommendations.append(f"修复 {page_name} 页面的问题")
                
                if result.get("errors"):
                    for error in result["errors"][:2]:  # 只显示前2个错误
                        recommendations.append(f"  - 解决错误: {error}")
                        
                if "tests" in result:
                    failed_tests = [test for test, passed in result["tests"].items() if not passed]
                    for test in failed_tests[:2]:  # 只显示前2个失败测试
                        recommendations.append(f"  - 修复测试: {test}")
        
        if not recommendations:
            recommendations.append("所有测试通过，系统运行良好！")
        
        return recommendations

async def main():
    """主函数"""
    tester = ChromeMCPTester()
    report = await tester.run_comprehensive_test()
    
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
            print(f"  {status_emoji} {page_result['page']}: {page_result['success_rate']} ({page_result['tests_passed']})")
    else:
        print(f"❌ 测试失败: {report['error']}")

if __name__ == "__main__":
    asyncio.run(main())