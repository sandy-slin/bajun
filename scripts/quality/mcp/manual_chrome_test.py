#!/usr/bin/env python3
"""
手动Chrome MCP前端功能测试
使用AppleScript控制Chrome进行完整的前端功能验证
"""

import json
import time
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
import re

class ManualChromeTester:
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.api_base_url = "http://localhost:8000"
        self.results = []
        
        # 测试页面配置
        self.test_pages = {
            "主页": "/",
            "仪表板": "/dashboard", 
            "板块分析": "/sectors",
            "股票推荐": "/stock-recommendation"
        }
        
        # 创建日志目录
        Path("logs/mcp_checks").mkdir(parents=True, exist_ok=True)

    def test_page_http_details(self, page_name: str, path: str) -> dict:
        """深度HTTP测试单个页面"""
        print(f"\n📄 深度测试页面: {page_name} ({path})")
        
        result = {
            "page": page_name,
            "url": f"{self.base_url}{path}",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "performance": {},
            "content_analysis": {},
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            # HTTP请求测试
            start_time = time.time()
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                load_time = time.time() - start_time
                content = response.read().decode('utf-8')
                status_code = response.getcode()
                headers = dict(response.headers)
                
                result["performance"]["load_time"] = round(load_time, 3)
                result["performance"]["status_code"] = status_code
                result["performance"]["content_length"] = len(content)
                result["performance"]["response_headers"] = {k: v for k, v in headers.items() if k.lower() in ['content-type', 'content-length', 'cache-control']}
                
                # 基础测试
                result["tests"]["HTTP状态正常"] = status_code == 200
                result["tests"]["页面加载时间"] = load_time < 5.0
                result["tests"]["响应内容非空"] = len(content) > 1000
                
                # HTML内容分析
                self.analyze_html_content(content, result)
                
                # 页面特定测试
                self.test_page_specific_content(page_name, content, result)
                
                print(f"   ✅ HTTP测试完成 - 状态: {status_code}, 加载时间: {load_time:.3f}s")
                
        except Exception as e:
            error_msg = f"HTTP测试失败: {str(e)}"
            result["errors"].append(error_msg)
            result["tests"]["HTTP状态正常"] = False
            print(f"   ❌ {error_msg}")
        
        # 计算状态
        if result["tests"]:
            passed_tests = sum(1 for test_result in result["tests"].values() if test_result)
            total_tests = len(result["tests"])
            success_rate = passed_tests / total_tests
            
            if success_rate >= 0.8 and not result["errors"]:
                result["status"] = "PASS"
            elif success_rate >= 0.6:
                result["status"] = "WARNING"  
            else:
                result["status"] = "FAIL"
        
        return result

    def analyze_html_content(self, content: str, result: dict):
        """分析HTML内容"""
        content_analysis = result["content_analysis"]
        
        # React应用检测
        react_patterns = [
            r'<div[^>]*id=["\']root["\'][^>]*>',
            r'react',
            r'ReactDOM',
            r'__REACT_DEVTOOLS_GLOBAL_HOOK__'
        ]
        
        react_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in react_patterns)
        result["tests"]["React应用存在"] = react_found
        content_analysis["react_detected"] = react_found
        
        # 检查必要的HTML结构
        result["tests"]["HTML文档结构完整"] = all([
            '<html' in content.lower(),
            '<head' in content.lower(),
            '<body' in content.lower(),
            '</html>' in content.lower()
        ])
        
        # 检查CSS和JS资源
        css_count = len(re.findall(r'<link[^>]*rel=["\']stylesheet["\']', content, re.IGNORECASE))
        js_count = len(re.findall(r'<script[^>]*src=', content, re.IGNORECASE))
        
        result["tests"]["CSS资源已加载"] = css_count > 0
        result["tests"]["JavaScript资源已加载"] = js_count > 0
        
        content_analysis["css_links_count"] = css_count
        content_analysis["js_scripts_count"] = js_count
        
        # 检查meta标签
        meta_count = len(re.findall(r'<meta[^>]*>', content, re.IGNORECASE))
        result["tests"]["Meta标签存在"] = meta_count > 0
        content_analysis["meta_tags_count"] = meta_count
        
        # 检查title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            result["tests"]["页面标题存在"] = bool(title)
            content_analysis["page_title"] = title
        else:
            result["tests"]["页面标题存在"] = False
            
        # 检查Antd组件
        antd_patterns = [
            r'ant-',
            r'antd',
            r'__antd__'
        ]
        antd_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in antd_patterns)
        result["tests"]["Antd UI库加载"] = antd_found
        content_analysis["antd_detected"] = antd_found

    def test_page_specific_content(self, page_name: str, content: str, result: dict):
        """测试页面特定内容"""
        if page_name == "主页":
            # 检查导航元素
            nav_patterns = [
                r'<nav[^>]*>',
                r'navigation',
                r'menu',
                r'nav-'
            ]
            nav_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in nav_patterns)
            result["tests"]["导航结构存在"] = nav_found
            
        elif page_name == "仪表板":
            # 检查仪表板特有元素
            dashboard_patterns = [
                r'dashboard',
                r'仪表板',
                r'card',
                r'统计',
                r'overview'
            ]
            dashboard_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in dashboard_patterns)
            result["tests"]["仪表板内容存在"] = dashboard_found
            
        elif page_name == "板块分析":
            # 检查板块分析相关内容
            sector_patterns = [
                r'sector',
                r'板块',
                r'行业',
                r'分析',
                r'评分'
            ]
            sector_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in sector_patterns)
            result["tests"]["板块分析内容存在"] = sector_found
            
        elif page_name == "股票推荐":
            # 检查股票推荐相关内容
            stock_patterns = [
                r'stock',
                r'股票',
                r'推荐',
                r'recommendation',
                r'评级'
            ]
            stock_found = any(re.search(pattern, content, re.IGNORECASE) for pattern in stock_patterns)
            result["tests"]["股票推荐内容存在"] = stock_found

    def test_api_functionality(self) -> dict:
        """测试API功能"""
        print("\n🔧 测试API功能...")
        
        api_result = {
            "test_name": "API功能测试",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "api_responses": {},
            "status": "UNKNOWN"
        }
        
        api_endpoints = {
            "健康检查": "/health",
            "板块数据": "/api/v1/sectors/top?months=6&limit=5",
            "API文档": "/docs"
        }
        
        for endpoint_name, path in api_endpoints.items():
            try:
                start_time = time.time()
                req = urllib.request.Request(
                    f"{self.api_base_url}{path}",
                    headers={'Accept': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    response_time = time.time() - start_time
                    status_code = response.getcode()
                    content = response.read().decode('utf-8')
                    
                    api_result["tests"][f"{endpoint_name}_可访问"] = status_code == 200
                    api_result["tests"][f"{endpoint_name}_响应时间"] = response_time < 3.0
                    
                    # 如果是JSON API，验证JSON格式
                    if path.startswith('/api/') or path == '/health':
                        try:
                            json_data = json.loads(content)
                            api_result["tests"][f"{endpoint_name}_JSON格式"] = True
                            api_result["api_responses"][endpoint_name] = {
                                "status_code": status_code,
                                "response_time": round(response_time, 3),
                                "data_keys": list(json_data.keys()) if isinstance(json_data, dict) else "array"
                            }
                        except json.JSONDecodeError:
                            api_result["tests"][f"{endpoint_name}_JSON格式"] = False
                            api_result["errors"].append(f"{endpoint_name} 返回非JSON格式")
                    
                    print(f"   ✅ {endpoint_name}: {status_code} - {response_time:.3f}s")
                    
            except Exception as e:
                api_result["tests"][f"{endpoint_name}_可访问"] = False
                api_result["errors"].append(f"{endpoint_name} 测试失败: {str(e)}")
                print(f"   ❌ {endpoint_name}: 失败 - {str(e)}")
        
        # 计算API测试状态
        if api_result["tests"]:
            passed_tests = sum(1 for test_result in api_result["tests"].values() if test_result)
            total_tests = len(api_result["tests"])
            success_rate = passed_tests / total_tests
            
            if success_rate >= 0.8:
                api_result["status"] = "PASS"
            elif success_rate >= 0.6:
                api_result["status"] = "WARNING"
            else:
                api_result["status"] = "FAIL"
        
        return api_result

    def test_javascript_functionality(self) -> dict:
        """使用AppleScript测试JavaScript功能"""
        print("\n🔍 测试JavaScript和用户交互...")
        
        js_result = {
            "test_name": "JavaScript和交互测试",
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "status": "UNKNOWN"
        }
        
        try:
            # 使用AppleScript控制Chrome
            applescript = '''
            tell application "Google Chrome"
                activate
                set myTab to make new tab at end of tabs of window 1
                set URL of myTab to "http://localhost:3000"
                delay 3
                
                -- 检查页面是否加载完成
                set pageTitle to title of myTab
                
                -- 尝试访问不同页面
                set URL of myTab to "http://localhost:3000/dashboard"
                delay 3
                set dashboardTitle to title of myTab
                
                set URL of myTab to "http://localhost:3000/sectors"
                delay 5
                set sectorsTitle to title of myTab
                
                close myTab
                
                return pageTitle & "|||" & dashboardTitle & "|||" & sectorsTitle
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', applescript], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                titles = result.stdout.strip().split('|||')
                js_result["tests"]["页面标题获取成功"] = len(titles) >= 3
                js_result["tests"]["多页面导航成功"] = all(title.strip() for title in titles)
                js_result["tests"]["Chrome自动化成功"] = True
                
                print(f"   ✅ 页面标题: {titles}")
                
            else:
                js_result["tests"]["Chrome自动化成功"] = False
                js_result["errors"].append(f"AppleScript执行失败: {result.stderr}")
                print(f"   ❌ AppleScript失败: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            js_result["tests"]["Chrome自动化成功"] = False
            js_result["errors"].append("Chrome自动化超时")
            print("   ❌ Chrome自动化超时")
        except Exception as e:
            js_result["tests"]["Chrome自动化成功"] = False
            js_result["errors"].append(f"Chrome自动化异常: {str(e)}")
            print(f"   ❌ Chrome自动化异常: {str(e)}")
        
        # 计算JavaScript测试状态
        if js_result["tests"]:
            passed_tests = sum(1 for test_result in js_result["tests"].values() if test_result)
            total_tests = len(js_result["tests"])
            success_rate = passed_tests / total_tests
            
            if success_rate >= 0.8:
                js_result["status"] = "PASS"
            elif success_rate >= 0.6:
                js_result["status"] = "WARNING"
            else:
                js_result["status"] = "FAIL"
        
        return js_result

    def run_comprehensive_test(self):
        """运行完整测试"""
        print("🌐 开始手动Chrome MCP完整前端功能测试")
        print("========================================================================")
        
        # 测试所有页面
        for page_name, path in self.test_pages.items():
            page_result = self.test_page_http_details(page_name, path)
            self.results.append(page_result)
        
        # 测试API功能
        api_result = self.test_api_functionality()
        self.results.append(api_result)
        
        # 测试JavaScript和交互
        js_result = self.test_javascript_functionality()
        self.results.append(js_result)
        
        # 生成综合报告
        report = self.generate_comprehensive_report()
        
        # 保存报告
        report_path = Path("logs/mcp_checks") / f"manual_chrome_comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 测试报告已保存: {report_path}")
        return report

    def generate_comprehensive_report(self) -> dict:
        """生成综合报告"""
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
                
                # 收集问题
                for error in result.get("errors", []):
                    all_issues.append(f"{result.get('page', result.get('test_name', 'Unknown'))}: {error}")
                
                failed_tests = [test for test, passed in result.get("tests", {}).items() if not passed]
                for test in failed_tests:
                    all_issues.append(f"{result.get('page', result.get('test_name', 'Unknown'))}: 测试失败 - {test}")
                
                page_results.append({
                    "page": result.get("page", result.get("test_name", "Unknown")),
                    "status": result["status"],
                    "tests_passed": f"{page_passed}/{page_tests}",
                    "success_rate": f"{(page_passed/page_tests*100):.1f}%" if page_tests > 0 else "0%",
                    "errors_count": page_errors,
                    "load_time": result.get("performance", {}).get("load_time", "N/A")
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
            "test_type": "手动Chrome MCP完整功能测试",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_status": overall_status,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{overall_success_rate:.1f}%",
                "total_errors": total_errors,
                "pages_tested": len([r for r in self.results if "page" in r])
            },
            "page_results": page_results,
            "detailed_results": self.results,
            "issues_found": all_issues,
            "recommendations": self.generate_recommendations()
        }

    def generate_recommendations(self) -> list:
        """生成修复建议"""
        recommendations = []
        
        # 分析所有问题
        all_issues = []
        for result in self.results:
            if result.get("status") in ["FAIL", "ERROR", "WARNING"]:
                all_issues.extend(result.get("errors", []))
                if "tests" in result:
                    failed_tests = [test for test, passed in result["tests"].items() if not passed]
                    all_issues.extend([f"测试失败: {test}" for test in failed_tests])
        
        # 生成分类建议
        if any("HTTP" in issue for issue in all_issues):
            recommendations.append("🔧 修复HTTP连接和服务器响应问题")
        
        if any("React" in issue or "应用" in issue for issue in all_issues):
            recommendations.append("⚛️ 检查React应用渲染和组件加载")
        
        if any("资源" in issue or "CSS" in issue or "JavaScript" in issue for issue in all_issues):
            recommendations.append("📦 优化静态资源加载（CSS/JS文件）")
        
        if any("API" in issue or "JSON" in issue for issue in all_issues):
            recommendations.append("🔌 修复后端API接口和数据格式")
        
        if any("加载时间" in issue or "响应时间" in issue for issue in all_issues):
            recommendations.append("⚡ 优化页面和API响应性能")
        
        if not all_issues:
            recommendations.append("✅ 所有测试通过，系统运行良好！")
        
        return recommendations

def main():
    """主函数"""
    tester = ManualChromeTester()
    report = tester.run_comprehensive_test()
    
    # 打印详细报告
    print("\n" + "="*80)
    print("📊 手动Chrome MCP测试完成")
    print("="*80)
    
    summary = report["summary"]
    print(f"整体状态: {summary['overall_status']}")
    print(f"成功率: {summary['success_rate']}")
    print(f"测试通过: {summary['passed_tests']}/{summary['total_tests']}")
    print(f"错误总数: {summary['total_errors']}")
    
    print("\n页面测试结果:")
    for page_result in report["page_results"]:
        status_emoji = "✅" if page_result["status"] == "PASS" else "⚠️" if page_result["status"] == "WARNING" else "❌"
        load_time = page_result.get("load_time", "N/A")
        print(f"  {status_emoji} {page_result['page']}: {page_result['success_rate']} ({page_result['tests_passed']}) - 加载: {load_time}s")
    
    if report.get("issues_found"):
        print(f"\n发现的问题:")
        for issue in report["issues_found"][:15]:
            print(f"  ❌ {issue}")
    
    if report.get("recommendations"):
        print(f"\n修复建议:")
        for rec in report["recommendations"]:
            print(f"  💡 {rec}")

if __name__ == "__main__":
    main()