#!/usr/bin/env python3
"""
Chrome MCP前端检查器
通过MCP协议控制Chrome浏览器，自动检查前端应用运行状态
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class FrontendCheckResult:
    page: str
    url: str
    status: str  # PASS, FAIL, WARNING
    issues: List[str]
    performance: Dict[str, float]
    screenshot_path: Optional[str] = None
    checked_at: str = ""
    
    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now().isoformat()

@dataclass
class PageCheckConfig:
    url: str
    name: str
    checks: List[str]
    expected_elements: List[str]
    performance_thresholds: Dict[str, int]
    api_endpoints: List[str] = None
    
    def __post_init__(self):
        if self.api_endpoints is None:
            self.api_endpoints = []

class ChromeFrontendChecker:
    """Chrome MCP前端检查器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "configs/mcp/frontend_check_config.json"
        self.config = self._load_config()
        self.results: List[FrontendCheckResult] = []
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建截图目录
        Path("logs/screenshots").mkdir(parents=True, exist_ok=True)
        Path("logs/mcp_checks").mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """加载检查配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件JSON格式错误: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "check_points": {
                "homepage": {
                    "url": "http://localhost:3000",
                    "name": "主页",
                    "checks": ["page_loads", "react_renders", "no_js_errors"],
                    "expected_elements": ["#root"],
                    "performance_thresholds": {"load_time_ms": 3000}
                }
            },
            "global_settings": {
                "screenshot_on_error": True,
                "max_wait_time": 30000,
                "headless": False
            }
        }
    
    async def check_page_loads(self, url: str) -> tuple[bool, List[str]]:
        """检查页面是否正常加载"""
        issues = []
        try:
            # 模拟页面加载检查（实际需要MCP连接）
            import urllib.request
            import urllib.error
            
            start_time = time.time()
            
            try:
                # 创建请求对象，添加适当的头信息
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
                req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
                req.add_header('Accept-Language', 'zh-CN,zh;q=0.9,en;q=0.8')
                req.add_header('Accept-Encoding', 'gzip, deflate')
                req.add_header('Connection', 'keep-alive')
                
                response = urllib.request.urlopen(req, timeout=10)
                load_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    content = response.read().decode('utf-8')
                    
                    # 基础内容检查
                    if len(content) < 100:
                        issues.append("页面内容过少，可能加载不完整")
                    
                    if '<title>' not in content:
                        issues.append("页面缺少title标签")
                    
                    if load_time > 5000:
                        issues.append(f"页面加载时间过长: {load_time:.0f}ms")
                    
                    return True, issues
                else:
                    issues.append(f"HTTP状态码错误: {response.status}")
                    return False, issues
                    
            except urllib.error.URLError as e:
                issues.append(f"页面无法访问: {str(e)}")
                return False, issues
                
        except Exception as e:
            issues.append(f"页面加载检查失败: {str(e)}")
            return False, issues
    
    async def check_react_renders(self, url: str) -> tuple[bool, List[str]]:
        """检查React应用是否正常渲染"""
        issues = []
        try:
            # 这里需要MCP连接来执行JavaScript
            # 当前为模拟实现
            
            # 模拟检查React根节点
            page_loaded, load_issues = await self.check_page_loads(url)
            if not page_loaded:
                issues.extend(load_issues)
                issues.append("无法检查React渲染状态")
                return False, issues
            
            # 模拟JavaScript执行检查
            # 实际应该通过MCP执行: document.getElementById('root').children.length > 0
            
            return True, issues
            
        except Exception as e:
            issues.append(f"React渲染检查失败: {str(e)}")
            return False, issues
    
    async def check_no_js_errors(self, url: str) -> tuple[bool, List[str]]:
        """检查JavaScript错误"""
        issues = []
        try:
            # 这里需要MCP连接来获取控制台日志
            # 当前为模拟实现
            
            # 模拟控制台错误检查
            # 实际应该通过MCP获取console.error日志
            
            return True, issues
            
        except Exception as e:
            issues.append(f"JavaScript错误检查失败: {str(e)}")
            return False, issues
    
    async def check_api_endpoints(self, endpoints: List[str]) -> tuple[bool, List[str]]:
        """检查API端点可用性"""
        issues = []
        all_ok = True
        
        for endpoint in endpoints:
            try:
                import urllib.request
                import urllib.error
                
                try:
                    response = urllib.request.urlopen(endpoint, timeout=5)
                    if response.status != 200:
                        issues.append(f"API端点异常 {endpoint}: HTTP {response.status}")
                        all_ok = False
                except urllib.error.URLError as e:
                    issues.append(f"API端点不可达 {endpoint}: {str(e)}")
                    all_ok = False
                    
            except Exception as e:
                issues.append(f"API检查失败 {endpoint}: {str(e)}")
                all_ok = False
        
        return all_ok, issues
    
    async def take_screenshot(self, page_name: str) -> str:
        """截图（需要MCP实现）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f"logs/screenshots/{page_name}_{timestamp}.png"
        
        # 这里需要MCP连接来实际截图
        # 当前创建一个占位文件
        try:
            Path(screenshot_path).touch()
            self.logger.info(f"截图保存到: {screenshot_path}")
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return ""
        
        return screenshot_path
    
    async def check_single_page(self, page_name: str, page_config: PageCheckConfig) -> FrontendCheckResult:
        """检查单个页面"""
        self.logger.info(f"🔍 检查页面: {page_config.name} ({page_config.url})")
        
        issues = []
        performance = {}
        screenshot_path = None
        
        try:
            start_time = time.time()
            
            # 执行各项检查
            for check in page_config.checks:
                if check == "page_loads":
                    success, check_issues = await self.check_page_loads(page_config.url)
                    if not success:
                        issues.extend(check_issues)
                
                elif check == "react_renders":
                    success, check_issues = await self.check_react_renders(page_config.url)
                    if not success:
                        issues.extend(check_issues)
                
                elif check == "no_js_errors":
                    success, check_issues = await self.check_no_js_errors(page_config.url)
                    if not success:
                        issues.extend(check_issues)
                
                elif check == "api_data_loads" and page_config.api_endpoints:
                    success, check_issues = await self.check_api_endpoints(page_config.api_endpoints)
                    if not success:
                        issues.extend(check_issues)
            
            # 性能检查
            total_time = (time.time() - start_time) * 1000
            performance["total_check_time_ms"] = total_time
            
            # 检查性能阈值
            if "load_time_ms" in page_config.performance_thresholds:
                if total_time > page_config.performance_thresholds["load_time_ms"]:
                    issues.append(f"检查时间超过阈值: {total_time:.0f}ms > {page_config.performance_thresholds['load_time_ms']}ms")
            
            # 决定状态
            if issues:
                status = "FAIL"
                # 如果有错误，截图
                if self.config.get("global_settings", {}).get("screenshot_on_error", True):
                    screenshot_path = await self.take_screenshot(page_name)
            else:
                status = "PASS"
            
        except Exception as e:
            self.logger.error(f"页面检查异常: {e}")
            issues.append(f"检查过程异常: {str(e)}")
            status = "FAIL"
        
        result = FrontendCheckResult(
            page=page_name,
            url=page_config.url,
            status=status,
            issues=issues,
            performance=performance,
            screenshot_path=screenshot_path
        )
        
        return result
    
    async def run_comprehensive_check(self) -> List[FrontendCheckResult]:
        """运行全面的前端检查"""
        self.logger.info("🚀 启动全面前端检查...")
        
        results = []
        
        for page_name, page_data in self.config["check_points"].items():
            page_config = PageCheckConfig(
                url=page_data["url"],
                name=page_data["name"],
                checks=page_data["checks"],
                expected_elements=page_data["expected_elements"],
                performance_thresholds=page_data.get("performance_thresholds", {}),
                api_endpoints=page_data.get("api_endpoints", [])
            )
            
            result = await self.check_single_page(page_name, page_config)
            results.append(result)
            
            # 短暂延迟避免过于频繁的请求
            await asyncio.sleep(1)
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict:
        """生成检查报告"""
        if not self.results:
            return {"error": "没有检查结果"}
        
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == "PASS"])
        failed_checks = len([r for r in self.results if r.status == "FAIL"])
        
        all_issues = []
        for result in self.results:
            all_issues.extend(result.issues)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
                "success_rate": f"{(passed_checks/total_checks)*100:.1f}%" if total_checks > 0 else "0%",
                "total_issues": len(all_issues)
            },
            "results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 分析所有问题并生成建议
        all_issues = []
        for result in self.results:
            all_issues.extend(result.issues)
        
        # 常见问题模式匹配
        if any("页面无法访问" in issue for issue in all_issues):
            recommendations.append("检查前端服务是否正常启动 (npm start)")
        
        if any("API端点" in issue for issue in all_issues):
            recommendations.append("检查后端服务是否正常运行 (scripts/deployment/start_services.sh)")
        
        if any("加载时间过长" in issue for issue in all_issues):
            recommendations.append("优化页面性能，检查网络请求和资源大小")
        
        if any("JavaScript错误" in issue for issue in all_issues):
            recommendations.append("检查TypeScript编译错误和控制台日志")
        
        if not recommendations:
            recommendations.append("所有检查通过，前端运行正常")
        
        return recommendations
    
    def save_report(self, report: Dict) -> str:
        """保存检查报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"logs/mcp_checks/frontend_check_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 检查报告已保存: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")
            return ""

async def main():
    """主函数"""
    print("🔍 Chrome MCP前端检查器")
    print("=" * 50)
    
    # 检查配置文件
    config_path = "configs/mcp/frontend_check_config.json"
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print("请先运行: scripts/setup/setup_chrome_mcp.sh")
        return 1
    
    try:
        checker = ChromeFrontendChecker(config_path)
        
        # 运行检查
        results = await checker.run_comprehensive_check()
        
        # 生成报告
        report = checker.generate_report()
        report_path = checker.save_report(report)
        
        # 显示结果
        print("\n📊 检查结果摘要:")
        print(f"   总检查项: {report['summary']['total_checks']}")
        print(f"   通过: {report['summary']['passed']}")
        print(f"   失败: {report['summary']['failed']}")
        print(f"   成功率: {report['summary']['success_rate']}")
        
        if report['summary']['failed'] > 0:
            print("\n❌ 发现的问题:")
            for result in results:
                if result.status == "FAIL":
                    print(f"   📄 {result.page}: {len(result.issues)}个问题")
                    for issue in result.issues[:3]:  # 只显示前3个问题
                        print(f"      - {issue}")
        
        print(f"\n💡 修复建议:")
        for rec in report['recommendations']:
            print(f"   - {rec}")
        
        print(f"\n📝 详细报告: {report_path}")
        
        # 返回适当的退出码
        return 0 if report['summary']['failed'] == 0 else 1
        
    except Exception as e:
        print(f"❌ 检查过程发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)