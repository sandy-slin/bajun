#!/bin/bash

# A股智能交易决策平台 - 前端UI设计检查脚本
# 基于YAML配置的UI符合性检查，包含浏览器页面检查

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径，然后切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🎨 启动前端UI设计检查...${NC}"
echo "========================================================"
echo "📍 项目根目录: $PROJECT_ROOT"

# 创建日志目录
mkdir -p logs

# 检查前置条件
echo -e "${YELLOW}🔍 检查前置条件...${NC}"

if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Python虚拟环境不存在${NC}"
    exit 1
fi

if [ ! -d "configs/ui_checker" ]; then
    echo -e "${RED}❌ UI检查器配置不存在${NC}"
    exit 1
fi

if [ ! -f "configs/ui_checker/ui_checker.py" ]; then
    echo -e "${RED}❌ UI检查器脚本不存在${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 前置条件检查通过${NC}"

# 激活虚拟环境
source venv/bin/activate

# 检查前端服务是否运行
echo -e "${YELLOW}🌐 检查前端服务状态...${NC}"

if ! curl -s -I http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ 前端服务未运行，尝试启动...${NC}"
    
    # 启动前端服务
    scripts/deployment/start_services.sh > logs/ui_check_startup.log 2>&1 &
    STARTUP_PID=$!
    
    # 等待服务启动
    echo -e "${YELLOW}⏳ 等待服务启动 (最多60秒)...${NC}"
    for i in {1..12}; do
        if curl -s -I http://localhost:3000 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ 前端服务已启动${NC}"
            break
        elif [ $i -eq 12 ]; then
            echo -e "${RED}❌ 前端服务启动失败${NC}"
            exit 1
        else
            echo "   等待中... ($i/12)"
            sleep 5
        fi
    done
    
    SERVICES_STARTED=true
else
    echo -e "${GREEN}✅ 前端服务已运行${NC}"
    SERVICES_STARTED=false
fi

# 安装页面检查依赖
echo -e "${YELLOW}📦 检查页面检查依赖...${NC}"
pip list | grep -E "(requests|beautifulsoup4)" > /dev/null 2>&1 || {
    echo "安装页面检查依赖..."
    pip install requests beautifulsoup4 > logs/ui_check_deps.log 2>&1
}

# 运行基础UI检查器测试
echo -e "${YELLOW}🎨 运行基础UI检查器测试...${NC}"
cd configs/ui_checker
python ui_checker.py > ../../logs/basic_ui_check.log 2>&1
UI_CHECKER_RESULT=$?
cd ../..

if [ $UI_CHECKER_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ 基础UI检查器测试通过${NC}"
else
    echo -e "${YELLOW}⚠️ 基础UI检查器测试发现问题${NC}"
fi

# 创建前端页面检查脚本
cat > scripts/quality/frontend_page_analyzer.py << 'EOF'
#!/usr/bin/env python3
"""
前端页面UI检查器
基于实际运行的前端应用进行UI符合性检查
"""

import requests
import json
import re
import sys
import os
from urllib.parse import urljoin
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class PageAnalysisResult:
    """页面分析结果"""
    url: str
    title: str
    status_code: int
    load_time: float
    component_count: int
    ui_issues: List[Dict[str, Any]]
    accessibility_score: float
    performance_score: float
    
class FrontendPageAnalyzer:
    """前端页面分析器"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.session = requests.Session()
            
    def analyze_page(self, path: str = "/") -> PageAnalysisResult:
        """分析指定页面"""
        url = urljoin(self.base_url, path)
        print(f"🔍 分析页面: {url}")
        
        start_time = datetime.now()
        
        try:
            response = self.session.get(url, timeout=10)
            load_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code != 200:
                print(f"❌ 页面访问失败: {response.status_code}")
                return PageAnalysisResult(
                    url=url,
                    title="",
                    status_code=response.status_code,
                    load_time=load_time,
                    component_count=0,
                    ui_issues=[],
                    accessibility_score=0.0,
                    performance_score=0.0
                )
                
            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ""
            
            # 分析页面结构
            ui_issues = self._analyze_page_structure(soup, url)
            component_count = self._count_components(soup)
            accessibility_score = self._calculate_accessibility_score(soup)
            performance_score = self._calculate_performance_score(load_time, response)
            
            return PageAnalysisResult(
                url=url,
                title=title,
                status_code=response.status_code,
                load_time=load_time,
                component_count=component_count,
                ui_issues=ui_issues,
                accessibility_score=accessibility_score,
                performance_score=performance_score
            )
            
        except Exception as e:
            print(f"❌ 页面分析失败: {e}")
            return PageAnalysisResult(
                url=url,
                title="",
                status_code=0,
                load_time=0.0,
                component_count=0,
                ui_issues=[{"error": str(e)}],
                accessibility_score=0.0,
                performance_score=0.0
            )
            
    def _analyze_page_structure(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """分析页面结构并进行UI检查"""
        issues = []
        
        # 检查React应用是否正确加载
        react_root = soup.find(id='root')
        if not react_root:
            issues.append({
                "type": "structure_error",
                "severity": "error",
                "message": "React应用根节点 #root 未找到",
                "component": "page_structure"
            })
        elif not react_root.get_text(strip=True):
            issues.append({
                "type": "structure_error", 
                "severity": "error",
                "message": "React应用未正确渲染，#root节点为空",
                "component": "page_structure"
            })
            
        # 检查是否有JavaScript错误（通过控制台日志API无法直接获取，改为检查基本结构）
        scripts = soup.find_all('script')
        if len(scripts) == 0:
            issues.append({
                "type": "resource_missing",
                "severity": "warning", 
                "message": "页面缺少JavaScript文件",
                "component": "page_resources"
            })
            
        # 检查CSS资源
        css_links = soup.find_all('link', rel='stylesheet')
        if len(css_links) == 0:
            issues.append({
                "type": "resource_missing",
                "severity": "warning",
                "message": "页面缺少CSS样式文件", 
                "component": "page_resources"
            })
            
        # 检查基本可访问性
        if not soup.find('title') or not soup.title.string.strip():
            issues.append({
                "type": "accessibility_error",
                "severity": "error",
                "message": "页面缺少有意义的title",
                "component": "page_head"
            })
            
        # 检查语言属性
        html_tag = soup.find('html')
        if not html_tag or not html_tag.get('lang'):
            issues.append({
                "type": "accessibility_error",
                "severity": "warning",
                "message": "html标签缺少lang属性",
                "component": "page_structure"
            })
            
        # 检查meta标签
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        if not viewport_meta:
            issues.append({
                "type": "responsive_error",
                "severity": "warning",
                "message": "缺少viewport meta标签",
                "component": "page_head"
            })
            
        return issues
        
    def _count_components(self, soup: BeautifulSoup) -> int:
        """统计页面中的组件数量"""
        # 基于常见的React/HTML结构模式统计
        components = 0
        
        # 统计可能的卡片组件
        components += len(soup.find_all(['div', 'article'], class_=re.compile(r'card', re.I)))
        
        # 统计表单组件
        components += len(soup.find_all(['form', 'div'], class_=re.compile(r'form', re.I)))
        
        # 统计按钮组件
        components += len(soup.find_all(['button', 'a'], class_=re.compile(r'btn|button', re.I)))
        
        # 统计输入组件
        components += len(soup.find_all(['input', 'select', 'textarea']))
        
        return components
        
    def _calculate_accessibility_score(self, soup: BeautifulSoup) -> float:
        """计算可访问性评分"""
        score = 100.0
        
        # 检查标题层次
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if not headings:
            score -= 20
            
        # 检查图片alt属性
        images = soup.find_all('img')
        images_without_alt = [img for img in images if not img.get('alt')]
        if images_without_alt:
            score -= min(30, len(images_without_alt) * 5)
            
        # 检查表单标签
        inputs = soup.find_all(['input', 'select', 'textarea'])
        for inp in inputs:
            if inp.get('type') not in ['hidden', 'submit'] and not (inp.get('aria-label') or inp.get('placeholder')):
                score -= 5
                
        # 检查按钮文本
        buttons = soup.find_all('button')
        for btn in buttons:
            if not btn.get_text(strip=True) and not btn.get('aria-label'):
                score -= 5
                
        return max(0.0, score)
        
    def _calculate_performance_score(self, load_time: float, response: requests.Response) -> float:
        """计算性能评分"""
        score = 100.0
        
        # 加载时间评分
        if load_time > 3.0:
            score -= 40
        elif load_time > 2.0:
            score -= 20
        elif load_time > 1.0:
            score -= 10
            
        # 响应大小评分
        content_length = len(response.content)
        if content_length > 1024 * 1024:  # 1MB
            score -= 20
        elif content_length > 512 * 1024:  # 512KB
            score -= 10
            
        return max(0.0, score)
        
    def analyze_multiple_pages(self) -> Dict[str, PageAnalysisResult]:
        """分析多个页面"""
        pages_to_check = [
            "/",
        ]
        
        results = {}
        
        for page_path in pages_to_check:
            try:
                result = self.analyze_page(page_path)
                results[page_path] = result
                print(f"✅ 页面分析完成: {page_path}")
            except Exception as e:
                print(f"❌ 页面分析失败: {page_path} - {e}")
                results[page_path] = None
                
        return results
        
    def generate_report(self, results: Dict[str, PageAnalysisResult]) -> Dict[str, Any]:
        """生成综合报告"""
        total_pages = len(results)
        successful_pages = len([r for r in results.values() if r and r.status_code == 200])
        failed_pages = total_pages - successful_pages
        
        total_issues = sum(len(r.ui_issues) for r in results.values() if r)
        error_issues = sum(len([issue for issue in r.ui_issues if issue.get('severity') == 'error']) 
                          for r in results.values() if r)
        warning_issues = total_issues - error_issues
        
        avg_accessibility = sum(r.accessibility_score for r in results.values() if r) / max(successful_pages, 1)
        avg_performance = sum(r.performance_score for r in results.values() if r) / max(successful_pages, 1)
        avg_load_time = sum(r.load_time for r in results.values() if r) / max(successful_pages, 1)
        
        report = {
            "summary": {
                "total_pages": total_pages,
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "total_issues": total_issues,
                "error_count": error_issues,
                "warning_count": warning_issues,
                "avg_accessibility_score": f"{avg_accessibility:.1f}%",
                "avg_performance_score": f"{avg_performance:.1f}%",
                "avg_load_time": f"{avg_load_time:.2f}s",
                "overall_status": "PASS" if error_issues == 0 and failed_pages == 0 else "FAIL"
            },
            "page_results": {path: asdict(result) if result else None for path, result in results.items()},
            "timestamp": datetime.now().isoformat()
        }
        
        return report

def main():
    """主函数"""
    print("🎨 启动前端UI检查器...")
    
    analyzer = FrontendPageAnalyzer()
    
    # 分析多个页面
    results = analyzer.analyze_multiple_pages()
    
    # 生成报告
    report = analyzer.generate_report(results)
    
    # 保存报告
    project_root = Path(__file__).parent.parent.parent
    report_file = project_root / 'logs' / 'frontend_ui_check_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    # 打印摘要
    summary = report["summary"]
    print("\n📊 前端UI检查摘要:")
    print(f"   总页面数: {summary['total_pages']}")
    print(f"   成功页面: {summary['successful_pages']}")
    print(f"   失败页面: {summary['failed_pages']}")
    print(f"   总问题数: {summary['total_issues']}")
    print(f"   错误数量: {summary['error_count']}")
    print(f"   警告数量: {summary['warning_count']}")
    print(f"   平均可访问性评分: {summary['avg_accessibility_score']}")
    print(f"   平均性能评分: {summary['avg_performance_score']}")
    print(f"   平均加载时间: {summary['avg_load_time']}")
    print(f"   整体状态: {summary['overall_status']}")
    
    # 输出详细问题
    if summary['total_issues'] > 0:
        print("\n🔍 发现的问题:")
        for page_path, result in results.items():
            if result and result.ui_issues:
                print(f"\n   页面 {page_path}:")
                for issue in result.ui_issues:
                    severity_color = '\033[0;31m' if issue.get('severity') == 'error' else '\033[1;33m'
                    print(f"     {severity_color}[{issue.get('severity', 'unknown').upper()}]\033[0m {issue.get('message', '')}")
    
    print(f"\n📝 详细报告已保存至: {report_file}")
    
    # 返回状态码
    return 0 if summary['overall_status'] == 'PASS' else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
EOF

# 赋予执行权限
chmod +x scripts/quality/frontend_page_analyzer.py

# 运行前端UI检查
echo -e "${YELLOW}🎨 执行前端页面检查...${NC}"
python scripts/quality/frontend_page_analyzer.py > logs/frontend_ui_check_output.log 2>&1
PAGE_CHECK_RESULT=$?

# 检查结果
if [ $PAGE_CHECK_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ 前端页面检查通过${NC}"
    UI_CHECK_RESULT="PASS"
else
    echo -e "${YELLOW}⚠️ 前端页面检查发现问题${NC}"
    UI_CHECK_RESULT="FAIL_WITH_ISSUES"
fi

# 显示检查结果摘要
if [ -f "logs/frontend_ui_check_report.json" ]; then
    echo -e "${BLUE}📊 前端UI检查结果摘要:${NC}"
    
    # 使用Python解析JSON并显示摘要
    python -c "
import json
try:
    with open('logs/frontend_ui_check_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    summary = report['summary']
    print(f'   总页面数: {summary[\"total_pages\"]}')
    print(f'   成功页面: {summary[\"successful_pages\"]}')
    print(f'   总问题数: {summary[\"total_issues\"]}')
    print(f'   错误数量: {summary[\"error_count\"]}')
    print(f'   警告数量: {summary[\"warning_count\"]}')
    print(f'   可访问性: {summary[\"avg_accessibility_score\"]}')
    print(f'   性能评分: {summary[\"avg_performance_score\"]}')
    print(f'   加载时间: {summary[\"avg_load_time\"]}')
    print(f'   整体状态: {summary[\"overall_status\"]}')
except Exception as e:
    print(f'无法解析报告: {e}')
"
fi

# 清理：如果是脚本启动的服务，则停止服务
if [ "$SERVICES_STARTED" = true ]; then
    echo -e "${YELLOW}🛑 停止测试服务...${NC}"
    scripts/deployment/stop_services.sh > /dev/null 2>&1
fi

echo -e "${BLUE}📋 检查完成!${NC}"
echo "========================================================"
echo -e "${BLUE}📝 日志文件:${NC}"
echo "   基础UI检查: logs/basic_ui_check.log"
echo "   页面检查输出: logs/frontend_ui_check_output.log"
echo "   详细报告: logs/frontend_ui_check_report.json"

# 综合判断结果
if [ "$UI_CHECK_RESULT" = "PASS" ] && [ $UI_CHECKER_RESULT -eq 0 ]; then
    echo -e "${GREEN}🎉 前端UI检查完全通过${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ 前端UI检查发现问题，请查看详细报告${NC}"
    exit 1
fi