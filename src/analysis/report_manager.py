"""
分析报告管理模块
负责将LLM分析结果格式化为Markdown并缓存到本地
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class ReportManager:
    """分析报告管理器"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
    
    def format_analysis_to_markdown(self, analysis_result: Dict, stock_code: str = None) -> str:
        """将分析结果格式化为Markdown"""
        stock_name = stock_code or "市场概览"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# {stock_name} - 股票分析报告

**分析时间**: {current_time}
**股票代码**: {stock_code or "全市场"}

---

## 📊 重大事件分析

{self._format_section(analysis_result.get('events_analysis', '暂无事件分析'))}

---

## 📈 技术面分析

{self._format_section(analysis_result.get('trend_analysis', '暂无技术分析'))}

---

## 💡 交易建议

{self._format_section(analysis_result.get('trading_advice', '暂无交易建议'))}

---

## ⚠️ 免责声明

本分析报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。
请根据自身风险承受能力，结合更多信息独立做出投资决策。

---

**报告生成时间**: {analysis_result.get('analysis_time', current_time)}
{self._format_note(analysis_result.get('note'))}
"""
        return markdown_content
    
    def save_report(self, markdown_content: str, stock_code: str = None) -> str:
        """保存分析报告到本地文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_code or 'market'}_{timestamp}.md"
        file_path = self.reports_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return str(file_path)
        except OSError as e:
            raise Exception(f"保存报告失败: {e}")
    
    def get_latest_report(self, stock_code: str = None) -> Optional[str]:
        """获取最新的分析报告"""
        pattern = f"{stock_code or 'market'}_*.md"
        matching_files = list(self.reports_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # 按修改时间排序，获取最新文件
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return f.read()
        except OSError:
            return None
    
    def list_reports(self, stock_code: str = None) -> list:
        """列出所有报告文件"""
        if stock_code:
            pattern = f"{stock_code}_*.md"
        else:
            pattern = "*.md"
        
        matching_files = list(self.reports_dir.glob(pattern))
        return sorted([f.name for f in matching_files], reverse=True)
    
    def clean_old_reports(self, keep_days: int = 30):
        """清理旧的报告文件"""
        import time
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        
        cleaned_count = 0
        for report_file in self.reports_dir.glob("*.md"):
            if report_file.stat().st_mtime < cutoff_time:
                try:
                    report_file.unlink()
                    cleaned_count += 1
                except OSError:
                    continue
        
        return cleaned_count
    
    def _format_section(self, content: str) -> str:
        """格式化章节内容"""
        if not content:
            return "*暂无内容*"
        
        # 确保内容格式正确
        content = content.strip()
        
        # 如果内容已经是Markdown格式，直接返回
        if any(marker in content for marker in ['**', '##', '###', '-', '*', '1.']):
            return content
        
        # 否则进行基础格式化
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith(('1.', '2.', '3.', '4.')):
                    # 数字列表
                    formatted_lines.append(f"### {line}")
                elif line.startswith(('•', '-', '*')):
                    # 项目符号
                    formatted_lines.append(line)
                elif len(line) > 50 and not line.endswith(('。', '.', ':', '：')):
                    # 长标题
                    formatted_lines.append(f"**{line}**")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def _format_note(self, note: str) -> str:
        """格式化备注信息"""
        if not note:
            return ""
        
        return f"\n> **说明**: {note}\n"