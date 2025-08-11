# -*- coding: utf-8 -*-
"""
Output utility module
"""

import json
from datetime import datetime
try:
    from typing import Dict, Any, List
except ImportError:
    Dict = dict
    Any = object
    List = list


class OutputFormatter:
    """输出格式化器"""
    
    def format_analysis_result(self, result):
        """格式化分析结果为可读文本"""
        lines = []
        lines.append(f"股票代码: {result['stock_code']}")
        lines.append(f"数据点数: {result['data_points']}")
        lines.append(f"最新价格: {result['latest_price']:.2f}")
        
        lines.append("\n--- 事件分析 ---")
        events = result.get('events_analysis', {})
        for key, value in events.items():
            lines.append(f"{key}: {value}")
        
        lines.append("\n--- 板块动向 ---")
        trends = result.get('sector_trends', {})
        for sector, trend in trends.items():
            lines.append(f"{sector}: {trend}")
        
        lines.append("\n--- 技术分析 ---")
        technical = result.get('technical_analysis', {})
        for key, value in technical.items():
            if isinstance(value, (int, float)):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
        
        lines.append("\n--- 交易建议 ---")
        lines.append(result.get('trading_advice', '暂无建议'))
        
        return "\n".join(lines)
    
    def save_results_to_file(self, results, filename=None):
        """保存结果到JSON文件"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"a_share_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        return filename
    
    def print_separator(self, length=50):
        """打印分隔线"""
        print("=" * length)
    
    def print_section_header(self, title):
        """打印章节标题"""
        print(f"\n--- {title} ---")