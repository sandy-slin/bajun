# -*- coding: utf-8 -*-
"""
LLM client module
"""

try:
    from typing import Dict, Any, Optional
except ImportError:
    Dict = dict
    Any = object
    Optional = None
from openai import OpenAI

from ..config import DEEPSEEK_CONFIG


class LLMClient:
    """DeepSeek LLM客户端"""
    
    def __init__(self, api_key=None, base_url=None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_CONFIG["api_key"],
            base_url=base_url or DEEPSEEK_CONFIG["base_url"]
        )
        self.model = DEEPSEEK_CONFIG["model"]
    
    def analyze_events(self, stock_name, stock_code):
        """分析股票重大事件"""
        prompt = f"""
        请分析股票{stock_name}({stock_code})的重要信息：
        
        1. 过去一个月的重大事件和公告
        2. 未来一个月可能影响股价的重要事件
        3. 所属行业的最新动向
        4. 相关政策影响
        
        请用中文回复，格式为JSON：
        {{
            "past_month_events": "过去一个月重大事件汇总",
            "future_month_events": "未来一个月重要事件预测",
            "industry_trends": "行业动向分析",
            "policy_impact": "政策影响分析"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return self._parse_json_response(response.choices[0].message.content)
            
        except Exception as e:
            return self._create_error_response(f"事件分析失败: {e}")
    
    def generate_trading_advice(self, stock_code, analysis_data):
        """生成交易建议"""
        prompt = f"""
        基于以下信息为股票{stock_code}生成未来3天的交易建议：
        
        技术指标：
        - 最新价格: {analysis_data.get('latest_price', 0):.2f}
        - 近期涨跌幅: {analysis_data.get('price_change_pct', 0):.2f}%
        - 成交量比: {analysis_data.get('volume_ratio', 0):.2f}
        
        基本面分析：
        - 过去一月事件: {analysis_data.get('past_month_events', '')}
        - 未来一月预期: {analysis_data.get('future_month_events', '')}
        - 行业趋势: {analysis_data.get('industry_trends', '')}
        
        请给出具体的交易建议，包括买入/卖出/持有建议，目标价位，止损位等。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"交易建议生成失败: {e}"
    
    def _parse_json_response(self, content):
        """解析JSON响应"""
        try:
            import json
            return json.loads(content)
        except json.JSONDecodeError:
            return self._create_error_response("JSON解析失败")
    
    def _create_error_response(self, error_msg):
        """创建错误响应"""
        return {
            "past_month_events": error_msg,
            "future_month_events": "分析失败",
            "industry_trends": "分析失败",
            "policy_impact": "分析失败"
        }