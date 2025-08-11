"""
LLM分析模块
使用DeepSeek API进行智能分析
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
from .report_manager import ReportManager
from .base_analyzer import BaseAnalyzer


class LLMAnalyzer:
    """LLM分析器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        self.report_manager = ReportManager()
        self.base_analyzer = BaseAnalyzer(self.report_manager)
    
    async def analyze_comprehensive(self, trading_data: List[Dict], stock_code: str = None) -> Dict:
        """综合分析"""
        try:
            # 分析过去和未来事件
            events_analysis = await self._analyze_events(stock_code)
            
            # 分析市场动向
            trend_analysis = await self._analyze_trends(trading_data)
            
            # 生成交易建议
            trading_advice = await self._generate_trading_advice(trading_data, stock_code)
            
            # 检查是否所有分析都失败了
            failed_analyses = [
                analysis for analysis in [events_analysis, trend_analysis, trading_advice]
                if analysis.startswith(('请求失败', 'API调用失败', '请求超时', '网络连接失败'))
            ]
            
            # 如果多个分析失败，提供降级分析
            if len(failed_analyses) >= 2:
                self.logger.warning("多个LLM分析失败，使用降级方案")
                return self.base_analyzer.get_fallback_analysis(trading_data, stock_code)
            
            analysis_result = {
                'events_analysis': events_analysis,
                'trend_analysis': trend_analysis,
                'trading_advice': trading_advice,
                'analysis_time': datetime.now().isoformat()
            }
            
            # 生成Markdown报告并缓存
            try:
                markdown_content = self.report_manager.format_analysis_to_markdown(
                    analysis_result, stock_code
                )
                report_path = self.report_manager.save_report(markdown_content, stock_code)
                analysis_result['report_path'] = report_path
                self.logger.info(f"分析报告已保存: {report_path}")
            except Exception as e:
                self.logger.warning(f"保存分析报告失败: {e}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"LLM分析失败: {e}")
            return self.base_analyzer.get_fallback_analysis(trading_data, stock_code)
    
    async def _analyze_events(self, stock_code: str) -> str:
        """分析重大事件"""
        prompt = f"""
        请分析{stock_code or "A股市场"}过去一个月和未来一个月的重大事件，包括：
        1. 政策变化影响
        2. 行业发展趋势  
        3. 公司重要公告
        4. 宏观经济因素
        
        请提供简洁的分析总结。
        """
        
        return await self._call_llm(prompt)
    
    async def _analyze_trends(self, trading_data: List[Dict]) -> str:
        """分析市场动向"""
        # 简化数据用于分析
        recent_prices = [item['close'] for item in trading_data[-7:]]
        
        prompt = f"""
        基于最近7天的收盘价数据：{recent_prices}
        
        请分析：
        1. 价格趋势方向
        2. 波动特征
        3. 技术面特点
        4. 风险提示
        
        请提供专业的技术分析。
        """
        
        return await self._call_llm(prompt)
    
    async def _generate_trading_advice(self, trading_data: List[Dict], stock_code: str) -> str:
        """生成交易建议"""
        latest_price = trading_data[-1]['close'] if trading_data else 0
        
        prompt = f"""
        基于{stock_code or "市场"}的交易数据，当前价格{latest_price}，请提供未来3天的交易建议：
        
        1. 买入/卖出/持有建议
        2. 风险控制要点
        3. 关键价位关注
        4. 操作时机提醒
        
        请提供实用的投资建议，注意风险提示。
        """
        
        return await self._call_llm(prompt)
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {
                    'role': 'system',
                    'content': '你是一个专业的股票分析师，请提供客观、专业的分析建议。'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                self.logger.debug(f"调用LLM API: {self.base_url}/chat/completions")
                
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        self.logger.debug(f"LLM响应成功，内容长度: {len(content)}")
                        return content
                    else:
                        error_text = await response.text()
                        self.logger.error(f"LLM API调用失败: {response.status}, {error_text}")
                        return f"API调用失败 (HTTP {response.status}): 请检查API密钥和网络连接"
                        
        except asyncio.TimeoutError:
            self.logger.error("LLM请求超时")
            return "请求超时: DeepSeek API响应时间过长，请稍后重试"
        except aiohttp.ClientError as e:
            self.logger.error(f"网络连接错误: {e}")
            return f"网络连接失败: {str(e)}"
        except Exception as e:
            self.logger.error(f"LLM请求异常: {type(e).__name__}: {e}")
            return f"请求失败: {type(e).__name__}: {str(e)}"
    
