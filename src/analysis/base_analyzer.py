"""
基础分析模块
提供降级分析和基础数据分析功能
"""

from datetime import datetime
from typing import Dict, List


class BaseAnalyzer:
    """基础分析器"""
    
    def __init__(self, report_manager):
        self.report_manager = report_manager
    
    def get_fallback_analysis(self, trading_data: List[Dict], stock_code: str) -> Dict:
        """降级分析方案"""
        # 基础数据分析
        if trading_data:
            recent_data = trading_data[-7:] if len(trading_data) >= 7 else trading_data
            prices = [item['close'] for item in recent_data]
            volumes = [item['volume'] for item in recent_data]
            
            price_change = (prices[-1] - prices[0]) / prices[0] * 100 if len(prices) > 1 else 0
            avg_volume = sum(volumes) / len(volumes)
            
            trend = "上涨" if price_change > 2 else "下跌" if price_change < -2 else "震荡"
            
            fallback_analysis = {
                'events_analysis': self._generate_basic_events_analysis(stock_code),
                'trend_analysis': self._generate_basic_trend_analysis(
                    trend, price_change, avg_volume, prices[-1]
                ),
                'trading_advice': self._generate_basic_trading_advice(),
                'analysis_time': datetime.now().isoformat(),
                'note': '本次分析为降级方案，建议结合更多信息进行投资决策'
            }
        else:
            fallback_analysis = {
                'events_analysis': '数据获取失败，无法进行事件分析',
                'trend_analysis': '数据获取失败，无法进行技术分析', 
                'trading_advice': '数据获取失败，无法提供交易建议',
                'analysis_time': datetime.now().isoformat(),
                'note': '由于数据问题，无法提供完整分析'
            }
        
        # 为降级分析生成Markdown报告
        try:
            markdown_content = self.report_manager.format_analysis_to_markdown(
                fallback_analysis, stock_code
            )
            report_path = self.report_manager.save_report(markdown_content, stock_code)
            fallback_analysis['report_path'] = report_path
        except Exception:
            pass
        
        return fallback_analysis
    
    def _generate_basic_events_analysis(self, stock_code: str) -> str:
        """生成基础事件分析"""
        return f"""
{stock_code or "目标股票"}基础分析：
• 近期政策环境：关注监管政策变化对行业影响
• 行业动态：需关注同行业竞争格局变化  
• 宏观环境：当前市场情绪谨慎，建议关注经济数据
• 公司动态：建议关注定期报告和重大公告
        """.strip()
    
    def _generate_basic_trend_analysis(self, trend: str, price_change: float, 
                                     avg_volume: float, current_price: float) -> str:
        """生成基础趋势分析"""
        return f"""
技术面基础分析：
• 价格走势：近7日{trend}趋势，变动幅度{price_change:.2f}%
• 成交量：平均成交量{avg_volume:,.0f}手
• 当前价格：{current_price:.2f}元
• 建议关注：支撑位和阻力位的突破情况
        """.strip()
    
    def _generate_basic_trading_advice(self) -> str:
        """生成基础交易建议"""
        return """
基础交易建议（仅供参考）：
• 操作建议：当前市场波动较大，建议谨慎操作
• 风险控制：严格设置止损位，控制仓位
• 关注要点：关注成交量变化和重要支撑阻力位
• 免责声明：以上仅为基础技术分析，不构成投资建议，请独立判断
        """.strip()