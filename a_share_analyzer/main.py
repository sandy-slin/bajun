# -*- coding: utf-8 -*-
"""
Main analyzer module
"""

try:
    from typing import Dict, Any, List
except ImportError:
    Dict = dict
    Any = object
    List = list

from .cache import DatabaseCache
from .data import StockDataProvider, SectorDataProvider  
from .analysis import LLMClient, TechnicalAnalyzer
from .utils import OutputFormatter


class AShareAnalyzer:
    """A股分析器主类"""
    
    def __init__(self, deepseek_api_key=None):
        self.cache = DatabaseCache()
        self.stock_provider = StockDataProvider(self.cache)
        self.sector_provider = SectorDataProvider()
        self.llm_client = LLMClient(deepseek_api_key)
        self.technical_analyzer = TechnicalAnalyzer()
        self.output_formatter = OutputFormatter()
    
    def analyze_stock(self, stock_code):
        """完整分析单个股票"""
        print("\n=== Analyzing stock: " + stock_code + " ===")
        
        # 获取股票数据
        stock_data = self.stock_provider.get_stock_data(stock_code)
        stock_name = self.stock_provider.get_stock_info(stock_code)
        
        # 事件分析
        events_cache = self.cache.get_events_cache(stock_code, 'analysis')
        if events_cache:
            events_analysis = events_cache
        else:
            events_analysis = self.llm_client.analyze_events(stock_name or stock_code, stock_code)
            self.cache.save_events_cache(stock_code, 'analysis', events_analysis)
        
        # 板块趋势分析
        sector_trends = self.sector_provider.analyze_sector_trends(stock_code)
        
        # 技术分析
        technical_metrics = self.technical_analyzer.analyze_price_trends(stock_data)
        moving_averages = self.technical_analyzer.calculate_moving_averages(stock_data)
        support_resistance = self.technical_analyzer.get_support_resistance(stock_data)
        
        technical_analysis = {}
        technical_analysis.update(technical_metrics)
        technical_analysis.update(moving_averages)
        technical_analysis.update(support_resistance)
        
        # 生成交易建议
        analysis_data = {}
        analysis_data.update(technical_metrics)
        analysis_data.update(events_analysis)
        trading_advice = self.llm_client.generate_trading_advice(stock_code, analysis_data)
        
        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "data_points": len(stock_data),
            "latest_price": technical_metrics.get("latest_price", 0),
            "events_analysis": events_analysis,
            "sector_trends": sector_trends,
            "technical_analysis": technical_analysis,
            "trading_advice": trading_advice
        }
    
    def analyze_multiple_stocks(self, stock_codes):
        """分析多个股票"""
        results = []
        
        for code in stock_codes:
            try:
                result = self.analyze_stock(code)
                results.append(result)
                
                # 打印结果
                formatted_output = self.output_formatter.format_analysis_result(result)
                self.output_formatter.print_separator()
                print(formatted_output)
                self.output_formatter.print_separator()
                
            except Exception as e:
                print("Analyzing stock " + code + " failed: " + str(e))
                results.append({
                    "stock_code": code,
                    "error": str(e)
                })
        
        return results
    
    def save_analysis_results(self, results):
        """保存分析结果"""
        filename = self.output_formatter.save_results_to_file(results)
        print("\nAnalysis results saved to: " + filename)
        return filename