# -*- coding: utf-8 -*-
"""
Technical analysis module
"""

try:
    from typing import Dict, Any
except ImportError:
    Dict = dict
    Any = object
import pandas as pd

from ..config import DATA_CONFIG


class TechnicalAnalyzer:
    """技术分析器"""
    
    def analyze_price_trends(self, stock_data):
        """分析价格趋势"""
        if stock_data.empty:
            return {"latest_price": 0, "price_change_pct": 0, "volume_ratio": 0}
        
        recent_points = DATA_CONFIG["recent_data_points"]
        recent_data = stock_data.tail(recent_points)
        
        latest_price = recent_data.iloc[-1]['close_price']
        
        # 计算价格变化
        if len(recent_data) > 1:
            start_price = recent_data.iloc[0]['close_price']
            price_change_pct = ((latest_price - start_price) / start_price) * 100
        else:
            price_change_pct = 0
        
        # 计算成交量比
        avg_volume = recent_data['volume'].mean()
        latest_volume = recent_data.iloc[-1]['volume']
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0
        
        return {
            "latest_price": latest_price,
            "price_change_pct": price_change_pct,
            "volume_ratio": volume_ratio
        }
    
    def get_support_resistance(self, stock_data):
        """计算支撑位和阻力位"""
        if stock_data.empty or len(stock_data) < 20:
            return {"support": 0, "resistance": 0}
        
        recent_data = stock_data.tail(20)
        
        # 简单的支撑阻力位计算
        support = recent_data['low_price'].min()
        resistance = recent_data['high_price'].max()
        
        return {
            "support": support,
            "resistance": resistance
        }
    
    def calculate_moving_averages(self, stock_data):
        """计算移动平均线"""
        if stock_data.empty or len(stock_data) < 20:
            return {"ma5": 0, "ma10": 0, "ma20": 0}
        
        prices = stock_data['close_price']
        
        ma5 = prices.tail(5).mean() if len(prices) >= 5 else 0
        ma10 = prices.tail(10).mean() if len(prices) >= 10 else 0
        ma20 = prices.tail(20).mean() if len(prices) >= 20 else 0
        
        return {
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20
        }