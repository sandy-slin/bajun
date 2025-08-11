# -*- coding: utf-8 -*-
"""
Stock data provider module
"""

from datetime import datetime, timedelta
try:
    from typing import Optional
except ImportError:
    Optional = None
import pandas as pd
import akshare as ak

from ..cache import DatabaseCache
from ..config import DATA_CONFIG


class StockDataProvider:
    """股票数据提供者"""
    
    def __init__(self, cache=None):
        self.cache = cache or DatabaseCache()
    
    def get_stock_data(self, stock_code, months=None):
        """获取股票交易数据，优先从缓存获取"""
        if months is None:
            months = DATA_CONFIG["default_months"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 检查缓存
        if self.cache.is_stock_data_fresh(stock_code):
            cached_data = self.cache.get_stock_data(stock_code, start_str, end_str)
            if len(cached_data) > 0:
                print(f"使用缓存数据: {stock_code}")
                return cached_data
        
        # 从网络获取
        try:
            print(f"从网络获取数据: {stock_code}")
            stock_data = self._fetch_from_akshare(stock_code, start_date, end_date)
            
            if not stock_data.empty:
                self.cache.save_stock_data(stock_data)
            
            return stock_data
            
        except Exception as e:
            print(f"获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def _fetch_from_akshare(self, stock_code, start_date, end_date):
        """从akshare获取数据"""
        stock_data = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        # 标准化列名
        stock_data['stock_code'] = stock_code
        stock_data['updated_at'] = datetime.now().isoformat()
        
        expected_columns = [
            'date', 'open_price', 'high_price', 'low_price', 
            'close_price', 'volume', 'turnover', 'amplitude',
            'change_pct', 'change_amount', 'turnover_rate', 
            'stock_code', 'updated_at'
        ]
        
        stock_data.columns = expected_columns
        
        return stock_data
    
    def get_stock_info(self, stock_code):
        """获取股票基本信息"""
        try:
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            stock_name = stock_info[stock_info['item'] == '股票简称']['value'].iloc[0]
            return stock_name
        except Exception as e:
            print(f"获取股票信息失败: {e}")
            return None