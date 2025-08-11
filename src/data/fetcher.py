"""
数据获取模块
负责从各种数据源获取A股交易数据
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

from cache.manager import CacheManager
from .real_data_fetcher import RealDataFetcher


class DataFetcher:
    """数据获取器"""
    
    def __init__(self, cache_manager: CacheManager, use_real_data: bool = True):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        self.use_real_data = use_real_data
        if use_real_data:
            self.real_fetcher = RealDataFetcher()
    
    async def get_trading_data(self, stock_code: Optional[str] = None) -> List[Dict]:
        """
        获取交易数据
        优先从缓存读取，缓存未命中则从API获取
        """
        cache_key = f"trading_data_{stock_code or 'all'}"
        
        # 尝试从缓存获取
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            self.logger.info(f"从缓存获取数据: {cache_key}")
            return cached_data
        
        # 缓存未命中，从API获取
        self.logger.info(f"从API获取数据: {stock_code or '全市场'}")
        fresh_data = await self._fetch_from_api(stock_code)
        
        # 缓存数据
        await self.cache_manager.set(cache_key, fresh_data)
        
        return fresh_data
    
    async def _fetch_from_api(self, stock_code: Optional[str]) -> List[Dict]:
        """从API获取数据"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self._fetch_real_data(stock_code)
        else:
            return await self._fetch_mock_data(stock_code)
    
    async def _fetch_real_data(self, stock_code: Optional[str]) -> List[Dict]:
        """从真实API获取数据"""
        if stock_code:
            # 获取单个股票数据
            data = await self.real_fetcher.get_stock_data(stock_code)
            if not data:
                self.logger.warning(f"真实数据获取失败，使用模拟数据: {stock_code}")
                return await self._fetch_mock_data(stock_code)
            return data
        else:
            # 获取市场概览数据
            data = await self.real_fetcher.get_market_indices()
            if not data:
                self.logger.warning("真实市场数据获取失败，使用模拟数据")
                return await self._fetch_mock_data(None)
            return data
    
    async def _fetch_mock_data(self, stock_code: Optional[str]) -> List[Dict]:
        """获取模拟数据（原逻辑）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6个月
        
        if stock_code:
            return await self._fetch_stock_data(stock_code, start_date, end_date)
        else:
            return await self._fetch_market_overview(start_date, end_date)
    
    async def _fetch_stock_data(self, code: str, start: datetime, end: datetime) -> List[Dict]:
        """获取特定股票数据"""
        # 模拟API调用 - 实际项目中替换为真实API
        sample_data = []
        current_date = start
        base_price = 10.0
        
        while current_date <= end:
            if current_date.weekday() < 5:  # 工作日
                price = base_price + (current_date.day % 10) * 0.1
                sample_data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'code': code,
                    'open': price,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price * 1.01,
                    'volume': 100000 + (current_date.day % 100) * 1000
                })
            current_date += timedelta(days=1)
        
        return sample_data
    
    async def _fetch_market_overview(self, start: datetime, end: datetime) -> List[Dict]:
        """获取市场概览数据"""
        # 模拟获取主要指数数据
        indices = ['000001', '399001', '399006']  # 上证、深证成指、创业板
        all_data = []
        
        for index_code in indices:
            index_data = await self._fetch_stock_data(index_code, start, end)
            all_data.extend(index_data)
        
        return all_data
    
    async def get_market_events(self, days: int = 30) -> List[Dict]:
        """获取市场事件数据"""
        cache_key = f"market_events_{days}"
        
        cached_events = await self.cache_manager.get(cache_key)
        if cached_events:
            return cached_events
        
        # 模拟事件数据
        events = [
            {
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'title': f'重要事件{i}',
                'description': f'这是第{i}天发生的重要市场事件',
                'impact': 'high' if i % 5 == 0 else 'medium'
            }
            for i in range(1, days + 1)
        ]
        
        await self.cache_manager.set(cache_key, events)
        return events
    
    async def get_current_price(self, stock_code: str) -> Optional[float]:
        """获取当前股价"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self.real_fetcher.get_current_price(stock_code)
        return None
    
    async def validate_stock_code(self, stock_code: str) -> bool:
        """验证股票代码是否有效"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self.real_fetcher.validate_stock_code(stock_code)
        return True  # 模拟模式下总是返回True
    
    async def get_stock_info(self, stock_code: str) -> Dict:
        """获取股票基本信息"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self.real_fetcher.get_stock_info(stock_code)
        return {}