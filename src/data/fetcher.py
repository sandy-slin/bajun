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
            self.logger.error("不允许使用模拟数据，系统必须使用真实数据")
            raise RuntimeError("系统配置为仅使用真实数据，不允许使用模拟数据")
    
    async def _fetch_real_data(self, stock_code: Optional[str]) -> List[Dict]:
        """从真实API获取数据"""
        if stock_code:
            # 获取单个股票数据
            data = await self.real_fetcher.get_stock_data(stock_code)
            if not data:
                self.logger.error(f"真实数据获取失败: {stock_code}")
                raise RuntimeError(f"无法获取股票{stock_code}的真实数据")
            return data
        else:
            # 获取市场概览数据
            data = await self.real_fetcher.get_market_indices()
            if not data:
                self.logger.error("真实市场数据获取失败")
                raise RuntimeError("无法获取真实市场数据")
            return data
    
    
    async def get_market_events(self, days: int = 30) -> List[Dict]:
        """获取市场事件数据"""
        cache_key = f"market_events_{days}"
        
        cached_events = await self.cache_manager.get(cache_key)
        if cached_events:
            return cached_events
        
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            try:
                events = await self.real_fetcher.get_market_events(days)
                await self.cache_manager.set(cache_key, events)
                return events
            except Exception as e:
                self.logger.error(f"获取市场事件数据失败: {e}")
                raise RuntimeError(f"无法获取市场事件数据: {e}")
        else:
            self.logger.error("不允许使用模拟事件数据")
            raise RuntimeError("系统配置为仅使用真实数据，无法获取市场事件数据")
    
    async def get_current_price(self, stock_code: str) -> Optional[float]:
        """获取当前股价"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self.real_fetcher.get_current_price(stock_code)
        return None
    
    async def validate_stock_code(self, stock_code: str) -> bool:
        """验证股票代码是否有效"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self.real_fetcher.validate_stock_code(stock_code)
        self.logger.error("不允许使用模拟股票代码验证")
        raise RuntimeError("系统配置为仅使用真实数据，无法验证股票代码")
    
    async def get_stock_info(self, stock_code: str) -> Dict:
        """获取股票基本信息"""
        if self.use_real_data and hasattr(self, 'real_fetcher'):
            return await self.real_fetcher.get_stock_info(stock_code)
        self.logger.error("不允许使用模拟股票信息")
        raise RuntimeError("系统配置为仅使用真实数据，无法获取股票信息")