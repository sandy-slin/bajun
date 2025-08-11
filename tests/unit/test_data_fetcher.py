"""
数据获取器单元测试
"""

import pytest
from unittest.mock import AsyncMock, Mock

from src.data.fetcher import DataFetcher
from src.cache.manager import CacheManager


@pytest.fixture
def mock_cache_manager():
    """模拟缓存管理器"""
    cache_manager = Mock(spec=CacheManager)
    cache_manager.get = AsyncMock()
    cache_manager.set = AsyncMock()
    return cache_manager


@pytest.fixture
def data_fetcher(mock_cache_manager):
    """数据获取器实例"""
    return DataFetcher(mock_cache_manager)


class TestDataFetcher:
    """数据获取器测试类"""
    
    @pytest.mark.asyncio
    async def test_get_trading_data_from_cache(self, data_fetcher, mock_cache_manager):
        """测试从缓存获取交易数据"""
        # 设置缓存返回数据
        cached_data = [
            {'date': '2024-01-01', 'code': '000001', 'close': 10.0}
        ]
        mock_cache_manager.get.return_value = cached_data
        
        # 获取数据
        result = await data_fetcher.get_trading_data('000001')
        
        # 验证结果
        assert result == cached_data
        mock_cache_manager.get.assert_called_once_with('trading_data_000001')
        mock_cache_manager.set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_trading_data_from_api(self, data_fetcher, mock_cache_manager):
        """测试从API获取交易数据"""
        # 设置缓存未命中
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True
        
        # 获取数据
        result = await data_fetcher.get_trading_data('000001')
        
        # 验证结果
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('date' in item for item in result)
        assert all('code' in item for item in result)
        
        # 验证缓存调用
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_market_overview(self, data_fetcher, mock_cache_manager):
        """测试获取市场概览"""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True
        
        # 获取市场概览（不指定股票代码）
        result = await data_fetcher.get_trading_data(None)
        
        # 验证结果包含多个指数数据
        assert isinstance(result, list)
        assert len(result) > 0
        
        # 验证包含主要指数
        codes = {item['code'] for item in result}
        expected_codes = {'000001', '399001', '399006'}
        assert expected_codes.issubset(codes)
    
    @pytest.mark.asyncio
    async def test_get_market_events(self, data_fetcher, mock_cache_manager):
        """测试获取市场事件"""
        mock_cache_manager.get.return_value = None
        mock_cache_manager.set.return_value = True
        
        # 获取市场事件
        events = await data_fetcher.get_market_events(30)
        
        # 验证结果
        assert isinstance(events, list)
        assert len(events) == 30
        assert all('date' in event for event in events)
        assert all('title' in event for event in events)
        assert all('impact' in event for event in events)
    
    @pytest.mark.asyncio
    async def test_fetch_stock_data_structure(self, data_fetcher):
        """测试股票数据结构"""
        # 通过直接调用内部方法测试数据结构
        from datetime import datetime, timedelta
        
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        data = await data_fetcher._fetch_stock_data('000001', start_date, end_date)
        
        # 验证数据结构
        assert isinstance(data, list)
        if data:  # 如果有数据
            sample = data[0]
            required_fields = {'date', 'code', 'open', 'high', 'low', 'close', 'volume'}
            assert required_fields.issubset(sample.keys())
            assert sample['code'] == '000001'