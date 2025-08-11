"""
系统集成测试
"""

import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.main import StockInfoSystem
from src.config.settings import Settings


@pytest.fixture
def temp_settings():
    """临时配置"""
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = Settings()
        settings.cache_dir = temp_dir
        yield settings


class TestSystemIntegration:
    """系统集成测试类"""
    
    @pytest.mark.asyncio
    async def test_stock_info_system_initialization(self, temp_settings):
        """测试系统初始化"""
        with patch('src.main.Settings', return_value=temp_settings):
            system = StockInfoSystem()
            
            # 验证组件初始化
            assert system.settings is not None
            assert system.cache_manager is not None
            assert system.data_fetcher is not None
            assert system.llm_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_run_analysis_success(self, temp_settings):
        """测试分析流程成功"""
        with patch('src.main.Settings', return_value=temp_settings):
            system = StockInfoSystem()
            
            # Mock LLM分析结果
            mock_analysis = {
                'events_analysis': '市场事件分析',
                'trend_analysis': '趋势分析',
                'trading_advice': '交易建议'
            }
            
            with patch.object(system.llm_analyzer, 'analyze_comprehensive', 
                            return_value=mock_analysis):
                
                result = await system.run_analysis('000001')
                
                # 验证结果
                assert result['status'] == 'success'
                assert 'data' in result
                assert 'analysis' in result
                assert isinstance(result['data'], list)
                assert result['analysis'] == mock_analysis
    
    @pytest.mark.asyncio
    async def test_run_analysis_with_exception(self, temp_settings):
        """测试分析流程异常处理"""
        with patch('src.main.Settings', return_value=temp_settings):
            system = StockInfoSystem()
            
            # Mock数据获取异常
            with patch.object(system.data_fetcher, 'get_trading_data', 
                            side_effect=Exception("API调用失败")):
                
                result = await system.run_analysis('000001')
                
                # 验证错误处理
                assert result['status'] == 'error'
                assert 'API调用失败' in result['message']
    
    @pytest.mark.asyncio
    async def test_data_flow_integration(self, temp_settings):
        """测试数据流集成"""
        with patch('src.main.Settings', return_value=temp_settings):
            system = StockInfoSystem()
            
            # 获取交易数据
            trading_data = await system.data_fetcher.get_trading_data('000001')
            
            # 验证数据格式
            assert isinstance(trading_data, list)
            if trading_data:
                sample = trading_data[0]
                assert 'date' in sample
                assert 'code' in sample
                assert 'close' in sample
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, temp_settings):
        """测试缓存集成"""
        with patch('src.main.Settings', return_value=temp_settings):
            system = StockInfoSystem()
            
            # 第一次获取数据（应该从API获取并缓存）
            data1 = await system.data_fetcher.get_trading_data('000001')
            
            # 第二次获取数据（应该从缓存获取）
            data2 = await system.data_fetcher.get_trading_data('000001')
            
            # 验证数据一致性
            assert data1 == data2