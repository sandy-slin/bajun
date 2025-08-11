"""
配置设置单元测试
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.config.settings import Settings


class TestSettings:
    """设置配置测试类"""
    
    def test_default_settings(self):
        """测试默认设置"""
        settings = Settings()
        
        # 验证默认值
        assert settings.deepseek_api_key == "sk-f4affcb7b78243f5a138e7c9bdbbd6ee"
        assert settings.deepseek_base_url == "https://api.deepseek.com/v1"
        assert settings.cache_dir == "cache"
        assert settings.cache_expiry_days == 7
        assert settings.trading_data_months == 6
        assert settings.event_analysis_days == 30
        assert settings.trend_analysis_days == 7
        assert settings.prediction_days == 3
        assert settings.request_timeout == 30
        assert settings.max_retries == 3
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖"""
        # 设置环境变量
        test_api_key = "test-api-key-123"
        os.environ['DEEPSEEK_API_KEY'] = test_api_key
        
        try:
            settings = Settings()
            assert settings.deepseek_api_key == test_api_key
        finally:
            # 清理环境变量
            del os.environ['DEEPSEEK_API_KEY']
    
    def test_cache_path_property(self):
        """测试缓存路径属性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.cache_dir = temp_dir
            
            cache_path = settings.cache_path
            assert isinstance(cache_path, Path)
            assert str(cache_path) == temp_dir
    
    def test_validate_success(self):
        """测试配置验证成功"""
        settings = Settings()
        assert settings.validate() is True
    
    def test_validate_empty_api_key(self):
        """测试API密钥为空的验证"""
        settings = Settings()
        settings.deepseek_api_key = ""
        
        with pytest.raises(ValueError, match="DeepSeek API密钥不能为空"):
            settings.validate()
    
    def test_validate_invalid_cache_expiry(self):
        """测试无效缓存过期时间"""
        settings = Settings()
        settings.cache_expiry_days = 0
        
        with pytest.raises(ValueError, match="缓存过期天数必须大于0"):
            settings.validate()
    
    def test_validate_invalid_trading_months(self):
        """测试无效交易数据月数"""
        settings = Settings()
        settings.trading_data_months = -1
        
        with pytest.raises(ValueError, match="交易数据月数必须大于0"):
            settings.validate()
    
    def test_cache_directory_creation(self):
        """测试缓存目录创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "test_cache")
            
            # 确保目录不存在
            assert not os.path.exists(cache_dir)
            
            # 创建设置实例
            settings = Settings()
            settings.cache_dir = cache_dir
            settings.__post_init__()
            
            # 验证目录已创建
            assert os.path.exists(cache_dir)
            assert os.path.isdir(cache_dir)