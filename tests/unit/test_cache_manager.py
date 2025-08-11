"""
缓存管理器单元测试
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.cache.manager import CacheManager


@pytest.fixture
def temp_cache_dir():
    """临时缓存目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def cache_manager(temp_cache_dir):
    """缓存管理器实例"""
    return CacheManager(cache_dir=temp_cache_dir, expiry_days=1)


class TestCacheManager:
    """缓存管理器测试类"""
    
    @pytest.mark.asyncio
    async def test_set_and_get_cache(self, cache_manager):
        """测试设置和获取缓存"""
        test_key = "test_key"
        test_data = {"name": "test", "value": 123}
        
        # 设置缓存
        result = await cache_manager.set(test_key, test_data)
        assert result is True
        
        # 获取缓存
        cached_data = await cache_manager.get(test_key)
        assert cached_data == test_data
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_cache(self, cache_manager):
        """测试获取不存在的缓存"""
        result = await cache_manager.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, temp_cache_dir):
        """测试缓存过期"""
        # 创建过期时间为0天的缓存管理器
        cache_manager = CacheManager(cache_dir=temp_cache_dir, expiry_days=0)
        
        test_key = "expired_key"
        test_data = {"expired": True}
        
        # 设置缓存
        await cache_manager.set(test_key, test_data)
        
        # 立即获取应该失败（因为过期时间为0）
        result = await cache_manager.get(test_key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_cache(self, cache_manager):
        """测试删除缓存"""
        test_key = "delete_test"
        test_data = {"to_delete": True}
        
        # 设置缓存
        await cache_manager.set(test_key, test_data)
        
        # 验证缓存存在
        cached_data = await cache_manager.get(test_key)
        assert cached_data == test_data
        
        # 删除缓存
        delete_result = await cache_manager.delete(test_key)
        assert delete_result is True
        
        # 验证缓存已删除
        cached_data = await cache_manager.get(test_key)
        assert cached_data is None
    
    @pytest.mark.asyncio
    async def test_clear_expired_caches(self, temp_cache_dir):
        """测试清理过期缓存"""
        cache_manager = CacheManager(cache_dir=temp_cache_dir, expiry_days=1)
        
        # 创建一个正常缓存
        await cache_manager.set("normal_key", {"normal": True})
        
        # 手动创建一个过期的缓存文件
        expired_file = Path(temp_cache_dir) / "expired_key.json"
        expired_data = {
            'timestamp': (datetime.now() - timedelta(days=2)).isoformat(),
            'data': {"expired": True}
        }
        with open(expired_file, 'w') as f:
            json.dump(expired_data, f)
        
        # 清理过期缓存
        cleared_count = await cache_manager.clear_expired()
        assert cleared_count == 1
        
        # 验证正常缓存仍存在
        normal_data = await cache_manager.get("normal_key")
        assert normal_data == {"normal": True}
        
        # 验证过期缓存已删除
        expired_data = await cache_manager.get("expired_key")
        assert expired_data is None