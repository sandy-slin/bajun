"""
缓存管理模块
提供高效的数据缓存和检索功能
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache", expiry_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.expiry_days = expiry_days
        self.logger = logging.getLogger(__name__)
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(exist_ok=True)
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查是否过期
            if self._is_expired(cache_data['timestamp']):
                self.logger.info(f"缓存已过期，删除: {key}")
                cache_file.unlink()
                return None
            
            self.logger.debug(f"缓存命中: {key}")
            return cache_data['data']
            
        except (json.JSONDecodeError, KeyError, OSError) as e:
            self.logger.warning(f"读取缓存失败: {key}, 错误: {e}")
            return None
    
    async def set(self, key: str, data: Any) -> bool:
        """设置缓存数据"""
        cache_file = self._get_cache_file(key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"缓存已保存: {key}")
            return True
            
        except (OSError, TypeError) as e:
            self.logger.error(f"保存缓存失败: {key}, 错误: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        cache_file = self._get_cache_file(key)
        
        try:
            if cache_file.exists():
                cache_file.unlink()
                self.logger.debug(f"缓存已删除: {key}")
                return True
            return False
            
        except OSError as e:
            self.logger.error(f"删除缓存失败: {key}, 错误: {e}")
            return False
    
    async def clear_expired(self) -> int:
        """清理过期缓存"""
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                if self._is_expired(cache_data['timestamp']):
                    cache_file.unlink()
                    cleared_count += 1
                    self.logger.debug(f"清理过期缓存: {cache_file.name}")
                    
            except (json.JSONDecodeError, KeyError, OSError):
                # 无法读取的文件也删除
                cache_file.unlink()
                cleared_count += 1
        
        if cleared_count > 0:
            self.logger.info(f"清理了 {cleared_count} 个过期缓存文件")
        
        return cleared_count
    
    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 将key转换为安全的文件名
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"
    
    def _is_expired(self, timestamp_str: str) -> bool:
        """检查时间戳是否过期"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            expire_time = timestamp + timedelta(days=self.expiry_days)
            return datetime.now() > expire_time
        except ValueError:
            return True  # 无法解析时间戳，视为过期