"""
系统配置管理
集中管理所有配置参数
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """系统配置类"""
    
    # API配置
    deepseek_api_key: str = "sk-f4affcb7b78243f5a138e7c9bdbbd6ee"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # 数据配置
    cache_dir: str = "cache"
    cache_expiry_days: int = 7
    
    # 分析配置
    trading_data_months: int = 6
    event_analysis_days: int = 30
    trend_analysis_days: int = 7
    prediction_days: int = 3
    
    # 请求配置
    request_timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量覆盖配置
        self.deepseek_api_key = os.getenv(
            'DEEPSEEK_API_KEY', 
            self.deepseek_api_key
        )
        
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(exist_ok=True)
    
    @property
    def cache_path(self) -> Path:
        """缓存路径"""
        return Path(self.cache_dir)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API密钥不能为空")
        
        if self.cache_expiry_days <= 0:
            raise ValueError("缓存过期天数必须大于0")
        
        if self.trading_data_months <= 0:
            raise ValueError("交易数据月数必须大于0")
        
        return True