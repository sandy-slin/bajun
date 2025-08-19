#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取器配置文件
控制不同数据获取策略的启用和参数
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class DataFetcherConfig:
    """数据获取器配置"""
    
    # 主要策略选择
    use_stable_fetcher: bool = True        # 是否使用稳定数据获取器
    fallback_to_original: bool = True      # 稳定获取器失败时是否回退到原始方案
    strict_real_data_only: bool = True     # 严格仅使用真实数据，禁止任何模拟数据
    
    # 重试策略配置
    max_retries: int = 5                   # 最大重试次数
    base_delay: float = 1.0                # 基础延迟时间（秒）
    max_delay: float = 16.0                # 最大延迟时间（秒）
    timeout: float = 30.0                  # 请求超时时间（秒）
    
    # 批处理配置
    batch_size: int = 100                  # 批量处理股票数量
    batch_delay: float = 0.5               # 批次间延迟时间（秒）
    
    # 缓存配置
    cache_duration: Dict[str, int] = None
    
    # 数据质量配置
    min_data_points: int = 5               # 最少数据点要求
    max_missing_ratio: float = 0.1         # 最大数据缺失率
    
    # 实时数据方案优先级
    realtime_priority: List[str] = None
    
    # 板块数据方案优先级  
    sector_priority: List[str] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.cache_duration is None:
            self.cache_duration = {
                'stock_basic_info': 3600,      # 股票基本信息：1小时
                'historical_data': 1800,       # 历史数据：30分钟
                'realtime_data': 300,          # 实时数据：5分钟
                'sector_data': 1800,           # 板块数据：30分钟
                'individual_info': 600,        # 个股信息：10分钟
            }
        
        if self.realtime_priority is None:
            self.realtime_priority = [
                'pseudo_realtime_from_hist',    # 历史数据伪实时（推荐）
                'individual_info_batch',        # 个股信息批量获取
                'realtime_from_163',           # 网易财经数据
                'spot_with_enhanced_retry',     # 增强重试的原始接口
            ]
        
        if self.sector_priority is None:
            self.sector_priority = [
                'index_hist_sw',               # 申万历史指数（推荐）
                'index_analysis_daily_sw',     # 申万每日分析数据
                'sw_index_spot',               # 申万实时数据
            ]

# 预设配置
CONFIGS = {
    # 生产环境配置 - 高稳定性
    'production': DataFetcherConfig(
        use_stable_fetcher=True,
        fallback_to_original=False,     # 生产环境不回退到不稳定的原始方案
        strict_real_data_only=True,
        max_retries=5,
        timeout=30.0,
        batch_size=50,                  # 生产环境较小批次
        batch_delay=1.0,                # 较长延迟避免被限流
    ),
    
    # 开发环境配置 - 平衡性能和稳定性
    'development': DataFetcherConfig(
        use_stable_fetcher=True,
        fallback_to_original=True,      # 开发环境允许回退测试
        strict_real_data_only=True,
        max_retries=3,
        timeout=20.0,
        batch_size=100,
        batch_delay=0.5,
    ),
    
    # 测试环境配置 - 快速响应
    'testing': DataFetcherConfig(
        use_stable_fetcher=True,
        fallback_to_original=True,
        strict_real_data_only=True,
        max_retries=2,
        timeout=10.0,
        batch_size=20,                  # 测试环境小批次
        batch_delay=0.2,
    ),
    
    # 紧急兼容配置 - 全部启用
    'emergency': DataFetcherConfig(
        use_stable_fetcher=True,
        fallback_to_original=True,
        strict_real_data_only=False,    # 紧急情况允许使用备用数据
        max_retries=10,
        timeout=60.0,
        batch_size=200,
        batch_delay=0.1,
    ),
}

def get_config(environment: str = 'development') -> DataFetcherConfig:
    """
    获取指定环境的配置
    
    Args:
        environment: 环境名称 ('production', 'development', 'testing', 'emergency')
        
    Returns:
        DataFetcherConfig: 配置对象
    """
    return CONFIGS.get(environment, CONFIGS['development'])

def get_akshare_alternatives() -> Dict[str, Dict[str, Any]]:
    """
    获取AKShare替代方案的详细信息
    
    Returns:
        Dict: 各种数据获取方案的详细配置
    """
    return {
        'realtime_stock_data': {
            'pseudo_realtime_from_hist': {
                'function': 'ak.stock_zh_a_hist',
                'stability': 5,
                'speed': 3,
                'data_freshness': 2,
                'description': '使用历史数据获取最新行情，稳定性最高',
                'pros': ['极高稳定性', '数据质量高', '批量获取效率高'],
                'cons': ['数据延迟1天', '不是真正实时'],
            },
            'individual_info_batch': {
                'function': 'ak.stock_individual_info_em',
                'stability': 4,
                'speed': 2,
                'data_freshness': 4,
                'description': '通过个股信息接口批量获取',
                'pros': ['数据详细', '相对稳定', '包含实时价格'],
                'cons': ['速度较慢', '不适合大批量'],
            },
            'realtime_from_163': {
                'function': 'ak.stock_zh_a_hist_163',
                'stability': 3,
                'speed': 4,
                'data_freshness': 4,
                'description': '网易财经数据源',
                'pros': ['数据更新较快', '覆盖全面'],
                'cons': ['稳定性中等', '可能有访问限制'],
            },
            'spot_with_enhanced_retry': {
                'function': 'ak.stock_zh_a_spot_em',
                'stability': 1,
                'speed': 5,
                'data_freshness': 5,
                'description': '原始实时接口增强重试',
                'pros': ['数据最新', '真正实时'],
                'cons': ['连接不稳定', '经常失败'],
            },
        },
        
        'sector_data': {
            'index_hist_sw': {
                'function': 'ak.index_hist_sw',
                'stability': 5,
                'speed': 4,
                'data_freshness': 3,
                'description': '申万行业历史指数数据',
                'pros': ['极高稳定性', '数据完整', '历史悠久'],
                'cons': ['更新频率日级别'],
            },
            'index_analysis_daily_sw': {
                'function': 'ak.index_analysis_daily_sw',
                'stability': 4,
                'speed': 3,
                'data_freshness': 3,
                'description': '申万行业每日分析数据',
                'pros': ['稳定性高', '分析维度多'],
                'cons': ['数据字段复杂'],
            },
            'sw_index_spot': {
                'function': 'ak.sw_index_spot',
                'stability': 2,
                'speed': 4,
                'data_freshness': 5,
                'description': '申万实时指数数据',
                'pros': ['数据最新', '实时更新'],
                'cons': ['稳定性一般'],
            },
        }
    }

# 当前环境配置（可通过环境变量或配置文件修改）
import os
CURRENT_ENVIRONMENT = os.getenv('DATA_FETCHER_ENV', 'development')
current_config = get_config(CURRENT_ENVIRONMENT)