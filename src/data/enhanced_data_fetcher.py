# -*- coding: utf-8 -*-
"""
增强数据获取模块
整合北向资金、融资融券、市场情绪等多维度数据，提升预测准确性
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from cache.manager import CacheManager


class EnhancedDataFetcher:
    """增强数据获取器 - 多维度市场数据整合"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
    async def get_northbound_capital_data(self, sector_name: str, 
                                        date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        获取北向资金数据
        
        Args:
            sector_name: 板块名称
            date_range: 日期范围
            
        Returns:
            Dict: 北向资金流向数据
        """
        cache_key = f"northbound_capital_{sector_name}_{date_range[0]}_{date_range[1]}"
        
        try:
            # 尝试从缓存获取
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
                
            if AKSHARE_AVAILABLE:
                # 获取北向资金行业流向数据
                try:
                    # 使用AKShare获取北向资金数据
                    df = ak.stock_connect_sector_summary_sw(
                        start_date=date_range[0], 
                        end_date=date_range[1]
                    )
                    
                    if df is not None and not df.empty:
                        # 筛选特定板块数据
                        sector_data = df[df['申万一级行业'] == sector_name]
                        
                        if not sector_data.empty:
                            northbound_analysis = {
                                'total_net_inflow': float(sector_data['净买入额'].sum()),
                                'avg_daily_inflow': float(sector_data['净买入额'].mean()),
                                'max_single_day_inflow': float(sector_data['净买入额'].max()),
                                'min_single_day_inflow': float(sector_data['净买入额'].min()),
                                'positive_days': int((sector_data['净买入额'] > 0).sum()),
                                'negative_days': int((sector_data['净买入额'] < 0).sum()),
                                'flow_trend': self._analyze_capital_trend(sector_data['净买入额'].tolist()),
                                'data_quality': 'real'
                            }
                        else:
                            self.logger.error("北向资金数据处理失败")
                            raise RuntimeError("北向资金数据处理失败，无法获取真实数据")
                    else:
                        self.logger.error("北向资金数据获取失败")
                        raise RuntimeError("北向资金数据获取失败，请检查数据源配置")
                        
                except Exception as e:
                    self.logger.error(f"获取北向资金数据失败: {e}")
                    raise RuntimeError(f"获取北向资金数据失败: {e}")
            else:
                self.logger.error("AKShare不可用，无法获取北向资金数据")
                raise RuntimeError("AKShare不可用，无法获取北向资金数据")
                
            # 缓存数据
            await self.cache_manager.set(cache_key, northbound_analysis)
            return northbound_analysis
            
        except Exception as e:
            self.logger.error(f"获取北向资金数据异常: {e}")
            raise RuntimeError(f"获取北向资金数据异常: {e}")
            
    async def get_margin_trading_data(self, sector_name: str, 
                                    date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        获取融资融券数据
        
        Args:
            sector_name: 板块名称  
            date_range: 日期范围
            
        Returns:
            Dict: 融资融券数据
        """
        cache_key = f"margin_trading_{sector_name}_{date_range[0]}_{date_range[1]}"
        
        try:
            # 尝试从缓存获取
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
                
            if AKSHARE_AVAILABLE:
                try:
                    # 获取融资融券数据（示例API，实际需要根据AKShare具体接口调整）
                    df = ak.stock_margin_detail_sw(
                        start_date=date_range[0],
                        end_date=date_range[1]
                    )
                    
                    if df is not None and not df.empty:
                        # 筛选特定板块
                        sector_data = df[df['申万一级行业'] == sector_name]
                        
                        if not sector_data.empty:
                            margin_analysis = {
                                'financing_balance': float(sector_data['融资余额'].iloc[-1]) if '融资余额' in sector_data.columns else 0,
                                'financing_change_5d': self._calculate_change_rate(
                                    sector_data['融资余额'].tolist(), 5
                                ),
                                'margin_lending_balance': float(sector_data['融券余额'].iloc[-1]) if '融券余额' in sector_data.columns else 0,
                                'net_financing_ratio': self._calculate_net_financing_ratio(sector_data),
                                'margin_activity_level': self._assess_margin_activity(sector_data),
                                'data_quality': 'real'
                            }
                        else:
                            self.logger.error("融资融券数据处理失败")
                            raise RuntimeError("融资融券数据处理失败，无法获取真实数据")
                    else:
                        self.logger.error("融资融券数据获取失败")
                        raise RuntimeError("融资融券数据获取失败，请检查数据源配置")
                        
                except Exception as e:
                    self.logger.error(f"获取融资融券数据失败: {e}")
                    raise RuntimeError(f"获取融资融券数据失败: {e}")
            else:
                self.logger.error("AKShare不可用，无法获取融资融券数据")
                raise RuntimeError("AKShare不可用，无法获取融资融券数据")
                
            await self.cache_manager.set(cache_key, margin_analysis)
            return margin_analysis
            
        except Exception as e:
            self.logger.error(f"获取融资融券数据异常: {e}")
            raise RuntimeError(f"获取融资融券数据异常: {e}")
            
    async def get_market_sentiment_data(self, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        获取市场情绪数据
        
        Args:
            date_range: 日期范围
            
        Returns:
            Dict: 市场情绪指标
        """
        cache_key = f"market_sentiment_{date_range[0]}_{date_range[1]}"
        
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
                
            if AKSHARE_AVAILABLE:
                try:
                    # 获取市场情绪相关数据
                    sentiment_data = {}
                    
                    # 1. 恐慌贪婪指数（如果有相关API）
                    try:
                        fear_greed_df = ak.index_fear_greed_sina()
                        if fear_greed_df is not None and not fear_greed_df.empty:
                            latest_value = fear_greed_df['value'].iloc[-1]
                            sentiment_data['fear_greed_index'] = float(latest_value)
                    except:
                        sentiment_data['fear_greed_index'] = 50  # 中性值
                        
                    # 2. 新股发行情况
                    try:
                        ipo_df = ak.stock_ipo_summary_cninfo()
                        if ipo_df is not None and not ipo_df.empty:
                            recent_ipos = len(ipo_df[ipo_df['上市日期'] >= date_range[0]])
                            sentiment_data['recent_ipo_count'] = recent_ipos
                    except:
                        sentiment_data['recent_ipo_count'] = 5
                        
                    # 3. 涨跌停数据
                    try:
                        limit_df = ak.stock_em_zt_pool_dtgc()
                        if limit_df is not None and not limit_df.empty:
                            sentiment_data['limit_up_count'] = len(limit_df)
                    except:
                        sentiment_data['limit_up_count'] = 20
                        
                    # 4. 计算综合情绪指数
                    sentiment_data.update(self._calculate_comprehensive_sentiment(sentiment_data))
                    sentiment_data['data_quality'] = 'real'
                    
                except Exception as e:
                    self.logger.error(f"获取市场情绪数据失败: {e}")
                    raise RuntimeError(f"获取市场情绪数据失败: {e}")
            else:
                self.logger.error("AKShare不可用，无法获取市场情绪数据")
                raise RuntimeError("AKShare不可用，无法获取市场情绪数据")
                
            await self.cache_manager.set(cache_key, sentiment_data)
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"获取市场情绪数据异常: {e}")
            raise RuntimeError(f"获取市场情绪数据异常: {e}")
            
    async def get_macro_environment_data(self, date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        获取宏观环境数据
        
        Args:
            date_range: 日期范围
            
        Returns:
            Dict: 宏观环境指标
        """
        cache_key = f"macro_environment_{date_range[0]}_{date_range[1]}"
        
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
                
            macro_data = {}
            
            if AKSHARE_AVAILABLE:
                try:
                    # 1. 沪深300指数表现
                    hs300_df = ak.index_zh_a_hist(symbol="000300", period="daily",
                                                start_date=date_range[0], end_date=date_range[1])
                    if hs300_df is not None and not hs300_df.empty:
                        macro_data['hs300_return_5d'] = self._calculate_return(
                            hs300_df['收盘'].tolist(), 5
                        )
                        macro_data['hs300_volatility'] = float(hs300_df['收盘'].pct_change().std() * 100)
                    
                    # 2. 创业板指数表现
                    cyb_df = ak.index_zh_a_hist(symbol="399006", period="daily", 
                                              start_date=date_range[0], end_date=date_range[1])
                    if cyb_df is not None and not cyb_df.empty:
                        macro_data['cyb_return_5d'] = self._calculate_return(
                            cyb_df['收盘'].tolist(), 5
                        )
                        
                    # 3. 科创50指数表现
                    kc50_df = ak.index_zh_a_hist(symbol="000688", period="daily",
                                               start_date=date_range[0], end_date=date_range[1])
                    if kc50_df is not None and not kc50_df.empty:
                        macro_data['kc50_return_5d'] = self._calculate_return(
                            kc50_df['收盘'].tolist(), 5
                        )
                        
                    # 4. 市场风格判断
                    macro_data['market_style'] = self._determine_market_style(macro_data)
                    macro_data['data_quality'] = 'real'
                    
                except Exception as e:
                    self.logger.error(f"获取宏观数据失败: {e}")
                    raise RuntimeError(f"获取宏观数据失败: {e}")
            else:
                self.logger.error("AKShare不可用，无法获取宏观数据")
                raise RuntimeError("AKShare不可用，无法获取宏观数据")
                
            await self.cache_manager.set(cache_key, macro_data)
            return macro_data
            
        except Exception as e:
            self.logger.error(f"获取宏观环境数据异常: {e}")
            raise RuntimeError(f"获取宏观环境数据异常: {e}")
            
    def _analyze_capital_trend(self, flow_data: List[float]) -> str:
        """分析资金流向趋势"""
        if not flow_data or len(flow_data) < 3:
            return "neutral"
            
        recent_3_days = flow_data[-3:]
        positive_days = sum(1 for x in recent_3_days if x > 0)
        
        if positive_days >= 2:
            return "inflow"
        elif positive_days <= 1:
            return "outflow"
        else:
            return "neutral"
            
    def _calculate_change_rate(self, data_series: List[float], periods: int) -> float:
        """计算变化率"""
        if len(data_series) < periods + 1:
            return 0.0
            
        current = data_series[-1]
        previous = data_series[-periods-1]
        
        if previous != 0:
            return round((current / previous - 1) * 100, 2)
        return 0.0
        
    def _calculate_net_financing_ratio(self, margin_data: pd.DataFrame) -> float:
        """计算净融资比率"""
        try:
            if '融资余额' in margin_data.columns and '融券余额' in margin_data.columns:
                financing = margin_data['融资余额'].iloc[-1]
                lending = margin_data['融券余额'].iloc[-1]
                
                total = financing + lending
                if total > 0:
                    return round((financing - lending) / total * 100, 2)
            return 0.0
        except:
            return 0.0
            
    def _assess_margin_activity(self, margin_data: pd.DataFrame) -> str:
        """评估融资融券活跃度"""
        try:
            if len(margin_data) >= 5:
                recent_avg = margin_data['融资余额'].tail(5).mean()
                historical_avg = margin_data['融资余额'].mean()
                
                if recent_avg > historical_avg * 1.1:
                    return "high"
                elif recent_avg < historical_avg * 0.9:
                    return "low"
                else:
                    return "normal"
            return "normal"
        except:
            return "normal"
            
    def _calculate_comprehensive_sentiment(self, sentiment_data: Dict) -> Dict[str, Any]:
        """计算综合情绪指数"""
        try:
            # 基于多个指标计算综合情绪
            fear_greed = sentiment_data.get('fear_greed_index', 50)
            ipo_activity = min(sentiment_data.get('recent_ipo_count', 5) * 10, 100)
            limit_up_activity = min(sentiment_data.get('limit_up_count', 20) * 2, 100)
            
            # 综合情绪指数 (0-100)
            comprehensive_sentiment = (fear_greed * 0.4 + ipo_activity * 0.3 + limit_up_activity * 0.3)
            
            if comprehensive_sentiment >= 70:
                sentiment_level = "optimistic"
            elif comprehensive_sentiment >= 50:
                sentiment_level = "neutral"
            elif comprehensive_sentiment >= 30:
                sentiment_level = "cautious"
            else:
                sentiment_level = "pessimistic"
                
            return {
                'comprehensive_sentiment_index': round(comprehensive_sentiment, 1),
                'sentiment_level': sentiment_level
            }
        except:
            return {
                'comprehensive_sentiment_index': 50.0,
                'sentiment_level': "neutral"
            }
            
    def _calculate_return(self, price_series: List[float], periods: int) -> float:
        """计算收益率"""
        if len(price_series) < periods + 1:
            return 0.0
            
        current = price_series[-1]
        previous = price_series[-periods-1]
        
        if previous > 0:
            return round((current / previous - 1) * 100, 2)
        return 0.0
        
    def _determine_market_style(self, macro_data: Dict) -> str:
        """判断市场风格"""
        try:
            hs300_return = macro_data.get('hs300_return_5d', 0)
            cyb_return = macro_data.get('cyb_return_5d', 0)
            kc50_return = macro_data.get('kc50_return_5d', 0)
            
            # 成长股相对价值股的表现
            growth_vs_value = (cyb_return + kc50_return) / 2 - hs300_return
            
            if growth_vs_value > 2:
                return "growth"
            elif growth_vs_value < -2:
                return "value"
            else:
                return "balanced"
        except:
            return "balanced"
            
