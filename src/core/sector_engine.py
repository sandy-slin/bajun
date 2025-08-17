#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块分析引擎 - Phase 1 MVP核心组件
基于统计模型的TOP5板块预测系统

功能:
1. 基于动量和相对强弱的板块评分
2. 权重配置: 涨跌幅度预测(80%) + 相对强弱指标(20%)
3. 输出TOP5板块排名及投资逻辑
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..data.sector_fetcher import SectorFetcher
from ..data.technical_calculator import TechnicalCalculator
from ..validation.real_data_validator import RealDataValidator


class SectorEngine:
    """板块分析引擎 - 效果优先的统计模型实现"""
    
    def __init__(self, sector_fetcher: SectorFetcher, tech_calculator: TechnicalCalculator):
        self.sector_fetcher = sector_fetcher
        self.tech_calculator = tech_calculator
        self.logger = logging.getLogger(__name__)
        
        # 真实数据验证器 - 严禁模拟数据
        self.data_validator = RealDataValidator()
        
        # 评分权重配置 (符合requirement.md规定)
        self.scoring_weights = {
            'momentum_prediction': 0.8,  # 涨跌幅度预测权重
            'relative_strength': 0.2     # 相对强弱指标权重
        }
        
        # 技术参数
        self.lookback_days = 120  # 6个月历史数据
        self.momentum_window = 20 # 动量计算窗口
        self.rsi_window = 14      # RSI计算窗口
        
    async def analyze_top_sectors(
        self, 
        lookback_months: int = 6,
        top_n: int = 5
    ) -> Dict:
        """
        分析并返回TOP N板块排名
        
        Args:
            lookback_months: 回望月数
            top_n: 返回前N个板块
            
        Returns:
            Dict: 包含TOP5板块排名和分析结果
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"开始板块分析，回望{lookback_months}个月，TOP{top_n}")
            
            # 1. 获取所有板块数据
            sectors = self.sector_fetcher.get_supported_sectors()
            sector_scores = []
            
            for sector_name in sectors:
                try:
                    score_result = await self._calculate_sector_score(
                        sector_name, lookback_months
                    )
                    if score_result:
                        sector_scores.append(score_result)
                        
                except Exception as e:
                    self.logger.warning(f"板块{sector_name}评分失败: {e}")
                    continue
            
            # 2. 排序并选择TOP N
            sector_scores.sort(key=lambda x: x['composite_score'], reverse=True)
            top_sectors = sector_scores[:top_n]
            
            # 3. 生成分析结果
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'analysis_params': {
                    'lookback_months': lookback_months,
                    'top_n': top_n,
                    'scoring_weights': self.scoring_weights
                },
                'total_sectors_analyzed': len(sector_scores),
                'top_sectors': top_sectors,
                'market_overview': await self._generate_market_overview(sector_scores),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            self.logger.info(f"板块分析完成，耗时{analysis_result['processing_time_seconds']:.1f}秒")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"板块分析失败: {e}")
            return {'error': str(e)}
    
    async def _calculate_sector_score(
        self, 
        sector_name: str, 
        lookback_months: int
    ) -> Optional[Dict]:
        """
        计算单个板块的综合评分
        
        Args:
            sector_name: 板块名称
            lookback_months: 回望月数
            
        Returns:
            Dict: 板块评分结果
        """
        try:
            # 获取板块历史数据
            sector_data = await self.sector_fetcher.get_sector_data(
                sector_name, 
                days=lookback_months * 30
            )
            
            if sector_data is None or len(sector_data) < 30:
                return None
            
            # 真实数据验证 - 严禁模拟数据
            try:
                validation_result = await self.data_validator.validate_and_ensure_real_data(
                    sector_data, f"sector_fetcher_{sector_name}"
                )
                self.logger.debug(f"板块{sector_name}数据验证通过，质量评分: {validation_result['quality_score']:.2f}")
            except ValueError as e:
                self.logger.error(f"板块{sector_name}数据验证失败: {e}")
                raise ValueError(f"检测到非真实数据，拒绝处理: {e}")
            except Exception as e:
                self.logger.warning(f"板块{sector_name}数据验证异常: {e}")
                # 验证异常时也拒绝处理，确保数据安全
                raise ValueError(f"数据验证异常，拒绝处理: {e}")
                
            df = pd.DataFrame(sector_data)
            
            # 1. 计算动量预测评分 (80%权重)
            momentum_score = self._calculate_momentum_prediction(df)
            
            # 2. 计算相对强弱评分 (20%权重)
            relative_strength_score = self._calculate_relative_strength(df)
            
            # 3. 计算综合评分
            composite_score = (
                momentum_score * self.scoring_weights['momentum_prediction'] +
                relative_strength_score * self.scoring_weights['relative_strength']
            )
            
            # 4. 生成投资逻辑
            investment_logic = self._generate_investment_logic(
                sector_name, momentum_score, relative_strength_score, composite_score
            )
            
            return {
                'sector_name': sector_name,
                'composite_score': composite_score,
                'momentum_score': momentum_score,
                'relative_strength_score': relative_strength_score,
                'investment_logic': investment_logic,
                'latest_price': float(df.iloc[-1]['close']) if 'close' in df.columns else 0.0,
                'price_change_5d': self._calculate_price_change(df, 5),
                'volume_trend': self._calculate_volume_trend(df),
                'risk_level': self._assess_risk_level(df),
                'confidence_level': self._calculate_confidence(momentum_score, relative_strength_score)
            }
            
        except Exception as e:
            self.logger.warning(f"计算板块{sector_name}评分失败: {e}")
            return None
    
    def _calculate_momentum_prediction(self, df: pd.DataFrame) -> float:
        """
        计算动量预测评分 (0-100分)
        
        基于价格动量、成交量确认和趋势强度
        """
        try:
            closes = df['close'].values
            volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(closes))
            
            # 价格动量评分
            price_momentum = self._calculate_price_momentum(closes)
            
            # 成交量确认评分
            volume_confirmation = self._calculate_volume_confirmation(closes, volumes)
            
            # 趋势强度评分
            trend_strength = self._calculate_trend_strength(closes)
            
            # 组合评分
            momentum_score = (
                price_momentum * 0.5 +      # 价格动量 50%
                volume_confirmation * 0.3 + # 成交量确认 30%
                trend_strength * 0.2        # 趋势强度 20%
            )
            
            return min(100, max(0, momentum_score))
            
        except Exception as e:
            self.logger.warning(f"计算动量预测评分失败: {e}")
            return 50.0  # 中性评分
    
    def _calculate_relative_strength(self, df: pd.DataFrame) -> float:
        """
        计算相对强弱评分 (0-100分)
        
        基于RSI指标和相对表现
        """
        try:
            closes = df['close'].values
            
            # RSI指标评分
            rsi_score = self._calculate_rsi_score(closes)
            
            # 相对市场表现评分 (简化版，后续可增强)
            relative_performance_score = self._calculate_relative_performance(closes)
            
            # 组合评分
            relative_strength_score = (
                rsi_score * 0.6 +                    # RSI权重 60%
                relative_performance_score * 0.4     # 相对表现权重 40%
            )
            
            return min(100, max(0, relative_strength_score))
            
        except Exception as e:
            self.logger.warning(f"计算相对强弱评分失败: {e}")
            return 50.0  # 中性评分
    
    def _calculate_price_momentum(self, closes: np.ndarray) -> float:
        """计算价格动量评分"""
        if len(closes) < self.momentum_window:
            return 50.0
            
        # 多时间框架动量
        momentum_5d = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
        momentum_10d = (closes[-1] / closes[-11] - 1) * 100 if len(closes) >= 11 else 0
        momentum_20d = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
        
        # 动量加权评分
        weighted_momentum = (
            momentum_5d * 0.5 +   # 5日动量权重最高
            momentum_10d * 0.3 +  # 10日动量
            momentum_20d * 0.2    # 20日动量
        )
        
        # 转换为0-100评分 (假设-10%到+10%的动量范围)
        score = 50 + weighted_momentum * 2.5
        return min(100, max(0, score))
    
    def _calculate_volume_confirmation(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """计算成交量确认评分"""
        if len(volumes) < 10:
            return 50.0
            
        try:
            # 计算价格变化和成交量变化的相关性
            price_changes = np.diff(closes[-20:]) if len(closes) >= 20 else np.diff(closes)
            volume_changes = np.diff(volumes[-20:]) if len(volumes) >= 20 else np.diff(volumes)
            
            if len(price_changes) > 5 and len(volume_changes) > 5:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                    
                # 相关性转换为评分
                score = 50 + correlation * 50
                return min(100, max(0, score))
            
            return 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_trend_strength(self, closes: np.ndarray) -> float:
        """计算趋势强度评分"""
        if len(closes) < 20:
            return 50.0
            
        # 计算移动平均线斜率
        ma_short = np.mean(closes[-5:])   # 5日均线
        ma_long = np.mean(closes[-20:])   # 20日均线
        
        # 趋势方向
        trend_direction = 1 if ma_short > ma_long else -1
        
        # 趋势强度 (基于价格与均线的距离)
        current_price = closes[-1]
        distance_from_ma = abs(current_price - ma_long) / ma_long
        
        # 转换为评分
        base_score = 50
        strength_bonus = distance_from_ma * 100 * trend_direction
        
        score = base_score + strength_bonus
        return min(100, max(0, score))
    
    def _calculate_rsi_score(self, closes: np.ndarray) -> float:
        """计算RSI评分"""
        if len(closes) < self.rsi_window + 1:
            return 50.0
            
        try:
            # 计算RSI
            deltas = np.diff(closes[-self.rsi_window-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # RSI转换为评分 (30-70为正常区间)
            if 30 <= rsi <= 70:
                score = 50 + (rsi - 50) * 0.5  # 中性偏好
            elif rsi < 30:
                score = 70 + (30 - rsi) * 1.0  # 超卖机会
            else:  # rsi > 70
                score = 30 - (rsi - 70) * 1.0  # 超买风险
                
            return min(100, max(0, score))
            
        except Exception:
            return 50.0
    
    def _calculate_relative_performance(self, closes: np.ndarray) -> float:
        """计算相对表现评分 (简化版)"""
        if len(closes) < 20:
            return 50.0
            
        # 简化实现：基于近期表现相对历史均值
        recent_avg = np.mean(closes[-5:])
        historical_avg = np.mean(closes[-20:-5]) if len(closes) >= 20 else np.mean(closes[:-5])
        
        if historical_avg > 0:
            relative_performance = (recent_avg / historical_avg - 1) * 100
            score = 50 + relative_performance * 2
            return min(100, max(0, score))
        
        return 50.0
    
    def _calculate_price_change(self, df: pd.DataFrame, days: int) -> float:
        """计算指定天数的价格变化百分比"""
        if len(df) < days + 1:
            return 0.0
            
        current_price = df.iloc[-1]['close']
        past_price = df.iloc[-(days+1)]['close']
        
        return (current_price / past_price - 1) * 100
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> str:
        """计算成交量趋势"""
        if 'volume' not in df.columns or len(df) < 10:
            return "unknown"
            
        recent_volume = np.mean(df['volume'].iloc[-5:])
        historical_volume = np.mean(df['volume'].iloc[-20:-5]) if len(df) >= 20 else np.mean(df['volume'].iloc[:-5])
        
        if recent_volume > historical_volume * 1.2:
            return "increasing"
        elif recent_volume < historical_volume * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _assess_risk_level(self, df: pd.DataFrame) -> str:
        """评估风险水平"""
        if len(df) < 20:
            return "unknown"
            
        # 基于价格波动率评估风险
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        if volatility < 0.2:
            return "low"
        elif volatility < 0.4:
            return "medium"
        else:
            return "high"
    
    def _calculate_confidence(self, momentum_score: float, relative_strength_score: float) -> float:
        """计算预测置信度"""
        # 基于两个评分的一致性和极端程度
        consistency = 100 - abs(momentum_score - relative_strength_score)
        extremeness = max(abs(momentum_score - 50), abs(relative_strength_score - 50))
        
        confidence = (consistency * 0.6 + extremeness * 0.4) / 100
        return min(1.0, max(0.0, confidence))
    
    def _generate_investment_logic(
        self, 
        sector_name: str, 
        momentum_score: float, 
        relative_strength_score: float, 
        composite_score: float
    ) -> str:
        """生成投资逻辑说明"""
        
        logic_parts = []
        
        # 综合评分逻辑
        if composite_score >= 75:
            logic_parts.append(f"【强烈推荐】{sector_name}板块综合评分{composite_score:.1f}分，处于高位")
        elif composite_score >= 60:
            logic_parts.append(f"【适度推荐】{sector_name}板块综合评分{composite_score:.1f}分，表现良好")
        elif composite_score >= 40:
            logic_parts.append(f"【中性观点】{sector_name}板块综合评分{composite_score:.1f}分，表现平稳")
        else:
            logic_parts.append(f"【谨慎观望】{sector_name}板块综合评分{composite_score:.1f}分，表现偏弱")
        
        # 动量分析逻辑
        if momentum_score >= 70:
            logic_parts.append(f"动量指标强劲({momentum_score:.1f}分)，价格趋势向上")
        elif momentum_score <= 30:
            logic_parts.append(f"动量指标偏弱({momentum_score:.1f}分)，短期承压")
        
        # 相对强弱逻辑
        if relative_strength_score >= 70:
            logic_parts.append(f"相对强弱指标优秀({relative_strength_score:.1f}分)，板块具备相对优势")
        elif relative_strength_score <= 30:
            logic_parts.append(f"相对强弱指标偏弱({relative_strength_score:.1f}分)，相对表现不佳")
        
        return "；".join(logic_parts)
    
    async def _generate_market_overview(self, sector_scores: List[Dict]) -> Dict:
        """生成市场概览"""
        if not sector_scores:
            return {}
            
        scores = [s['composite_score'] for s in sector_scores]
        
        return {
            'total_sectors': len(sector_scores),
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'strong_sectors_count': len([s for s in scores if s >= 70]),
            'weak_sectors_count': len([s for s in scores if s <= 40]),
            'market_sentiment': self._assess_market_sentiment(np.mean(scores))
        }
    
    def _assess_market_sentiment(self, avg_score: float) -> str:
        """评估市场情绪"""
        if avg_score >= 65:
            return "optimistic"
        elif avg_score >= 50:
            return "neutral" 
        elif avg_score >= 35:
            return "cautious"
        else:
            return "pessimistic"