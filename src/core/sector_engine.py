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
        
    async def analyze_all_sectors(
        self, 
        lookback_months: int = 6,
        top_n: int = 5
    ) -> Dict:
        """
        分析所有板块并返回完整排名（突出显示TOP N）
        
        Args:
            lookback_months: 回望月数
            top_n: 返回前N个板块数量（用于突出显示）
            
        Returns:
            Dict: 包含所有板块排名和分析结果，TOP N突出显示
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
                'data_period': {
                    'start_date': (datetime.now() - timedelta(days=lookback_months * 30)).strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'lookback_months': lookback_months
                },
                'analysis_params': {
                    'lookback_months': lookback_months,
                    'top_n': top_n,
                    'scoring_weights': self.scoring_weights,
                    'algorithm_version': 'Enhanced-v1.3.0'
                },
                'total_sectors_analyzed': len(sector_scores),
                'all_sectors': sector_scores,  # 完整列表
                'top_sectors': top_sectors,    # TOP N突出显示
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
                'sector_code': self.sw_sectors.get(sector_name, 'N/A'),
                'composite_score': round(composite_score, 1),
                'momentum_score': round(momentum_score, 1),
                'relative_strength_score': round(relative_strength_score, 1),
                'investment_logic': investment_logic,
                'latest_price': float(df.iloc[-1]['close']) if 'close' in df.columns else 0.0,
                'price_change_5d': round(self._calculate_price_change(df, 5), 2),
                'price_change_10d': round(self._calculate_price_change(df, 10), 2),
                'price_change_20d': round(self._calculate_price_change(df, 20), 2),
                'volume_trend': self._calculate_volume_trend(df),
                'volume_ratio': self._calculate_volume_ratio(df),
                'risk_level': self._assess_risk_level(df),
                'volatility': round(self._calculate_volatility(df), 3),
                'confidence_level': round(self._calculate_confidence(momentum_score, relative_strength_score), 3),
                'data_quality': self._assess_data_quality(df),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        
        if recent_volume > historical_volume * 1.5:
            return "surge"      # 放量
        elif recent_volume > historical_volume * 1.2:
            return "increasing" # 温和放量
        elif recent_volume < historical_volume * 0.7:
            return "shrinking"  # 缩量
        elif recent_volume < historical_volume * 0.8:
            return "decreasing" # 温和缩量
        else:
            return "stable"     # 平稳
            
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """计算成交量比率"""
        if 'volume' not in df.columns or len(df) < 10:
            return 1.0
            
        recent_volume = np.mean(df['volume'].iloc[-5:])
        historical_volume = np.mean(df['volume'].iloc[-20:-5]) if len(df) >= 20 else np.mean(df['volume'].iloc[:-5])
        
        return round(recent_volume / historical_volume if historical_volume > 0 else 1.0, 2)
        
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """计算年化波动率"""
        if len(df) < 20:
            return 0.0
            
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        return volatility
        
    def _assess_data_quality(self, df: pd.DataFrame) -> str:
        """评估数据质量"""
        if len(df) < 10:
            return "poor"
        elif len(df) < 30:
            return "fair"
        elif len(df) < 90:
            return "good"
        else:
            return "excellent"
    
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
        """生成详细投资逻辑说明"""
        
        logic_parts = []
        
        # 1. 综合评分与投资建议
        if composite_score >= 80:
            logic_parts.append(f"【强烈推荐⭐⭐⭐】{sector_name}板块综合评分{composite_score:.1f}分，属于优质投资标的")
            logic_parts.append("建议重点关注，可适当增加配置权重")
        elif composite_score >= 70:
            logic_parts.append(f"【积极推荐⭐⭐】{sector_name}板块综合评分{composite_score:.1f}分，投资价值较高")
            logic_parts.append("建议标准配置，密切跟踪")
        elif composite_score >= 60:
            logic_parts.append(f"【适度推荐⭐】{sector_name}板块综合评分{composite_score:.1f}分，表现稳健")
            logic_parts.append("可考虑适量配置")
        elif composite_score >= 40:
            logic_parts.append(f"【中性观点】{sector_name}板块综合评分{composite_score:.1f}分，走势平稳")
            logic_parts.append("建议观望，等待更好时机")
        else:
            logic_parts.append(f"【谨慎观望】{sector_name}板块综合评分{composite_score:.1f}分，短期承压")
            logic_parts.append("不建议新增投资，考虑减仓")
        
        # 2. 技术面分析
        if momentum_score >= 70:
            if momentum_score >= 85:
                logic_parts.append(f"技术面：动量指标极强({momentum_score:.1f}分)，多头趋势确立，短期上涨动能充足")
            else:
                logic_parts.append(f"技术面：动量指标强劲({momentum_score:.1f}分)，价格趋势向上，技术面支撑较好")
        elif momentum_score >= 50:
            logic_parts.append(f"技术面：动量指标中性({momentum_score:.1f}分)，价格波动平稳，缺乏明确方向")
        elif momentum_score >= 30:
            logic_parts.append(f"技术面：动量指标偏弱({momentum_score:.1f}分)，价格承压，技术面偏空")
        else:
            logic_parts.append(f"技术面：动量指标很弱({momentum_score:.1f}分)，下行压力较大，技术面恶化")
        
        # 3. 相对强弱分析
        if relative_strength_score >= 70:
            logic_parts.append(f"相对表现：板块强弱指标优秀({relative_strength_score:.1f}分)，相比大盘具备明显优势，资金青睐")
        elif relative_strength_score >= 50:
            logic_parts.append(f"相对表现：板块强弱指标平稳({relative_strength_score:.1f}分)，与大盘同步波动")
        elif relative_strength_score >= 30:
            logic_parts.append(f"相对表现：板块强弱指标偏弱({relative_strength_score:.1f}分)，跑输大盘，资金流出")
        else:
            logic_parts.append(f"相对表现：板块强弱指标很弱({relative_strength_score:.1f}分)，大幅跑输大盘，避险情绪浓厚")
        
        # 4. 风险提示与操作建议
        risk_level = "高" if composite_score >= 75 else "中" if composite_score >= 50 else "低"
        logic_parts.append(f"风险收益比：{risk_level}，建议采用{'积极' if composite_score >= 70 else '稳健' if composite_score >= 50 else '保守'}型投资策略")
        
        return "；".join(logic_parts)
    
    async def _generate_market_overview(self, sector_scores: List[Dict]) -> Dict:
        """生成详细市场概览"""
        if not sector_scores:
            return {}
            
        scores = [s['composite_score'] for s in sector_scores]
        avg_score = np.mean(scores)
        
        # 分级统计
        excellent_count = len([s for s in scores if s >= 80])  # 优秀
        good_count = len([s for s in scores if 70 <= s < 80])   # 良好
        fair_count = len([s for s in scores if 50 <= s < 70])   # 一般
        poor_count = len([s for s in scores if s < 50])         # 较差
        
        return {
            'total_sectors': len(sector_scores),
            'average_score': round(avg_score, 1),
            'score_std': round(np.std(scores), 1),
            'score_distribution': {
                'excellent': {'count': excellent_count, 'percentage': round(excellent_count/len(scores)*100, 1)},
                'good': {'count': good_count, 'percentage': round(good_count/len(scores)*100, 1)},
                'fair': {'count': fair_count, 'percentage': round(fair_count/len(scores)*100, 1)},
                'poor': {'count': poor_count, 'percentage': round(poor_count/len(scores)*100, 1)}
            },
            'strong_sectors_count': excellent_count + good_count,
            'weak_sectors_count': poor_count,
            'market_sentiment': self._assess_market_sentiment(avg_score),
            'market_analysis': self._generate_market_analysis(avg_score, sector_scores),
            'top_3_sectors': [s['sector_name'] for s in sorted(sector_scores, key=lambda x: x['composite_score'], reverse=True)[:3]],
            'bottom_3_sectors': [s['sector_name'] for s in sorted(sector_scores, key=lambda x: x['composite_score'])[:3]]
        }
        
    def _generate_market_analysis(self, avg_score: float, sector_scores: List[Dict]) -> str:
        """生成市场分析报告"""
        analysis_parts = []
        
        # 整体市场判断
        if avg_score >= 70:
            analysis_parts.append(f"市场整体表现强劲，平均得分{avg_score:.1f}分，多数板块呈现积极态势")
        elif avg_score >= 60:
            analysis_parts.append(f"市场整体表现稳健，平均得分{avg_score:.1f}分，板块分化程度适中")
        elif avg_score >= 50:
            analysis_parts.append(f"市场整体表现平稳，平均得分{avg_score:.1f}分，板块走势相对均衡")
        else:
            analysis_parts.append(f"市场整体承压，平均得分{avg_score:.1f}分，多数板块表现低迷")
        
        # 板块表现分析
        high_momentum_sectors = [s for s in sector_scores if s['momentum_score'] >= 70]
        if high_momentum_sectors:
            analysis_parts.append(f"技术面较强的板块包括：{', '.join([s['sector_name'] for s in high_momentum_sectors[:3]])}等")
        
        # 投资建议
        if avg_score >= 65:
            analysis_parts.append("建议：积极布局优质板块，把握结构性机会")
        elif avg_score >= 50:
            analysis_parts.append("建议：稳健配置，关注龙头板块的投资机会")
        else:
            analysis_parts.append("建议：保持谨慎，重点关注防御性板块")
            
        return "；".join(analysis_parts)
    
    def _assess_market_sentiment(self, avg_score: float) -> str:
        """评估市场情绪"""
        if avg_score >= 75:
            return "very_optimistic"   # 非常乐观
        elif avg_score >= 65:
            return "optimistic"        # 乐观
        elif avg_score >= 55:
            return "neutral_positive"  # 中性偏乐观
        elif avg_score >= 45:
            return "neutral"           # 中性
        elif avg_score >= 35:
            return "cautious"          # 谨慎
        else:
            return "pessimistic"       # 悲观