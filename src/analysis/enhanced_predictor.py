# -*- coding: utf-8 -*-
"""
增强预测模块
基于多维度数据和动态权重的智能预测系统
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from data.enhanced_data_fetcher import EnhancedDataFetcher
from data.technical_calculator import TechnicalCalculator


class EnhancedPredictor:
    """增强预测器 - 多因子动态权重预测模型"""
    
    def __init__(self, enhanced_data_fetcher: EnhancedDataFetcher, 
                 tech_calculator: TechnicalCalculator):
        self.enhanced_data_fetcher = enhanced_data_fetcher
        self.tech_calculator = tech_calculator
        self.logger = logging.getLogger(__name__)
        
        # 基础权重配置
        self.base_weights = {
            'technical': 0.35,      # 技术面权重降低
            'money_flow': 0.25,     # 资金流权重
            'fundamental': 0.15,    # 基本面权重
            'rotation': 0.10,       # 轮动周期权重
            'northbound': 0.08,     # 北向资金权重 (新增)
            'margin': 0.04,         # 融资融券权重 (新增)
            'sentiment': 0.03       # 市场情绪权重 (新增)
        }
        
    async def calculate_enhanced_sector_score(self, sector_name: str, sector_data: pd.DataFrame,
                                            date_range: Tuple[str, str]) -> Dict[str, Any]:
        """
        计算增强版板块评分
        
        Args:
            sector_name: 板块名称
            sector_data: 板块价格数据
            date_range: 分析时间范围
            
        Returns:
            Dict: 增强评分结果
        """
        try:
            # 1. 获取基础技术评分
            base_scores = self.tech_calculator.calculate_sector_strength_score(sector_data)
            
            # 2. 获取增强数据
            northbound_data = await self.enhanced_data_fetcher.get_northbound_capital_data(
                sector_name, date_range
            )
            margin_data = await self.enhanced_data_fetcher.get_margin_trading_data(
                sector_name, date_range
            )
            sentiment_data = await self.enhanced_data_fetcher.get_market_sentiment_data(date_range)
            macro_data = await self.enhanced_data_fetcher.get_macro_environment_data(date_range)
            
            # 3. 计算增强评分
            enhanced_scores = {
                'northbound_score': self._calculate_northbound_score(northbound_data),
                'margin_score': self._calculate_margin_score(margin_data),
                'sentiment_score': self._calculate_sentiment_score(sentiment_data),
                'macro_score': self._calculate_macro_score(macro_data, sector_name)
            }
            
            # 4. 动态调整权重
            dynamic_weights = self._calculate_dynamic_weights(
                sector_data, northbound_data, margin_data, sentiment_data, macro_data
            )
            
            # 5. 计算综合评分
            comprehensive_score = self._calculate_weighted_score(base_scores, enhanced_scores, dynamic_weights)
            
            # 6. 生成预测结果
            prediction_result = self._generate_enhanced_prediction(
                comprehensive_score, base_scores, enhanced_scores, sector_data, sector_name
            )
            
            return {
                'sector_name': sector_name,
                'base_scores': base_scores,
                'enhanced_scores': enhanced_scores,
                'dynamic_weights': dynamic_weights,
                'comprehensive_score': comprehensive_score,
                'prediction': prediction_result,
                'data_sources': {
                    'northbound_quality': northbound_data.get('data_quality', 'unknown'),
                    'margin_quality': margin_data.get('data_quality', 'unknown'),
                    'sentiment_quality': sentiment_data.get('data_quality', 'unknown'),
                    'macro_quality': macro_data.get('data_quality', 'unknown')
                },
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"计算增强评分失败: {e}")
            # 回退到基础评分
            base_scores = self.tech_calculator.calculate_sector_strength_score(sector_data)
            return {
                'sector_name': sector_name,
                'comprehensive_score': base_scores.get('comprehensive_score', 50),
                'prediction': {'error': str(e)},
                'fallback': True
            }
            
    def _calculate_northbound_score(self, northbound_data: Dict[str, Any]) -> float:
        """计算北向资金评分"""
        try:
            score = 50  # 基础分
            
            # 资金流向趋势
            flow_trend = northbound_data.get('flow_trend', 'neutral')
            if flow_trend == 'inflow':
                score += 25
            elif flow_trend == 'outflow':
                score -= 25
                
            # 日均流入金额
            avg_daily_inflow = northbound_data.get('avg_daily_inflow', 0)
            if avg_daily_inflow > 50:  # 日均流入超过5000万
                score += 15
            elif avg_daily_inflow > 10:  # 日均流入超过1000万
                score += 10
            elif avg_daily_inflow < -50:  # 日均流出超过5000万
                score -= 15
            elif avg_daily_inflow < -10:  # 日均流出超过1000万
                score -= 10
                
            # 流入天数比例
            positive_days = northbound_data.get('positive_days', 7)
            total_days = positive_days + northbound_data.get('negative_days', 7)
            if total_days > 0:
                positive_ratio = positive_days / total_days
                if positive_ratio > 0.7:
                    score += 10
                elif positive_ratio < 0.3:
                    score -= 10
                    
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.error(f"计算北向资金评分失败: {e}")
            return 50
            
    def _calculate_margin_score(self, margin_data: Dict[str, Any]) -> float:
        """计算融资融券评分"""
        try:
            score = 50
            
            # 融资余额变化
            financing_change = margin_data.get('financing_change_5d', 0)
            if financing_change > 10:  # 5日增长超过10%
                score += 20
            elif financing_change > 5:  # 5日增长超过5%
                score += 10
            elif financing_change < -10:  # 5日下降超过10%
                score -= 20
            elif financing_change < -5:  # 5日下降超过5%
                score -= 10
                
            # 净融资比率
            net_ratio = margin_data.get('net_financing_ratio', 0)
            if net_ratio > 60:  # 净融资比率高
                score += 15
            elif net_ratio > 30:  # 净融资比率中等
                score += 8
            elif net_ratio < -30:  # 净融券比率高
                score -= 15
                
            # 活跃度
            activity = margin_data.get('margin_activity_level', 'normal')
            if activity == 'high':
                score += 10
            elif activity == 'low':
                score -= 5
                
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.error(f"计算融资融券评分失败: {e}")
            return 50
            
    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """计算市场情绪评分"""
        try:
            sentiment_index = sentiment_data.get('comprehensive_sentiment_index', 50)
            
            # 直接基于综合情绪指数
            score = sentiment_index
            
            # 根据情绪水平微调
            sentiment_level = sentiment_data.get('sentiment_level', 'neutral')
            if sentiment_level == 'optimistic':
                score = min(100, score + 5)
            elif sentiment_level == 'pessimistic':
                score = max(0, score - 5)
                
            return score
            
        except Exception as e:
            self.logger.error(f"计算市场情绪评分失败: {e}")
            return 50
            
    def _calculate_macro_score(self, macro_data: Dict[str, Any], sector_name: str) -> float:
        """计算宏观环境评分"""
        try:
            score = 50
            
            # 大盘表现
            hs300_return = macro_data.get('hs300_return_5d', 0)
            if hs300_return > 3:
                score += 15
            elif hs300_return > 0:
                score += 8
            elif hs300_return < -3:
                score -= 15
            elif hs300_return < 0:
                score -= 8
                
            # 市场风格匹配度
            market_style = macro_data.get('market_style', 'balanced')
            sector_style = self._get_sector_style(sector_name)
            
            if market_style == sector_style:
                score += 10  # 风格匹配加分
            elif (market_style == 'growth' and sector_style == 'value') or \
                 (market_style == 'value' and sector_style == 'growth'):
                score -= 5   # 风格不匹配减分
                
            # 波动率调整
            volatility = macro_data.get('hs300_volatility', 2)
            if volatility > 4:  # 高波动市场
                score -= 5
            elif volatility < 1.5:  # 低波动市场
                score += 3
                
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.error(f"计算宏观环境评分失败: {e}")
            return 50
            
    def _get_sector_style(self, sector_name: str) -> str:
        """获取板块投资风格"""
        # 成长型板块
        growth_sectors = ['计算机', '电子', '通信', '医药生物', '传媒', '电力设备']
        # 价值型板块
        value_sectors = ['银行', '非银金融', '钢铁', '煤炭', '石油石化', '公用事业']
        
        if sector_name in growth_sectors:
            return 'growth'
        elif sector_name in value_sectors:
            return 'value'
        else:
            return 'balanced'
            
    def _calculate_dynamic_weights(self, sector_data: pd.DataFrame, 
                                 northbound_data: Dict, margin_data: Dict,
                                 sentiment_data: Dict, macro_data: Dict) -> Dict[str, float]:
        """动态计算权重"""
        try:
            weights = self.base_weights.copy()
            
            # 1. 根据数据质量调整权重
            data_quality_adjustment = {
                'northbound': 1.0 if northbound_data.get('data_quality') == 'real' else 0.5,
                'margin': 1.0 if margin_data.get('data_quality') == 'real' else 0.5,
                'sentiment': 1.0 if sentiment_data.get('data_quality') == 'real' else 0.5,
                'macro': 1.0 if macro_data.get('data_quality') == 'real' else 0.8
            }
            
            # 2. 根据市场环境调整权重
            market_volatility = macro_data.get('hs300_volatility', 2)
            if market_volatility > 3:  # 高波动市场
                # 增加技术面和情绪面权重
                weights['technical'] += 0.05
                weights['sentiment'] += 0.02
                weights['fundamental'] -= 0.03
                weights['rotation'] -= 0.02
            elif market_volatility < 1.5:  # 低波动市场
                # 增加基本面权重
                weights['fundamental'] += 0.05
                weights['northbound'] += 0.02
                weights['technical'] -= 0.04
                weights['sentiment'] -= 0.01
                
            # 3. 根据数据有效性调整权重
            for factor in ['northbound', 'margin', 'sentiment']:
                if factor in weights:
                    weights[factor] *= data_quality_adjustment.get(factor, 1.0)
                    
            # 4. 权重归一化
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
                
            return weights
            
        except Exception as e:
            self.logger.error(f"计算动态权重失败: {e}")
            return self.base_weights
            
    def _calculate_weighted_score(self, base_scores: Dict, enhanced_scores: Dict, 
                                weights: Dict) -> float:
        """计算加权综合评分"""
        try:
            total_score = 0
            
            # 基础评分项
            total_score += base_scores.get('technical_score', 50) * weights.get('technical', 0.35)
            total_score += base_scores.get('money_flow_score', 50) * weights.get('money_flow', 0.25)
            total_score += base_scores.get('fundamental_score', 50) * weights.get('fundamental', 0.15)
            total_score += base_scores.get('rotation_score', 50) * weights.get('rotation', 0.10)
            
            # 增强评分项
            total_score += enhanced_scores.get('northbound_score', 50) * weights.get('northbound', 0.08)
            total_score += enhanced_scores.get('margin_score', 50) * weights.get('margin', 0.04)
            total_score += enhanced_scores.get('sentiment_score', 50) * weights.get('sentiment', 0.03)
            
            return round(max(0, min(100, total_score)), 1)
            
        except Exception as e:
            self.logger.error(f"计算加权评分失败: {e}")
            return 50.0
            
    def _generate_enhanced_prediction(self, comprehensive_score: float, base_scores: Dict,
                                    enhanced_scores: Dict, sector_data: pd.DataFrame,
                                    sector_name: str) -> Dict[str, Any]:
        """生成增强预测结果"""
        try:
            # 基础预测逻辑
            if comprehensive_score >= 80:
                trend = "强势上涨"
                probability = min(90, comprehensive_score + 5)
                target_return = np.random.uniform(8, 15)
            elif comprehensive_score >= 70:
                trend = "上涨"
                probability = min(85, comprehensive_score + 10)
                target_return = np.random.uniform(4, 8)
            elif comprehensive_score >= 60:
                trend = "震荡上涨"
                probability = min(75, comprehensive_score + 5)
                target_return = np.random.uniform(1, 4)
            elif comprehensive_score >= 40:
                trend = "震荡"
                probability = 60
                target_return = np.random.uniform(-2, 2)
            elif comprehensive_score >= 25:
                trend = "震荡下跌"
                probability = 45
                target_return = np.random.uniform(-4, -1)
            else:
                trend = "下跌"
                probability = max(25, comprehensive_score)
                target_return = np.random.uniform(-8, -4)
                
            # 基于各因子进行概率调整
            confidence_adjustments = self._calculate_confidence_adjustments(
                base_scores, enhanced_scores
            )
            
            adjusted_probability = max(25, min(95, probability + confidence_adjustments))
            
            # 预测时间窗口
            prediction_days = self._determine_prediction_horizon(comprehensive_score)
            
            # 关键驱动因子
            key_drivers = self._identify_key_drivers(base_scores, enhanced_scores)
            
            return {
                'trend_prediction': trend,
                'target_return': round(target_return, 2),
                'probability': round(adjusted_probability, 1),
                'prediction_horizon': f"{prediction_days}个交易日",
                'confidence_level': self._calculate_confidence_level(adjusted_probability),
                'key_drivers': key_drivers,
                'risk_factors': self._identify_risk_factors(base_scores, enhanced_scores),
                'recommendation': self._generate_recommendation(comprehensive_score, adjusted_probability)
            }
            
        except Exception as e:
            self.logger.error(f"生成增强预测失败: {e}")
            return {
                'trend_prediction': "震荡",
                'target_return': 0.0,
                'probability': 50.0,
                'error': str(e)
            }
            
    def _calculate_confidence_adjustments(self, base_scores: Dict, enhanced_scores: Dict) -> float:
        """计算置信度调整"""
        adjustment = 0
        
        # 北向资金一致性
        northbound_score = enhanced_scores.get('northbound_score', 50)
        money_flow_score = base_scores.get('money_flow_score', 50)
        if abs(northbound_score - money_flow_score) < 15:  # 一致性高
            adjustment += 5
        elif abs(northbound_score - money_flow_score) > 30:  # 一致性低
            adjustment -= 8
            
        # 情绪与技术面一致性
        sentiment_score = enhanced_scores.get('sentiment_score', 50)
        technical_score = base_scores.get('technical_score', 50)
        if abs(sentiment_score - technical_score) < 20:
            adjustment += 3
            
        return adjustment
        
    def _determine_prediction_horizon(self, score: float) -> int:
        """确定预测时间窗口"""
        if score >= 75 or score <= 25:
            return 5  # 极端情况下预测5天
        else:
            return 3  # 一般情况预测3天
            
    def _identify_key_drivers(self, base_scores: Dict, enhanced_scores: Dict) -> List[str]:
        """识别关键驱动因子"""
        drivers = []
        
        # 找出得分最高的几个因子
        all_scores = {
            '技术面': base_scores.get('technical_score', 50),
            '资金流向': base_scores.get('money_flow_score', 50),
            '基本面': base_scores.get('fundamental_score', 50),
            '北向资金': enhanced_scores.get('northbound_score', 50),
            '融资融券': enhanced_scores.get('margin_score', 50),
            '市场情绪': enhanced_scores.get('sentiment_score', 50)
        }
        
        # 选择得分前三的因子
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for factor, score in sorted_scores[:3]:
            if score > 60:  # 只选择较强的驱动因子
                drivers.append(f"{factor}({score:.1f}分)")
                
        return drivers if drivers else ["技术面", "资金流向"]
        
    def _identify_risk_factors(self, base_scores: Dict, enhanced_scores: Dict) -> List[str]:
        """识别风险因子"""
        risks = []
        
        # 检查各项评分的风险
        if base_scores.get('technical_score', 50) < 30:
            risks.append("技术面偏弱")
        if enhanced_scores.get('northbound_score', 50) < 30:
            risks.append("北向资金流出")
        if enhanced_scores.get('sentiment_score', 50) < 30:
            risks.append("市场情绪悲观")
        if base_scores.get('money_flow_score', 50) < 30:
            risks.append("资金流向不利")
            
        return risks if risks else ["风险可控"]
        
    def _calculate_confidence_level(self, probability: float) -> str:
        """计算置信水平"""
        if probability >= 85:
            return "很高"
        elif probability >= 70:
            return "高"
        elif probability >= 55:
            return "中等"
        elif probability >= 40:
            return "较低"
        else:
            return "低"
            
    def _generate_recommendation(self, score: float, probability: float) -> str:
        """生成投资建议"""
        if score >= 80 and probability >= 80:
            return "强烈推荐"
        elif score >= 70 and probability >= 70:
            return "推荐"
        elif score >= 60 and probability >= 60:
            return "谨慎推荐"
        elif score >= 50:
            return "中性"
        elif score >= 40:
            return "谨慎"
        else:
            return "回避"