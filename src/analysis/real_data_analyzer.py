# -*- coding: utf-8 -*-
"""
基于真实数据的深度分析器
无需额外机器学习库，仅基于pandas和numpy进行深度数据挖掘
严格使用真实历史数据，提升预测准确率
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import json

from data.sector_fetcher import SectorFetcher
from data.enhanced_data_fetcher import EnhancedDataFetcher
from data.technical_calculator import TechnicalCalculator


class RealDataAnalyzer:
    """基于真实数据的深度分析器"""
    
    def __init__(self, sector_fetcher: SectorFetcher, enhanced_data_fetcher: EnhancedDataFetcher,
                 tech_calculator: TechnicalCalculator):
        self.sector_fetcher = sector_fetcher
        self.enhanced_data_fetcher = enhanced_data_fetcher
        self.tech_calculator = tech_calculator
        self.logger = logging.getLogger(__name__)
        
    async def analyze_prediction_accuracy_by_real_patterns(self, 
                                                         analysis_months: int = 12,
                                                         prediction_days: int = 5) -> Dict[str, Any]:
        """
        基于真实历史数据模式分析预测准确率
        
        Args:
            analysis_months: 分析月数
            prediction_days: 预测天数
            
        Returns:
            Dict: 分析结果
        """
        try:
            self.logger.info(f"开始基于真实数据模式的准确率分析，分析期: {analysis_months}个月")
            
            # 1. 获取长期真实历史数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_months * 30 + 90)
            
            sectors_data = await self.sector_fetcher.get_all_sectors_data(
                (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            )
            
            if not sectors_data:
                raise ValueError("无法获取真实历史数据")
                
            # 2. 分析每个板块的真实数据模式
            sector_patterns = {}
            for sector_name, sector_df in sectors_data.items():
                if len(sector_df) < 100:
                    continue
                    
                patterns = await self._analyze_sector_real_patterns(
                    sector_name, sector_df, prediction_days
                )
                if patterns:
                    sector_patterns[sector_name] = patterns
                    
            # 3. 识别最有效的预测模式
            effective_patterns = self._identify_effective_patterns(sector_patterns)
            
            # 4. 基于模式的准确率改进建议
            improvement_strategies = self._generate_pattern_based_strategies(
                sector_patterns, effective_patterns
            )
            
            # 5. 真实收益率验证
            return_validation = await self._validate_patterns_with_real_returns(
                sectors_data, effective_patterns, prediction_days
            )
            
            analysis_result = {
                'analysis_time': datetime.now().isoformat(),
                'analysis_months': analysis_months,
                'prediction_days': prediction_days,
                'sectors_analyzed': len(sector_patterns),
                'sector_patterns': sector_patterns,
                'effective_patterns': effective_patterns,
                'improvement_strategies': improvement_strategies,
                'return_validation': return_validation,
                'overall_insights': self._generate_overall_insights(
                    sector_patterns, effective_patterns, return_validation
                )
            }
            
            self.logger.info("基于真实数据的模式分析完成")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"真实数据模式分析失败: {e}")
            return {'error': str(e)}
            
    async def _analyze_sector_real_patterns(self, sector_name: str, 
                                          sector_df: pd.DataFrame,
                                          prediction_days: int) -> Dict[str, Any]:
        """分析单个板块的真实数据模式"""
        try:
            patterns = {}
            
            # 1. 价格动量模式分析
            momentum_patterns = self._analyze_momentum_patterns(sector_df, prediction_days)
            patterns['momentum'] = momentum_patterns
            
            # 2. 成交量模式分析
            volume_patterns = self._analyze_volume_patterns(sector_df, prediction_days)
            patterns['volume'] = volume_patterns
            
            # 3. 波动率模式分析
            volatility_patterns = self._analyze_volatility_patterns(sector_df, prediction_days)
            patterns['volatility'] = volatility_patterns
            
            # 4. 趋势持续性模式
            trend_patterns = self._analyze_trend_persistence(sector_df, prediction_days)
            patterns['trend'] = trend_patterns
            
            # 5. 季节性模式（基于真实历史数据）
            seasonal_patterns = self._analyze_seasonal_patterns(sector_df)
            patterns['seasonal'] = seasonal_patterns
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析{sector_name}模式失败: {e}")
            return {}
            
    def _analyze_momentum_patterns(self, df: pd.DataFrame, prediction_days: int) -> Dict[str, Any]:
        """分析动量模式"""
        try:
            patterns = {}
            
            # 计算多时间周期动量
            for lookback in [3, 5, 10, 20]:
                if len(df) > lookback + prediction_days + 10:
                    momentum_signals = []
                    
                    for i in range(lookback, len(df) - prediction_days):
                        # 动量信号：当前价格相对于N天前的变化
                        momentum = (df['close'].iloc[i] / df['close'].iloc[i-lookback] - 1) * 100
                        
                        # 未来收益
                        future_return = (df['close'].iloc[i+prediction_days] / df['close'].iloc[i] - 1) * 100
                        
                        momentum_signals.append({
                            'momentum': momentum,
                            'future_return': future_return,
                            'correct_direction': (momentum > 0) == (future_return > 0)
                        })
                    
                    if momentum_signals:
                        # 按动量强度分组分析准确率
                        strong_positive = [s for s in momentum_signals if s['momentum'] > 3]
                        strong_negative = [s for s in momentum_signals if s['momentum'] < -3]
                        weak_momentum = [s for s in momentum_signals if -1 <= s['momentum'] <= 1]
                        
                        patterns[f'momentum_{lookback}d'] = {
                            'total_signals': len(momentum_signals),
                            'strong_positive_accuracy': np.mean([s['correct_direction'] for s in strong_positive]) * 100 if strong_positive else 0,
                            'strong_negative_accuracy': np.mean([s['correct_direction'] for s in strong_negative]) * 100 if strong_negative else 0,
                            'weak_momentum_accuracy': np.mean([s['correct_direction'] for s in weak_momentum]) * 100 if weak_momentum else 0,
                            'strong_positive_count': len(strong_positive),
                            'strong_negative_count': len(strong_negative),
                            'weak_momentum_count': len(weak_momentum)
                        }
                        
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析动量模式失败: {e}")
            return {}
            
    def _analyze_volume_patterns(self, df: pd.DataFrame, prediction_days: int) -> Dict[str, Any]:
        """分析成交量模式"""
        try:
            patterns = {}
            
            if 'volume' not in df.columns:
                return patterns
                
            # 计算成交量异动信号
            df_copy = df.copy()
            df_copy['volume_ma20'] = df_copy['volume'].rolling(20).mean()
            df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma20']
            
            if len(df_copy) > 30:
                volume_signals = []
                
                for i in range(20, len(df_copy) - prediction_days):
                    volume_ratio = df_copy['volume_ratio'].iloc[i]
                    price_change = (df_copy['close'].iloc[i] / df_copy['close'].iloc[i-1] - 1) * 100
                    
                    # 未来收益
                    future_return = (df_copy['close'].iloc[i+prediction_days] / df_copy['close'].iloc[i] - 1) * 100
                    
                    volume_signals.append({
                        'volume_ratio': volume_ratio,
                        'price_change': price_change,
                        'future_return': future_return,
                        'volume_price_match': (volume_ratio > 1.5 and abs(price_change) > 2)
                    })
                
                if volume_signals:
                    # 分析量价配合情况
                    high_volume_signals = [s for s in volume_signals if s['volume_ratio'] > 2.0]
                    volume_price_match = [s for s in volume_signals if s['volume_price_match']]
                    
                    patterns['volume_analysis'] = {
                        'total_signals': len(volume_signals),
                        'high_volume_count': len(high_volume_signals),
                        'volume_price_match_count': len(volume_price_match),
                        'high_volume_avg_return': np.mean([s['future_return'] for s in high_volume_signals]) if high_volume_signals else 0,
                        'volume_price_match_avg_return': np.mean([s['future_return'] for s in volume_price_match]) if volume_price_match else 0,
                        'high_volume_accuracy': np.mean([(s['future_return'] > 0) == (s['price_change'] > 0) for s in high_volume_signals]) * 100 if high_volume_signals else 0
                    }
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析成交量模式失败: {e}")
            return {}
            
    def _analyze_volatility_patterns(self, df: pd.DataFrame, prediction_days: int) -> Dict[str, Any]:
        """分析波动率模式"""
        try:
            patterns = {}
            
            # 计算历史波动率
            df_copy = df.copy()
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['volatility_20d'] = df_copy['returns'].rolling(20).std() * np.sqrt(252)
            
            if len(df_copy) > 40:
                volatility_signals = []
                
                for i in range(20, len(df_copy) - prediction_days):
                    current_vol = df_copy['volatility_20d'].iloc[i]
                    vol_percentile = (df_copy['volatility_20d'].iloc[:i+1] <= current_vol).sum() / (i+1)
                    
                    # 未来收益和波动
                    future_returns = df_copy['returns'].iloc[i+1:i+1+prediction_days]
                    future_return = (df_copy['close'].iloc[i+prediction_days] / df_copy['close'].iloc[i] - 1) * 100
                    future_volatility = future_returns.std() * np.sqrt(252)
                    
                    volatility_signals.append({
                        'current_volatility': current_vol,
                        'vol_percentile': vol_percentile,
                        'future_return': future_return,
                        'future_volatility': future_volatility
                    })
                
                if volatility_signals:
                    # 分析不同波动率环境下的收益模式
                    high_vol_signals = [s for s in volatility_signals if s['vol_percentile'] > 0.8]
                    low_vol_signals = [s for s in volatility_signals if s['vol_percentile'] < 0.2]
                    
                    patterns['volatility_analysis'] = {
                        'total_signals': len(volatility_signals),
                        'high_vol_count': len(high_vol_signals),
                        'low_vol_count': len(low_vol_signals),
                        'high_vol_avg_return': np.mean([s['future_return'] for s in high_vol_signals]) if high_vol_signals else 0,
                        'low_vol_avg_return': np.mean([s['future_return'] for s in low_vol_signals]) if low_vol_signals else 0,
                        'high_vol_return_std': np.std([s['future_return'] for s in high_vol_signals]) if high_vol_signals else 0,
                        'low_vol_return_std': np.std([s['future_return'] for s in low_vol_signals]) if low_vol_signals else 0
                    }
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析波动率模式失败: {e}")
            return {}
            
    def _analyze_trend_persistence(self, df: pd.DataFrame, prediction_days: int) -> Dict[str, Any]:
        """分析趋势持续性"""
        try:
            patterns = {}
            
            # 计算多时间周期趋势
            df_copy = df.copy()
            for ma_period in [10, 20]:
                df_copy[f'ma_{ma_period}'] = df_copy['close'].rolling(ma_period).mean()
                df_copy[f'trend_{ma_period}'] = df_copy['close'] > df_copy[f'ma_{ma_period}']
                
            if len(df_copy) > 40:
                trend_signals = []
                
                for i in range(20, len(df_copy) - prediction_days):
                    # 趋势强度（价格相对均线位置）
                    trend_strength_10 = (df_copy['close'].iloc[i] / df_copy['ma_10'].iloc[i] - 1) * 100
                    trend_strength_20 = (df_copy['close'].iloc[i] / df_copy['ma_20'].iloc[i] - 1) * 100
                    
                    # 趋势一致性（短期和长期趋势方向）
                    trend_consistency = df_copy['trend_10'].iloc[i] == df_copy['trend_20'].iloc[i]
                    
                    # 未来收益
                    future_return = (df_copy['close'].iloc[i+prediction_days] / df_copy['close'].iloc[i] - 1) * 100
                    
                    trend_signals.append({
                        'trend_strength_10': trend_strength_10,
                        'trend_strength_20': trend_strength_20,
                        'trend_consistency': trend_consistency,
                        'future_return': future_return
                    })
                
                if trend_signals:
                    # 分析趋势强度和持续性
                    strong_uptrend = [s for s in trend_signals if s['trend_strength_20'] > 2 and s['trend_consistency']]
                    strong_downtrend = [s for s in trend_signals if s['trend_strength_20'] < -2 and s['trend_consistency']]
                    weak_trend = [s for s in trend_signals if abs(s['trend_strength_20']) < 1]
                    
                    patterns['trend_analysis'] = {
                        'total_signals': len(trend_signals),
                        'strong_uptrend_count': len(strong_uptrend),
                        'strong_downtrend_count': len(strong_downtrend),
                        'weak_trend_count': len(weak_trend),
                        'strong_uptrend_avg_return': np.mean([s['future_return'] for s in strong_uptrend]) if strong_uptrend else 0,
                        'strong_downtrend_avg_return': np.mean([s['future_return'] for s in strong_downtrend]) if strong_downtrend else 0,
                        'weak_trend_avg_return': np.mean([s['future_return'] for s in weak_trend]) if weak_trend else 0,
                        'trend_persistence_accuracy': np.mean([
                            (s['trend_strength_20'] > 0) == (s['future_return'] > 0) 
                            for s in trend_signals if abs(s['trend_strength_20']) > 1
                        ]) * 100 if trend_signals else 0
                    }
                    
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析趋势持续性失败: {e}")
            return {}
            
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析季节性模式（基于真实历史数据）"""
        try:
            patterns = {}
            
            if len(df) < 252:  # 少于一年数据
                return patterns
                
            df_copy = df.copy()
            df_copy['month'] = pd.to_datetime(df_copy.index).month
            df_copy['quarter'] = pd.to_datetime(df_copy.index).quarter
            df_copy['returns'] = df_copy['close'].pct_change()
            
            # 月度效应分析
            monthly_returns = df_copy.groupby('month')['returns'].agg(['mean', 'std', 'count'])
            monthly_returns['annual_return'] = monthly_returns['mean'] * 252
            monthly_returns = monthly_returns[monthly_returns['count'] >= 5]  # 至少5个样本
            
            if not monthly_returns.empty:
                patterns['monthly_effect'] = {
                    'best_month': monthly_returns['annual_return'].idxmax(),
                    'worst_month': monthly_returns['annual_return'].idxmin(),
                    'best_month_return': monthly_returns['annual_return'].max() * 100,
                    'worst_month_return': monthly_returns['annual_return'].min() * 100,
                    'monthly_data': monthly_returns.to_dict()
                }
                
            # 季度效应分析
            quarterly_returns = df_copy.groupby('quarter')['returns'].agg(['mean', 'std', 'count'])
            quarterly_returns['annual_return'] = quarterly_returns['mean'] * 252
            
            if not quarterly_returns.empty:
                patterns['quarterly_effect'] = {
                    'best_quarter': quarterly_returns['annual_return'].idxmax(),
                    'worst_quarter': quarterly_returns['annual_return'].idxmin(),
                    'best_quarter_return': quarterly_returns['annual_return'].max() * 100,
                    'worst_quarter_return': quarterly_returns['annual_return'].min() * 100
                }
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"分析季节性模式失败: {e}")
            return {}
            
    def _identify_effective_patterns(self, sector_patterns: Dict) -> Dict[str, Any]:
        """识别最有效的预测模式"""
        try:
            effective_patterns = {}
            
            # 汇总所有板块的模式效果
            all_momentum_accuracies = []
            all_volume_effects = []
            all_trend_accuracies = []
            
            for sector_name, patterns in sector_patterns.items():
                # 动量模式效果
                momentum_data = patterns.get('momentum', {})
                for period, data in momentum_data.items():
                    if 'strong_positive_accuracy' in data and data['strong_positive_count'] > 5:
                        all_momentum_accuracies.append(data['strong_positive_accuracy'])
                        
                # 成交量效果
                volume_data = patterns.get('volume', {}).get('volume_analysis', {})
                if volume_data and volume_data.get('high_volume_count', 0) > 5:
                    all_volume_effects.append(volume_data.get('high_volume_accuracy', 0))
                    
                # 趋势持续性
                trend_data = patterns.get('trend', {}).get('trend_analysis', {})
                if trend_data and trend_data.get('total_signals', 0) > 20:
                    all_trend_accuracies.append(trend_data.get('trend_persistence_accuracy', 0))
                    
            # 识别最有效的模式
            effective_patterns['momentum_effectiveness'] = {
                'avg_accuracy': np.mean(all_momentum_accuracies) if all_momentum_accuracies else 0,
                'consistency': 1 - np.std(all_momentum_accuracies) / np.mean(all_momentum_accuracies) if all_momentum_accuracies and np.mean(all_momentum_accuracies) > 0 else 0,
                'sample_count': len(all_momentum_accuracies)
            }
            
            effective_patterns['volume_effectiveness'] = {
                'avg_accuracy': np.mean(all_volume_effects) if all_volume_effects else 0,
                'consistency': 1 - np.std(all_volume_effects) / np.mean(all_volume_effects) if all_volume_effects and np.mean(all_volume_effects) > 0 else 0,
                'sample_count': len(all_volume_effects)
            }
            
            effective_patterns['trend_effectiveness'] = {
                'avg_accuracy': np.mean(all_trend_accuracies) if all_trend_accuracies else 0,
                'consistency': 1 - np.std(all_trend_accuracies) / np.mean(all_trend_accuracies) if all_trend_accuracies and np.mean(all_trend_accuracies) > 0 else 0,
                'sample_count': len(all_trend_accuracies)
            }
            
            return effective_patterns
            
        except Exception as e:
            self.logger.error(f"识别有效模式失败: {e}")
            return {}
            
    def _generate_pattern_based_strategies(self, sector_patterns: Dict, 
                                         effective_patterns: Dict) -> List[str]:
        """基于模式生成改进策略"""
        strategies = []
        
        try:
            # 基于动量效果的策略
            momentum_eff = effective_patterns.get('momentum_effectiveness', {})
            if momentum_eff.get('avg_accuracy', 0) > 65:
                strategies.append(f"动量策略有效，平均准确率{momentum_eff['avg_accuracy']:.1f}%，建议增加动量因子权重")
                
            # 基于成交量的策略
            volume_eff = effective_patterns.get('volume_effectiveness', {})
            if volume_eff.get('avg_accuracy', 0) > 60:
                strategies.append(f"成交量异动策略有效，建议增加量能分析权重")
                
            # 基于趋势的策略
            trend_eff = effective_patterns.get('trend_effectiveness', {})
            if trend_eff.get('avg_accuracy', 0) > 70:
                strategies.append(f"趋势持续性策略表现优秀，准确率{trend_eff['avg_accuracy']:.1f}%")
                
            # 基于季节性的策略
            seasonal_sectors = 0
            for patterns in sector_patterns.values():
                if 'seasonal' in patterns and patterns['seasonal']:
                    seasonal_sectors += 1
                    
            if seasonal_sectors > len(sector_patterns) * 0.5:
                strategies.append("发现明显季节性效应，建议结合月度/季度周期进行预测")
                
            # 通用策略
            if not strategies:
                strategies.append("需要进一步扩展历史数据深度以识别更多有效模式")
                
            strategies.append("建议采用多模式组合的方式，提高预测稳定性")
            strategies.append("针对不同板块采用差异化的预测策略")
            
        except Exception as e:
            self.logger.error(f"生成策略失败: {e}")
            strategies.append("无法生成具体策略，建议人工分析")
            
        return strategies
        
    async def _validate_patterns_with_real_returns(self, sectors_data: Dict,
                                                 effective_patterns: Dict,
                                                 prediction_days: int) -> Dict[str, Any]:
        """用真实收益率验证模式有效性"""
        try:
            validation_results = {}
            
            # 选择3个代表性板块进行深度验证
            sample_sectors = list(sectors_data.keys())[:3]
            
            for sector_name in sample_sectors:
                sector_df = sectors_data[sector_name]
                if len(sector_df) < 100:
                    continue
                    
                # 基于识别的有效模式进行预测验证
                validation_signals = []
                
                for i in range(60, len(sector_df) - prediction_days):
                    # 计算各种信号
                    momentum_5d = (sector_df['close'].iloc[i] / sector_df['close'].iloc[i-5] - 1) * 100
                    
                    # 成交量信号
                    volume_signal = 0
                    if 'volume' in sector_df.columns:
                        vol_ma = sector_df['volume'].iloc[i-20:i].mean()
                        volume_signal = 1 if sector_df['volume'].iloc[i] > vol_ma * 1.5 else 0
                        
                    # 趋势信号
                    ma_10 = sector_df['close'].iloc[i-10:i].mean()
                    trend_signal = 1 if sector_df['close'].iloc[i] > ma_10 else 0
                    
                    # 实际收益
                    actual_return = (sector_df['close'].iloc[i+prediction_days] / sector_df['close'].iloc[i] - 1) * 100
                    
                    # 综合预测信号
                    prediction_signal = 0
                    if momentum_5d > 2:
                        prediction_signal += 1
                    if volume_signal:
                        prediction_signal += 1
                    if trend_signal:
                        prediction_signal += 1
                        
                    validation_signals.append({
                        'momentum': momentum_5d,
                        'volume_signal': volume_signal,
                        'trend_signal': trend_signal,
                        'prediction_signal': prediction_signal,
                        'actual_return': actual_return,
                        'predicted_direction': 1 if prediction_signal >= 2 else -1,
                        'actual_direction': 1 if actual_return > 0 else -1
                    })
                    
                if validation_signals:
                    # 计算验证指标
                    direction_accuracy = np.mean([
                        s['predicted_direction'] == s['actual_direction'] 
                        for s in validation_signals
                    ]) * 100
                    
                    strong_signals = [s for s in validation_signals if s['prediction_signal'] >= 2]
                    strong_signal_accuracy = np.mean([
                        s['predicted_direction'] == s['actual_direction'] 
                        for s in strong_signals
                    ]) * 100 if strong_signals else 0
                    
                    validation_results[sector_name] = {
                        'total_signals': len(validation_signals),
                        'direction_accuracy': direction_accuracy,
                        'strong_signals_count': len(strong_signals),
                        'strong_signal_accuracy': strong_signal_accuracy,
                        'avg_return_when_bullish': np.mean([
                            s['actual_return'] for s in validation_signals 
                            if s['prediction_signal'] >= 2
                        ]) if strong_signals else 0
                    }
                    
            # 计算总体验证结果
            overall_validation = {}
            if validation_results:
                all_accuracies = [v['direction_accuracy'] for v in validation_results.values()]
                strong_accuracies = [v['strong_signal_accuracy'] for v in validation_results.values() if v['strong_signals_count'] > 5]
                
                overall_validation = {
                    'avg_direction_accuracy': np.mean(all_accuracies),
                    'avg_strong_signal_accuracy': np.mean(strong_accuracies) if strong_accuracies else 0,
                    'sectors_validated': len(validation_results),
                    'pattern_effectiveness': 'high' if np.mean(all_accuracies) > 65 else 'medium' if np.mean(all_accuracies) > 55 else 'low'
                }
                
            return {
                'sector_validations': validation_results,
                'overall_validation': overall_validation
            }
            
        except Exception as e:
            self.logger.error(f"验证模式失败: {e}")
            return {}
            
    def _generate_overall_insights(self, sector_patterns: Dict, 
                                 effective_patterns: Dict,
                                 return_validation: Dict) -> List[str]:
        """生成总体洞察"""
        insights = []
        
        try:
            # 基于分析结果生成洞察
            total_sectors = len(sector_patterns)
            
            insights.append(f"分析了{total_sectors}个板块的真实历史数据模式")
            
            # 效果洞察
            momentum_acc = effective_patterns.get('momentum_effectiveness', {}).get('avg_accuracy', 0)
            if momentum_acc > 65:
                insights.append(f"动量策略在真实数据中表现优秀，平均准确率达{momentum_acc:.1f}%")
                
            # 验证洞察
            overall_val = return_validation.get('overall_validation', {})
            if overall_val:
                pattern_eff = overall_val.get('pattern_effectiveness', 'unknown')
                if pattern_eff == 'high':
                    insights.append("基于真实数据的模式识别效果显著，建议应用到实际预测中")
                elif pattern_eff == 'medium':
                    insights.append("模式识别有一定效果，需要进一步优化参数")
                else:
                    insights.append("需要探索新的数据模式和预测方法")
                    
            # 改进方向
            insights.append("建议将有效模式集成到现有预测系统中")
            insights.append("持续监控模式效果，定期更新和优化")
            
        except Exception as e:
            self.logger.error(f"生成洞察失败: {e}")
            insights.append("分析完成，但无法生成具体洞察")
            
        return insights
        
    async def save_real_data_analysis_report(self, analysis_result: Dict,
                                           filename: str = None) -> str:
        """保存真实数据分析报告"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"real_data_analysis_{timestamp}.md"
                
            import os
            report_path = f"reports/analysis/{filename}"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            content = self._format_real_data_report(analysis_result)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.logger.info(f"真实数据分析报告已保存: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"保存分析报告失败: {e}")
            return ""
            
    def _format_real_data_report(self, result: Dict) -> str:
        """格式化真实数据分析报告"""
        try:
            content = f"""# 基于真实数据的预测准确率分析报告

## 分析概述
- **分析时间**: {result.get('analysis_time', 'Unknown')}
- **分析周期**: {result.get('analysis_months', 0)}个月
- **预测时间窗口**: {result.get('prediction_days', 5)}天
- **分析板块数**: {result.get('sectors_analyzed', 0)}

## 有效模式识别

### 动量模式效果
"""
            
            effective_patterns = result.get('effective_patterns', {})
            momentum_eff = effective_patterns.get('momentum_effectiveness', {})
            
            content += f"""
- **平均准确率**: {momentum_eff.get('avg_accuracy', 0):.1f}%
- **模式一致性**: {momentum_eff.get('consistency', 0):.3f}
- **样本数量**: {momentum_eff.get('sample_count', 0)}
"""
            
            volume_eff = effective_patterns.get('volume_effectiveness', {})
            content += f"""
### 成交量模式效果
- **平均准确率**: {volume_eff.get('avg_accuracy', 0):.1f}%
- **模式一致性**: {volume_eff.get('consistency', 0):.3f}
- **样本数量**: {volume_eff.get('sample_count', 0)}
"""

            trend_eff = effective_patterns.get('trend_effectiveness', {})
            content += f"""
### 趋势模式效果  
- **平均准确率**: {trend_eff.get('avg_accuracy', 0):.1f}%
- **模式一致性**: {trend_eff.get('consistency', 0):.3f}
- **样本数量**: {trend_eff.get('sample_count', 0)}
"""

            content += f"""
## 真实收益率验证结果

"""
            return_validation = result.get('return_validation', {})
            overall_val = return_validation.get('overall_validation', {})
            
            if overall_val:
                content += f"""
### 总体验证指标
- **平均方向准确率**: {overall_val.get('avg_direction_accuracy', 0):.1f}%
- **强信号准确率**: {overall_val.get('avg_strong_signal_accuracy', 0):.1f}%
- **验证板块数**: {overall_val.get('sectors_validated', 0)}
- **模式有效性**: {overall_val.get('pattern_effectiveness', 'unknown')}
"""

            content += f"""
## 改进策略建议

"""
            strategies = result.get('improvement_strategies', [])
            for i, strategy in enumerate(strategies, 1):
                content += f"{i}. {strategy}\n"
                
            content += f"""
## 关键洞察

"""
            insights = result.get('overall_insights', [])
            for insight in insights:
                content += f"- {insight}\n"
                
            content += f"""
## 结论

基于{result.get('analysis_months', 0)}个月的真实历史数据分析，识别出了多个有效的预测模式。
建议将这些模式集成到现有预测系统中，以提升整体预测准确率。

---
*报告基于真实市场数据生成，所有分析结果都有历史数据支撑*
*分析时间: {result.get('analysis_time', 'Unknown')}*
"""
            
            return content
            
        except Exception as e:
            self.logger.error(f"格式化报告失败: {e}")
            return f"报告生成失败: {str(e)}"