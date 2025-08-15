"""
高级信号过滤系统 - 多层信号质量控制和动态阈值优化
专门解决信号质量低和假信号过多的问题
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedSignalFilter:
    """高级信号过滤器，实现多层信号质量控制"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.signal_history = {}
        self.threshold_history = {}
        self.performance_metrics = {}
        
    def filter_high_quality_signals(self, data: pd.DataFrame, 
                                   selected_features: List[str],
                                   prediction_days: int = 5) -> Dict:
        """
        多层信号过滤，只保留高质量信号
        """
        try:
            if data.empty or not selected_features:
                return {'signals': {}, 'confidence': 0.0, 'quality_metrics': {}}
            
            # 1. 基础信号生成
            raw_signals = self._generate_raw_signals(data, selected_features)
            
            # 2. 信号强度评估
            signal_strengths = self._evaluate_signal_strength(data, raw_signals, selected_features)
            
            # 3. 信号一致性检验
            consistency_scores = self._check_signal_consistency(raw_signals, signal_strengths)
            
            # 4. 历史准确率验证
            historical_accuracy = self._validate_historical_accuracy(data, raw_signals, prediction_days)
            
            # 5. 动态阈值优化
            optimized_thresholds = self._optimize_dynamic_thresholds(
                signal_strengths, consistency_scores, historical_accuracy
            )
            
            # 6. 最终信号过滤
            filtered_signals = self._apply_advanced_filtering(
                raw_signals, signal_strengths, consistency_scores, 
                historical_accuracy, optimized_thresholds
            )
            
            # 7. 信号质量评估
            quality_metrics = self._assess_signal_quality(filtered_signals, data)
            
            result = {
                'signals': filtered_signals,
                'confidence': self._calculate_overall_confidence(filtered_signals, quality_metrics),
                'quality_metrics': quality_metrics,
                'threshold_info': optimized_thresholds,
                'signal_count': len([s for s in filtered_signals.values() if s['direction'] != 0])
            }
            
            self.logger.info(f"信号过滤完成，保留{result['signal_count']}个高质量信号，整体置信度: {result['confidence']:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"信号过滤失败: {e}")
            return {'signals': {}, 'confidence': 0.0, 'quality_metrics': {}}
    
    def _generate_raw_signals(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Dict]:
        """生成基础信号"""
        try:
            signals = {}
            
            for feature in features:
                if feature not in data.columns:
                    continue
                    
                feature_data = data[feature].fillna(0)
                if len(feature_data) < 10:
                    continue
                
                # 计算信号指标
                current_value = feature_data.iloc[-1]
                rolling_mean = feature_data.rolling(window=min(20, len(feature_data)//2)).mean().iloc[-1]
                rolling_std = feature_data.rolling(window=min(20, len(feature_data)//2)).std().iloc[-1]
                
                if np.isnan(rolling_mean) or np.isnan(rolling_std) or rolling_std == 0:
                    continue
                
                # Z-score标准化
                z_score = (current_value - rolling_mean) / rolling_std
                
                # 信号方向判断
                if abs(z_score) < 0.5:
                    direction = 0  # 无明确信号
                elif z_score > 0.5:
                    direction = 1  # 看涨信号
                else:
                    direction = -1  # 看跌信号
                
                signals[feature] = {
                    'direction': direction,
                    'strength': abs(z_score),
                    'z_score': z_score,
                    'current_value': current_value,
                    'baseline': rolling_mean,
                    'volatility': rolling_std
                }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成基础信号失败: {e}")
            return {}
    
    def _evaluate_signal_strength(self, data: pd.DataFrame, 
                                 signals: Dict[str, Dict], 
                                 features: List[str]) -> Dict[str, float]:
        """评估信号强度"""
        try:
            strengths = {}
            
            for feature, signal in signals.items():
                if feature not in data.columns:
                    continue
                
                feature_data = data[feature].fillna(0)
                strength_score = 0.0
                
                # 1. 基于Z-score的强度
                z_strength = min(abs(signal['z_score']) / 3.0, 1.0)  # 归一化到0-1
                
                # 2. 基于趋势一致性的强度
                if len(feature_data) >= 5:
                    recent_trend = np.sign(np.diff(feature_data.tail(5))).mean()
                    trend_strength = abs(recent_trend) * 0.3
                else:
                    trend_strength = 0.0
                
                # 3. 基于波动率的强度调整
                volatility_factor = 1.0 / (1.0 + signal['volatility'] / abs(signal['baseline']) if signal['baseline'] != 0 else 1.0)
                
                # 4. 基于信号持续性的强度
                persistence_strength = self._calculate_signal_persistence(feature_data, signal['direction'])
                
                # 综合强度计算
                strength_score = (z_strength * 0.4 + trend_strength + 
                                persistence_strength * 0.3) * volatility_factor
                
                strengths[feature] = min(strength_score, 1.0)
            
            return strengths
            
        except Exception as e:
            self.logger.error(f"信号强度评估失败: {e}")
            return {}
    
    def _calculate_signal_persistence(self, data: pd.Series, direction: int) -> float:
        """计算信号持续性"""
        try:
            if len(data) < 5 or direction == 0:
                return 0.0
            
            # 检查最近几个周期的信号方向一致性
            recent_changes = np.diff(data.tail(5))
            if direction > 0:
                consistency = (recent_changes > 0).mean()
            else:
                consistency = (recent_changes < 0).mean()
            
            return consistency
            
        except:
            return 0.0
    
    def _check_signal_consistency(self, signals: Dict[str, Dict], 
                                 strengths: Dict[str, float]) -> Dict[str, float]:
        """检查信号一致性"""
        try:
            consistency_scores = {}
            
            # 获取所有有效信号
            valid_signals = {k: v for k, v in signals.items() if v['direction'] != 0}
            
            if len(valid_signals) < 2:
                return {k: 0.5 for k in signals.keys()}
            
            for feature, signal in signals.items():
                if signal['direction'] == 0:
                    consistency_scores[feature] = 0.0
                    continue
                
                # 计算与其他信号的一致性
                agreement_count = 0
                total_comparisons = 0
                
                for other_feature, other_signal in valid_signals.items():
                    if other_feature != feature and other_signal['direction'] != 0:
                        # 权重相同方向的信号
                        weight = strengths.get(other_feature, 0.5)
                        if signal['direction'] == other_signal['direction']:
                            agreement_count += weight
                        total_comparisons += weight
                
                if total_comparisons > 0:
                    consistency = agreement_count / total_comparisons
                else:
                    consistency = 0.5
                
                consistency_scores[feature] = consistency
            
            return consistency_scores
            
        except Exception as e:
            self.logger.error(f"信号一致性检查失败: {e}")
            return {}
    
    def _validate_historical_accuracy(self, data: pd.DataFrame, 
                                    signals: Dict[str, Dict], 
                                    prediction_days: int) -> Dict[str, float]:
        """验证历史准确率"""
        try:
            accuracies = {}
            
            if 'close' not in data.columns or len(data) < prediction_days + 20:
                return {k: 0.5 for k in signals.keys()}
            
            for feature, signal in signals.items():
                if feature not in data.columns or signal['direction'] == 0:
                    accuracies[feature] = 0.5
                    continue
                
                feature_data = data[feature].fillna(0)
                price_data = data['close']
                
                # 计算历史信号准确率
                correct_predictions = 0
                total_predictions = 0
                
                # 回测最近的信号
                for i in range(prediction_days, min(50 + prediction_days, len(data))):
                    try:
                        # 历史信号生成
                        hist_data = feature_data.iloc[:i-prediction_days]
                        if len(hist_data) < 10:
                            continue
                        
                        hist_mean = hist_data.rolling(window=min(20, len(hist_data)//2)).mean().iloc[-1]
                        hist_std = hist_data.rolling(window=min(20, len(hist_data)//2)).std().iloc[-1]
                        
                        if np.isnan(hist_mean) or np.isnan(hist_std) or hist_std == 0:
                            continue
                        
                        hist_z = (feature_data.iloc[i-prediction_days] - hist_mean) / hist_std
                        
                        if abs(hist_z) > 0.5:  # 有效信号阈值
                            hist_direction = 1 if hist_z > 0 else -1
                            
                            # 实际价格变化
                            actual_change = price_data.iloc[i] - price_data.iloc[i-prediction_days]
                            actual_direction = 1 if actual_change > 0 else -1
                            
                            if hist_direction == actual_direction:
                                correct_predictions += 1
                            total_predictions += 1
                    
                    except:
                        continue
                
                if total_predictions > 0:
                    accuracy = correct_predictions / total_predictions
                else:
                    accuracy = 0.5
                
                accuracies[feature] = accuracy
            
            return accuracies
            
        except Exception as e:
            self.logger.error(f"历史准确率验证失败: {e}")
            return {}
    
    def _optimize_dynamic_thresholds(self, strengths: Dict[str, float], 
                                   consistency: Dict[str, float],
                                   accuracy: Dict[str, float]) -> Dict[str, Dict]:
        """优化动态阈值 - 阶段二增强版"""
        try:
            thresholds = {}
            
            # 市场环境适应性阈值调整
            market_volatility = self._calculate_market_volatility()
            market_trend = self._identify_market_trend()
            
            for feature in strengths.keys():
                strength = strengths.get(feature, 0.0)
                consist = consistency.get(feature, 0.0)
                acc = accuracy.get(feature, 0.5)
                
                # 1. 基础阈值（根据特征类型动态调整）
                base_threshold = self._get_adaptive_base_threshold(feature, market_volatility)
                
                # 2. 历史表现调整（更精细化）
                accuracy_weight = 0.6 if acc > 0.6 else 0.3  # 高准确率特征权重更大
                accuracy_adjustment = (acc - 0.5) * accuracy_weight
                
                # 3. 一致性调整（考虑市场环境）
                consistency_weight = 0.3 if market_volatility < 0.02 else 0.2  # 低波动市场更重视一致性
                consistency_adjustment = consist * consistency_weight
                
                # 4. 市场趋势调整
                trend_adjustment = self._get_trend_adjustment(feature, market_trend)
                
                # 5. 信号强度调整
                strength_adjustment = (strength - 0.5) * 0.2
                
                # 最终阈值计算
                final_threshold = max(0.05, min(0.8, 
                    base_threshold - accuracy_adjustment - consistency_adjustment 
                    + trend_adjustment - strength_adjustment
                ))
                
                # 个性化阈值设置
                thresholds[feature] = {
                    'strength_threshold': final_threshold,
                    'consistency_threshold': max(0.2, 0.7 - acc * 0.8),
                    'accuracy_threshold': max(0.52, 0.6 - market_volatility * 10),  # 波动大时降低准确率要求
                    'combined_threshold': final_threshold * 0.6,
                    'market_adjusted': True,
                    'volatility_factor': market_volatility,
                    'trend_factor': market_trend,
                    'adaptive_weight': accuracy_weight + consistency_weight
                }
            
            return thresholds
            
        except Exception as e:
            self.logger.error(f"动态阈值优化失败: {e}")
            return {}
    
    def _apply_advanced_filtering(self, signals: Dict[str, Dict],
                                 strengths: Dict[str, float],
                                 consistency: Dict[str, float],
                                 accuracy: Dict[str, float],
                                 thresholds: Dict[str, Dict]) -> Dict[str, Dict]:
        """应用高级信号过滤"""
        try:
            filtered_signals = {}
            
            for feature, signal in signals.items():
                if signal['direction'] == 0:
                    filtered_signals[feature] = signal
                    continue
                
                strength = strengths.get(feature, 0.0)
                consist = consistency.get(feature, 0.0)
                acc = accuracy.get(feature, 0.5)
                thresh = thresholds.get(feature, {})
                
                # 多重过滤条件
                conditions = []
                
                # 1. 强度过滤
                conditions.append(strength >= thresh.get('strength_threshold', 0.3))
                
                # 2. 一致性过滤
                conditions.append(consist >= thresh.get('consistency_threshold', 0.3))
                
                # 3. 准确率过滤
                conditions.append(acc >= thresh.get('accuracy_threshold', 0.55))
                
                # 4. 综合得分过滤
                combined_score = strength * 0.4 + consist * 0.3 + (acc - 0.5) * 2 * 0.3
                conditions.append(combined_score >= thresh.get('combined_threshold', 0.2))
                
                # 通过过滤的信号
                if sum(conditions) >= 3:  # 至少满足3个条件
                    filtered_signal = signal.copy()
                    filtered_signal.update({
                        'quality_score': combined_score,
                        'strength_score': strength,
                        'consistency_score': consist,
                        'accuracy_score': acc,
                        'filter_passed': True
                    })
                    filtered_signals[feature] = filtered_signal
                else:
                    # 不合格信号设为中性
                    filtered_signal = signal.copy()
                    filtered_signal['direction'] = 0
                    filtered_signal['filter_passed'] = False
                    filtered_signals[feature] = filtered_signal
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"高级信号过滤失败: {e}")
            return signals
    
    def _assess_signal_quality(self, signals: Dict[str, Dict], data: pd.DataFrame) -> Dict:
        """评估信号质量"""
        try:
            quality_metrics = {}
            
            valid_signals = [s for s in signals.values() if s['direction'] != 0 and s.get('filter_passed', False)]
            
            if not valid_signals:
                return {
                    'signal_count': 0,
                    'average_quality': 0.0,
                    'quality_distribution': {},
                    'confidence_level': 'very_low'
                }
            
            # 质量指标统计
            quality_scores = [s.get('quality_score', 0.0) for s in valid_signals]
            strength_scores = [s.get('strength_score', 0.0) for s in valid_signals]
            consistency_scores = [s.get('consistency_score', 0.0) for s in valid_signals]
            accuracy_scores = [s.get('accuracy_score', 0.5) for s in valid_signals]
            
            # 方向一致性
            directions = [s['direction'] for s in valid_signals]
            bullish_signals = sum(1 for d in directions if d > 0)
            bearish_signals = sum(1 for d in directions if d < 0)
            
            direction_consensus = max(bullish_signals, bearish_signals) / len(directions) if directions else 0
            
            quality_metrics = {
                'signal_count': len(valid_signals),
                'average_quality': np.mean(quality_scores),
                'average_strength': np.mean(strength_scores),
                'average_consistency': np.mean(consistency_scores),
                'average_accuracy': np.mean(accuracy_scores),
                'direction_consensus': direction_consensus,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'quality_distribution': self._categorize_signal_quality(quality_scores),
                'confidence_level': self._determine_confidence_level(np.mean(quality_scores), direction_consensus)
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"信号质量评估失败: {e}")
            return {}
    
    def _categorize_signal_quality(self, quality_scores: List[float]) -> Dict:
        """分类信号质量"""
        if not quality_scores:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high_quality = sum(1 for q in quality_scores if q >= 0.7)
        medium_quality = sum(1 for q in quality_scores if 0.4 <= q < 0.7)
        low_quality = sum(1 for q in quality_scores if q < 0.4)
        
        return {
            'high': high_quality,
            'medium': medium_quality, 
            'low': low_quality
        }
    
    def _determine_confidence_level(self, avg_quality: float, direction_consensus: float) -> str:
        """确定置信水平"""
        combined_score = avg_quality * 0.7 + direction_consensus * 0.3
        
        if combined_score >= 0.8:
            return 'very_high'
        elif combined_score >= 0.65:
            return 'high'
        elif combined_score >= 0.5:
            return 'medium'
        elif combined_score >= 0.35:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_overall_confidence(self, signals: Dict[str, Dict], 
                                    quality_metrics: Dict) -> float:
        """计算整体置信度"""
        try:
            if not signals or quality_metrics.get('signal_count', 0) == 0:
                return 0.0
            
            avg_quality = quality_metrics.get('average_quality', 0.0)
            direction_consensus = quality_metrics.get('direction_consensus', 0.0)
            signal_count = quality_metrics.get('signal_count', 0)
            
            # 基于信号数量的置信度调整
            count_factor = min(signal_count / 10.0, 1.0)  # 最多10个信号时置信度最高
            
            # 综合置信度
            confidence = (avg_quality * 0.5 + direction_consensus * 0.3 + count_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"计算整体置信度失败: {e}")
            return 0.0
    
    # ==================== 阶段二优化：高级动态阈值和信号过滤 ====================
    
    def _calculate_market_volatility(self) -> float:
        """计算市场波动率 - 阶段二优化"""
        try:
            # 如果有历史数据，基于历史数据计算
            if hasattr(self, 'current_data') and self.current_data is not None:
                close_col = 'close' if 'close' in self.current_data.columns else None
                if close_col and len(self.current_data) > 10:
                    returns = self.current_data[close_col].pct_change().dropna()
                    volatility = returns.rolling(10).std().iloc[-1]
                    return volatility if not np.isnan(volatility) else 0.02
            
            # 默认市场波动率（基于A股历史经验）
            return 0.025  # 2.5%日波动率
            
        except Exception as e:
            self.logger.warning(f"计算市场波动率失败: {e}")
            return 0.025
    
    def _identify_market_trend(self) -> str:
        """识别市场趋势 - 阶段二优化"""
        try:
            # 如果有历史数据，基于价格趋势分析
            if hasattr(self, 'current_data') and self.current_data is not None:
                close_col = 'close' if 'close' in self.current_data.columns else None
                if close_col and len(self.current_data) > 20:
                    data = self.current_data[close_col]
                    short_ma = data.rolling(5).mean().iloc[-1]
                    long_ma = data.rolling(20).mean().iloc[-1]
                    
                    if short_ma > long_ma * 1.02:
                        return 'bullish'
                    elif short_ma < long_ma * 0.98:
                        return 'bearish'
                    else:
                        return 'sideways'
            
            # 默认趋势
            return 'sideways'
            
        except Exception as e:
            self.logger.warning(f"识别市场趋势失败: {e}")
            return 'sideways'
    
    def _get_adaptive_base_threshold(self, feature: str, market_volatility: float) -> float:
        """获取自适应基础阈值 - 阶段二优化"""
        try:
            # 特征类型分类阈值
            feature_lower = feature.lower()
            
            # 1. 动量类特征
            if any(word in feature_lower for word in ['momentum', 'roc', 'return', 'change']):
                base = 0.25 + market_volatility * 5  # 波动大时阈值更高
            
            # 2. 趋势类特征  
            elif any(word in feature_lower for word in ['trend', 'ma_', 'ema', 'slope']):
                base = 0.20 + market_volatility * 3
            
            # 3. 波动率类特征
            elif any(word in feature_lower for word in ['volatility', 'std', 'atr', 'bbands']):
                base = 0.35 + market_volatility * 2  # 波动率特征需要更高阈值
            
            # 4. 成交量类特征
            elif any(word in feature_lower for word in ['volume', 'obv', 'mfi', 'money_flow']):
                base = 0.30 + market_volatility * 4
            
            # 5. 市场情绪类特征（阶段一新增）
            elif any(word in feature_lower for word in ['sentiment', 'fear', 'northbound', 'margin']):
                base = 0.28 + market_volatility * 6  # 情绪类特征对波动更敏感
            
            # 6. 技术指标类
            elif any(word in feature_lower for word in ['rsi', 'macd', 'kdj', 'cci', 'williams']):
                base = 0.25 + market_volatility * 3
            
            # 默认阈值
            else:
                base = 0.30 + market_volatility * 4
            
            return max(0.1, min(0.6, base))
            
        except Exception as e:
            self.logger.warning(f"获取自适应基础阈值失败: {e}")
            return 0.30
    
    def _get_trend_adjustment(self, feature: str, market_trend: str) -> float:
        """获取趋势调整系数 - 阶段二优化"""
        try:
            feature_lower = feature.lower()
            
            # 趋势调整系数
            if market_trend == 'bullish':
                # 牛市环境：适当放宽买入信号阈值，收紧卖出信号阈值
                if any(word in feature_lower for word in ['momentum', 'trend', 'ma_']):
                    return -0.05  # 降低阈值，更容易触发买入信号
                elif any(word in feature_lower for word in ['sentiment', 'fear']):
                    return -0.08  # 情绪类在牛市中信号更可靠
                else:
                    return -0.03
                    
            elif market_trend == 'bearish':
                # 熊市环境：收紧买入信号阈值，适当放宽卖出信号阈值
                if any(word in feature_lower for word in ['momentum', 'trend']):
                    return +0.05  # 提高阈值，避免假突破
                elif any(word in feature_lower for word in ['volatility', 'fear']):
                    return -0.05  # 波动率和恐慌指标在熊市更重要
                else:
                    return +0.03
                    
            else:  # sideways
                # 震荡市：均衡策略，稍微收紧阈值
                return +0.02
                
        except Exception as e:
            self.logger.warning(f"获取趋势调整失败: {e}")
            return 0.0
    
    def _apply_advanced_filtering(self, raw_signals: Dict[str, Dict],
                                signal_strengths: Dict[str, float],
                                consistency_scores: Dict[str, float],
                                historical_accuracy: Dict[str, float],
                                optimized_thresholds: Dict[str, Dict]) -> Dict[str, Dict]:
        """应用高级过滤 - 阶段二增强版"""
        try:
            filtered_signals = {}
            
            # 1. 信号相关性分析
            correlation_matrix = self._calculate_signal_correlations(raw_signals)
            
            # 2. 信号权重动态调整
            dynamic_weights = self._calculate_dynamic_weights(
                historical_accuracy, consistency_scores, signal_strengths
            )
            
            # 3. 市场环境分类过滤
            market_environment = self._classify_market_environment()
            
            for feature, signal_data in raw_signals.items():
                if feature not in optimized_thresholds:
                    continue
                
                thresholds = optimized_thresholds[feature]
                strength = signal_strengths.get(feature, 0.0)
                consistency = consistency_scores.get(feature, 0.0)
                accuracy = historical_accuracy.get(feature, 0.5)
                
                # 基础过滤条件
                passes_strength = strength >= thresholds['strength_threshold']
                passes_consistency = consistency >= thresholds['consistency_threshold']
                passes_accuracy = accuracy >= thresholds['accuracy_threshold']
                
                # 相关性过滤：避免过度相关的信号
                passes_correlation = self._check_correlation_filter(
                    feature, filtered_signals, correlation_matrix
                )
                
                # 市场环境适应性过滤
                passes_environment = self._check_environment_filter(
                    feature, signal_data, market_environment
                )
                
                # 综合质量评分
                quality_score = (
                    strength * 0.3 + 
                    consistency * 0.25 + 
                    accuracy * 0.25 + 
                    dynamic_weights.get(feature, 0.5) * 0.2
                )
                
                # 最终过滤决策
                if (passes_strength and passes_consistency and passes_accuracy and 
                    passes_correlation and passes_environment and quality_score > 0.5):
                    
                    filtered_signals[feature] = {
                        'direction': signal_data['direction'],
                        'strength': strength,
                        'consistency': consistency,
                        'accuracy': accuracy,
                        'quality_score': quality_score,
                        'dynamic_weight': dynamic_weights.get(feature, 0.5),
                        'filter_passed': True,
                        'environment_adapted': True
                    }
            
            self.logger.info(f"高级过滤完成: {len(raw_signals)} -> {len(filtered_signals)} 信号")
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"高级过滤失败: {e}")
            return raw_signals
    
    def _calculate_signal_correlations(self, signals: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """计算信号间相关性 - 阶段二优化"""
        try:
            correlations = {}
            signal_names = list(signals.keys())
            
            for i, signal1 in enumerate(signal_names):
                correlations[signal1] = {}
                for j, signal2 in enumerate(signal_names):
                    if i != j:
                        # 基于信号方向和强度计算相关性
                        dir1 = signals[signal1].get('direction', 0)
                        dir2 = signals[signal2].get('direction', 0)
                        str1 = signals[signal1].get('raw_strength', 0)
                        str2 = signals[signal2].get('raw_strength', 0)
                        
                        # 简化相关性计算
                        correlation = (dir1 * dir2) * min(abs(str1), abs(str2))
                        correlations[signal1][signal2] = correlation
                    else:
                        correlations[signal1][signal2] = 1.0
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f"计算信号相关性失败: {e}")
            return {}
    
    def _calculate_dynamic_weights(self, accuracy: Dict[str, float],
                                 consistency: Dict[str, float],
                                 strength: Dict[str, float]) -> Dict[str, float]:
        """计算动态权重 - 阶段二优化"""
        try:
            weights = {}
            
            for feature in accuracy.keys():
                acc = accuracy.get(feature, 0.5)
                cons = consistency.get(feature, 0.5)
                str_val = strength.get(feature, 0.5)
                
                # 历史成功率驱动的权重
                success_weight = (acc - 0.5) * 2  # 准确率超过50%的部分放大
                
                # 一致性加权
                consistency_weight = cons * 0.5
                
                # 信号强度加权
                strength_weight = str_val * 0.3
                
                # 综合动态权重
                dynamic_weight = max(0.1, min(1.0, 
                    0.5 + success_weight + consistency_weight + strength_weight
                ))
                
                weights[feature] = dynamic_weight
                
            return weights
            
        except Exception as e:
            self.logger.warning(f"计算动态权重失败: {e}")
            return {}
    
    def _classify_market_environment(self) -> str:
        """分类市场环境 - 阶段二优化"""
        try:
            trend = self._identify_market_trend()
            volatility = self._calculate_market_volatility()
            
            # 基于趋势和波动率分类
            if trend == 'bullish' and volatility < 0.02:
                return 'stable_bull'
            elif trend == 'bullish' and volatility >= 0.02:
                return 'volatile_bull'
            elif trend == 'bearish' and volatility < 0.02:
                return 'stable_bear'
            elif trend == 'bearish' and volatility >= 0.02:
                return 'volatile_bear'
            elif volatility < 0.015:
                return 'low_volatility_sideways'
            else:
                return 'high_volatility_sideways'
                
        except Exception as e:
            self.logger.warning(f"分类市场环境失败: {e}")
            return 'neutral'
    
    def _check_correlation_filter(self, feature: str, existing_signals: Dict[str, Dict],
                                correlation_matrix: Dict[str, Dict[str, float]]) -> bool:
        """检查相关性过滤 - 阶段二优化"""
        try:
            if not existing_signals or feature not in correlation_matrix:
                return True
            
            max_correlation = 0.0
            for existing_feature in existing_signals.keys():
                if existing_feature in correlation_matrix[feature]:
                    correlation = abs(correlation_matrix[feature][existing_feature])
                    max_correlation = max(max_correlation, correlation)
            
            # 如果与已有信号相关性过高，则过滤掉
            return max_correlation < 0.85
            
        except Exception as e:
            self.logger.warning(f"相关性过滤检查失败: {e}")
            return True
    
    def _check_environment_filter(self, feature: str, signal_data: Dict,
                                market_environment: str) -> bool:
        """检查环境过滤 - 阶段二优化"""
        try:
            feature_lower = feature.lower()
            direction = signal_data.get('direction', 0)
            
            # 根据市场环境调整信号可信度
            if market_environment in ['stable_bull', 'volatile_bull']:
                # 牛市环境：看涨信号更可信
                if direction > 0:
                    return True
                elif any(word in feature_lower for word in ['fear', 'volatility']):
                    return False  # 牛市中恐慌和高波动信号不可信
                    
            elif market_environment in ['stable_bear', 'volatile_bear']:
                # 熊市环境：看跌信号更可信
                if direction < 0:
                    return True
                elif any(word in feature_lower for word in ['momentum', 'trend']):
                    return abs(direction) > 0.5  # 熊市中需要更强的信号
                    
            elif 'sideways' in market_environment:
                # 震荡市：均衡策略，需要更强的信号
                return abs(direction) > 0.3
                
            return True
            
        except Exception as e:
            self.logger.warning(f"环境过滤检查失败: {e}")
            return True
    
    def filter_high_quality_signals_enhanced(self, data: pd.DataFrame, 
                                           selected_features: List[str],
                                           prediction_days: int = 5) -> Dict:
        """增强版信号过滤入口 - 阶段二完整版"""
        try:
            # 保存当前数据用于市场环境分析
            self.current_data = data
            
            # 调用增强版过滤流程
            result = self.filter_high_quality_signals(data, selected_features, prediction_days)
            
            # 添加阶段二特有的质量指标
            if result['signals']:
                enhanced_metrics = self._calculate_enhanced_quality_metrics(result['signals'])
                result['quality_metrics'].update(enhanced_metrics)
                result['stage_two_enhanced'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"增强版信号过滤失败: {e}")
            return self.filter_high_quality_signals(data, selected_features, prediction_days)
    
    def _calculate_enhanced_quality_metrics(self, signals: Dict[str, Dict]) -> Dict:
        """计算增强质量指标 - 阶段二专用"""
        try:
            if not signals:
                return {}
            
            # 权重分布分析
            weights = [s.get('dynamic_weight', 0.5) for s in signals.values()]
            weight_variance = np.var(weights) if weights else 0
            
            # 环境适应性评分
            environment_adapted = sum(1 for s in signals.values() if s.get('environment_adapted', False))
            adaptation_rate = environment_adapted / len(signals)
            
            # 质量评分分布
            quality_scores = [s.get('quality_score', 0.5) for s in signals.values()]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.5
            quality_std = np.std(quality_scores) if quality_scores else 0
            
            return {
                'dynamic_weight_variance': weight_variance,
                'environment_adaptation_rate': adaptation_rate,
                'average_quality_score': avg_quality,
                'quality_score_stability': 1.0 - min(quality_std, 0.3) / 0.3,
                'enhanced_filtering_applied': True
            }
            
        except Exception as e:
            self.logger.warning(f"计算增强质量指标失败: {e}")
            return {}