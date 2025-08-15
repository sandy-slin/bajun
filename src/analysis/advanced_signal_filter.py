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
        """优化动态阈值"""
        try:
            thresholds = {}
            
            for feature in strengths.keys():
                strength = strengths.get(feature, 0.0)
                consist = consistency.get(feature, 0.0)
                acc = accuracy.get(feature, 0.5)
                
                # 基于历史表现调整阈值
                base_threshold = 0.3  # 基础阈值
                
                # 准确率越高，阈值越低（更容易触发）
                accuracy_adjustment = (acc - 0.5) * 0.4
                
                # 一致性越高，阈值越低
                consistency_adjustment = consist * 0.2
                
                # 最终阈值
                final_threshold = max(0.1, base_threshold - accuracy_adjustment - consistency_adjustment)
                
                thresholds[feature] = {
                    'strength_threshold': final_threshold,
                    'consistency_threshold': max(0.3, 0.6 - acc * 0.6),
                    'accuracy_threshold': 0.55,  # 最低准确率要求
                    'combined_threshold': final_threshold * 0.7
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