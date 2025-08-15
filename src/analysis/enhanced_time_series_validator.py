"""
阶段三：增强时间序列验证器 - 考虑季节性因素和市场环境
专门为集成学习优化的时序交叉验证系统
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class EnhancedTimeSeriesValidator:
    """阶段三增强时间序列验证器，支持季节性感知的交叉验证"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_history = {}
        self.seasonal_patterns = {}
        
    def perform_seasonal_aware_validation(self, data: pd.DataFrame, 
                                        features: List[str], 
                                        target_col: str = 'future_return',
                                        n_splits: int = 5,
                                        test_size: int = 30) -> Dict:
        """
        执行季节性感知的时间序列交叉验证
        
        Args:
            data: 训练数据
            features: 特征列表
            target_col: 目标列名
            n_splits: 交叉验证折数
            test_size: 测试集天数
            
        Returns:
            详细的验证结果
        """
        try:
            self.logger.info(f"开始季节性感知时序验证，数据量: {len(data)}")
            
            if len(data) < 100:
                self.logger.warning(f"数据量较少: {len(data)}，可能影响验证效果")
            
            # 1. 检测季节性模式
            seasonal_info = self._detect_seasonal_patterns(data, target_col)
            
            # 2. 创建季节性感知的分割
            splits = self._create_seasonal_aware_splits(data, n_splits, test_size, seasonal_info)
            
            # 3. 执行交叉验证
            validation_results = self._execute_cross_validation(data, features, target_col, splits)
            
            # 4. 分析验证结果
            performance_analysis = self._analyze_validation_performance(validation_results, seasonal_info)
            
            result = {
                'validation_type': 'seasonal_aware_cross_validation',
                'n_splits': len(splits),
                'seasonal_info': seasonal_info,
                'validation_results': validation_results,
                'performance_analysis': performance_analysis,
                'overall_accuracy': performance_analysis.get('mean_accuracy', 0),
                'stability_score': performance_analysis.get('stability_score', 0)
            }
            
            self.logger.info(f"季节性验证完成，平均准确率: {result['overall_accuracy']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"季节性验证失败: {e}")
            return {'error': str(e)}
    
    def _detect_seasonal_patterns(self, data: pd.DataFrame, target_col: str) -> Dict:
        """检测数据中的季节性模式"""
        try:
            seasonal_info = {
                'has_seasonality': False,
                'seasonal_strength': 0.0,
                'monthly_patterns': {},
                'quarterly_patterns': {},
                'market_cycle_info': {}
            }
            
            if target_col not in data.columns or len(data) < 60:
                return seasonal_info
            
            # 假设数据有日期索引或可以推断日期
            if hasattr(data.index, 'month'):
                # 按月分析
                monthly_performance = data.groupby(data.index.month)[target_col].agg(['mean', 'std', 'count'])
                seasonal_info['monthly_patterns'] = monthly_performance.to_dict()
                
                # 计算季节性强度
                monthly_means = monthly_performance['mean'].values
                seasonal_strength = np.std(monthly_means) / np.mean(np.abs(monthly_means)) if np.mean(np.abs(monthly_means)) > 0 else 0
                seasonal_info['seasonal_strength'] = seasonal_strength
                seasonal_info['has_seasonality'] = seasonal_strength > 0.1
                
            else:
                # 使用滑动窗口检测周期性模式
                window_size = min(30, len(data) // 3)
                if window_size > 5:
                    rolling_means = data[target_col].rolling(window=window_size).mean()
                    pattern_stability = 1 - (rolling_means.std() / rolling_means.mean()) if rolling_means.mean() != 0 else 0
                    seasonal_info['seasonal_strength'] = max(0, pattern_stability)
                    seasonal_info['has_seasonality'] = pattern_stability > 0.2
            
            # 检测市场周期（基于收益率波动）
            if len(data) > 20:
                volatility_periods = self._identify_market_regimes(data, target_col)
                seasonal_info['market_cycle_info'] = volatility_periods
            
            self.logger.info(f"季节性检测完成，季节性强度: {seasonal_info['seasonal_strength']:.3f}")
            return seasonal_info
            
        except Exception as e:
            self.logger.error(f"季节性检测失败: {e}")
            return {'has_seasonality': False, 'seasonal_strength': 0.0}
    
    def _identify_market_regimes(self, data: pd.DataFrame, target_col: str) -> Dict:
        """识别市场环境（牛市/熊市/震荡）"""
        try:
            regimes = {
                'bull_market_periods': [],
                'bear_market_periods': [],
                'sideways_periods': [],
                'current_regime': 'unknown'
            }
            
            # 计算累积收益率
            cumulative_returns = data[target_col].cumsum()
            
            # 使用滑动窗口识别趋势
            window = min(20, len(data) // 4)
            if window < 5:
                return regimes
                
            rolling_trend = cumulative_returns.rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # 分类市场环境
            bull_threshold = rolling_trend.quantile(0.7)
            bear_threshold = rolling_trend.quantile(0.3)
            
            for i in range(len(rolling_trend)):
                if rolling_trend.iloc[i] > bull_threshold:
                    regimes['bull_market_periods'].append(i)
                elif rolling_trend.iloc[i] < bear_threshold:
                    regimes['bear_market_periods'].append(i)
                else:
                    regimes['sideways_periods'].append(i)
            
            # 确定当前市场环境
            if len(rolling_trend) > 0:
                latest_trend = rolling_trend.iloc[-1]
                if latest_trend > bull_threshold:
                    regimes['current_regime'] = 'bull'
                elif latest_trend < bear_threshold:
                    regimes['current_regime'] = 'bear'
                else:
                    regimes['current_regime'] = 'sideways'
            
            return regimes
            
        except Exception as e:
            self.logger.error(f"市场环境识别失败: {e}")
            return {}
    
    def _create_seasonal_aware_splits(self, data: pd.DataFrame, 
                                    n_splits: int, test_size: int, 
                                    seasonal_info: Dict) -> List[Tuple]:
        """创建季节性感知的数据分割"""
        try:
            splits = []
            data_length = len(data)
            
            if data_length < n_splits * test_size * 2:
                # 数据不足，使用简单时序分割
                tscv = TimeSeriesSplit(n_splits=min(n_splits, data_length // (test_size * 2)))
                for train_idx, test_idx in tscv.split(data):
                    splits.append((train_idx, test_idx))
            else:
                # 考虑季节性的智能分割
                if seasonal_info.get('has_seasonality', False):
                    # 确保每个分割都包含不同季节的数据
                    seasonal_step = max(1, data_length // (n_splits * 2))
                    
                    for i in range(n_splits):
                        # 训练集：从开始到测试集之前
                        train_end = data_length - (n_splits - i) * test_size - test_size
                        train_start = max(0, train_end - seasonal_step * 4)  # 包含多个季节
                        
                        # 测试集：固定大小的最新数据
                        test_start = train_end
                        test_end = min(data_length, test_start + test_size)
                        
                        if train_start < train_end and test_start < test_end:
                            train_idx = np.arange(train_start, train_end)
                            test_idx = np.arange(test_start, test_end)
                            splits.append((train_idx, test_idx))
                else:
                    # 标准时序分割
                    step_size = (data_length - test_size) // n_splits
                    
                    for i in range(n_splits):
                        train_end = min(data_length - test_size - (n_splits - i - 1) * (test_size // 2), 
                                      data_length - test_size)
                        train_start = max(0, train_end - step_size * 2)
                        
                        test_start = train_end
                        test_end = min(data_length, test_start + test_size)
                        
                        if train_start < train_end and test_start < test_end:
                            train_idx = np.arange(train_start, train_end)
                            test_idx = np.arange(test_start, test_end)
                            splits.append((train_idx, test_idx))
            
            self.logger.info(f"创建了{len(splits)}个季节性感知分割")
            return splits
            
        except Exception as e:
            self.logger.error(f"季节性分割创建失败: {e}")
            return []
    
    def _execute_cross_validation(self, data: pd.DataFrame, features: List[str], 
                                target_col: str, splits: List[Tuple]) -> List[Dict]:
        """执行交叉验证"""
        try:
            results = []
            
            for i, (train_idx, test_idx) in enumerate(splits):
                fold_result = {
                    'fold': i + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_period': (int(train_idx[0]), int(train_idx[-1])) if len(train_idx) > 0 else (0, 0),
                    'test_period': (int(test_idx[0]), int(test_idx[-1])) if len(test_idx) > 0 else (0, 0)
                }
                
                try:
                    # 准备训练和测试数据
                    train_data = data.iloc[train_idx]
                    test_data = data.iloc[test_idx]
                    
                    X_train = train_data[features].fillna(0)
                    y_train = train_data[target_col].fillna(0)
                    X_test = test_data[features].fillna(0)
                    y_test = test_data[target_col].fillna(0)
                    
                    # 简单线性回归验证（可以替换为更复杂的模型）
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    
                    # 预测和评估
                    y_pred = model.predict(X_test)
                    
                    # 计算方向准确率
                    direction_accuracy = self._calculate_direction_accuracy(y_test.values, y_pred)
                    
                    # 计算回归指标
                    mse = np.mean((y_test.values - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test.values - y_pred))
                    
                    fold_result.update({
                        'direction_accuracy': direction_accuracy,
                        'mse': mse,
                        'mae': mae,
                        'rmse': np.sqrt(mse),
                        'status': 'success'
                    })
                    
                except Exception as fold_error:
                    fold_result.update({
                        'status': 'error',
                        'error': str(fold_error),
                        'direction_accuracy': 0.0,
                        'mse': float('inf'),
                        'mae': float('inf')
                    })
                
                results.append(fold_result)
                
            self.logger.info(f"交叉验证完成，{len(results)}个折的验证")
            return results
            
        except Exception as e:
            self.logger.error(f"交叉验证执行失败: {e}")
            return []
    
    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算方向预测准确率"""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0
            
            # 转换为方向（涨跌）
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            
            # 计算一致性
            correct = np.sum(true_direction == pred_direction)
            total = len(y_true)
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"方向准确率计算失败: {e}")
            return 0.0
    
    def _analyze_validation_performance(self, validation_results: List[Dict], 
                                      seasonal_info: Dict) -> Dict:
        """分析验证性能"""
        try:
            if not validation_results:
                return {'mean_accuracy': 0.0, 'stability_score': 0.0}
            
            successful_results = [r for r in validation_results if r.get('status') == 'success']
            
            if not successful_results:
                return {'mean_accuracy': 0.0, 'stability_score': 0.0}
            
            # 基础统计
            accuracies = [r['direction_accuracy'] for r in successful_results]
            rmses = [r['rmse'] for r in successful_results]
            
            analysis = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'mean_rmse': np.mean(rmses),
                'std_rmse': np.std(rmses),
                'successful_folds': len(successful_results),
                'total_folds': len(validation_results)
            }
            
            # 稳定性分数（变异系数的倒数）
            cv_accuracy = analysis['std_accuracy'] / analysis['mean_accuracy'] if analysis['mean_accuracy'] > 0 else float('inf')
            analysis['stability_score'] = 1 / (1 + cv_accuracy)
            
            # 季节性影响分析
            if seasonal_info.get('has_seasonality', False):
                analysis['seasonal_impact'] = self._assess_seasonal_impact(successful_results, seasonal_info)
            else:
                analysis['seasonal_impact'] = {'impact_detected': False}
            
            # 趋势分析
            analysis['performance_trend'] = self._analyze_performance_trend(accuracies)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"验证性能分析失败: {e}")
            return {'mean_accuracy': 0.0, 'stability_score': 0.0}
    
    def _assess_seasonal_impact(self, results: List[Dict], seasonal_info: Dict) -> Dict:
        """评估季节性对性能的影响"""
        try:
            impact_analysis = {
                'impact_detected': False,
                'seasonal_performance_variance': 0.0,
                'best_season_accuracy': 0.0,
                'worst_season_accuracy': 0.0
            }
            
            # 简单的季节性影响分析
            accuracies = [r['direction_accuracy'] for r in results]
            
            if len(accuracies) > 2:
                # 检查性能是否有明显的周期性变化
                performance_variance = np.var(accuracies)
                mean_variance = np.mean(accuracies) * 0.1  # 10%基准
                
                impact_analysis.update({
                    'impact_detected': performance_variance > mean_variance,
                    'seasonal_performance_variance': performance_variance,
                    'best_season_accuracy': np.max(accuracies),
                    'worst_season_accuracy': np.min(accuracies)
                })
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"季节性影响评估失败: {e}")
            return {'impact_detected': False}
    
    def _analyze_performance_trend(self, accuracies: List[float]) -> Dict:
        """分析性能趋势"""
        try:
            trend_analysis = {
                'trend_direction': 'stable',
                'trend_strength': 0.0,
                'is_improving': False,
                'is_declining': False
            }
            
            if len(accuracies) < 3:
                return trend_analysis
            
            # 计算趋势斜率
            x = np.arange(len(accuracies))
            slope = np.polyfit(x, accuracies, 1)[0]
            
            # 判断趋势方向
            if slope > 0.01:
                trend_analysis['trend_direction'] = 'improving'
                trend_analysis['is_improving'] = True
            elif slope < -0.01:
                trend_analysis['trend_direction'] = 'declining'  
                trend_analysis['is_declining'] = True
            else:
                trend_analysis['trend_direction'] = 'stable'
            
            trend_analysis['trend_strength'] = abs(slope)
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"性能趋势分析失败: {e}")
            return {'trend_direction': 'stable', 'trend_strength': 0.0}
    
    def validate_ensemble_models(self, ensemble_engine, data: pd.DataFrame, 
                               features: List[str], target_col: str = 'future_return') -> Dict:
        """验证集成模型的性能"""
        try:
            self.logger.info("开始集成模型验证")
            
            # 执行季节性感知验证
            validation_result = self.perform_seasonal_aware_validation(
                data, features, target_col
            )
            
            if 'error' in validation_result:
                return validation_result
            
            # 添加集成模型特定的分析
            ensemble_analysis = {
                'ensemble_validation': validation_result,
                'model_consistency': self._analyze_model_consistency(ensemble_engine),
                'feature_importance': self._get_feature_importance_from_ensemble(ensemble_engine),
                'recommendation': self._generate_validation_recommendation(validation_result)
            }
            
            return ensemble_analysis
            
        except Exception as e:
            self.logger.error(f"集成模型验证失败: {e}")
            return {'error': str(e)}
    
    def _analyze_model_consistency(self, ensemble_engine) -> Dict:
        """分析集成模型的一致性"""
        try:
            consistency_analysis = {
                'weight_distribution': ensemble_engine.model_weights,
                'performance_variance': 0.0,
                'top_performer': None,
                'consistency_score': 0.0
            }
            
            if ensemble_engine.model_performance:
                accuracies = [perf['direction_accuracy'] for perf in ensemble_engine.model_performance.values()]
                consistency_analysis['performance_variance'] = np.var(accuracies)
                
                # 找出最佳表现模型
                best_model = max(ensemble_engine.model_performance.items(), 
                               key=lambda x: x[1]['direction_accuracy'])
                consistency_analysis['top_performer'] = {
                    'model': best_model[0],
                    'accuracy': best_model[1]['direction_accuracy'],
                    'weight': ensemble_engine.model_weights.get(best_model[0], 0)
                }
                
                # 计算一致性分数
                weight_entropy = -sum(w * np.log(w + 1e-10) for w in ensemble_engine.model_weights.values())
                max_entropy = np.log(len(ensemble_engine.model_weights))
                consistency_analysis['consistency_score'] = 1 - (weight_entropy / max_entropy) if max_entropy > 0 else 0
            
            return consistency_analysis
            
        except Exception as e:
            self.logger.error(f"模型一致性分析失败: {e}")
            return {}
    
    def _get_feature_importance_from_ensemble(self, ensemble_engine) -> Dict:
        """从集成模型获取特征重要性"""
        try:
            if hasattr(ensemble_engine, 'get_feature_importance_analysis'):
                return ensemble_engine.get_feature_importance_analysis()
            else:
                return {'status': 'feature_importance_not_available'}
                
        except Exception as e:
            self.logger.error(f"特征重要性获取失败: {e}")
            return {'error': str(e)}
    
    def _generate_validation_recommendation(self, validation_result: Dict) -> Dict:
        """生成验证建议"""
        try:
            performance = validation_result.get('performance_analysis', {})
            mean_accuracy = performance.get('mean_accuracy', 0)
            stability_score = performance.get('stability_score', 0)
            
            recommendation = {
                'overall_assessment': 'unknown',
                'specific_recommendations': [],
                'risk_level': 'medium'
            }
            
            # 整体评估
            if mean_accuracy > 0.65 and stability_score > 0.8:
                recommendation['overall_assessment'] = 'excellent'
                recommendation['risk_level'] = 'low'
            elif mean_accuracy > 0.6 and stability_score > 0.7:
                recommendation['overall_assessment'] = 'good'
                recommendation['risk_level'] = 'medium'
            elif mean_accuracy > 0.55:
                recommendation['overall_assessment'] = 'acceptable'
                recommendation['risk_level'] = 'medium'
            else:
                recommendation['overall_assessment'] = 'needs_improvement'
                recommendation['risk_level'] = 'high'
            
            # 具体建议
            if mean_accuracy < 0.6:
                recommendation['specific_recommendations'].append("考虑增加更多特征或改进特征工程")
            
            if stability_score < 0.7:
                recommendation['specific_recommendations'].append("模型稳定性需要改善，考虑调整模型权重")
            
            seasonal_info = validation_result.get('seasonal_info', {})
            if seasonal_info.get('has_seasonality', False):
                recommendation['specific_recommendations'].append("检测到季节性模式，建议在不同季节分别优化模型")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"验证建议生成失败: {e}")
            return {'overall_assessment': 'unknown'}