#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime, timedelta

class TimeSeriesValidator:
    """
    时序验证器 - 实现walk-forward验证和时序交叉验证
    专门针对时间序列数据的验证机制，避免数据泄露
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results = {}
        self.walk_forward_results = {}
        
    def walk_forward_validation(self, data: pd.DataFrame, 
                               features: List[str],
                               target_col: str = 'future_return',
                               prediction_days: int = 5,
                               initial_train_size: int = 60,
                               step_size: int = 5,
                               max_train_size: Optional[int] = None) -> Dict:
        """
        Walk-forward验证 - 模拟真实交易环境
        
        Args:
            data: 时序数据
            features: 特征列表
            target_col: 目标变量
            prediction_days: 预测天数
            initial_train_size: 初始训练集大小
            step_size: 每次前进步长
            max_train_size: 最大训练集大小
            
        Returns:
            验证结果和性能指标
        """
        try:
            self.logger.info(f"开始walk-forward验证，初始训练: {initial_train_size}，步长: {step_size}")
            
            if len(data) < initial_train_size + prediction_days + 20:
                return {"status": "insufficient_data", "required_size": initial_train_size + prediction_days + 20}
            
            # 准备目标变量
            data_clean = data.copy()
            data_clean[target_col] = data_clean['close'].shift(-prediction_days) / data_clean['close'] - 1
            data_clean = data_clean.iloc[:-prediction_days]  # 移除没有目标值的行
            
            walk_results = []
            train_start = 0
            
            # Walk-forward循环
            while train_start + initial_train_size + step_size < len(data_clean):
                # 确定训练和测试窗口
                train_end = train_start + initial_train_size
                if max_train_size and train_end - train_start > max_train_size:
                    train_start = train_end - max_train_size
                
                test_start = train_end
                test_end = min(test_start + step_size, len(data_clean))
                
                if test_end <= test_start:
                    break
                
                # 分割数据
                train_data = data_clean.iloc[train_start:train_end]
                test_data = data_clean.iloc[test_start:test_end]
                
                # 验证数据质量
                if len(train_data) < 30 or len(test_data) < 3:
                    train_start += step_size
                    continue
                
                # 单步验证
                step_result = self._validate_single_step(
                    train_data, test_data, features, target_col, 
                    train_start, test_start, test_end
                )
                
                if step_result['status'] == 'success':
                    walk_results.append(step_result)
                
                # 移动窗口
                train_start += step_size
                
                # 避免过度循环
                if len(walk_results) > 20:
                    break
            
            # 聚合结果
            if walk_results:
                aggregated = self._aggregate_walk_forward_results(walk_results)
                self.walk_forward_results = aggregated
                return aggregated
            else:
                return {"status": "no_valid_steps", "total_attempts": len(walk_results)}
                
        except Exception as e:
            self.logger.error(f"Walk-forward验证失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _validate_single_step(self, train_data: pd.DataFrame, 
                             test_data: pd.DataFrame,
                             features: List[str],
                             target_col: str,
                             train_start: int, test_start: int, test_end: int) -> Dict:
        """单步验证"""
        try:
            # 准备训练数据
            X_train = train_data[features].fillna(0)
            y_train = train_data[target_col].fillna(0)
            
            # 准备测试数据
            X_test = test_data[features].fillna(0)
            y_test = test_data[target_col].fillna(0)
            
            # 简单线性预测模型（避免过拟合）
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            
            # 训练
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 转换为方向预测
            y_true_direction = np.sign(y_test)
            y_pred_direction = np.sign(y_pred)
            
            # 计算指标
            direction_accuracy = accuracy_score(y_true_direction, y_pred_direction)
            
            # 强信号准确率（只考虑预测幅度较大的）
            strong_signals_mask = np.abs(y_pred) > np.std(y_pred) * 0.5
            if np.sum(strong_signals_mask) > 0:
                strong_accuracy = accuracy_score(
                    y_true_direction[strong_signals_mask], 
                    y_pred_direction[strong_signals_mask]
                )
            else:
                strong_accuracy = 0.0
            
            # 计算收益相关指标
            returns_corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
            if np.isnan(returns_corr):
                returns_corr = 0
            
            return {
                'status': 'success',
                'train_period': (train_start, train_start + len(train_data)),
                'test_period': (test_start, test_end),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'direction_accuracy': direction_accuracy,
                'strong_signal_accuracy': strong_accuracy,
                'returns_correlation': returns_corr,
                'strong_signals_count': np.sum(strong_signals_mask),
                'test_predictions': y_pred.tolist(),
                'test_actual': y_test.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"单步验证失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'train_period': (train_start, train_start + len(train_data)),
                'test_period': (test_start, test_end)
            }
    
    def _aggregate_walk_forward_results(self, walk_results: List[Dict]) -> Dict:
        """聚合walk-forward结果"""
        try:
            if not walk_results:
                return {"status": "no_results"}
            
            # 提取指标
            direction_accuracies = [r['direction_accuracy'] for r in walk_results if r.get('direction_accuracy') is not None]
            strong_accuracies = [r['strong_signal_accuracy'] for r in walk_results if r.get('strong_signal_accuracy') is not None]
            correlations = [r['returns_correlation'] for r in walk_results if r.get('returns_correlation') is not None]
            
            # 计算稳定性指标
            direction_std = np.std(direction_accuracies) if direction_accuracies else 0
            stability_score = max(0, 100 - direction_std * 100)  # 稳定性得分
            
            # 计算改进趋势
            if len(direction_accuracies) >= 3:
                recent_acc = np.mean(direction_accuracies[-3:])
                early_acc = np.mean(direction_accuracies[:3])
                improvement_trend = recent_acc - early_acc
            else:
                improvement_trend = 0
            
            # 计算权重平均（更近期的结果权重更高）
            if direction_accuracies:
                weights = np.linspace(0.5, 1.0, len(direction_accuracies))
                weighted_accuracy = np.average(direction_accuracies, weights=weights)
            else:
                weighted_accuracy = 0
            
            aggregated = {
                'status': 'success',
                'total_steps': len(walk_results),
                'successful_steps': len([r for r in walk_results if r['status'] == 'success']),
                'direction_accuracy': {
                    'mean': np.mean(direction_accuracies) if direction_accuracies else 0,
                    'std': direction_std,
                    'min': np.min(direction_accuracies) if direction_accuracies else 0,
                    'max': np.max(direction_accuracies) if direction_accuracies else 0,
                    'weighted_mean': weighted_accuracy,
                    'values': direction_accuracies
                },
                'strong_signal_accuracy': {
                    'mean': np.mean(strong_accuracies) if strong_accuracies else 0,
                    'std': np.std(strong_accuracies) if strong_accuracies else 0,
                    'values': strong_accuracies
                },
                'returns_correlation': {
                    'mean': np.mean(correlations) if correlations else 0,
                    'std': np.std(correlations) if correlations else 0,
                    'values': correlations
                },
                'stability_metrics': {
                    'stability_score': stability_score,
                    'improvement_trend': improvement_trend,
                    'consistency_ratio': len([acc for acc in direction_accuracies if acc > 0.5]) / len(direction_accuracies) if direction_accuracies else 0
                },
                'detailed_results': walk_results
            }
            
            self.logger.info(f"Walk-forward验证完成: {len(walk_results)}步，平均方向准确率: {aggregated['direction_accuracy']['mean']:.3f}")
            return aggregated
            
        except Exception as e:
            self.logger.error(f"结果聚合失败: {e}")
            return {"status": "aggregation_error", "error": str(e)}
    
    def time_series_cross_validation(self, data: pd.DataFrame,
                                   features: List[str],
                                   target_col: str = 'future_return',
                                   n_splits: int = 5,
                                   prediction_days: int = 5) -> Dict:
        """
        时序交叉验证 - 使用sklearn的TimeSeriesSplit
        """
        try:
            self.logger.info(f"开始时序交叉验证，分割数: {n_splits}")
            
            # 准备数据
            data_clean = data.copy()
            data_clean[target_col] = data_clean['close'].shift(-prediction_days) / data_clean['close'] - 1
            data_clean = data_clean.iloc[:-prediction_days]
            
            X = data_clean[features].fillna(0)
            y = data_clean[target_col].fillna(0)
            
            if len(X) < n_splits * 20:
                return {"status": "insufficient_data", "required_size": n_splits * 20}
            
            # 时序分割
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                try:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # 训练简单模型
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 计算指标
                    y_true_dir = np.sign(y_test)
                    y_pred_dir = np.sign(y_pred)
                    
                    fold_accuracy = accuracy_score(y_true_dir, y_pred_dir)
                    fold_corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
                    if np.isnan(fold_corr):
                        fold_corr = 0
                    
                    cv_results.append({
                        'fold': fold,
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'direction_accuracy': fold_accuracy,
                        'returns_correlation': fold_corr
                    })
                    
                    self.logger.debug(f"Fold {fold}: 准确率={fold_accuracy:.3f}, 相关性={fold_corr:.3f}")
                    
                except Exception as e:
                    self.logger.warning(f"Fold {fold}失败: {e}")
                    continue
            
            if cv_results:
                # 聚合CV结果
                accuracies = [r['direction_accuracy'] for r in cv_results]
                correlations = [r['returns_correlation'] for r in cv_results]
                
                cv_summary = {
                    'status': 'success',
                    'n_folds': len(cv_results),
                    'direction_accuracy': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'min': np.min(accuracies),
                        'max': np.max(accuracies),
                        'values': accuracies
                    },
                    'returns_correlation': {
                        'mean': np.mean(correlations),
                        'std': np.std(correlations),
                        'values': correlations
                    },
                    'fold_results': cv_results
                }
                
                self.validation_results = cv_summary
                self.logger.info(f"时序交叉验证完成: 平均准确率={cv_summary['direction_accuracy']['mean']:.3f}")
                return cv_summary
            else:
                return {"status": "all_folds_failed"}
                
        except Exception as e:
            self.logger.error(f"时序交叉验证失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def comprehensive_validation(self, data: pd.DataFrame,
                               features: List[str],
                               target_col: str = 'future_return',
                               prediction_days: int = 5) -> Dict:
        """
        综合验证 - 结合walk-forward和时序交叉验证
        """
        try:
            self.logger.info("开始综合时序验证")
            
            # Walk-forward验证
            wf_results = self.walk_forward_validation(
                data, features, target_col, prediction_days,
                initial_train_size=60, step_size=5
            )
            
            # 时序交叉验证
            cv_results = self.time_series_cross_validation(
                data, features, target_col, n_splits=5, prediction_days=prediction_days
            )
            
            # 结合两种验证结果
            comprehensive = {
                'validation_timestamp': datetime.now().isoformat(),
                'data_info': {
                    'total_samples': len(data),
                    'features_count': len(features),
                    'prediction_days': prediction_days
                },
                'walk_forward_validation': wf_results,
                'time_series_cv': cv_results,
                'combined_metrics': self._combine_validation_metrics(wf_results, cv_results)
            }
            
            return comprehensive
            
        except Exception as e:
            self.logger.error(f"综合验证失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _combine_validation_metrics(self, wf_results: Dict, cv_results: Dict) -> Dict:
        """结合两种验证的指标"""
        try:
            combined = {
                'validation_consistency': 'unknown',
                'overall_confidence': 0.0,
                'recommendation': 'insufficient_data'
            }
            
            # 提取准确率
            wf_acc = wf_results.get('direction_accuracy', {}).get('mean', 0) if wf_results.get('status') == 'success' else 0
            cv_acc = cv_results.get('direction_accuracy', {}).get('mean', 0) if cv_results.get('status') == 'success' else 0
            
            if wf_acc > 0 and cv_acc > 0:
                # 计算一致性
                acc_diff = abs(wf_acc - cv_acc)
                if acc_diff < 0.05:
                    combined['validation_consistency'] = 'high'
                    confidence_boost = 0.2
                elif acc_diff < 0.10:
                    combined['validation_consistency'] = 'medium'
                    confidence_boost = 0.1
                else:
                    combined['validation_consistency'] = 'low'
                    confidence_boost = 0.0
                
                # 综合置信度
                avg_accuracy = (wf_acc + cv_acc) / 2
                stability = wf_results.get('stability_metrics', {}).get('stability_score', 0) / 100
                
                combined['overall_confidence'] = min(
                    avg_accuracy + confidence_boost + stability * 0.1, 1.0
                )
                
                # 生成建议
                if avg_accuracy >= 0.65 and combined['validation_consistency'] in ['high', 'medium']:
                    combined['recommendation'] = 'excellent_model'
                elif avg_accuracy >= 0.60:
                    combined['recommendation'] = 'good_model'
                elif avg_accuracy >= 0.55:
                    combined['recommendation'] = 'acceptable_model'
                else:
                    combined['recommendation'] = 'needs_improvement'
                
                combined.update({
                    'walk_forward_accuracy': wf_acc,
                    'cross_validation_accuracy': cv_acc,
                    'average_accuracy': avg_accuracy,
                    'accuracy_difference': acc_diff
                })
            
            return combined
            
        except Exception as e:
            self.logger.error(f"指标结合失败: {e}")
            return {"validation_consistency": "error", "error": str(e)}
    
    def get_validation_summary(self) -> Dict:
        """获取验证摘要"""
        return {
            'walk_forward_results': self.walk_forward_results,
            'cv_results': self.validation_results,
            'last_validation': datetime.now().isoformat()
        }