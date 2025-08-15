#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from datetime import datetime, timedelta

class EnsemblePredictionEngine:
    """
    集成预测引擎 - 实现多模型协同预测和加权投票系统
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.ensemble_results = {}
        
        # 初始化基础模型
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """初始化基础预测模型"""
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        self.logger.info("已初始化5个基础预测模型")
    
    def train_ensemble_models(self, training_data: pd.DataFrame, 
                            features: List[str], target_col: str = 'future_return') -> Dict:
        """
        训练集成模型
        
        Args:
            training_data: 训练数据
            features: 特征列表
            target_col: 目标列名
            
        Returns:
            训练结果和性能指标
        """
        try:
            self.logger.info(f"开始训练集成模型，使用{len(features)}个特征")
            
            # 准备训练数据
            X = training_data[features].fillna(0)
            y = training_data[target_col].fillna(0)
            
            if len(X) < 50:
                self.logger.warning(f"训练数据量较少: {len(X)}条")
                return {"status": "insufficient_data", "data_size": len(X)}
            
            # 训练各个基础模型
            model_scores = {}
            trained_models = {}
            
            for model_name, model in self.base_models.items():
                try:
                    # 训练模型
                    model.fit(X, y)
                    trained_models[model_name] = model
                    
                    # 预测并评估
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    # 计算方向准确率
                    direction_accuracy = self._calculate_direction_accuracy(y, y_pred)
                    
                    model_scores[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'direction_accuracy': direction_accuracy,
                        'rmse': np.sqrt(mse)
                    }
                    
                    self.logger.info(f"{model_name}: 方向准确率={direction_accuracy:.3f}, RMSE={np.sqrt(mse):.4f}")
                    
                except Exception as e:
                    self.logger.error(f"{model_name}训练失败: {e}")
                    continue
            
            # 计算模型权重
            self.models = trained_models
            self.model_performance = model_scores
            self.model_weights = self._calculate_model_weights(model_scores)
            
            # 生成集成预测
            ensemble_pred = self._generate_ensemble_prediction(X)
            ensemble_accuracy = self._calculate_direction_accuracy(y, ensemble_pred)
            
            results = {
                'status': 'success',
                'models_trained': len(trained_models),
                'model_performance': model_scores,
                'model_weights': self.model_weights,
                'ensemble_accuracy': ensemble_accuracy,
                'training_size': len(X),
                'features_used': len(features)
            }
            
            self.logger.info(f"集成模型训练完成，集成准确率: {ensemble_accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"集成模型训练失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算方向预测准确率"""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0
            
            # 将连续值转换为方向（涨跌）
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            
            # 计算方向一致性
            correct_directions = np.sum(true_direction == pred_direction)
            total_predictions = len(y_true)
            
            return correct_directions / total_predictions if total_predictions > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"方向准确率计算失败: {e}")
            return 0.0
    
    def _calculate_model_weights(self, model_scores: Dict) -> Dict:
        """
        基于模型性能计算权重
        综合考虑RMSE、MAE和方向准确率
        """
        try:
            if not model_scores:
                return {}
            
            weights = {}
            
            # 提取各项指标
            rmse_scores = {name: scores['rmse'] for name, scores in model_scores.items()}
            direction_scores = {name: scores['direction_accuracy'] for name, scores in model_scores.items()}
            
            # 归一化RMSE分数（越小越好）
            max_rmse = max(rmse_scores.values()) if rmse_scores.values() else 1.0
            normalized_rmse = {name: 1 - (score / max_rmse) for name, score in rmse_scores.items()}
            
            # 计算综合权重
            for model_name in model_scores.keys():
                # 综合权重 = 0.3 * (1-归一化RMSE) + 0.7 * 方向准确率
                rmse_weight = normalized_rmse.get(model_name, 0)
                direction_weight = direction_scores.get(model_name, 0)
                
                composite_score = 0.3 * rmse_weight + 0.7 * direction_weight
                weights[model_name] = max(composite_score, 0.01)  # 确保最小权重
            
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}
            
            self.logger.info(f"模型权重分配: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"模型权重计算失败: {e}")
            return {}
    
    def _generate_ensemble_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """生成集成预测结果"""
        try:
            if not self.models or not self.model_weights:
                return np.zeros(len(X))
            
            ensemble_pred = np.zeros(len(X))
            
            for model_name, model in self.models.items():
                weight = self.model_weights.get(model_name, 0)
                if weight > 0:
                    try:
                        pred = model.predict(X)
                        ensemble_pred += weight * pred
                    except Exception as e:
                        self.logger.warning(f"{model_name}预测失败: {e}")
                        continue
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"集成预测生成失败: {e}")
            return np.zeros(len(X))
    
    def predict(self, data: pd.DataFrame, features: List[str]) -> Dict:
        """
        使用集成模型进行预测
        
        Args:
            data: 预测数据
            features: 特征列表
            
        Returns:
            预测结果和信心度
        """
        try:
            if not self.models:
                return {"status": "no_trained_models", "predictions": []}
            
            X = data[features].fillna(0)
            
            # 各模型预测结果
            model_predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    model_predictions[model_name] = pred
                except Exception as e:
                    self.logger.warning(f"{model_name}预测失败: {e}")
                    continue
            
            # 集成预测
            ensemble_pred = self._generate_ensemble_prediction(X)
            
            # 计算预测信心度
            confidence_scores = self._calculate_prediction_confidence(model_predictions, ensemble_pred)
            
            # 生成预测信号
            signals = self._generate_prediction_signals(ensemble_pred, confidence_scores)
            
            results = {
                'status': 'success',
                'ensemble_predictions': ensemble_pred.tolist(),
                'model_predictions': {name: pred.tolist() for name, pred in model_predictions.items()},
                'confidence_scores': confidence_scores.tolist(),
                'prediction_signals': signals,
                'model_weights': self.model_weights
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"集成预测失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_prediction_confidence(self, model_predictions: Dict, ensemble_pred: np.ndarray) -> np.ndarray:
        """计算预测信心度"""
        try:
            if not model_predictions or len(ensemble_pred) == 0:
                return np.zeros(len(ensemble_pred))
            
            confidence = np.zeros(len(ensemble_pred))
            
            for i in range(len(ensemble_pred)):
                # 收集所有模型在位置i的预测
                predictions_at_i = []
                for model_name, pred_array in model_predictions.items():
                    if i < len(pred_array):
                        predictions_at_i.append(pred_array[i])
                
                if len(predictions_at_i) > 1:
                    # 计算预测一致性（标准差的倒数）
                    pred_std = np.std(predictions_at_i)
                    consistency = 1 / (1 + pred_std) if pred_std > 0 else 1.0
                    
                    # 计算预测强度（绝对值）
                    strength = abs(ensemble_pred[i])
                    
                    # 综合信心度
                    confidence[i] = 0.6 * consistency + 0.4 * min(strength, 1.0)
                else:
                    confidence[i] = 0.5  # 默认信心度
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"信心度计算失败: {e}")
            return np.zeros(len(ensemble_pred))
    
    def _generate_prediction_signals(self, predictions: np.ndarray, confidence: np.ndarray) -> List[Dict]:
        """生成预测信号"""
        try:
            signals = []
            
            for i, (pred, conf) in enumerate(zip(predictions, confidence)):
                # 根据预测值和信心度生成信号
                if conf > 0.6:  # 高信心度
                    if pred > 0.02:  # 预期上涨超过2%
                        signal_type = "strong_buy"
                        signal_strength = min(conf * abs(pred) * 10, 1.0)
                    elif pred < -0.02:  # 预期下跌超过2%
                        signal_type = "strong_sell"
                        signal_strength = min(conf * abs(pred) * 10, 1.0)
                    elif pred > 0.005:  # 预期上涨0.5-2%
                        signal_type = "buy"
                        signal_strength = min(conf * abs(pred) * 20, 0.8)
                    elif pred < -0.005:  # 预期下跌0.5-2%
                        signal_type = "sell"
                        signal_strength = min(conf * abs(pred) * 20, 0.8)
                    else:
                        signal_type = "hold"
                        signal_strength = conf * 0.3
                else:
                    signal_type = "hold"
                    signal_strength = conf * 0.2
                
                signals.append({
                    'index': i,
                    'signal_type': signal_type,
                    'signal_strength': signal_strength,
                    'predicted_return': pred,
                    'confidence': conf,
                    'timestamp': datetime.now().isoformat()
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"信号生成失败: {e}")
            return []
    
    def get_model_performance_summary(self) -> Dict:
        """获取模型性能摘要"""
        try:
            if not self.model_performance:
                return {"status": "no_performance_data"}
            
            summary = {
                'total_models': len(self.model_performance),
                'model_details': self.model_performance,
                'best_model': None,
                'worst_model': None,
                'average_accuracy': 0.0
            }
            
            # 找出最佳和最差模型
            accuracies = {name: perf['direction_accuracy'] 
                         for name, perf in self.model_performance.items()}
            
            if accuracies:
                best_model = max(accuracies, key=accuracies.get)
                worst_model = min(accuracies, key=accuracies.get)
                avg_accuracy = sum(accuracies.values()) / len(accuracies)
                
                summary.update({
                    'best_model': {
                        'name': best_model,
                        'accuracy': accuracies[best_model],
                        'weight': self.model_weights.get(best_model, 0)
                    },
                    'worst_model': {
                        'name': worst_model,
                        'accuracy': accuracies[worst_model],
                        'weight': self.model_weights.get(worst_model, 0)
                    },
                    'average_accuracy': avg_accuracy
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"性能摘要生成失败: {e}")
            return {"status": "error", "error": str(e)}