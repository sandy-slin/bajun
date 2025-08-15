# -*- coding: utf-8 -*-
"""
机器学习预测模型
基于真实历史数据训练多种机器学习模型进行板块预测
严格禁止使用任何模拟数据，所有训练和预测都基于真实市场数据
"""

import asyncio
import logging
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 如果有XGBoost和LightGBM可以取消注释
# try:
#     import xgboost as xgb
#     import lightgbm as lgb
#     HAS_BOOST_LIBS = True
# except ImportError:
#     HAS_BOOST_LIBS = False

from data.sector_fetcher import SectorFetcher
from data.enhanced_feature_engineer import EnhancedFeatureEngineer
from data.technical_calculator import TechnicalCalculator


class MLPredictor:
    """机器学习预测器 - 基于真实数据的多模型集成预测"""
    
    def __init__(self, sector_fetcher: SectorFetcher, tech_calculator: TechnicalCalculator):
        self.sector_fetcher = sector_fetcher
        self.tech_calculator = tech_calculator
        self.feature_engineer = EnhancedFeatureEngineer()
        self.logger = logging.getLogger(__name__)
        
        # 模型存储
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # 模型配置
        self.model_configs = self._get_model_configurations()
        
        # 数据存储路径
        self.model_save_path = "models/"
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def _get_model_configurations(self) -> Dict:
        """获取模型配置"""
        return {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'scaler': RobustScaler()
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ),
                'scaler': StandardScaler()
            },
            'ridge_regression': {
                'model': Ridge(alpha=1.0, random_state=42),
                'scaler': StandardScaler()
            },
            'elastic_net': {
                'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                'scaler': StandardScaler()
            },
            'svr': {
                'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'scaler': StandardScaler()
            }
        }
        
    async def train_models_on_real_data(self, 
                                       training_months: int = 24,
                                       prediction_horizon: int = 5) -> Dict[str, Any]:
        """
        基于真实历史数据训练所有模型
        
        Args:
            training_months: 训练数据月数
            prediction_horizon: 预测天数
            
        Returns:
            Dict: 训练结果和模型性能
        """
        try:
            self.logger.info(f"开始基于真实数据训练模型，训练期: {training_months}个月")
            
            # 1. 获取真实训练数据
            training_data = await self._prepare_real_training_data(
                training_months, prediction_horizon
            )
            
            if not training_data:
                raise ValueError("无法获取足够的真实训练数据")
                
            # 2. 训练所有模型
            model_performances = {}
            
            for model_name, config in self.model_configs.items():
                self.logger.info(f"训练模型: {model_name}")
                
                performance = await self._train_single_model(
                    model_name, config, training_data, prediction_horizon
                )
                
                model_performances[model_name] = performance
                
            # 3. 选择最佳模型
            best_model = self._select_best_model(model_performances)
            
            # 4. 保存模型和配置
            self._save_models_and_configs(model_performances, best_model)
            
            training_result = {
                'training_time': datetime.now().isoformat(),
                'training_months': training_months,
                'prediction_horizon': prediction_horizon,
                'models_trained': len(model_performances),
                'model_performances': model_performances,
                'best_model': best_model,
                'training_data_summary': self._summarize_training_data(training_data)
            }
            
            self.logger.info(f"模型训练完成，最佳模型: {best_model}")
            return training_result
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return {'error': str(e)}
            
    async def _prepare_real_training_data(self, training_months: int, 
                                        prediction_horizon: int) -> Dict[str, Any]:
        """准备真实训练数据"""
        try:
            end_date = datetime.now() - timedelta(days=prediction_horizon)
            start_date = end_date - timedelta(days=training_months * 30 + 60)
            
            # 获取真实板块数据
            sectors_data = await self.sector_fetcher.get_all_sectors_data(
                (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            )
            
            if not sectors_data:
                return {}
                
            # 构建训练样本
            training_samples = []
            feature_names = None
            
            for sector_name, sector_df in sectors_data.items():
                if sector_df.empty or len(sector_df) < 100:
                    continue
                    
                # 为每个时间点创建训练样本
                for i in range(60, len(sector_df) - prediction_horizon):
                    # 特征：基于前60天的真实数据
                    feature_data = sector_df.iloc[i-60:i].copy()
                    
                    if len(feature_data) < 60:
                        continue
                        
                    # 使用真实数据创建特征
                    features = self.feature_engineer.create_comprehensive_features(
                        feature_data, sector_name
                    )
                    
                    if not features:
                        continue
                        
                    # 标签：未来N天的真实收益率
                    current_price = sector_df.iloc[i]['close']
                    future_price = sector_df.iloc[i + prediction_horizon]['close']
                    target_return = (future_price / current_price - 1) * 100
                    
                    # 创建训练样本
                    sample = {
                        'sector': sector_name,
                        'date': sector_df.index[i],
                        'features': features,
                        'target': target_return,
                        'current_price': current_price,
                        'future_price': future_price
                    }
                    
                    training_samples.append(sample)
                    
                    if feature_names is None:
                        feature_names = list(features.keys())
                        
            return {
                'samples': training_samples,
                'feature_names': feature_names,
                'total_samples': len(training_samples),
                'unique_sectors': len(set(s['sector'] for s in training_samples)),
                'date_range': {
                    'start': min(s['date'] for s in training_samples) if training_samples else None,
                    'end': max(s['date'] for s in training_samples) if training_samples else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {e}")
            return {}
            
    async def _train_single_model(self, model_name: str, config: Dict, 
                                training_data: Dict, prediction_horizon: int) -> Dict[str, Any]:
        """训练单个模型"""
        try:
            samples = training_data['samples']
            feature_names = training_data['feature_names']
            
            if not samples or not feature_names:
                return {'error': '训练数据为空'}
                
            # 构建特征矩阵和标签向量
            X = []
            y = []
            dates = []
            
            for sample in samples:
                features = sample['features']
                feature_vector = [features.get(fname, 0) for fname in feature_names]
                
                X.append(feature_vector)
                y.append(sample['target'])
                dates.append(sample['date'])
                
            X = np.array(X)
            y = np.array(y)
            dates = pd.to_datetime(dates)
            
            # 数据预处理
            scaler = config['scaler']
            X_scaled = scaler.fit_transform(X)
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                config['model'], X_scaled, y, 
                cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            # 训练最终模型（使用所有数据）
            model = config['model']
            model.fit(X_scaled, y)
            
            # 预测评估
            y_pred = model.predict(X_scaled)
            
            # 计算性能指标
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # 方向准确率
            direction_accuracy = np.mean(
                (y > 0) == (y_pred > 0)
            ) * 100
            
            # 特征重要性（如果模型支持）
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    feature_names, model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(
                    feature_names, np.abs(model.coef_)
                ))
                
            # 存储模型和缩放器
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.feature_importance[model_name] = feature_importance
            
            performance = {
                'model_name': model_name,
                'cv_mse_mean': -cv_scores.mean(),
                'cv_mse_std': cv_scores.std(),
                'train_mse': mse,
                'train_mae': mae,
                'train_r2': r2,
                'direction_accuracy': direction_accuracy,
                'feature_importance': feature_importance,
                'training_samples': len(samples),
                'feature_count': len(feature_names)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"训练{model_name}模型失败: {e}")
            return {'error': str(e)}
            
    def _select_best_model(self, model_performances: Dict) -> str:
        """选择最佳模型"""
        try:
            valid_performances = {
                name: perf for name, perf in model_performances.items()
                if 'error' not in perf
            }
            
            if not valid_performances:
                return list(model_performances.keys())[0]
                
            # 综合评分：方向准确率 * 0.6 + R² * 0.4
            best_model = None
            best_score = -1
            
            for model_name, perf in valid_performances.items():
                direction_acc = perf.get('direction_accuracy', 0) / 100
                r2 = max(0, perf.get('train_r2', 0))  # R²可能为负
                
                composite_score = direction_acc * 0.6 + r2 * 0.4
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model = model_name
                    
            return best_model or list(valid_performances.keys())[0]
            
        except Exception as e:
            self.logger.error(f"选择最佳模型失败: {e}")
            return list(model_performances.keys())[0]
            
    def _save_models_and_configs(self, model_performances: Dict, best_model: str):
        """保存模型和配置"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存所有模型
            for model_name in self.models:
                model_file = f"{self.model_save_path}ml_model_{model_name}_{timestamp}.joblib"
                scaler_file = f"{self.model_save_path}scaler_{model_name}_{timestamp}.joblib"
                
                joblib.dump(self.models[model_name], model_file)
                joblib.dump(self.scalers[model_name], scaler_file)
                
            # 保存模型性能和配置
            config_file = f"{self.model_save_path}model_config_{timestamp}.json"
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_performances': model_performances,
                    'best_model': best_model,
                    'feature_importance': {k: v for k, v in self.feature_importance.items() if v},
                    'timestamp': timestamp
                }, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"模型已保存到: {self.model_save_path}")
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            
    async def predict_with_ml_models(self, sector_name: str, 
                                   sector_data: pd.DataFrame,
                                   ensemble_method: str = 'weighted_average') -> Dict[str, Any]:
        """
        使用机器学习模型进行预测
        
        Args:
            sector_name: 板块名称
            sector_data: 真实板块数据
            ensemble_method: 集成方法
            
        Returns:
            Dict: 预测结果
        """
        try:
            if not self.models:
                return {'error': '模型未训练'}
                
            # 基于真实数据创建特征
            features = self.feature_engineer.create_comprehensive_features(
                sector_data, sector_name
            )
            
            if not features:
                return {'error': '无法创建特征'}
                
            # 模型预测
            predictions = {}
            weights = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name not in self.scalers:
                        continue
                        
                    # 准备特征向量
                    feature_names = list(self.feature_importance.get(model_name, {}).keys())
                    if not feature_names:
                        continue
                        
                    feature_vector = [features.get(fname, 0) for fname in feature_names]
                    feature_vector = np.array(feature_vector).reshape(1, -1)
                    
                    # 标准化
                    scaler = self.scalers[model_name]
                    feature_vector_scaled = scaler.transform(feature_vector)
                    
                    # 预测
                    pred = model.predict(feature_vector_scaled)[0]
                    predictions[model_name] = pred
                    
                    # 权重（基于交叉验证性能）
                    weights[model_name] = 1.0  # 简化，实际中应基于模型性能
                    
                except Exception as e:
                    self.logger.warning(f"{model_name}预测失败: {e}")
                    continue
                    
            if not predictions:
                return {'error': '所有模型预测失败'}
                
            # 集成预测结果
            if ensemble_method == 'weighted_average':
                total_weight = sum(weights.values())
                ensemble_pred = sum(
                    pred * weights[name] for name, pred in predictions.items()
                ) / total_weight
                
            elif ensemble_method == 'median':
                ensemble_pred = np.median(list(predictions.values()))
                
            else:  # simple_average
                ensemble_pred = np.mean(list(predictions.values()))
                
            # 计算预测置信度
            pred_std = np.std(list(predictions.values()))
            confidence = max(0, 1 - pred_std / abs(ensemble_pred)) if ensemble_pred != 0 else 0.5
            
            # 生成预测结果
            prediction_result = {
                'sector_name': sector_name,
                'ensemble_prediction': round(ensemble_pred, 3),
                'individual_predictions': predictions,
                'prediction_confidence': round(confidence, 3),
                'prediction_std': round(pred_std, 3),
                'ensemble_method': ensemble_method,
                'models_used': list(predictions.keys()),
                'trend_prediction': '上涨' if ensemble_pred > 1 else '下跌' if ensemble_pred < -1 else '震荡',
                'prediction_strength': '强' if abs(ensemble_pred) > 3 else '中' if abs(ensemble_pred) > 1 else '弱'
            }
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"ML模型预测失败: {e}")
            return {'error': str(e)}
            
    def _summarize_training_data(self, training_data: Dict) -> Dict:
        """汇总训练数据信息"""
        try:
            if not training_data or 'samples' not in training_data:
                return {}
                
            samples = training_data['samples']
            
            # 收益率分布
            returns = [s['target'] for s in samples]
            
            return {
                'total_samples': len(samples),
                'unique_sectors': len(set(s['sector'] for s in samples)),
                'date_range': training_data.get('date_range'),
                'return_statistics': {
                    'mean': round(np.mean(returns), 3),
                    'std': round(np.std(returns), 3),
                    'min': round(np.min(returns), 3),
                    'max': round(np.max(returns), 3),
                    'positive_ratio': round(np.mean(np.array(returns) > 0), 3)
                },
                'feature_count': len(training_data.get('feature_names', []))
            }
            
        except Exception as e:
            self.logger.error(f"汇总训练数据失败: {e}")
            return {}
            
    def load_pretrained_models(self, model_timestamp: str = None) -> bool:
        """加载预训练模型"""
        try:
            if model_timestamp is None:
                # 找最新的模型
                model_files = [f for f in os.listdir(self.model_save_path) 
                             if f.startswith('ml_model_') and f.endswith('.joblib')]
                if not model_files:
                    return False
                    
                # 提取时间戳
                timestamps = []
                for f in model_files:
                    parts = f.split('_')
                    if len(parts) >= 4:
                        timestamp = parts[-1].replace('.joblib', '')
                        timestamps.append(timestamp)
                        
                model_timestamp = max(timestamps) if timestamps else None
                
            if not model_timestamp:
                return False
                
            # 加载模型
            loaded_models = {}
            loaded_scalers = {}
            
            for model_name in self.model_configs.keys():
                model_file = f"{self.model_save_path}ml_model_{model_name}_{model_timestamp}.joblib"
                scaler_file = f"{self.model_save_path}scaler_{model_name}_{model_timestamp}.joblib"
                
                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    loaded_models[model_name] = joblib.load(model_file)
                    loaded_scalers[model_name] = joblib.load(scaler_file)
                    
            if loaded_models:
                self.models = loaded_models
                self.scalers = loaded_scalers
                self.logger.info(f"已加载{len(loaded_models)}个预训练模型")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"加载预训练模型失败: {e}")
            return False
            
    async def evaluate_model_performance_on_real_data(self, 
                                                    evaluation_months: int = 6) -> Dict[str, Any]:
        """基于真实数据评估模型性能"""
        try:
            if not self.models:
                return {'error': '模型未加载'}
                
            # 获取评估数据（最近的真实数据）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=evaluation_months * 30)
            
            sectors_data = await self.sector_fetcher.get_all_sectors_data(
                (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            )
            
            if not sectors_data:
                return {'error': '无法获取评估数据'}
                
            # 对每个板块进行预测和评估
            evaluation_results = {}
            
            for sector_name, sector_df in list(sectors_data.items())[:5]:  # 限制评估板块数量
                if len(sector_df) < 60:
                    continue
                    
                sector_results = []
                
                # 在历史数据上进行预测验证
                for i in range(60, len(sector_df) - 5):
                    hist_data = sector_df.iloc[:i]
                    
                    # 进行预测
                    pred_result = await self.predict_with_ml_models(sector_name, hist_data)
                    
                    if 'error' not in pred_result:
                        # 获取实际结果
                        actual_return = (sector_df.iloc[i+5]['close'] / sector_df.iloc[i]['close'] - 1) * 100
                        
                        sector_results.append({
                            'predicted': pred_result['ensemble_prediction'],
                            'actual': actual_return,
                            'confidence': pred_result['prediction_confidence']
                        })
                        
                if sector_results:
                    evaluation_results[sector_name] = self._calculate_evaluation_metrics(sector_results)
                    
            return {
                'evaluation_time': datetime.now().isoformat(),
                'evaluation_months': evaluation_months,
                'sectors_evaluated': len(evaluation_results),
                'sector_results': evaluation_results,
                'overall_performance': self._calculate_overall_evaluation_metrics(evaluation_results)
            }
            
        except Exception as e:
            self.logger.error(f"评估模型性能失败: {e}")
            return {'error': str(e)}
            
    def _calculate_evaluation_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """计算评估指标"""
        try:
            predicted = [r['predicted'] for r in results]
            actual = [r['actual'] for r in results]
            
            # 基本指标
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            # 方向准确率
            direction_accuracy = np.mean(
                (np.array(actual) > 0) == (np.array(predicted) > 0)
            ) * 100
            
            # 相关系数
            correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
            
            return {
                'mse': round(mse, 4),
                'mae': round(mae, 4),
                'r2': round(r2, 4),
                'direction_accuracy': round(direction_accuracy, 2),
                'correlation': round(correlation, 4),
                'sample_count': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"计算评估指标失败: {e}")
            return {}
            
    def _calculate_overall_evaluation_metrics(self, evaluation_results: Dict) -> Dict[str, float]:
        """计算总体评估指标"""
        try:
            if not evaluation_results:
                return {}
                
            all_metrics = list(evaluation_results.values())
            
            return {
                'avg_direction_accuracy': round(np.mean([m.get('direction_accuracy', 0) for m in all_metrics]), 2),
                'avg_correlation': round(np.mean([m.get('correlation', 0) for m in all_metrics]), 4),
                'avg_r2': round(np.mean([m.get('r2', 0) for m in all_metrics]), 4),
                'avg_mae': round(np.mean([m.get('mae', 0) for m in all_metrics]), 4),
                'best_sector_accuracy': max([m.get('direction_accuracy', 0) for m in all_metrics]),
                'worst_sector_accuracy': min([m.get('direction_accuracy', 0) for m in all_metrics]),
                'total_samples': sum([m.get('sample_count', 0) for m in all_metrics])
            }
            
        except Exception as e:
            self.logger.error(f"计算总体评估指标失败: {e}")
            return {}