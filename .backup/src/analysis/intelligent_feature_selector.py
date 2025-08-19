"""
智能特征选择器 - 基于信息熵和预测能力的特征优化
专门解决特征噪音和信号质量问题
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class IntelligentFeatureSelector:
    """智能特征选择器，基于多维度评估选择最优特征"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.selected_features = []
        self.feature_scores = {}
        self.feature_stability = {}
        
    def select_optimal_features(self, data: pd.DataFrame, 
                               target_col: str = 'future_return',
                               prediction_days: int = 5,
                               max_features: int = 50) -> List[str]:
        """
        选择最优特征集合，基于信息增益、预测能力和稳定性
        """
        try:
            if data.empty or len(data) < 30:
                self.logger.warning("数据不足，无法进行特征选择")
                return []
            
            # 准备目标变量
            data = self._prepare_target_variable(data, prediction_days)
            
            # 1. 基础特征过滤
            valid_features = self._basic_feature_filtering(data, target_col)
            
            # 2. 信息增益评估
            info_gain_scores = self._calculate_information_gain(data, valid_features, target_col)
            
            # 3. 预测能力评估
            prediction_scores = self._evaluate_prediction_power(data, valid_features, target_col)
            
            # 4. 特征稳定性评估
            stability_scores = self._assess_feature_stability(data, valid_features)
            
            # 5. 相关性去冗余
            correlation_filtered = self._remove_redundant_features(data, valid_features)
            
            # 6. 综合评分和排序
            final_features = self._rank_and_select_features(
                correlation_filtered, info_gain_scores, prediction_scores, 
                stability_scores, max_features
            )
            
            self.selected_features = final_features
            self.logger.info(f"特征选择完成，从{len(data.columns)}个特征中选择了{len(final_features)}个优质特征")
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"特征选择失败: {e}")
            return []
    
    def _prepare_target_variable(self, data: pd.DataFrame, prediction_days: int) -> pd.DataFrame:
        """准备目标变量"""
        try:
            data_copy = data.copy()
            
            # 计算未来收益
            if 'close' in data_copy.columns:
                future_price = data_copy['close'].shift(-prediction_days)
                current_price = data_copy['close']
                future_return = (future_price - current_price) / current_price
                
                # 转换为分类标签（方向预测）
                data_copy['future_return'] = (future_return > 0).astype(int)
                data_copy['future_return_magnitude'] = future_return.abs()
                
                # 去除无效数据
                data_copy = data_copy.dropna(subset=['future_return'])
            
            return data_copy
            
        except Exception as e:
            self.logger.error(f"目标变量准备失败: {e}")
            return data
    
    def _basic_feature_filtering(self, data: pd.DataFrame, target_col: str) -> List[str]:
        """基础特征过滤"""
        try:
            valid_features = []
            
            # 排除非数值列和目标列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            exclude_cols = [target_col, 'future_return_magnitude', 'open', 'high', 'low', 'close', 'volume']
            
            for col in numeric_cols:
                if col in exclude_cols:
                    continue
                
                # 检查数据质量
                col_data = data[col]
                
                # 1. 缺失值检查
                missing_rate = col_data.isnull().sum() / len(col_data)
                if missing_rate > 0.3:  # 缺失率超过30%
                    continue
                
                # 2. 方差检查
                if col_data.var() == 0 or np.isnan(col_data.var()):
                    continue
                
                # 3. 异常值检查
                q99 = col_data.quantile(0.99)
                q01 = col_data.quantile(0.01)
                if np.isinf(q99) or np.isinf(q01):
                    continue
                
                # 4. 数值范围检查
                if col_data.abs().max() > 1e6:  # 过大的数值可能是错误数据
                    continue
                
                valid_features.append(col)
            
            self.logger.info(f"基础过滤后保留{len(valid_features)}个特征")
            return valid_features
            
        except Exception as e:
            self.logger.error(f"基础特征过滤失败: {e}")
            return []
    
    def _calculate_information_gain(self, data: pd.DataFrame, 
                                   features: List[str], target_col: str) -> Dict[str, float]:
        """计算信息增益"""
        try:
            info_gains = {}
            target = data[target_col]
            
            for feature in features:
                feature_data = data[feature].fillna(0)
                
                # 计算互信息
                try:
                    # 离散化连续特征
                    if len(np.unique(feature_data)) > 20:
                        feature_bins = pd.qcut(feature_data.rank(method='first'), 
                                             q=10, duplicates='drop')
                    else:
                        feature_bins = feature_data
                    
                    # 计算条件熵
                    info_gain = self._mutual_information(feature_bins, target)
                    info_gains[feature] = info_gain
                    
                except Exception as e:
                    info_gains[feature] = 0.0
                    
            return info_gains
            
        except Exception as e:
            self.logger.error(f"信息增益计算失败: {e}")
            return {}
    
    def _mutual_information(self, X, y):
        """计算互信息"""
        try:
            # 创建联合分布表
            joint_prob = pd.crosstab(X, y, normalize=True)
            marginal_x = joint_prob.sum(axis=1)
            marginal_y = joint_prob.sum(axis=0)
            
            mi = 0.0
            for i in joint_prob.index:
                for j in joint_prob.columns:
                    if joint_prob.loc[i, j] > 0:
                        mi += joint_prob.loc[i, j] * np.log2(
                            joint_prob.loc[i, j] / (marginal_x[i] * marginal_y[j])
                        )
            
            return mi
            
        except:
            return 0.0
    
    def _evaluate_prediction_power(self, data: pd.DataFrame, 
                                  features: List[str], target_col: str) -> Dict[str, float]:
        """评估单特征预测能力"""
        try:
            prediction_scores = {}
            target = data[target_col]
            
            for feature in features:
                feature_data = data[feature].fillna(0)
                if not feature_data.empty:
                    feature_data = feature_data.fillna(feature_data.median())
                
                try:
                    # 计算特征与目标的相关性
                    correlation = abs(np.corrcoef(feature_data, target)[0, 1])
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    # 计算单变量预测准确率
                    feature_median = feature_data.median()
                    predictions = (feature_data > feature_median).astype(int)
                    accuracy = (predictions == target).mean()
                    
                    # 组合得分
                    prediction_scores[feature] = correlation * 0.6 + (abs(accuracy - 0.5) * 2) * 0.4
                    
                except Exception:
                    prediction_scores[feature] = 0.0
            
            return prediction_scores
            
        except Exception as e:
            self.logger.error(f"预测能力评估失败: {e}")
            return {}
    
    def _assess_feature_stability(self, data: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """评估特征稳定性"""
        try:
            stability_scores = {}
            
            for feature in features:
                feature_data = data[feature].fillna(0)
                
                # 计算时序稳定性
                if len(feature_data) > 20:
                    # 分段计算特征统计量的稳定性
                    segment_size = len(feature_data) // 4
                    segments = [
                        feature_data[i*segment_size:(i+1)*segment_size] 
                        for i in range(4)
                    ]
                    
                    means = [seg.mean() for seg in segments]
                    stds = [seg.std() for seg in segments]
                    
                    # 稳定性得分（方差越小越稳定）
                    mean_stability = 1 / (1 + np.std(means)) if np.std(means) > 0 else 1.0
                    std_stability = 1 / (1 + np.std(stds)) if np.std(stds) > 0 else 1.0
                    
                    stability_scores[feature] = (mean_stability + std_stability) / 2
                else:
                    stability_scores[feature] = 0.5
            
            return stability_scores
            
        except Exception as e:
            self.logger.error(f"特征稳定性评估失败: {e}")
            return {}
    
    def _remove_redundant_features(self, data: pd.DataFrame, features: List[str]) -> List[str]:
        """去除高度相关的冗余特征"""
        try:
            if len(features) <= 10:
                return features
                
            correlation_matrix = data[features].corr().abs()
            
            # 找出高度相关的特征对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.85:  # 高相关阈值
                        high_corr_pairs.append((correlation_matrix.columns[i], 
                                              correlation_matrix.columns[j]))
            
            # 移除冗余特征
            features_to_remove = set()
            for feature1, feature2 in high_corr_pairs:
                # 优先保留信息增益更高的特征
                if feature1 in self.feature_scores and feature2 in self.feature_scores:
                    if self.feature_scores[feature1] < self.feature_scores[feature2]:
                        features_to_remove.add(feature1)
                    else:
                        features_to_remove.add(feature2)
                else:
                    features_to_remove.add(feature2)  # 默认移除第二个
            
            filtered_features = [f for f in features if f not in features_to_remove]
            self.logger.info(f"相关性过滤后保留{len(filtered_features)}个特征")
            
            return filtered_features
            
        except Exception as e:
            self.logger.error(f"冗余特征移除失败: {e}")
            return features
    
    def _rank_and_select_features(self, features: List[str], 
                                 info_gains: Dict[str, float],
                                 prediction_scores: Dict[str, float],
                                 stability_scores: Dict[str, float],
                                 max_features: int) -> List[str]:
        """综合评分并选择最优特征"""
        try:
            feature_rankings = []
            
            for feature in features:
                info_gain = info_gains.get(feature, 0.0)
                pred_score = prediction_scores.get(feature, 0.0)
                stability = stability_scores.get(feature, 0.0)
                
                # 综合得分 = 信息增益*0.4 + 预测能力*0.4 + 稳定性*0.2
                composite_score = info_gain * 0.4 + pred_score * 0.4 + stability * 0.2
                
                feature_rankings.append((feature, composite_score))
                self.feature_scores[feature] = composite_score
            
            # 按得分排序
            feature_rankings.sort(key=lambda x: x[1], reverse=True)
            
            # 选择top特征
            selected = [feature for feature, _ in feature_rankings[:max_features]]
            
            # 记录特征得分
            for feature, score in feature_rankings[:max_features]:
                self.logger.debug(f"特征 {feature}: 得分 {score:.4f}")
            
            return selected
            
        except Exception as e:
            self.logger.error(f"特征排序失败: {e}")
            return features[:max_features]
    
    def get_feature_importance_report(self) -> Dict:
        """获取特征重要性报告"""
        try:
            if not self.feature_scores:
                return {}
            
            sorted_features = sorted(self.feature_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return {
                'top_features': sorted_features[:20],
                'feature_count': len(self.selected_features),
                'average_score': np.mean(list(self.feature_scores.values())),
                'score_std': np.std(list(self.feature_scores.values()))
            }
            
        except Exception as e:
            self.logger.error(f"生成特征重要性报告失败: {e}")
            return {}