#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块走势预测系统
- 通过历史数据预测未来3-7天板块涨跌可能性
- 标记TOP5预期表现板块
- 评估算法历史性能
- 为个股遴选提供板块输入
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

from data.sector_fetcher import SectorFetcher
from data.enhanced_data_fetcher import EnhancedDataFetcher
from data.technical_calculator import TechnicalCalculator
from analysis.optimized_data_analyzer import OptimizedDataAnalyzer
from cache.manager import CacheManager
from config.settings import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SectorTrendPredictor:
    """板块走势预测器"""
    
    def __init__(self):
        """初始化预测器组件"""
        # 初始化基础组件
        self.settings = Settings()
        self.cache_manager = CacheManager(self.settings.cache_dir)
        self.sector_fetcher = SectorFetcher(self.cache_manager)
        self.enhanced_data_fetcher = EnhancedDataFetcher(self.cache_manager)
        self.technical_calculator = TechnicalCalculator()
        
        # 初始化优化分析器
        self.analyzer = OptimizedDataAnalyzer(
            sector_fetcher=self.sector_fetcher,
            enhanced_data_fetcher=self.enhanced_data_fetcher,
            technical_calculator=self.technical_calculator,
            logger=logger
        )
        
        self.logger = logger
    
    async def predict_sector_trends(self, 
                                  prediction_start_date: str = None,
                                  prediction_days: int = 5,
                                  training_months: int = 6) -> Dict:
        """
        预测板块未来走势
        
        Args:
            prediction_start_date: 预测起始日期 (YYYYMMDD), 默认为明天
            prediction_days: 预测天数 (3-7天)
            training_months: 训练数据月数
            
        Returns:
            板块预测结果和TOP5推荐
        """
        try:
            self.logger.info("🚀 开始板块走势预测")
            
            # 设置预测日期
            if prediction_start_date is None:
                prediction_start_date = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            
            self.logger.info(f"📊 预测参数: 起始日期={prediction_start_date}, 预测天数={prediction_days}, 训练月数={training_months}")
            
            # 获取板块数据
            sectors_data = await self._fetch_sector_data_for_prediction(training_months)
            if not sectors_data:
                return {'error': '无法获取板块数据'}
            
            # 执行板块预测
            sector_predictions = await self._analyze_all_sectors(sectors_data, prediction_days)
            
            # 计算预测结果
            prediction_results = self._calculate_sector_predictions(sector_predictions)
            
            # 标记TOP5板块
            top5_sectors = self._identify_top5_sectors(prediction_results)
            
            # 生成最终预测报告
            final_result = {
                'prediction_info': {
                    'prediction_start_date': prediction_start_date,
                    'prediction_end_date': self._calculate_end_date(prediction_start_date, prediction_days),
                    'prediction_days': prediction_days,
                    'training_months': training_months,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'sector_predictions': prediction_results,
                'top5_recommendations': top5_sectors,
                'prediction_summary': self._generate_prediction_summary(prediction_results),
                'performance_metrics': await self._calculate_prediction_confidence(sector_predictions)
            }
            
            # 保存预测结果
            await self._save_prediction_results(final_result)
            
            self.logger.info(f"✅ 板块预测完成，分析了{len(prediction_results)}个板块")
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 板块预测失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    async def evaluate_historical_performance(self,
                                            validation_start_date: str = "20250702",
                                            validation_end_date: str = "20250705",
                                            training_months: int = 6) -> Dict:
        """
        评估算法历史性能
        
        Args:
            validation_start_date: 验证开始日期
            validation_end_date: 验证结束日期
            training_months: 训练数据月数
            
        Returns:
            历史性能评估结果
        """
        try:
            self.logger.info(f"📈 开始历史性能评估: {validation_start_date} - {validation_end_date}")
            
            # 计算预测天数
            start_dt = datetime.strptime(validation_start_date, "%Y%m%d")
            end_dt = datetime.strptime(validation_end_date, "%Y%m%d")
            prediction_days = (end_dt - start_dt).days
            
            # 模拟历史预测
            historical_predictions = await self._simulate_historical_prediction(
                validation_start_date, prediction_days, training_months
            )
            
            # 获取实际数据
            actual_results = await self._get_actual_sector_performance(
                validation_start_date, validation_end_date
            )
            
            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics(
                historical_predictions, actual_results
            )
            
            # 生成评估报告
            evaluation_result = {
                'evaluation_period': {
                    'start_date': validation_start_date,
                    'end_date': validation_end_date,
                    'prediction_days': prediction_days,
                    'training_months': training_months
                },
                'historical_predictions': historical_predictions,
                'actual_results': actual_results,
                'performance_metrics': performance_metrics,
                'sector_accuracy': self._calculate_sector_accuracy(historical_predictions, actual_results),
                'top5_accuracy': self._calculate_top5_accuracy(historical_predictions, actual_results),
                'evaluation_summary': self._generate_evaluation_summary(performance_metrics)
            }
            
            self.logger.info(f"✅ 历史性能评估完成")
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"❌ 历史性能评估失败: {e}")
            return {'error': str(e)}
    
    async def _fetch_sector_data_for_prediction(self, months: int) -> Dict:
        """获取板块预测数据"""
        try:
            # 计算数据获取日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30 + 30)  # 多取30天缓冲
            
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            self.logger.info(f"📊 获取板块数据: {start_str} - {end_str}")
            
            # 获取所有板块数据
            sectors_data = await self.sector_fetcher.get_all_sectors_data((start_str, end_str))
            
            # 过滤高质量板块数据
            quality_sectors = {}
            for sector_name, data in sectors_data.items():
                if isinstance(data, pd.DataFrame) and len(data) >= 60:  # 至少60天数据
                    # 检查数据完整性
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        # 检查数据质量
                        if not data[required_cols].isnull().all().any():
                            quality_sectors[sector_name] = data
            
            self.logger.info(f"✅ 获取到{len(quality_sectors)}个高质量板块数据")
            return quality_sectors
            
        except Exception as e:
            self.logger.error(f"❌ 获取板块数据失败: {e}")
            return {}
    
    async def _analyze_all_sectors(self, sectors_data: Dict, prediction_days: int) -> Dict:
        """分析所有板块预测"""
        try:
            sector_results = {}
            
            for sector_name, data in sectors_data.items():
                try:
                    self.logger.info(f"🔍 分析板块: {sector_name}")
                    
                    # 应用技术分析
                    # 使用技术计算器增强数据特征
                    strength_score = self.technical_calculator.calculate_sector_strength_score(data, benchmark_data=data)
                    enhanced_data = data.copy()
                    enhanced_data['sector_strength'] = strength_score.get('overall_score', 0)
                    
                    # 生成特征
                    features = self._generate_sector_features(enhanced_data)
                    
                    # 预测板块走势
                    prediction_result = await self._predict_single_sector(
                        enhanced_data, features, prediction_days
                    )
                    
                    sector_results[sector_name] = {
                        'data_quality': self._assess_data_quality(enhanced_data),
                        'feature_count': len(features),
                        'prediction': prediction_result,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 板块 {sector_name} 分析失败: {e}")
                    continue
            
            return sector_results
            
        except Exception as e:
            self.logger.error(f"❌ 板块分析失败: {e}")
            return {}
    
    def _generate_sector_features(self, data: pd.DataFrame) -> List[str]:
        """生成板块特征"""
        try:
            features = []
            
            # 基础价格特征
            price_features = ['open', 'high', 'low', 'close', 'volume']
            features.extend([f for f in price_features if f in data.columns])
            
            # 技术指标特征
            technical_features = [
                'ma_5', 'ma_10', 'ma_20', 'ma_50',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower',
                'cci', 'stoch_k', 'stoch_d',
                'atr', 'adx', 'williams_r'
            ]
            features.extend([f for f in technical_features if f in data.columns])
            
            # 价格动量特征
            momentum_features = [
                'price_change', 'price_change_pct',
                'high_low_ratio', 'close_open_ratio'
            ]
            features.extend([f for f in momentum_features if f in data.columns])
            
            # 成交量特征
            volume_features = [
                'volume_ma', 'volume_ratio', 'obv'
            ]
            features.extend([f for f in volume_features if f in data.columns])
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ 特征生成失败: {e}")
            return []
    
    async def _predict_single_sector(self, data: pd.DataFrame, features: List[str], prediction_days: int) -> Dict:
        """预测单个板块走势"""
        try:
            if len(data) < 50 or len(features) < 10:
                return {
                    'trend_direction': 'neutral',
                    'confidence': 0.3,
                    'expected_return': 0.0,
                    'risk_level': 'high',
                    'prediction_strength': 'weak'
                }
            
            # 使用优化分析器进行预测
            analysis_result = await self.analyzer.analyze_prediction_accuracy_optimized(
                analysis_months=2,
                prediction_days=prediction_days
            )
            
            # 提取预测结果
            if 'effective_patterns' in analysis_result:
                patterns = analysis_result['effective_patterns'].get('enhanced_momentum', {}).get('patterns', [])
                
                # 寻找当前板块的预测结果
                current_sector_result = None
                for pattern in patterns:
                    if pattern.get('sector') == data.name or pattern.get('accuracy', 0) > 60:
                        current_sector_result = pattern
                        break
                
                if current_sector_result:
                    accuracy = current_sector_result.get('accuracy', 50)
                    confidence = current_sector_result.get('confidence', 0.5)
                    
                    # 基于准确率判断趋势方向
                    if accuracy > 70:
                        trend_direction = 'bullish'
                        expected_return = min(0.15, accuracy / 100 * 0.2)
                        prediction_strength = 'strong'
                    elif accuracy > 60:
                        trend_direction = 'bullish'
                        expected_return = min(0.08, accuracy / 100 * 0.15)
                        prediction_strength = 'moderate'
                    elif accuracy > 50:
                        trend_direction = 'neutral'
                        expected_return = 0.02
                        prediction_strength = 'weak'
                    else:
                        trend_direction = 'bearish'
                        expected_return = -min(0.05, (50 - accuracy) / 100 * 0.1)
                        prediction_strength = 'moderate'
                    
                    # 计算风险等级
                    if confidence > 0.7:
                        risk_level = 'low'
                    elif confidence > 0.5:
                        risk_level = 'medium'
                    else:
                        risk_level = 'high'
                    
                    return {
                        'trend_direction': trend_direction,
                        'confidence': confidence,
                        'expected_return': expected_return,
                        'risk_level': risk_level,
                        'prediction_strength': prediction_strength,
                        'accuracy': accuracy,
                        'sample_size': current_sector_result.get('sample_size', 0)
                    }
            
            # 默认预测结果
            return {
                'trend_direction': 'neutral',
                'confidence': 0.4,
                'expected_return': 0.0,
                'risk_level': 'medium',
                'prediction_strength': 'weak'
            }
            
        except Exception as e:
            self.logger.error(f"❌ 单板块预测失败: {e}")
            return {
                'trend_direction': 'neutral',
                'confidence': 0.3,
                'expected_return': 0.0,
                'risk_level': 'high',
                'prediction_strength': 'weak'
            }
    
    def _calculate_sector_predictions(self, sector_analyses: Dict) -> List[Dict]:
        """计算板块预测结果"""
        try:
            predictions = []
            
            for sector_name, analysis in sector_analyses.items():
                prediction = analysis.get('prediction', {})
                
                sector_prediction = {
                    'sector_name': sector_name,
                    'trend_direction': prediction.get('trend_direction', 'neutral'),
                    'confidence': round(prediction.get('confidence', 0.5), 3),
                    'expected_return': round(prediction.get('expected_return', 0.0), 4),
                    'risk_level': prediction.get('risk_level', 'medium'),
                    'prediction_strength': prediction.get('prediction_strength', 'weak'),
                    'data_quality': round(analysis.get('data_quality', 0.5), 3),
                    'feature_count': analysis.get('feature_count', 0),
                    'accuracy': round(prediction.get('accuracy', 50), 1),
                    'sample_size': prediction.get('sample_size', 0),
                    'composite_score': self._calculate_composite_score(prediction)
                }
                
                predictions.append(sector_prediction)
            
            # 按综合评分排序
            predictions.sort(key=lambda x: x['composite_score'], reverse=True)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ 计算板块预测失败: {e}")
            return []
    
    def _calculate_composite_score(self, prediction: Dict) -> float:
        """计算综合评分"""
        try:
            confidence = prediction.get('confidence', 0.5)
            expected_return = abs(prediction.get('expected_return', 0.0))
            accuracy = prediction.get('accuracy', 50) / 100
            
            # 风险调整
            risk_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.8}
            risk_adj = risk_multiplier.get(prediction.get('risk_level', 'medium'), 1.0)
            
            # 强度调整
            strength_multiplier = {'strong': 1.3, 'moderate': 1.1, 'weak': 0.9}
            strength_adj = strength_multiplier.get(prediction.get('prediction_strength', 'weak'), 1.0)
            
            # 综合评分
            composite_score = (0.4 * confidence + 0.3 * accuracy + 0.3 * expected_return) * risk_adj * strength_adj
            
            return round(composite_score, 4)
            
        except Exception as e:
            self.logger.error(f"❌ 综合评分计算失败: {e}")
            return 0.5
    
    def _identify_top5_sectors(self, predictions: List[Dict]) -> List[Dict]:
        """标记TOP5板块"""
        try:
            # 筛选看涨板块
            bullish_sectors = [p for p in predictions if p['trend_direction'] == 'bullish']
            
            # 按综合评分排序
            bullish_sectors.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # 选择TOP5
            top5 = bullish_sectors[:5]
            
            # 添加排名信息
            for i, sector in enumerate(top5):
                sector['rank'] = i + 1
                sector['top5_recommended'] = True
                sector['recommendation_reason'] = self._generate_recommendation_reason(sector)
            
            return top5
            
        except Exception as e:
            self.logger.error(f"❌ TOP5识别失败: {e}")
            return []
    
    def _generate_recommendation_reason(self, sector: Dict) -> str:
        """生成推荐理由"""
        try:
            reasons = []
            
            # 置信度理由
            confidence = sector['confidence']
            if confidence > 0.7:
                reasons.append("高置信度预测")
            elif confidence > 0.6:
                reasons.append("较高置信度")
            
            # 预期收益理由
            expected_return = sector['expected_return']
            if expected_return > 0.1:
                reasons.append("高预期收益")
            elif expected_return > 0.05:
                reasons.append("良好预期收益")
            
            # 风险理由
            if sector['risk_level'] == 'low':
                reasons.append("低风险")
            elif sector['risk_level'] == 'medium':
                reasons.append("适中风险")
            
            # 强度理由
            if sector['prediction_strength'] == 'strong':
                reasons.append("强预测信号")
            elif sector['prediction_strength'] == 'moderate':
                reasons.append("中等预测信号")
            
            return "、".join(reasons) if reasons else "综合评分较高"
            
        except Exception as e:
            self.logger.error(f"❌ 推荐理由生成失败: {e}")
            return "综合评分较高"
    
    def _generate_prediction_summary(self, predictions: List[Dict]) -> Dict:
        """生成预测摘要"""
        try:
            if not predictions:
                return {}
            
            total_sectors = len(predictions)
            bullish_count = len([p for p in predictions if p['trend_direction'] == 'bullish'])
            bearish_count = len([p for p in predictions if p['trend_direction'] == 'bearish'])
            neutral_count = total_sectors - bullish_count - bearish_count
            
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            avg_expected_return = np.mean([p['expected_return'] for p in predictions])
            
            high_confidence_count = len([p for p in predictions if p['confidence'] > 0.7])
            strong_predictions = len([p for p in predictions if p['prediction_strength'] == 'strong'])
            
            return {
                'total_sectors_analyzed': total_sectors,
                'bullish_sectors': bullish_count,
                'bearish_sectors': bearish_count,
                'neutral_sectors': neutral_count,
                'bullish_percentage': round(bullish_count / total_sectors * 100, 1),
                'average_confidence': round(avg_confidence, 3),
                'average_expected_return': round(avg_expected_return, 4),
                'high_confidence_sectors': high_confidence_count,
                'strong_prediction_sectors': strong_predictions,
                'market_sentiment': self._determine_market_sentiment(bullish_count, bearish_count, total_sectors)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 预测摘要生成失败: {e}")
            return {}
    
    def _determine_market_sentiment(self, bullish: int, bearish: int, total: int) -> str:
        """判断市场情绪"""
        bullish_pct = bullish / total if total > 0 else 0
        bearish_pct = bearish / total if total > 0 else 0
        
        if bullish_pct > 0.6:
            return "强烈看涨"
        elif bullish_pct > 0.4:
            return "温和看涨"
        elif bearish_pct > 0.6:
            return "强烈看跌"
        elif bearish_pct > 0.4:
            return "温和看跌"
        else:
            return "中性"
    
    async def _calculate_prediction_confidence(self, sector_analyses: Dict) -> Dict:
        """计算预测置信度指标"""
        try:
            if not sector_analyses:
                return {}
            
            confidences = []
            accuracies = []
            
            for analysis in sector_analyses.values():
                prediction = analysis.get('prediction', {})
                confidence = prediction.get('confidence', 0.5)
                accuracy = prediction.get('accuracy', 50)
                
                confidences.append(confidence)
                accuracies.append(accuracy)
            
            return {
                'overall_confidence': round(np.mean(confidences), 3),
                'confidence_std': round(np.std(confidences), 3),
                'overall_accuracy': round(np.mean(accuracies), 1),
                'accuracy_std': round(np.std(accuracies), 1),
                'high_confidence_ratio': round(len([c for c in confidences if c > 0.7]) / len(confidences), 3),
                'high_accuracy_ratio': round(len([a for a in accuracies if a > 70]) / len(accuracies), 3)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 置信度计算失败: {e}")
            return {}
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """评估数据质量"""
        try:
            if data.empty:
                return 0.0
            
            # 数据完整性
            completeness = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
            
            # 数据长度
            length_score = min(1.0, len(data) / 100)
            
            # 价格数据连续性
            price_continuity = 1.0
            if 'close' in data.columns:
                price_changes = data['close'].pct_change().abs()
                extreme_changes = (price_changes > 0.2).sum()
                price_continuity = max(0.5, 1 - extreme_changes / len(data))
            
            # 综合评分
            quality_score = 0.4 * completeness + 0.3 * length_score + 0.3 * price_continuity
            
            return round(quality_score, 3)
            
        except Exception as e:
            self.logger.error(f"❌ 数据质量评估失败: {e}")
            return 0.5
    
    def _calculate_end_date(self, start_date: str, days: int) -> str:
        """计算结束日期"""
        try:
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = start_dt + timedelta(days=days-1)
            return end_dt.strftime("%Y%m%d")
        except:
            return start_date
    
    async def _save_prediction_results(self, results: Dict) -> None:
        """保存预测结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sector_predictions_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 预测结果已保存: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存预测结果失败: {e}")
    
    # ============= 历史性能评估方法 =============
    
    async def _simulate_historical_prediction(self, target_date: str, prediction_days: int, training_months: int) -> Dict:
        """模拟历史预测"""
        try:
            self.logger.info(f"🔄 模拟历史预测: {target_date}")
            
            # 设置历史数据范围（在目标日期之前）
            target_dt = datetime.strptime(target_date, "%Y%m%d")
            train_end_dt = target_dt - timedelta(days=1)
            train_start_dt = train_end_dt - timedelta(days=training_months * 30)
            
            # 获取历史训练数据
            historical_data = await self.sector_fetcher.get_all_sectors_data((
                train_start_dt.strftime("%Y%m%d"),
                train_end_dt.strftime("%Y%m%d")
            ))
            
            # 使用历史数据进行预测
            historical_analyses = await self._analyze_all_sectors(historical_data, prediction_days)
            
            # 生成历史预测结果
            historical_predictions = self._calculate_sector_predictions(historical_analyses)
            
            return {
                'prediction_date': target_date,
                'training_period': f"{train_start_dt.strftime('%Y%m%d')} - {train_end_dt.strftime('%Y%m%d')}",
                'predictions': historical_predictions
            }
            
        except Exception as e:
            self.logger.error(f"❌ 历史预测模拟失败: {e}")
            return {}
    
    async def _get_actual_sector_performance(self, start_date: str, end_date: str) -> Dict:
        """获取实际板块表现"""
        try:
            self.logger.info(f"📊 获取实际表现: {start_date} - {end_date}")
            
            # 获取实际数据
            actual_data = await self.sector_fetcher.get_all_sectors_data((start_date, end_date))
            
            actual_performance = {}
            for sector_name, data in actual_data.items():
                if isinstance(data, pd.DataFrame) and len(data) >= 2:
                    start_price = data.iloc[0]['close']
                    end_price = data.iloc[-1]['close']
                    actual_return = (end_price - start_price) / start_price
                    
                    actual_performance[sector_name] = {
                        'actual_return': round(actual_return, 4),
                        'actual_direction': 'bullish' if actual_return > 0.01 else ('bearish' if actual_return < -0.01 else 'neutral'),
                        'start_price': round(start_price, 2),
                        'end_price': round(end_price, 2),
                        'price_change': round(end_price - start_price, 2)
                    }
            
            return actual_performance
            
        except Exception as e:
            self.logger.error(f"❌ 获取实际表现失败: {e}")
            return {}
    
    def _calculate_performance_metrics(self, predictions: Dict, actuals: Dict) -> Dict:
        """计算性能指标"""
        try:
            if not predictions.get('predictions') or not actuals:
                return {}
            
            predicted_sectors = {p['sector_name']: p for p in predictions['predictions']}
            
            correct_direction = 0
            total_predictions = 0
            return_errors = []
            
            for sector_name, actual in actuals.items():
                if sector_name in predicted_sectors:
                    pred = predicted_sectors[sector_name]
                    
                    # 方向准确性
                    if pred['trend_direction'] == actual['actual_direction']:
                        correct_direction += 1
                    total_predictions += 1
                    
                    # 收益率误差
                    pred_return = pred['expected_return']
                    actual_return = actual['actual_return']
                    return_errors.append(abs(pred_return - actual_return))
            
            direction_accuracy = correct_direction / total_predictions if total_predictions > 0 else 0
            mean_return_error = np.mean(return_errors) if return_errors else 0
            
            return {
                'direction_accuracy': round(direction_accuracy, 3),
                'direction_accuracy_percentage': round(direction_accuracy * 100, 1),
                'total_predictions': total_predictions,
                'correct_predictions': correct_direction,
                'mean_return_error': round(mean_return_error, 4),
                'return_error_std': round(np.std(return_errors), 4) if return_errors else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ 性能指标计算失败: {e}")
            return {}
    
    def _calculate_sector_accuracy(self, predictions: Dict, actuals: Dict) -> List[Dict]:
        """计算各板块准确性"""
        try:
            if not predictions.get('predictions') or not actuals:
                return []
            
            predicted_sectors = {p['sector_name']: p for p in predictions['predictions']}
            sector_accuracies = []
            
            for sector_name, actual in actuals.items():
                if sector_name in predicted_sectors:
                    pred = predicted_sectors[sector_name]
                    
                    direction_correct = pred['trend_direction'] == actual['actual_direction']
                    return_error = abs(pred['expected_return'] - actual['actual_return'])
                    
                    sector_accuracies.append({
                        'sector_name': sector_name,
                        'predicted_direction': pred['trend_direction'],
                        'actual_direction': actual['actual_direction'],
                        'direction_correct': direction_correct,
                        'predicted_return': pred['expected_return'],
                        'actual_return': actual['actual_return'],
                        'return_error': round(return_error, 4),
                        'prediction_confidence': pred['confidence']
                    })
            
            return sector_accuracies
            
        except Exception as e:
            self.logger.error(f"❌ 板块准确性计算失败: {e}")
            return []
    
    def _calculate_top5_accuracy(self, predictions: Dict, actuals: Dict) -> Dict:
        """计算TOP5准确性"""
        try:
            if not predictions.get('predictions') or not actuals:
                return {}
            
            # 获取预测的TOP5
            predicted_sectors = predictions['predictions']
            bullish_predictions = [p for p in predicted_sectors if p['trend_direction'] == 'bullish']
            bullish_predictions.sort(key=lambda x: x['composite_score'], reverse=True)
            top5_predicted = bullish_predictions[:5]
            
            # 获取实际的TOP5表现
            actual_returns = [(name, data['actual_return']) for name, data in actuals.items()]
            actual_returns.sort(key=lambda x: x[1], reverse=True)
            top5_actual = [name for name, _ in actual_returns[:5]]
            
            # 计算重叠度
            top5_predicted_names = [p['sector_name'] for p in top5_predicted]
            overlap = len(set(top5_predicted_names) & set(top5_actual))
            
            return {
                'top5_predicted': top5_predicted_names,
                'top5_actual': top5_actual,
                'overlap_count': overlap,
                'overlap_percentage': round(overlap / 5 * 100, 1),
                'prediction_quality': 'excellent' if overlap >= 4 else ('good' if overlap >= 3 else ('fair' if overlap >= 2 else 'poor'))
            }
            
        except Exception as e:
            self.logger.error(f"❌ TOP5准确性计算失败: {e}")
            return {}
    
    def _generate_evaluation_summary(self, metrics: Dict) -> str:
        """生成评估摘要"""
        try:
            if not metrics:
                return "评估数据不足"
            
            direction_acc = metrics.get('direction_accuracy_percentage', 0)
            return_error = metrics.get('mean_return_error', 0)
            
            summary_parts = []
            
            # 方向准确性评价
            if direction_acc >= 70:
                summary_parts.append(f"方向预测准确率{direction_acc}%，表现优秀")
            elif direction_acc >= 60:
                summary_parts.append(f"方向预测准确率{direction_acc}%，表现良好")
            elif direction_acc >= 50:
                summary_parts.append(f"方向预测准确率{direction_acc}%，表现一般")
            else:
                summary_parts.append(f"方向预测准确率{direction_acc}%，需要改进")
            
            # 收益率误差评价
            if return_error <= 0.03:
                summary_parts.append("收益率预测精度较高")
            elif return_error <= 0.05:
                summary_parts.append("收益率预测精度中等")
            else:
                summary_parts.append("收益率预测精度有待提升")
            
            return "；".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"❌ 评估摘要生成失败: {e}")
            return "评估摘要生成失败"

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='板块走势预测系统')
    parser.add_argument('--mode', choices=['predict', 'evaluate'], default='predict', help='运行模式')
    parser.add_argument('--start-date', type=str, help='预测起始日期 (YYYYMMDD)')
    parser.add_argument('--days', type=int, default=5, help='预测天数 (3-7)')
    parser.add_argument('--training-months', type=int, default=6, help='训练数据月数')
    parser.add_argument('--validation-start', type=str, default='20250702', help='验证开始日期')
    parser.add_argument('--validation-end', type=str, default='20250705', help='验证结束日期')
    
    args = parser.parse_args()
    
    predictor = SectorTrendPredictor()
    
    if args.mode == 'predict':
        print("🚀 开始板块走势预测...")
        result = await predictor.predict_sector_trends(
            prediction_start_date=args.start_date,
            prediction_days=args.days,
            training_months=args.training_months
        )
        
        if 'error' not in result:
            print(f"\n📊 预测完成！分析了{len(result['sector_predictions'])}个板块")
            print(f"🏆 TOP5推荐板块:")
            for i, sector in enumerate(result['top5_recommendations'], 1):
                print(f"  {i}. {sector['sector_name']} - 预期收益: {sector['expected_return']:.2%}, 置信度: {sector['confidence']:.1%}")
        else:
            print(f"❌ 预测失败: {result['error']}")
    
    elif args.mode == 'evaluate':
        print("📈 开始历史性能评估...")
        result = await predictor.evaluate_historical_performance(
            validation_start_date=args.validation_start,
            validation_end_date=args.validation_end,
            training_months=args.training_months
        )
        
        if 'error' not in result and 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"\n📊 历史性能评估结果:")
            print(f"  方向预测准确率: {metrics.get('direction_accuracy_percentage', 0)}%")
            print(f"  收益率预测误差: {metrics.get('mean_return_error', 0):.2%}")
            print(f"  评估摘要: {result.get('evaluation_summary', '无')}")
        else:
            print(f"❌ 评估失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    asyncio.run(main())