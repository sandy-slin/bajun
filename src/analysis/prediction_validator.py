# -*- coding: utf-8 -*-
"""
预测验证模块
用于验证增强预测系统的准确性，提供预测效果评估和改进建议
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
from analysis.enhanced_predictor import EnhancedPredictor
from data.technical_calculator import TechnicalCalculator


class PredictionValidator:
    """预测验证器 - 评估增强预测系统的准确性"""
    
    def __init__(self, sector_fetcher: SectorFetcher, enhanced_data_fetcher: EnhancedDataFetcher,
                 enhanced_predictor: EnhancedPredictor, tech_calculator: TechnicalCalculator):
        self.sector_fetcher = sector_fetcher
        self.enhanced_data_fetcher = enhanced_data_fetcher
        self.enhanced_predictor = enhanced_predictor
        self.tech_calculator = tech_calculator
        self.logger = logging.getLogger(__name__)
        
    async def validate_predictions(self, validation_periods: int = 10, 
                                  prediction_days: int = 3) -> Dict[str, Any]:
        """
        验证预测准确性
        
        Args:
            validation_periods: 验证期数(每期间隔prediction_days)
            prediction_days: 预测天数
            
        Returns:
            Dict: 验证结果和统计数据
        """
        try:
            self.logger.info(f"开始预测验证，验证期数: {validation_periods}, 预测天数: {prediction_days}")
            
            # 计算验证时间范围
            end_date = datetime.now() - timedelta(days=prediction_days)  # 确保有足够的验证数据
            start_date = end_date - timedelta(days=validation_periods * prediction_days + 30)  # 额外30天用于计算基础数据
            
            # 获取所有板块数据
            date_range = (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            sectors_data = await self.sector_fetcher.get_all_sectors_data(date_range)
            
            if not sectors_data:
                raise ValueError("无法获取验证数据")
                
            # 生成验证时间点
            validation_dates = self._generate_validation_dates(start_date, end_date, validation_periods, prediction_days)
            
            # 执行预测验证
            validation_results = []
            for i, (pred_date, verify_date) in enumerate(validation_dates):
                self.logger.info(f"验证进度: {i+1}/{len(validation_dates)} - 预测日期: {pred_date.strftime('%Y-%m-%d')}")
                
                period_result = await self._validate_single_period(
                    sectors_data, pred_date, verify_date, prediction_days
                )
                validation_results.append(period_result)
                
            # 计算总体统计
            overall_stats = self._calculate_overall_statistics(validation_results)
            
            # 分析预测因子效果
            factor_analysis = self._analyze_prediction_factors(validation_results)
            
            # 生成优化建议
            optimization_suggestions = self._generate_optimization_suggestions(overall_stats, factor_analysis)
            
            validation_summary = {
                'validation_time': datetime.now().isoformat(),
                'validation_periods': validation_periods,
                'prediction_days': prediction_days,
                'total_predictions': len(validation_results) * len(sectors_data),
                'overall_statistics': overall_stats,
                'factor_analysis': factor_analysis,
                'optimization_suggestions': optimization_suggestions,
                'detailed_results': validation_results[:5]  # 只保存前5期的详细结果
            }
            
            self.logger.info(f"预测验证完成，总体准确率: {overall_stats.get('accuracy', 0):.1%}")
            return validation_summary
            
        except Exception as e:
            self.logger.error(f"预测验证失败: {e}")
            return {'error': str(e)}
            
    def _generate_validation_dates(self, start_date: datetime, end_date: datetime, 
                                 periods: int, prediction_days: int) -> List[Tuple[datetime, datetime]]:
        """生成验证日期对"""
        dates = []
        current_date = start_date + timedelta(days=30)  # 跳过初始30天
        
        for _ in range(periods):
            pred_date = current_date
            verify_date = current_date + timedelta(days=prediction_days)
            
            if verify_date <= end_date:
                dates.append((pred_date, verify_date))
                current_date += timedelta(days=prediction_days)
            else:
                break
                
        return dates
        
    async def _validate_single_period(self, sectors_data: Dict[str, pd.DataFrame], 
                                    pred_date: datetime, verify_date: datetime, 
                                    prediction_days: int) -> Dict[str, Any]:
        """验证单个时期的预测"""
        try:
            period_results = []
            
            for sector_name, full_data in sectors_data.items():
                try:
                    # 转换日期索引为字符串进行比较
                    full_data_copy = full_data.copy()
                    full_data_copy['date_str'] = full_data_copy.index.astype(str)
                    pred_date_str = pred_date.strftime('%Y-%m-%d')
                    verify_date_str = verify_date.strftime('%Y-%m-%d')
                    
                    # 截取预测日期之前的数据用于预测
                    pred_mask = full_data_copy['date_str'] <= pred_date_str
                    pred_data = full_data_copy[pred_mask].drop('date_str', axis=1)
                    if len(pred_data) < 20:  # 数据不足
                        continue
                        
                    # 获取验证日期的实际数据
                    verify_mask = (full_data_copy['date_str'] > pred_date_str) & \
                                 (full_data_copy['date_str'] <= verify_date_str)
                    verify_data = full_data_copy[verify_mask].drop('date_str', axis=1)
                    if verify_data.empty:
                        continue
                        
                    # 进行预测
                    date_range = (
                        (pred_date - timedelta(days=30)).strftime("%Y%m%d"),
                        pred_date.strftime("%Y%m%d")
                    )
                    
                    prediction_result = await self.enhanced_predictor.calculate_enhanced_sector_score(
                        sector_name, pred_data, date_range
                    )
                    
                    # 计算实际收益率
                    start_price = pred_data['close'].iloc[-1]
                    end_price = verify_data['close'].iloc[-1]
                    actual_return = (end_price / start_price - 1) * 100
                    
                    # 提取预测信息
                    prediction = prediction_result.get('prediction', {})
                    predicted_return = prediction.get('target_return', 0)
                    predicted_trend = prediction.get('trend_prediction', '震荡')
                    prediction_probability = prediction.get('probability', 50)
                    
                    # 评估预测准确性
                    accuracy_metrics = self._evaluate_prediction_accuracy(
                        predicted_return, actual_return, predicted_trend, prediction_probability
                    )
                    
                    sector_result = {
                        'sector_name': sector_name,
                        'pred_date': pred_date.strftime('%Y-%m-%d'),
                        'verify_date': verify_date.strftime('%Y-%m-%d'),
                        'predicted_return': predicted_return,
                        'actual_return': actual_return,
                        'predicted_trend': predicted_trend,
                        'prediction_probability': prediction_probability,
                        'comprehensive_score': prediction_result.get('comprehensive_score', 50),
                        'accuracy_metrics': accuracy_metrics,
                        'enhanced_scores': prediction_result.get('enhanced_scores', {}),
                        'prediction_error': abs(predicted_return - actual_return)
                    }
                    
                    period_results.append(sector_result)
                    
                except Exception as e:
                    self.logger.warning(f"验证{sector_name}时出错: {e}")
                    continue
                    
            # 计算本期统计数据
            period_stats = self._calculate_period_statistics(period_results)
            
            return {
                'period_date': pred_date.strftime('%Y-%m-%d'),
                'total_predictions': len(period_results),
                'period_statistics': period_stats,
                'sector_results': period_results
            }
            
        except Exception as e:
            self.logger.error(f"单期验证失败: {e}")
            return {'error': str(e)}
            
    def _evaluate_prediction_accuracy(self, predicted_return: float, actual_return: float,
                                    predicted_trend: str, prediction_probability: float) -> Dict[str, Any]:
        """评估单个预测的准确性"""
        # 1. 收益率误差
        return_error = abs(predicted_return - actual_return)
        return_accuracy = max(0, 1 - return_error / max(abs(actual_return), 1)) * 100
        
        # 2. 趋势方向准确性
        predicted_direction = 1 if predicted_return > 0 else -1 if predicted_return < 0 else 0
        actual_direction = 1 if actual_return > 0 else -1 if actual_return < 0 else 0
        direction_correct = predicted_direction == actual_direction
        
        # 3. 趋势强度匹配
        trend_mapping = {'强势上涨': 2, '上涨': 1, '震荡上涨': 0.5, '震荡': 0, '震荡下跌': -0.5, '下跌': -1}
        predicted_strength = trend_mapping.get(predicted_trend, 0)
        
        if abs(actual_return) >= 8:
            actual_strength = 2 if actual_return > 0 else -2
        elif abs(actual_return) >= 4:
            actual_strength = 1 if actual_return > 0 else -1
        elif abs(actual_return) >= 1:
            actual_strength = 0.5 if actual_return > 0 else -0.5
        else:
            actual_strength = 0
            
        strength_accuracy = max(0, 1 - abs(predicted_strength - actual_strength) / 4) * 100
        
        # 4. 概率校准(基于实际结果是否符合预测趋势)
        probability_calibration = prediction_probability if direction_correct else (100 - prediction_probability)
        
        return {
            'return_accuracy': round(return_accuracy, 2),
            'direction_correct': direction_correct,
            'strength_accuracy': round(strength_accuracy, 2),
            'probability_calibration': round(probability_calibration, 2),
            'return_error': round(return_error, 2)
        }
        
    def _calculate_period_statistics(self, period_results: List[Dict]) -> Dict[str, float]:
        """计算单期统计数据"""
        if not period_results:
            return {}
            
        return_accuracies = [r['accuracy_metrics']['return_accuracy'] for r in period_results]
        direction_correct = [r['accuracy_metrics']['direction_correct'] for r in period_results]
        strength_accuracies = [r['accuracy_metrics']['strength_accuracy'] for r in period_results]
        return_errors = [r['accuracy_metrics']['return_error'] for r in period_results]
        
        return {
            'avg_return_accuracy': round(np.mean(return_accuracies), 2),
            'direction_accuracy': round(np.mean(direction_correct) * 100, 2),
            'avg_strength_accuracy': round(np.mean(strength_accuracies), 2),
            'avg_return_error': round(np.mean(return_errors), 2),
            'median_return_error': round(np.median(return_errors), 2)
        }
        
    def _calculate_overall_statistics(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """计算总体统计数据"""
        try:
            all_period_stats = [r.get('period_statistics', {}) for r in validation_results if 'period_statistics' in r]
            
            if not all_period_stats:
                return {}
                
            # 提取各项指标
            return_accuracies = [s.get('avg_return_accuracy', 0) for s in all_period_stats]
            direction_accuracies = [s.get('direction_accuracy', 0) for s in all_period_stats]
            strength_accuracies = [s.get('avg_strength_accuracy', 0) for s in all_period_stats]
            return_errors = [s.get('avg_return_error', 0) for s in all_period_stats]
            
            # 计算总体指标
            overall_stats = {
                'accuracy': round(np.mean(direction_accuracies) / 100, 3),
                'avg_return_accuracy': round(np.mean(return_accuracies), 2),
                'avg_direction_accuracy': round(np.mean(direction_accuracies), 2),
                'avg_strength_accuracy': round(np.mean(strength_accuracies), 2),
                'avg_return_error': round(np.mean(return_errors), 2),
                'stability': round(1 - np.std(direction_accuracies) / 100, 3),  # 稳定性
                'total_periods': len(validation_results),
                'successful_periods': len(all_period_stats)
            }
            
            # 分级评价
            if overall_stats['accuracy'] >= 0.7:
                overall_stats['grade'] = '优秀'
            elif overall_stats['accuracy'] >= 0.6:
                overall_stats['grade'] = '良好'
            elif overall_stats['accuracy'] >= 0.5:
                overall_stats['grade'] = '一般'
            else:
                overall_stats['grade'] = '较差'
                
            return overall_stats
            
        except Exception as e:
            self.logger.error(f"计算总体统计失败: {e}")
            return {}
            
    def _analyze_prediction_factors(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """分析各预测因子的效果"""
        try:
            # 收集所有预测结果
            all_results = []
            for period_result in validation_results:
                if 'sector_results' in period_result:
                    all_results.extend(period_result['sector_results'])
                    
            if not all_results:
                return {}
                
            # 按因子分组分析
            factor_performance = {}
            
            # 分析技术面因子
            tech_scores = [r.get('enhanced_scores', {}).get('northbound_score', 50) for r in all_results]
            tech_accuracies = [r['accuracy_metrics']['direction_correct'] for r in all_results]
            
            # 按技术面评分分组
            high_tech = [acc for score, acc in zip(tech_scores, tech_accuracies) if score >= 70]
            low_tech = [acc for score, acc in zip(tech_scores, tech_accuracies) if score < 50]
            
            factor_performance['technical_factor'] = {
                'high_score_accuracy': round(np.mean(high_tech) * 100, 1) if high_tech else 0,
                'low_score_accuracy': round(np.mean(low_tech) * 100, 1) if low_tech else 0,
                'effectiveness': 'high' if len(high_tech) > 0 and np.mean(high_tech) > np.mean(low_tech) else 'low'
            }
            
            # 分析北向资金因子效果
            nb_scores = [r.get('enhanced_scores', {}).get('northbound_score', 50) for r in all_results]
            nb_accuracies = [r['accuracy_metrics']['direction_correct'] for r in all_results]
            
            high_nb = [acc for score, acc in zip(nb_scores, nb_accuracies) if score >= 70]
            low_nb = [acc for score, acc in zip(nb_scores, nb_accuracies) if score < 50]
            
            factor_performance['northbound_factor'] = {
                'high_score_accuracy': round(np.mean(high_nb) * 100, 1) if high_nb else 0,
                'low_score_accuracy': round(np.mean(low_nb) * 100, 1) if low_nb else 0,
                'effectiveness': 'high' if len(high_nb) > 0 and np.mean(high_nb) > np.mean(low_nb) else 'low'
            }
            
            return factor_performance
            
        except Exception as e:
            self.logger.error(f"因子分析失败: {e}")
            return {}
            
    def _generate_optimization_suggestions(self, overall_stats: Dict, factor_analysis: Dict) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        try:
            accuracy = overall_stats.get('accuracy', 0)
            
            # 基于总体准确率的建议
            if accuracy < 0.5:
                suggestions.append("预测准确率较低，建议重新评估基础评分算法")
                suggestions.append("考虑增加更多历史数据进行模型训练")
            elif accuracy < 0.6:
                suggestions.append("预测准确率中等，建议优化权重配置")
            
            # 基于因子分析的建议
            for factor_name, factor_data in factor_analysis.items():
                effectiveness = factor_data.get('effectiveness', 'low')
                if effectiveness == 'low':
                    suggestions.append(f"因子 {factor_name} 效果不佳，建议调整或替换")
                    
            # 基于稳定性的建议
            stability = overall_stats.get('stability', 0)
            if stability < 0.7:
                suggestions.append("预测稳定性不足，建议增加数据平滑处理")
                
            # 基于误差的建议
            avg_error = overall_stats.get('avg_return_error', 0)
            if avg_error > 5:
                suggestions.append("预测误差较大，建议调整目标收益率计算方法")
                
            if not suggestions:
                suggestions.append("当前预测模型表现良好，建议保持现有配置")
                
        except Exception as e:
            self.logger.error(f"生成优化建议失败: {e}")
            suggestions.append("无法生成优化建议")
            
        return suggestions
        
    async def save_validation_report(self, validation_result: Dict, filename: str = None) -> str:
        """保存验证报告"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prediction_validation_{timestamp}.md"
                
            import os
            report_path = f"reports/validation/{filename}"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # 生成报告内容
            content = self._format_validation_report(validation_result)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.logger.info(f"验证报告已保存: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"保存验证报告失败: {e}")
            return ""
            
    def _format_validation_report(self, result: Dict) -> str:
        """格式化验证报告"""
        try:
            overall_stats = result.get('overall_statistics', {})
            factor_analysis = result.get('factor_analysis', {})
            suggestions = result.get('optimization_suggestions', [])
            
            content = f"""# 增强预测系统验证报告

## 验证概述
- **验证时间**: {result.get('validation_time', 'Unknown')}
- **验证期数**: {result.get('validation_periods', 0)}
- **预测天数**: {result.get('prediction_days', 3)}
- **总预测次数**: {result.get('total_predictions', 0)}

## 总体表现

### 准确性指标
- **总体准确率**: {overall_stats.get('accuracy', 0):.1%}
- **方向准确率**: {overall_stats.get('avg_direction_accuracy', 0):.1f}%
- **收益率准确率**: {overall_stats.get('avg_return_accuracy', 0):.1f}%
- **趋势强度准确率**: {overall_stats.get('avg_strength_accuracy', 0):.1f}%

### 误差指标
- **平均收益率误差**: {overall_stats.get('avg_return_error', 0):.2f}%
- **预测稳定性**: {overall_stats.get('stability', 0):.1%}
- **综合评级**: {overall_stats.get('grade', '未知')}

## 因子效果分析

"""
            
            for factor_name, factor_data in factor_analysis.items():
                content += f"""### {factor_name}
- **高分时准确率**: {factor_data.get('high_score_accuracy', 0):.1f}%
- **低分时准确率**: {factor_data.get('low_score_accuracy', 0):.1f}%
- **因子有效性**: {factor_data.get('effectiveness', 'unknown')}

"""
            
            content += f"""## 优化建议

"""
            for i, suggestion in enumerate(suggestions, 1):
                content += f"{i}. {suggestion}\n"
                
            content += f"""
## 结论

基于{result.get('validation_periods', 0)}期验证结果，增强预测系统总体准确率为{overall_stats.get('accuracy', 0):.1%}，评级为{overall_stats.get('grade', '未知')}。

建议根据以上分析结果调整模型参数，以提升预测准确性和稳定性。

---
*报告生成时间: {result.get('validation_time', 'Unknown')}*
*验证数据来源: 历史交易数据*
"""
            return content
            
        except Exception as e:
            self.logger.error(f"格式化验证报告失败: {e}")
            return f"报告生成失败: {str(e)}"