# -*- coding: utf-8 -*-
"""
历史数据验证模块
负责获取历史数据用于回测验证
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from data.sector_fetcher import SectorFetcher
from cache.manager import CacheManager


class HistoricalValidator:
    """历史数据验证器"""
    
    def __init__(self, sector_fetcher: SectorFetcher, cache_manager: CacheManager):
        self.sector_fetcher = sector_fetcher
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
    async def get_historical_segments(self, start_date: str, end_date: str, 
                                    validation_period: str = "2weeks") -> List[Dict[str, Any]]:
        """
        生成历史分段数据用于回测验证
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD  
            validation_period: 验证周期
            
        Returns:
            List[Dict]: 历史分段数据
        """
        try:
            self.logger.info(f"生成历史分段数据: {start_date} - {end_date}")
            
            # 解析验证周期
            validation_days = self._parse_validation_period(validation_period)
            
            # 生成时间段
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            segments = []
            current_date = start_dt
            
            while current_date + timedelta(days=validation_days * 2) <= end_dt:
                # 训练期：30天用于分析
                train_start = current_date
                train_end = current_date + timedelta(days=30)
                
                # 验证期：validation_days天用于验证预测
                valid_start = train_end + timedelta(days=1)
                valid_end = train_end + timedelta(days=validation_days)
                
                segment = {
                    'segment_id': f"{train_start.strftime('%Y%m%d')}_{valid_end.strftime('%Y%m%d')}",
                    'train_period': {
                        'start_date': train_start.strftime("%Y%m%d"),
                        'end_date': train_end.strftime("%Y%m%d")
                    },
                    'validation_period': {
                        'start_date': valid_start.strftime("%Y%m%d"), 
                        'end_date': valid_end.strftime("%Y%m%d")
                    }
                }
                
                segments.append(segment)
                
                # 移动到下一个时间段（重叠50%）
                current_date += timedelta(days=validation_days)
                
            self.logger.info(f"生成了{len(segments)}个历史分段")
            return segments
            
        except Exception as e:
            self.logger.error(f"生成历史分段失败: {e}")
            return []
            
    async def get_segment_data(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取单个分段的数据
        
        Args:
            segment: 分段信息
            
        Returns:
            Dict: 分段数据
        """
        try:
            train_period = segment['train_period']
            validation_period = segment['validation_period']
            
            # 获取训练期数据
            train_data = await self.sector_fetcher.get_all_sectors_data(
                (train_period['start_date'], train_period['end_date'])
            )
            
            # 获取验证期数据  
            validation_data = await self.sector_fetcher.get_all_sectors_data(
                (validation_period['start_date'], validation_period['end_date'])
            )
            
            return {
                'segment_id': segment['segment_id'],
                'train_data': train_data,
                'validation_data': validation_data,
                'train_period': train_period,
                'validation_period': validation_period
            }
            
        except Exception as e:
            self.logger.error(f"获取分段数据失败: {e}")
            return {
                'segment_id': segment.get('segment_id', 'unknown'),
                'error': str(e)
            }
            
    def calculate_actual_performance(self, validation_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        计算验证期实际表现
        
        Args:
            validation_data: 验证期数据
            
        Returns:
            Dict: 各板块实际表现
        """
        actual_performance = {}
        
        try:
            for sector_name, sector_df in validation_data.items():
                if sector_df.empty:
                    continue
                    
                # 计算验证期收益率
                start_price = sector_df['close'].iloc[0]
                end_price = sector_df['close'].iloc[-1]
                total_return = (end_price / start_price - 1) * 100 if start_price > 0 else 0
                
                # 计算波动率
                daily_returns = sector_df['pct_change'].dropna()
                volatility = daily_returns.std() * 100 if len(daily_returns) > 1 else 0
                
                # 计算最大涨跌幅
                max_price = sector_df['close'].max()
                min_price = sector_df['close'].min()
                max_gain = (max_price / start_price - 1) * 100 if start_price > 0 else 0
                max_loss = (min_price / start_price - 1) * 100 if start_price > 0 else 0
                
                # 计算趋势方向
                if total_return > 2:
                    trend = "上涨"
                elif total_return < -2:
                    trend = "下跌"
                else:
                    trend = "震荡"
                    
                actual_performance[sector_name] = {
                    'total_return': round(total_return, 2),
                    'volatility': round(volatility, 2),
                    'max_gain': round(max_gain, 2),
                    'max_loss': round(max_loss, 2),
                    'trend': trend,
                    'trading_days': len(sector_df),
                    'start_price': float(start_price),
                    'end_price': float(end_price)
                }
                
        except Exception as e:
            self.logger.error(f"计算实际表现失败: {e}")
            
        return actual_performance
        
    def compare_prediction_vs_actual(self, predictions: Dict[str, Dict], 
                                   actual_performance: Dict[str, Dict]) -> Dict[str, Any]:
        """
        对比预测与实际结果
        
        Args:
            predictions: 预测结果
            actual_performance: 实际表现
            
        Returns:
            Dict: 对比结果
        """
        try:
            comparison_results = {}
            accuracy_metrics = {
                'trend_accuracy': 0,
                'return_accuracy': 0,
                'recommendation_accuracy': 0,
                'total_sectors': 0
            }
            
            for sector_name in predictions.keys():
                if sector_name not in actual_performance:
                    continue
                    
                pred = predictions[sector_name]
                actual = actual_performance[sector_name]
                
                # 趋势方向准确性
                pred_trend = pred.get('trend_prediction', 'Unknown')
                actual_trend = actual.get('trend', 'Unknown')
                trend_correct = pred_trend == actual_trend
                
                # 收益率偏差
                pred_return = pred.get('expected_return', 0)
                actual_return = actual.get('total_return', 0)
                return_deviation = abs(pred_return - actual_return)
                
                # 推荐准确性（基于实际表现评判推荐质量）
                recommendation = pred.get('recommendation', 'Hold')
                actual_return = actual.get('total_return', 0)
                
                rec_correct = False
                if recommendation in ['Strong Buy', 'Buy'] and actual_return > 3:
                    rec_correct = True
                elif recommendation == 'Hold' and -3 <= actual_return <= 3:
                    rec_correct = True
                elif recommendation in ['Hold-', 'Avoid'] and actual_return < -3:
                    rec_correct = True
                    
                comparison_results[sector_name] = {
                    'predicted_trend': pred_trend,
                    'actual_trend': actual_trend,
                    'trend_correct': trend_correct,
                    'predicted_return': pred_return,
                    'actual_return': actual_return,
                    'return_deviation': round(return_deviation, 2),
                    'recommendation': recommendation,
                    'recommendation_correct': rec_correct,
                    'score': pred.get('comprehensive_score', 0)
                }
                
                # 累计准确率统计
                if trend_correct:
                    accuracy_metrics['trend_accuracy'] += 1
                if return_deviation <= 5:  # 5%以内误差视为准确
                    accuracy_metrics['return_accuracy'] += 1
                if rec_correct:
                    accuracy_metrics['recommendation_accuracy'] += 1
                accuracy_metrics['total_sectors'] += 1
                
            # 计算准确率百分比
            if accuracy_metrics['total_sectors'] > 0:
                total = accuracy_metrics['total_sectors']
                accuracy_metrics['trend_accuracy_pct'] = round(accuracy_metrics['trend_accuracy'] / total * 100, 1)
                accuracy_metrics['return_accuracy_pct'] = round(accuracy_metrics['return_accuracy'] / total * 100, 1)
                accuracy_metrics['recommendation_accuracy_pct'] = round(accuracy_metrics['recommendation_accuracy'] / total * 100, 1)
            else:
                accuracy_metrics['trend_accuracy_pct'] = 0
                accuracy_metrics['return_accuracy_pct'] = 0
                accuracy_metrics['recommendation_accuracy_pct'] = 0
                
            return {
                'sector_comparisons': comparison_results,
                'accuracy_metrics': accuracy_metrics,
                'summary': {
                    'best_predicted_sectors': self._get_best_predictions(comparison_results),
                    'worst_predicted_sectors': self._get_worst_predictions(comparison_results),
                    'overall_accuracy': round((accuracy_metrics['trend_accuracy_pct'] + 
                                             accuracy_metrics['return_accuracy_pct'] + 
                                             accuracy_metrics['recommendation_accuracy_pct']) / 3, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"对比预测结果失败: {e}")
            return {'error': str(e)}
            
    def _parse_validation_period(self, period: str) -> int:
        """解析验证周期为天数"""
        if "week" in period.lower():
            if "2" in period:
                return 14
            else:
                return 7
        elif "month" in period.lower():
            return 30
        else:
            return 14  # 默认2周
            
    def _get_best_predictions(self, comparisons: Dict[str, Dict]) -> List[Dict]:
        """获取预测最准确的板块"""
        try:
            scored_sectors = []
            for sector_name, comp in comparisons.items():
                accuracy_score = 0
                if comp.get('trend_correct', False):
                    accuracy_score += 40
                if comp.get('return_deviation', 100) <= 5:
                    accuracy_score += 30
                if comp.get('recommendation_correct', False):
                    accuracy_score += 30
                    
                scored_sectors.append({
                    'sector_name': sector_name,
                    'accuracy_score': accuracy_score,
                    'trend_correct': comp.get('trend_correct', False),
                    'return_deviation': comp.get('return_deviation', 0)
                })
                
            # 按准确率排序
            scored_sectors.sort(key=lambda x: x['accuracy_score'], reverse=True)
            return scored_sectors[:5]  # 返回前5个
            
        except Exception as e:
            self.logger.error(f"获取最佳预测失败: {e}")
            return []
            
    def _get_worst_predictions(self, comparisons: Dict[str, Dict]) -> List[Dict]:
        """获取预测最不准确的板块"""
        try:
            scored_sectors = []
            for sector_name, comp in comparisons.items():
                accuracy_score = 0
                if comp.get('trend_correct', False):
                    accuracy_score += 40
                if comp.get('return_deviation', 100) <= 5:
                    accuracy_score += 30
                if comp.get('recommendation_correct', False):
                    accuracy_score += 30
                    
                scored_sectors.append({
                    'sector_name': sector_name,
                    'accuracy_score': accuracy_score,
                    'trend_correct': comp.get('trend_correct', False),
                    'return_deviation': comp.get('return_deviation', 0)
                })
                
            # 按准确率排序（升序）
            scored_sectors.sort(key=lambda x: x['accuracy_score'])
            return scored_sectors[:5]  # 返回后5个
            
        except Exception as e:
            self.logger.error(f"获取最差预测失败: {e}")
            return []
            
    async def validate_historical_predictions(self, sectors_to_test: List[str] = None) -> Dict[str, Any]:
        """
        验证历史预测的简化版本（用于演示）
        
        Args:
            sectors_to_test: 要测试的板块列表
            
        Returns:
            Dict: 验证结果
        """
        try:
            if not sectors_to_test:
                sectors_to_test = ["银行", "医药生物", "电子", "计算机", "食品饮料"]
                
            self.logger.info(f"开始验证历史预测，测试板块: {sectors_to_test}")
            
            # 生成模拟的历史预测结果
            historical_predictions = {}
            for sector_name in sectors_to_test:
                historical_predictions[sector_name] = {
                    'trend_prediction': self._generate_mock_prediction(sector_name)['trend'],
                    'expected_return': self._generate_mock_prediction(sector_name)['return'],
                    'recommendation': self._generate_mock_prediction(sector_name)['recommendation'],
                    'comprehensive_score': self._generate_mock_prediction(sector_name)['score']
                }
                
            # 生成模拟的实际表现
            actual_performance = {}
            for sector_name in sectors_to_test:
                actual_performance[sector_name] = self._generate_mock_actual_performance(sector_name)
                
            # 对比结果
            comparison = self.compare_prediction_vs_actual(historical_predictions, actual_performance)
            
            return {
                'validation_type': 'Historical Simulation',
                'test_sectors': sectors_to_test,
                'test_period': '过去6个月模拟',
                'predictions': historical_predictions,
                'actual_performance': actual_performance,
                'comparison_results': comparison,
                'validation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"历史预测验证失败: {e}")
            return {'error': str(e)}
            
    def _generate_mock_prediction(self, sector_name: str) -> Dict[str, Any]:
        """生成模拟预测（基于板块特征）"""
        import random
        
        # 根据板块名称设置不同的特征
        sector_characteristics = {
            "银行": {"volatility": 0.5, "growth_tendency": 0.3},
            "医药生物": {"volatility": 0.8, "growth_tendency": 0.7},
            "电子": {"volatility": 0.9, "growth_tendency": 0.6},
            "计算机": {"volatility": 0.9, "growth_tendency": 0.8},
            "食品饮料": {"volatility": 0.4, "growth_tendency": 0.4}
        }
        
        char = sector_characteristics.get(sector_name, {"volatility": 0.6, "growth_tendency": 0.5})
        
        # 生成预测
        expected_return = random.uniform(-10, 15) * char["growth_tendency"]
        
        if expected_return > 5:
            trend = "上涨"
            recommendation = "Buy"
            score = random.uniform(70, 90)
        elif expected_return > 0:
            trend = "震荡上涨"
            recommendation = "Hold+"
            score = random.uniform(55, 75)
        elif expected_return > -3:
            trend = "震荡"
            recommendation = "Hold"
            score = random.uniform(45, 65)
        else:
            trend = "下跌"
            recommendation = "Avoid"
            score = random.uniform(20, 45)
            
        return {
            'trend': trend,
            'return': round(expected_return, 2),
            'recommendation': recommendation,
            'score': round(score, 1)
        }
        
    def _generate_mock_actual_performance(self, sector_name: str) -> Dict[str, Any]:
        """生成模拟实际表现"""
        import random
        
        # 基于预测生成相关的实际表现（添加一些随机误差）
        pred = self._generate_mock_prediction(sector_name)
        expected_return = pred['return']
        
        # 添加随机误差
        actual_return = expected_return + random.uniform(-8, 8)
        
        # 确定实际趋势
        if actual_return > 2:
            trend = "上涨"
        elif actual_return < -2:
            trend = "下跌"
        else:
            trend = "震荡"
            
        return {
            'total_return': round(actual_return, 2),
            'volatility': round(random.uniform(1, 6), 2),
            'max_gain': round(actual_return + random.uniform(2, 8), 2),
            'max_loss': round(actual_return - random.uniform(2, 8), 2),
            'trend': trend,
            'trading_days': 14,
            'start_price': 100.0,
            'end_price': round(100 * (1 + actual_return/100), 2)
        }