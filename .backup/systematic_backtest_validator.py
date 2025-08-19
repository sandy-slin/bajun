#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统性历史回测验证框架
支持多时间点(T, period)批量验证，全面评估预测算法准确性
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
from concurrent.futures import ThreadPoolExecutor
import argparse

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

class SystematicBacktestValidator:
    """系统性历史回测验证器"""
    
    def __init__(self):
        """初始化验证器组件"""
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
        
    async def run_systematic_backtest(self, 
                                    test_periods: List[Tuple[str, int]],
                                    training_months: int = 6,
                                    min_training_days: int = 90) -> Dict:
        """
        运行系统性回测验证
        
        Args:
            test_periods: 测试时间点和周期列表 [(T1, period1), (T2, period2), ...]
                         T格式: YYYYMMDD, period: 预测天数
            training_months: 训练数据月数
            min_training_days: 最小训练天数
            
        Returns:
            综合验证结果和统计分析
        """
        try:
            self.logger.info(f"🚀 开始系统性历史回测验证，共{len(test_periods)}个测试点")
            
            # 验证结果存储
            validation_results = []
            overall_stats = {
                'total_tests': len(test_periods),
                'successful_tests': 0,
                'failed_tests': 0,
                'accuracy_scores': [],
                'sector_performance': {},
                'period_performance': {}
            }
            
            # 批量执行验证
            for i, (test_date, prediction_period) in enumerate(test_periods):
                self.logger.info(f"📊 执行测试 {i+1}/{len(test_periods)}: T={test_date}, period={prediction_period}天")
                
                try:
                    # 执行单个时间点的回测
                    single_result = await self._run_single_backtest(
                        test_date, prediction_period, training_months, min_training_days
                    )
                    
                    if single_result and 'error' not in single_result:
                        validation_results.append(single_result)
                        overall_stats['successful_tests'] += 1
                        
                        # 收集统计数据
                        if 'overall_accuracy' in single_result:
                            overall_stats['accuracy_scores'].append(single_result['overall_accuracy'])
                        
                        # 按预测周期统计
                        period_key = f"{prediction_period}天"
                        if period_key not in overall_stats['period_performance']:
                            overall_stats['period_performance'][period_key] = []
                        overall_stats['period_performance'][period_key].append(single_result['overall_accuracy'])
                        
                        # 按板块统计
                        sector_perf = single_result.get('sector_performance', {})
                        for sector, perf in sector_perf.items():
                            if sector not in overall_stats['sector_performance']:
                                overall_stats['sector_performance'][sector] = []
                            overall_stats['sector_performance'][sector].append(perf.get('accuracy', 0))
                            
                        self.logger.info(f"✅ 测试{i+1}完成，准确率: {single_result.get('overall_accuracy', 0):.1f}%")
                    else:
                        overall_stats['failed_tests'] += 1
                        self.logger.warning(f"❌ 测试{i+1}失败: {single_result.get('error', '未知错误')}")
                        
                except Exception as e:
                    overall_stats['failed_tests'] += 1
                    self.logger.error(f"❌ 测试{i+1}异常: {e}")
                    continue
            
            # 计算综合统计
            comprehensive_stats = self._calculate_comprehensive_statistics(validation_results, overall_stats)
            
            # 生成最终报告
            final_report = {
                'validation_summary': {
                    'test_framework': 'SystematicBacktestValidator',
                    'test_timestamp': datetime.now().isoformat(),
                    'total_test_periods': len(test_periods),
                    'successful_validations': overall_stats['successful_tests'],
                    'failed_validations': overall_stats['failed_tests'],
                    'success_rate': (overall_stats['successful_tests'] / len(test_periods)) * 100
                },
                'performance_metrics': comprehensive_stats,
                'detailed_results': validation_results,
                'test_periods': test_periods
            }
            
            # 保存验证报告
            await self._save_validation_report(final_report)
            
            self.logger.info(f"🎉 系统性回测验证完成！成功率: {final_report['validation_summary']['success_rate']:.1f}%")
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"❌ 系统性回测验证失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    async def _run_single_backtest(self, 
                                 test_date: str, 
                                 prediction_period: int,
                                 training_months: int,
                                 min_training_days: int) -> Dict:
        """执行单个时间点的回测验证"""
        try:
            # 解析测试日期
            test_dt = datetime.strptime(test_date, "%Y%m%d")
            
            # 计算训练数据时间范围
            train_end_dt = test_dt - timedelta(days=1)
            train_start_dt = train_end_dt - timedelta(days=training_months * 30)
            
            # 检查训练数据是否充足
            training_days = (train_end_dt - train_start_dt).days
            if training_days < min_training_days:
                return {'error': f'训练数据不足: {training_days}天 < {min_training_days}天'}
            
            # 计算预测目标时间范围
            prediction_start_dt = test_dt
            prediction_end_dt = test_dt + timedelta(days=prediction_period)
            
            self.logger.debug(f"📅 训练期: {train_start_dt.strftime('%Y%m%d')} - {train_end_dt.strftime('%Y%m%d')}")
            self.logger.debug(f"🎯 预测期: {prediction_start_dt.strftime('%Y%m%d')} - {prediction_end_dt.strftime('%Y%m%d')}")
            
            # 获取训练数据
            train_data = await self.sector_fetcher.get_all_sectors_data((
                train_start_dt.strftime("%Y%m%d"),
                train_end_dt.strftime("%Y%m%d")
            ))
            
            if not train_data:
                return {'error': '无法获取训练数据'}
            
            # 获取实际结果数据
            actual_data = await self.sector_fetcher.get_all_sectors_data((
                prediction_start_dt.strftime("%Y%m%d"),
                prediction_end_dt.strftime("%Y%m%d")
            ))
            
            if not actual_data:
                return {'error': '无法获取验证数据'}
            
            # 执行预测分析
            sector_predictions = await self._analyze_sectors_for_backtest(train_data, prediction_period)
            
            # 计算实际表现
            actual_performance = self._calculate_actual_performance(actual_data, prediction_period)
            
            # 对比预测和实际结果
            validation_results = self._compare_predictions_with_actual(
                sector_predictions, actual_performance
            )
            
            # 计算准确率
            accuracy_metrics = self._calculate_accuracy_metrics(validation_results)
            
            return {
                'test_date': test_date,
                'prediction_period': prediction_period,
                'training_period': f"{train_start_dt.strftime('%Y%m%d')}-{train_end_dt.strftime('%Y%m%d')}",
                'prediction_target_period': f"{prediction_start_dt.strftime('%Y%m%d')}-{prediction_end_dt.strftime('%Y%m%d')}",
                'training_days': training_days,
                'sectors_analyzed': len(sector_predictions),
                'overall_accuracy': accuracy_metrics['overall_accuracy'],
                'sector_performance': validation_results,
                'accuracy_metrics': accuracy_metrics
            }
            
        except Exception as e:
            self.logger.error(f"单点回测失败 T={test_date}: {e}")
            return {'error': str(e)}
    
    async def _analyze_sectors_for_backtest(self, train_data: Dict, prediction_period: int) -> Dict:
        """使用训练数据分析板块预测"""
        sector_predictions = {}
        
        for sector_name, data in train_data.items():
            try:
                if isinstance(data, pd.DataFrame) and len(data) >= 30:  # 至少30天数据
                    # 使用技术分析计算板块强度
                    strength_score = self.technical_calculator.calculate_sector_strength_score(data)
                    
                    # 基于强度评分生成预测
                    overall_score = strength_score.get('overall_score', 0)
                    
                    # 简化的预测逻辑（基于技术强度评分）
                    if overall_score > 0.6:
                        predicted_direction = '上涨'
                        confidence = min(overall_score * 1.2, 1.0)
                    elif overall_score < 0.4:
                        predicted_direction = '下跌'
                        confidence = min((1 - overall_score) * 1.2, 1.0)
                    else:
                        predicted_direction = '横盘'
                        confidence = 0.5
                    
                    sector_predictions[sector_name] = {
                        'predicted_direction': predicted_direction,
                        'confidence': confidence,
                        'strength_score': overall_score,
                        'data_points': len(data)
                    }
                    
            except Exception as e:
                self.logger.warning(f"板块{sector_name}预测失败: {e}")
                continue
        
        return sector_predictions
    
    def _calculate_actual_performance(self, actual_data: Dict, prediction_period: int) -> Dict:
        """计算实际表现"""
        actual_performance = {}
        
        for sector_name, data in actual_data.items():
            try:
                if isinstance(data, pd.DataFrame) and len(data) >= 2:
                    # 计算期间收益率
                    start_price = data.iloc[0]['close']
                    end_price = data.iloc[-1]['close']
                    
                    change_pct = ((end_price - start_price) / start_price) * 100
                    
                    # 判断实际方向
                    if change_pct > 2:
                        actual_direction = '上涨'
                    elif change_pct < -2:
                        actual_direction = '下跌'
                    else:
                        actual_direction = '横盘'
                    
                    actual_performance[sector_name] = {
                        'actual_direction': actual_direction,
                        'change_percent': change_pct,
                        'start_price': start_price,
                        'end_price': end_price,
                        'data_points': len(data)
                    }
                    
            except Exception as e:
                self.logger.warning(f"板块{sector_name}实际表现计算失败: {e}")
                continue
        
        return actual_performance
    
    def _compare_predictions_with_actual(self, predictions: Dict, actual: Dict) -> Dict:
        """对比预测和实际结果"""
        comparison_results = {}
        
        for sector_name in predictions.keys():
            if sector_name in actual:
                pred = predictions[sector_name]
                act = actual[sector_name]
                
                # 判断预测是否正确
                predicted_dir = pred['predicted_direction']
                actual_dir = act['actual_direction']
                
                is_correct = predicted_dir == actual_dir
                
                comparison_results[sector_name] = {
                    'predicted_direction': predicted_dir,
                    'actual_direction': actual_dir,
                    'is_correct': is_correct,
                    'confidence': pred['confidence'],
                    'actual_change': act['change_percent'],
                    'accuracy': 100.0 if is_correct else 0.0
                }
        
        return comparison_results
    
    def _calculate_accuracy_metrics(self, validation_results: Dict) -> Dict:
        """计算准确率指标"""
        if not validation_results:
            return {'overall_accuracy': 0, 'total_predictions': 0}
        
        total_predictions = len(validation_results)
        correct_predictions = sum(1 for result in validation_results.values() if result['is_correct'])
        
        overall_accuracy = (correct_predictions / total_predictions) * 100
        
        # 按方向统计
        direction_stats = {}
        for direction in ['上涨', '下跌', '横盘']:
            dir_predictions = [r for r in validation_results.values() if r['predicted_direction'] == direction]
            dir_correct = sum(1 for r in dir_predictions if r['is_correct'])
            
            direction_stats[direction] = {
                'total': len(dir_predictions),
                'correct': dir_correct,
                'accuracy': (dir_correct / len(dir_predictions)) * 100 if dir_predictions else 0
            }
        
        # 置信度加权准确率
        weighted_accuracy = 0
        total_confidence = 0
        for result in validation_results.values():
            weight = result['confidence']
            accuracy = 100.0 if result['is_correct'] else 0.0
            weighted_accuracy += accuracy * weight
            total_confidence += weight
        
        if total_confidence > 0:
            weighted_accuracy = weighted_accuracy / total_confidence
        
        return {
            'overall_accuracy': overall_accuracy,
            'weighted_accuracy': weighted_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'direction_statistics': direction_stats
        }
    
    def _calculate_comprehensive_statistics(self, validation_results: List, overall_stats: Dict) -> Dict:
        """计算综合统计数据"""
        if not overall_stats['accuracy_scores']:
            return {'error': '无有效验证结果'}
        
        accuracy_scores = overall_stats['accuracy_scores']
        
        comprehensive_stats = {
            'accuracy_statistics': {
                'mean_accuracy': np.mean(accuracy_scores),
                'median_accuracy': np.median(accuracy_scores),
                'std_accuracy': np.std(accuracy_scores),
                'min_accuracy': np.min(accuracy_scores),
                'max_accuracy': np.max(accuracy_scores),
                'accuracy_range': np.max(accuracy_scores) - np.min(accuracy_scores)
            },
            'period_performance_summary': {},
            'sector_performance_summary': {},
            'consistency_metrics': {
                'accuracy_variance': np.var(accuracy_scores),
                'consistent_performance_ratio': sum(1 for acc in accuracy_scores if acc >= 60) / len(accuracy_scores)
            }
        }
        
        # 按预测周期统计
        for period, accuracies in overall_stats['period_performance'].items():
            comprehensive_stats['period_performance_summary'][period] = {
                'mean_accuracy': np.mean(accuracies),
                'test_count': len(accuracies),
                'success_rate': sum(1 for acc in accuracies if acc >= 50) / len(accuracies)
            }
        
        # 按板块统计
        for sector, accuracies in overall_stats['sector_performance'].items():
            if len(accuracies) >= 3:  # 至少3次测试
                comprehensive_stats['sector_performance_summary'][sector] = {
                    'mean_accuracy': np.mean(accuracies),
                    'test_count': len(accuracies),
                    'consistency': 1 - (np.std(accuracies) / 100)  # 一致性评分
                }
        
        return comprehensive_stats
    
    async def _save_validation_report(self, report: Dict):
        """保存验证报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"systematic_backtest_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"📄 验证报告已保存: {filename}")
            
            # 同时生成简化的摘要报告
            summary_filename = f"backtest_summary_{timestamp}.txt"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("系统性历史回测验证报告摘要\n")
                f.write("=" * 80 + "\n\n")
                
                summary = report['validation_summary']
                f.write(f"测试时间: {summary['test_timestamp']}\n")
                f.write(f"测试点数: {summary['total_test_periods']}\n")
                f.write(f"成功验证: {summary['successful_validations']}\n")
                f.write(f"失败验证: {summary['failed_validations']}\n")
                f.write(f"成功率: {summary['success_rate']:.1f}%\n\n")
                
                metrics = report['performance_metrics']
                if 'accuracy_statistics' in metrics:
                    acc_stats = metrics['accuracy_statistics']
                    f.write("准确率统计:\n")
                    f.write(f"  平均准确率: {acc_stats['mean_accuracy']:.1f}%\n")
                    f.write(f"  中位准确率: {acc_stats['median_accuracy']:.1f}%\n")
                    f.write(f"  标准差: {acc_stats['std_accuracy']:.1f}%\n")
                    f.write(f"  最高准确率: {acc_stats['max_accuracy']:.1f}%\n")
                    f.write(f"  最低准确率: {acc_stats['min_accuracy']:.1f}%\n")
            
            self.logger.info(f"📋 摘要报告已保存: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"保存验证报告失败: {e}")

def generate_test_periods(start_date: str, end_date: str, 
                         intervals_days: int = 30, 
                         prediction_periods: List[int] = [3, 5, 7]) -> List[Tuple[str, int]]:
    """
    生成测试时间点列表
    
    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD  
        intervals_days: 测试点间隔天数
        prediction_periods: 预测周期列表
    
    Returns:
        [(T, period), ...] 测试点列表
    """
    test_periods = []
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    current_dt = start_dt
    while current_dt <= end_dt:
        for period in prediction_periods:
            # 确保有足够的未来数据用于验证
            if current_dt + timedelta(days=period) <= end_dt:
                test_periods.append((current_dt.strftime("%Y%m%d"), period))
        
        current_dt += timedelta(days=intervals_days)
    
    return test_periods

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='系统性历史回测验证')
    parser.add_argument('--start-date', default='20250501', help='测试开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', default='20250801', help='测试结束日期 (YYYYMMDD)')
    parser.add_argument('--intervals', type=int, default=30, help='测试点间隔天数')
    parser.add_argument('--periods', nargs='+', type=int, default=[3, 5, 7], help='预测周期列表')
    parser.add_argument('--training-months', type=int, default=6, help='训练数据月数')
    
    args = parser.parse_args()
    
    # 生成测试时间点
    test_periods = generate_test_periods(
        args.start_date, args.end_date, 
        args.intervals, args.periods
    )
    
    print(f"🎯 系统性回测验证计划:")
    print(f"   测试期间: {args.start_date} - {args.end_date}")
    print(f"   测试点数: {len(test_periods)}")
    print(f"   预测周期: {args.periods}")
    print(f"   训练月数: {args.training_months}")
    print(f"   测试间隔: {args.intervals}天")
    
    # 执行验证
    validator = SystematicBacktestValidator()
    result = await validator.run_systematic_backtest(
        test_periods=test_periods,
        training_months=args.training_months
    )
    
    if result and 'error' not in result:
        summary = result['validation_summary']
        metrics = result['performance_metrics']
        
        print(f"\n🎉 验证完成!")
        print(f"   成功率: {summary['success_rate']:.1f}%")
        print(f"   成功验证: {summary['successful_validations']}/{summary['total_test_periods']}")
        
        if 'accuracy_statistics' in metrics:
            acc_stats = metrics['accuracy_statistics']
            print(f"   平均准确率: {acc_stats['mean_accuracy']:.1f}%")
            print(f"   准确率范围: {acc_stats['min_accuracy']:.1f}% - {acc_stats['max_accuracy']:.1f}%")
    else:
        print(f"❌ 验证失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    asyncio.run(main())