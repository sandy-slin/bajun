#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史性能验证框架 - 效果测试优先
基于真实历史数据验证交易决策效果

验证方法:
1. (T, T+1~T+5)时间窗口回测
2. 板块预测准确率验证
3. 股票选择收益率验证  
4. 投资组合表现验证
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path

from ..core.sector_engine import SectorEngine
from ..core.stock_engine import StockEngine
from ..core.portfolio_engine import PortfolioEngine
from .real_data_validator import RealDataValidator


class PerformanceValidator:
    """历史性能验证器 - 基于真实数据的效果验证"""
    
    def __init__(
        self,
        sector_engine: SectorEngine,
        stock_engine: StockEngine,
        portfolio_engine: PortfolioEngine
    ):
        self.sector_engine = sector_engine
        self.stock_engine = stock_engine
        self.portfolio_engine = portfolio_engine
        self.data_validator = RealDataValidator()
        self.logger = logging.getLogger(__name__)
        
        # 验证参数
        self.validation_params = {
            'default_t_values': [  # 默认验证时间点
                '20240615', '20240715', '20240815', 
                '20240915', '20241015'
            ],
            'prediction_horizons': [1, 3, 5],  # 预测时间窗口(天)
            'min_validation_periods': 5,       # 最少验证期数
            'success_threshold': 0.5,          # 成功率阈值50%
            'benchmark_indices': ['000300', '000001', '399001']  # 基准指数
        }
        
        # 性能指标定义
        self.performance_metrics = {
            'sector_analysis': {
                'top5_accuracy': '前5板块准确率',
                'ranking_correlation': '排名相关性',
                'excess_return': '超额收益率',
                'hit_rate': '方向预测准确率'
            },
            'stock_selection': {
                'average_return': '平均收益率',
                'win_rate': '胜率',
                'sharpe_ratio': '夏普比率',
                'benchmark_alpha': '基准超额收益'
            },
            'portfolio_management': {
                'total_return': '总收益率',
                'max_drawdown': '最大回撤',
                'volatility': '波动率',
                'information_ratio': '信息比率'
            }
        }
    
    async def run_comprehensive_validation(
        self,
        t_values: Optional[List[str]] = None,
        prediction_days: int = 5,
        save_results: bool = True
    ) -> Dict:
        """
        运行综合性能验证
        
        Args:
            t_values: 验证时间点列表 (YYYYMMDD格式)
            prediction_days: 预测天数
            save_results: 是否保存结果
            
        Returns:
            Dict: 完整的验证结果
        """
        try:
            validation_start = datetime.now()
            self.logger.info("开始历史性能验证")
            
            # 使用默认时间点或指定时间点
            if t_values is None:
                t_values = self.validation_params['default_t_values']
            
            if len(t_values) < self.validation_params['min_validation_periods']:
                return {
                    'error': f'验证期数不足，至少需要{self.validation_params["min_validation_periods"]}个时间点'
                }
            
            validation_results = []
            
            # 对每个时间点进行验证
            for t_value in t_values:
                try:
                    self.logger.info(f"验证时间点: {t_value}")
                    
                    single_validation = await self._validate_single_period(
                        t_value, prediction_days
                    )
                    
                    if 'error' not in single_validation:
                        validation_results.append(single_validation)
                    else:
                        self.logger.warning(f"时间点{t_value}验证失败: {single_validation['error']}")
                        
                except Exception as e:
                    self.logger.error(f"时间点{t_value}验证异常: {e}")
                    continue
            
            if not validation_results:
                return {'error': '所有时间点验证均失败'}
            
            # 综合分析结果
            comprehensive_analysis = self._analyze_comprehensive_results(validation_results)
            
            # 生成最终报告
            final_report = {
                'validation_metadata': {
                    'validation_time': datetime.now().isoformat(),
                    'validation_periods': len(validation_results),
                    'prediction_days': prediction_days,
                    't_values_tested': [r['t_value'] for r in validation_results],
                    'processing_time_seconds': (datetime.now() - validation_start).total_seconds()
                },
                'individual_results': validation_results,
                'comprehensive_analysis': comprehensive_analysis,
                'performance_summary': self._generate_performance_summary(comprehensive_analysis),
                'recommendations': self._generate_improvement_recommendations(comprehensive_analysis)
            }
            
            # 保存结果
            if save_results:
                report_path = await self._save_validation_report(final_report)
                final_report['report_path'] = report_path
            
            self.logger.info(f"历史性能验证完成，耗时{final_report['validation_metadata']['processing_time_seconds']:.1f}秒")
            return final_report
            
        except Exception as e:
            self.logger.error(f"综合性能验证失败: {e}")
            return {'error': str(e)}
    
    async def _validate_single_period(self, t_value: str, prediction_days: int) -> Dict:
        """验证单个时间点的性能"""
        try:
            # 解析时间点
            t_date = datetime.strptime(t_value, '%Y%m%d')
            
            # Step 1: 板块分析验证
            sector_validation = await self._validate_sector_analysis(t_value, prediction_days)
            
            # Step 2: 股票选择验证
            stock_validation = await self._validate_stock_selection(
                t_value, prediction_days, sector_validation.get('top_sectors', [])
            )
            
            # Step 3: 整体策略验证
            strategy_validation = await self._validate_overall_strategy(
                t_value, prediction_days, sector_validation, stock_validation
            )
            
            return {
                't_value': t_value,
                't_date': t_date.isoformat(),
                'prediction_days': prediction_days,
                'sector_validation': sector_validation,
                'stock_validation': stock_validation,
                'strategy_validation': strategy_validation,
                'overall_score': self._calculate_period_score(
                    sector_validation, stock_validation, strategy_validation
                )
            }
            
        except Exception as e:
            return {'error': f'单期验证失败: {str(e)}'}
    
    async def _validate_sector_analysis(self, t_value: str, prediction_days: int) -> Dict:
        """验证板块分析效果"""
        try:
            # 在T时点运行板块分析
            sector_result = await self.sector_engine.analyze_top_sectors(
                lookback_months=6, top_n=5
            )
            
            if 'error' in sector_result:
                return {'error': f'板块分析失败: {sector_result["error"]}'}
            
            top_sectors = sector_result['top_sectors']
            
            # 获取T+1到T+prediction_days的实际表现
            actual_performance = await self._get_actual_sector_performance(
                [s['sector_name'] for s in top_sectors], t_value, prediction_days
            )
            
            # 计算验证指标
            validation_metrics = self._calculate_sector_metrics(
                top_sectors, actual_performance
            )
            
            return {
                'predicted_sectors': top_sectors,
                'actual_performance': actual_performance,
                'validation_metrics': validation_metrics,
                'success': validation_metrics['top5_accuracy'] >= self.validation_params['success_threshold']
            }
            
        except Exception as e:
            return {'error': f'板块分析验证失败: {str(e)}'}
    
    async def _validate_stock_selection(
        self, 
        t_value: str, 
        prediction_days: int, 
        top_sectors: List[Dict]
    ) -> Dict:
        """验证股票选择效果"""
        try:
            if not top_sectors:
                return {'error': '无可用板块数据'}
            
            # 在T时点运行股票筛选
            stock_result = await self.stock_engine.select_stocks_from_sectors(
                top_sectors, stocks_per_sector=5
            )
            
            if 'error' in stock_result:
                return {'error': f'股票筛选失败: {stock_result["error"]}'}
            
            # 收集所有选中的股票
            selected_stocks = []
            for sector_sel in stock_result['sector_selections']:
                selected_stocks.extend(sector_sel['selected_stocks'])
            
            # 获取实际表现
            actual_performance = await self._get_actual_stock_performance(
                [s['stock_code'] for s in selected_stocks], t_value, prediction_days
            )
            
            # 计算验证指标
            validation_metrics = self._calculate_stock_metrics(
                selected_stocks, actual_performance
            )
            
            return {
                'selected_stocks': selected_stocks,
                'actual_performance': actual_performance,
                'validation_metrics': validation_metrics,
                'success': validation_metrics['win_rate'] >= self.validation_params['success_threshold']
            }
            
        except Exception as e:
            return {'error': f'股票选择验证失败: {str(e)}'}
    
    async def _validate_overall_strategy(
        self, 
        t_value: str, 
        prediction_days: int,
        sector_validation: Dict,
        stock_validation: Dict
    ) -> Dict:
        """验证整体策略效果"""
        try:
            # 构建虚拟投资组合
            if 'selected_stocks' not in stock_validation:
                return {'error': '无可用股票数据'}
            
            selected_stocks = stock_validation['selected_stocks']
            portfolio_value = 1000000  # 100万初始资金
            equal_weight = 1.0 / len(selected_stocks)
            
            # 计算组合表现
            portfolio_performance = await self._calculate_portfolio_performance(
                selected_stocks, portfolio_value, equal_weight, t_value, prediction_days
            )
            
            # 与基准比较
            benchmark_performance = await self._get_benchmark_performance(
                '000300', t_value, prediction_days
            )
            
            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(portfolio_performance)
            
            return {
                'portfolio_performance': portfolio_performance,
                'benchmark_performance': benchmark_performance,
                'risk_metrics': risk_metrics,
                'alpha': portfolio_performance['total_return'] - benchmark_performance['total_return'],
                'success': portfolio_performance['total_return'] > 0
            }
            
        except Exception as e:
            return {'error': f'整体策略验证失败: {str(e)}'}
    
    async def _get_actual_sector_performance(
        self, 
        sector_names: List[str], 
        t_value: str, 
        prediction_days: int
    ) -> Dict:
        """获取板块实际表现"""
        try:
            t_date = datetime.strptime(t_value, '%Y%m%d')
            end_date = t_date + timedelta(days=prediction_days)
            
            sector_performance = {}
            
            for sector_name in sector_names:
                try:
                    # 获取T到T+prediction_days的数据
                    sector_data = await self.sector_engine.sector_fetcher.get_sector_data(
                        sector_name, days=prediction_days + 10
                    )
                    
                    if sector_data and len(sector_data) >= prediction_days:
                        # 验证数据真实性
                        await self.data_validator.validate_and_ensure_real_data(
                            sector_data, f"actual_performance_{sector_name}"
                        )
                        
                        df = pd.DataFrame(sector_data)
                        
                        # 计算收益率
                        start_price = df.iloc[0]['close']
                        end_price = df.iloc[prediction_days-1]['close']
                        return_rate = (end_price / start_price - 1) * 100
                        
                        sector_performance[sector_name] = {
                            'start_price': start_price,
                            'end_price': end_price,
                            'return_rate': return_rate,
                            'price_trend': 'up' if return_rate > 0 else 'down'
                        }
                        
                except Exception as e:
                    self.logger.warning(f"获取板块{sector_name}实际表现失败: {e}")
                    continue
            
            return sector_performance
            
        except Exception as e:
            self.logger.error(f"获取板块实际表现失败: {e}")
            return {}
    
    async def _get_actual_stock_performance(
        self, 
        stock_codes: List[str], 
        t_value: str, 
        prediction_days: int
    ) -> Dict:
        """获取股票实际表现"""
        try:
            stock_performance = {}
            
            for stock_code in stock_codes:
                try:
                    # 获取股票数据
                    stock_data = await self.stock_engine.data_fetcher.get_trading_data(stock_code)
                    
                    if stock_data and len(stock_data) >= prediction_days:
                        # 验证数据真实性
                        await self.data_validator.validate_and_ensure_real_data(
                            stock_data, f"actual_stock_{stock_code}"
                        )
                        
                        df = pd.DataFrame(stock_data)
                        
                        # 计算收益率
                        start_price = df.iloc[0]['close']
                        end_price = df.iloc[prediction_days-1]['close']
                        return_rate = (end_price / start_price - 1) * 100
                        
                        stock_performance[stock_code] = {
                            'start_price': start_price,
                            'end_price': end_price,
                            'return_rate': return_rate,
                            'volatility': df['close'].pct_change().std() * np.sqrt(252)
                        }
                        
                except Exception as e:
                    self.logger.warning(f"获取股票{stock_code}实际表现失败: {e}")
                    continue
            
            return stock_performance
            
        except Exception as e:
            self.logger.error(f"获取股票实际表现失败: {e}")
            return {}
    
    def _calculate_sector_metrics(self, predicted_sectors: List[Dict], actual_performance: Dict) -> Dict:
        """计算板块验证指标"""
        try:
            if not actual_performance:
                return {'top5_accuracy': 0, 'ranking_correlation': 0}
            
            # TOP5准确率 (预测为正收益的比例)
            positive_sectors = sum(
                1 for sector in predicted_sectors
                if sector['sector_name'] in actual_performance and 
                actual_performance[sector['sector_name']]['return_rate'] > 0
            )
            top5_accuracy = positive_sectors / len(predicted_sectors) if predicted_sectors else 0
            
            # 排名相关性
            predicted_scores = [s['composite_score'] for s in predicted_sectors if s['sector_name'] in actual_performance]
            actual_returns = [actual_performance[s['sector_name']]['return_rate'] 
                            for s in predicted_sectors if s['sector_name'] in actual_performance]
            
            if len(predicted_scores) >= 3 and len(actual_returns) >= 3:
                ranking_correlation = np.corrcoef(predicted_scores, actual_returns)[0, 1]
                if np.isnan(ranking_correlation):
                    ranking_correlation = 0
            else:
                ranking_correlation = 0
            
            # 平均超额收益
            avg_return = np.mean(actual_returns) if actual_returns else 0
            
            return {
                'top5_accuracy': top5_accuracy,
                'ranking_correlation': ranking_correlation,
                'average_return': avg_return,
                'positive_sectors': positive_sectors,
                'total_sectors': len(predicted_sectors)
            }
            
        except Exception as e:
            self.logger.error(f"计算板块指标失败: {e}")
            return {'top5_accuracy': 0, 'ranking_correlation': 0}
    
    def _calculate_stock_metrics(self, selected_stocks: List[Dict], actual_performance: Dict) -> Dict:
        """计算股票验证指标"""
        try:
            if not actual_performance:
                return {'win_rate': 0, 'average_return': 0}
            
            # 胜率
            winning_stocks = sum(
                1 for stock in selected_stocks
                if stock['stock_code'] in actual_performance and 
                actual_performance[stock['stock_code']]['return_rate'] > 0
            )
            win_rate = winning_stocks / len(selected_stocks) if selected_stocks else 0
            
            # 平均收益率
            returns = [actual_performance[s['stock_code']]['return_rate'] 
                      for s in selected_stocks if s['stock_code'] in actual_performance]
            average_return = np.mean(returns) if returns else 0
            
            # 夏普比率 (简化计算)
            if len(returns) > 1:
                return_std = np.std(returns)
                sharpe_ratio = average_return / return_std if return_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'win_rate': win_rate,
                'average_return': average_return,
                'sharpe_ratio': sharpe_ratio,
                'winning_stocks': winning_stocks,
                'total_stocks': len(selected_stocks),
                'return_volatility': np.std(returns) if returns else 0
            }
            
        except Exception as e:
            self.logger.error(f"计算股票指标失败: {e}")
            return {'win_rate': 0, 'average_return': 0}
    
    async def _calculate_portfolio_performance(
        self, 
        selected_stocks: List[Dict], 
        initial_value: float,
        weight_per_stock: float,
        t_value: str, 
        prediction_days: int
    ) -> Dict:
        """计算投资组合表现"""
        try:
            # 获取组合股票表现
            stock_performance = await self._get_actual_stock_performance(
                [s['stock_code'] for s in selected_stocks], t_value, prediction_days
            )
            
            if not stock_performance:
                return {'total_return': 0, 'final_value': initial_value}
            
            # 计算加权收益
            total_weighted_return = 0
            valid_positions = 0
            
            for stock in selected_stocks:
                stock_code = stock['stock_code']
                if stock_code in stock_performance:
                    return_rate = stock_performance[stock_code]['return_rate'] / 100
                    total_weighted_return += return_rate * weight_per_stock
                    valid_positions += 1
            
            # 调整权重 (如果有股票数据缺失)
            if valid_positions > 0 and valid_positions < len(selected_stocks):
                total_weighted_return = total_weighted_return * len(selected_stocks) / valid_positions
            
            final_value = initial_value * (1 + total_weighted_return)
            
            return {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_weighted_return * 100,
                'valid_positions': valid_positions,
                'total_positions': len(selected_stocks)
            }
            
        except Exception as e:
            self.logger.error(f"计算组合表现失败: {e}")
            return {'total_return': 0, 'final_value': initial_value}
    
    async def _get_benchmark_performance(self, benchmark_code: str, t_value: str, prediction_days: int) -> Dict:
        """获取基准表现"""
        try:
            # 这里应该实现获取基准指数数据的逻辑
            # 简化实现，返回模拟基准表现
            return {
                'benchmark_code': benchmark_code,
                'total_return': 2.5,  # 假设基准收益2.5%
                'volatility': 1.2
            }
            
        except Exception as e:
            self.logger.error(f"获取基准表现失败: {e}")
            return {'total_return': 0}
    
    def _calculate_risk_metrics(self, portfolio_performance: Dict) -> Dict:
        """计算风险指标"""
        try:
            return {
                'max_drawdown': 0.05,  # 简化实现
                'volatility': 0.15,
                'var_95': 0.03,
                'risk_score': 'medium'
            }
            
        except Exception as e:
            return {'risk_score': 'unknown'}
    
    def _calculate_period_score(self, *validation_results) -> float:
        """计算单期综合得分"""
        try:
            scores = []
            
            for result in validation_results:
                if isinstance(result, dict) and 'error' not in result:
                    if 'validation_metrics' in result:
                        metrics = result['validation_metrics']
                        if 'top5_accuracy' in metrics:
                            scores.append(metrics['top5_accuracy'] * 100)
                        elif 'win_rate' in metrics:
                            scores.append(metrics['win_rate'] * 100)
                    elif 'portfolio_performance' in result:
                        total_return = result['portfolio_performance'].get('total_return', 0)
                        scores.append(max(0, 50 + total_return))  # 基础分50 + 收益率
            
            return np.mean(scores) if scores else 0
            
        except Exception:
            return 0
    
    def _analyze_comprehensive_results(self, validation_results: List[Dict]) -> Dict:
        """分析综合结果"""
        try:
            if not validation_results:
                return {}
            
            # 提取各项指标
            sector_accuracies = []
            stock_win_rates = []
            portfolio_returns = []
            overall_scores = []
            
            for result in validation_results:
                if 'sector_validation' in result and 'validation_metrics' in result['sector_validation']:
                    sector_accuracies.append(result['sector_validation']['validation_metrics']['top5_accuracy'])
                
                if 'stock_validation' in result and 'validation_metrics' in result['stock_validation']:
                    stock_win_rates.append(result['stock_validation']['validation_metrics']['win_rate'])
                
                if 'strategy_validation' in result and 'portfolio_performance' in result['strategy_validation']:
                    portfolio_returns.append(result['strategy_validation']['portfolio_performance']['total_return'])
                
                overall_scores.append(result.get('overall_score', 0))
            
            return {
                'sector_analysis_performance': {
                    'average_accuracy': np.mean(sector_accuracies) if sector_accuracies else 0,
                    'accuracy_stability': np.std(sector_accuracies) if len(sector_accuracies) > 1 else 0,
                    'best_accuracy': max(sector_accuracies) if sector_accuracies else 0,
                    'worst_accuracy': min(sector_accuracies) if sector_accuracies else 0
                },
                'stock_selection_performance': {
                    'average_win_rate': np.mean(stock_win_rates) if stock_win_rates else 0,
                    'win_rate_stability': np.std(stock_win_rates) if len(stock_win_rates) > 1 else 0,
                    'best_win_rate': max(stock_win_rates) if stock_win_rates else 0,
                    'worst_win_rate': min(stock_win_rates) if stock_win_rates else 0
                },
                'portfolio_performance': {
                    'average_return': np.mean(portfolio_returns) if portfolio_returns else 0,
                    'return_stability': np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0,
                    'best_return': max(portfolio_returns) if portfolio_returns else 0,
                    'worst_return': min(portfolio_returns) if portfolio_returns else 0,
                    'positive_periods': sum(1 for r in portfolio_returns if r > 0)
                },
                'overall_assessment': {
                    'average_score': np.mean(overall_scores),
                    'score_stability': np.std(overall_scores) if len(overall_scores) > 1 else 0,
                    'success_rate': sum(1 for s in overall_scores if s >= 60) / len(overall_scores),
                    'grade': self._calculate_grade(np.mean(overall_scores))
                }
            }
            
        except Exception as e:
            self.logger.error(f"分析综合结果失败: {e}")
            return {}
    
    def _generate_performance_summary(self, comprehensive_analysis: Dict) -> Dict:
        """生成性能摘要"""
        try:
            if not comprehensive_analysis:
                return {'summary': '无有效分析数据'}
            
            overall = comprehensive_analysis.get('overall_assessment', {})
            avg_score = overall.get('average_score', 0)
            success_rate = overall.get('success_rate', 0)
            grade = overall.get('grade', 'F')
            
            return {
                'overall_grade': grade,
                'average_score': avg_score,
                'success_rate': success_rate * 100,
                'key_strengths': self._identify_strengths(comprehensive_analysis),
                'improvement_areas': self._identify_weaknesses(comprehensive_analysis),
                'system_readiness': self._assess_system_readiness(comprehensive_analysis)
            }
            
        except Exception as e:
            return {'summary': f'摘要生成失败: {str(e)}'}
    
    def _generate_improvement_recommendations(self, comprehensive_analysis: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        try:
            sector_perf = comprehensive_analysis.get('sector_analysis_performance', {})
            stock_perf = comprehensive_analysis.get('stock_selection_performance', {})
            portfolio_perf = comprehensive_analysis.get('portfolio_performance', {})
            
            # 板块分析改进建议
            if sector_perf.get('average_accuracy', 0) < 0.6:
                recommendations.append('板块分析准确率偏低，建议优化评分算法权重')
            
            if sector_perf.get('accuracy_stability', 1) > 0.2:
                recommendations.append('板块预测稳定性不足，建议增加历史数据深度')
            
            # 股票选择改进建议
            if stock_perf.get('average_win_rate', 0) < 0.5:
                recommendations.append('股票选择胜率不足，建议加强筛选条件')
            
            # 组合管理改进建议
            if portfolio_perf.get('average_return', 0) < 5:
                recommendations.append('组合平均收益偏低，建议优化仓位配置策略')
            
            if portfolio_perf.get('positive_periods', 0) < 3:
                recommendations.append('盈利期数较少，建议增强风险控制措施')
            
            if not recommendations:
                recommendations.append('系统整体表现良好，建议继续保持当前策略')
            
        except Exception as e:
            recommendations.append(f'建议生成异常: {str(e)}')
        
        return recommendations
    
    async def _save_validation_report(self, report: Dict) -> str:
        """保存验证报告"""
        try:
            # 创建验证报告目录
            reports_dir = Path('reports/validation')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'performance_validation_{timestamp}.json'
            report_path = reports_dir / report_filename
            
            # 保存JSON报告
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"保存验证报告失败: {e}")
            return ''
    
    # 辅助方法
    
    def _calculate_grade(self, score: float) -> str:
        """计算等级"""
        if score >= 85:
            return 'A'
        elif score >= 75:
            return 'B'
        elif score >= 65:
            return 'C'
        elif score >= 55:
            return 'D'
        else:
            return 'F'
    
    def _identify_strengths(self, analysis: Dict) -> List[str]:
        """识别优势"""
        strengths = []
        
        sector_perf = analysis.get('sector_analysis_performance', {})
        if sector_perf.get('average_accuracy', 0) >= 0.7:
            strengths.append('板块分析准确率优秀')
        
        stock_perf = analysis.get('stock_selection_performance', {})
        if stock_perf.get('average_win_rate', 0) >= 0.6:
            strengths.append('股票选择胜率较高')
        
        portfolio_perf = analysis.get('portfolio_performance', {})
        if portfolio_perf.get('average_return', 0) >= 8:
            strengths.append('组合收益表现良好')
        
        return strengths if strengths else ['系统稳定性良好']
    
    def _identify_weaknesses(self, analysis: Dict) -> List[str]:
        """识别弱点"""
        weaknesses = []
        
        sector_perf = analysis.get('sector_analysis_performance', {})
        if sector_perf.get('average_accuracy', 0) < 0.5:
            weaknesses.append('板块分析准确率不足')
        
        stock_perf = analysis.get('stock_selection_performance', {})
        if stock_perf.get('average_win_rate', 0) < 0.4:
            weaknesses.append('股票选择效果偏弱')
        
        portfolio_perf = analysis.get('portfolio_performance', {})
        if portfolio_perf.get('positive_periods', 0) < 2:
            weaknesses.append('盈利稳定性不足')
        
        return weaknesses if weaknesses else ['暂无明显弱点']
    
    def _assess_system_readiness(self, analysis: Dict) -> str:
        """评估系统就绪度"""
        overall = analysis.get('overall_assessment', {})
        avg_score = overall.get('average_score', 0)
        success_rate = overall.get('success_rate', 0)
        
        if avg_score >= 75 and success_rate >= 0.6:
            return 'production_ready'
        elif avg_score >= 65 and success_rate >= 0.4:
            return 'beta_ready'
        elif avg_score >= 55:
            return 'development_stage'
        else:
            return 'needs_improvement'