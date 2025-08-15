# -*- coding: utf-8 -*-
"""
板块回测模块
基于真实历史数据实现板块策略回测
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from data.sector_fetcher import SectorFetcher
from data.technical_calculator import TechnicalCalculator
from analysis.sector_screener import SectorScreener
from cache.manager import CacheManager


class SectorBacktester:
    """板块回测器"""
    
    def __init__(self, sector_fetcher: SectorFetcher, cache_manager: CacheManager):
        self.sector_fetcher = sector_fetcher
        self.cache_manager = cache_manager
        self.tech_calculator = TechnicalCalculator()
        # 创建组件
        from analysis.report_manager import ReportManager
        from data.enhanced_data_fetcher import EnhancedDataFetcher
        from analysis.enhanced_predictor import EnhancedPredictor
        
        report_manager = ReportManager()
        enhanced_data_fetcher = EnhancedDataFetcher(cache_manager)
        enhanced_predictor = EnhancedPredictor(enhanced_data_fetcher, self.tech_calculator)
        
        # 为回测使用基础评分，避免对可能不存在的增强数据的依赖
        self.sector_screener = SectorScreener(
            sector_fetcher, 
            self.tech_calculator, 
            enhanced_data_fetcher, 
            enhanced_predictor,
            report_manager
        )
        # 关闭增强预测用于回测
        self.sector_screener.use_enhanced_prediction = False
        self.logger = logging.getLogger(__name__)
        
    async def run_backtest(self, 
                          start_date: str, 
                          end_date: str, 
                          rebalance_frequency: str = "weekly",
                          top_n: int = 5,
                          initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """
        运行板块回测
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            rebalance_frequency: 再平衡频率 (daily, weekly, monthly)
            top_n: 每次筛选的Top N板块
            initial_capital: 初始资金
            
        Returns:
            Dict: 回测结果
        """
        try:
            self.logger.info(f"开始板块回测: {start_date} - {end_date}, Top {top_n}, 频率: {rebalance_frequency}")
            
            # 1. 生成回测时间点
            rebalance_dates = self._generate_rebalance_dates(start_date, end_date, rebalance_frequency)
            
            # 2. 初始化回测状态
            backtest_state = {
                'current_capital': initial_capital,
                'initial_capital': initial_capital,
                'portfolio': {},  # 当前持仓
                'transactions': [],  # 交易记录
                'daily_returns': [],  # 每日收益率
                'sector_allocations': [],  # 板块配置记录
                'performance_metrics': {}
            }
            
            # 3. 执行回测
            for i, rebalance_date in enumerate(rebalance_dates):
                self.logger.info(f"执行第{i+1}次再平衡: {rebalance_date}")
                
                # 获取当前时间点的板块数据
                current_data = await self._get_sector_data_at_date(rebalance_date)
                if not current_data:
                    continue
                    
                # 执行板块筛选
                screening_result = await self._run_sector_screening_at_date(current_data, top_n)
                if not screening_result:
                    continue
                    
                # 执行再平衡
                await self._rebalance_portfolio(backtest_state, screening_result, rebalance_date)
                
                # 记录配置
                backtest_state['sector_allocations'].append({
                    'date': rebalance_date,
                    'top_sectors': screening_result['top_sectors'],
                    'portfolio': backtest_state['portfolio'].copy()
                })
                
                # 计算到下一个再平衡日期的收益
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    await self._calculate_period_returns(backtest_state, rebalance_date, next_date)
            
            # 4. 计算最终回测指标
            final_metrics = self._calculate_final_metrics(backtest_state)
            backtest_state['performance_metrics'] = final_metrics
            
            # 5. 生成回测报告
            report_path = await self._generate_backtest_report(backtest_state, start_date, end_date)
            
            return {
                'backtest_summary': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'rebalance_frequency': rebalance_frequency,
                    'top_n': top_n,
                    'initial_capital': initial_capital,
                    'final_capital': backtest_state['current_capital'],
                    'total_return': final_metrics['total_return'],
                    'annualized_return': final_metrics['annualized_return'],
                    'max_drawdown': final_metrics['max_drawdown'],
                    'sharpe_ratio': final_metrics['sharpe_ratio']
                },
                'detailed_results': backtest_state,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"板块回测失败: {e}")
            return {'error': str(e)}
    
    def _generate_rebalance_dates(self, start_date: str, end_date: str, frequency: str) -> List[str]:
        """生成再平衡日期列表"""
        try:
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            dates = []
            current_date = start_dt
            
            if frequency == "daily":
                delta = timedelta(days=1)
            elif frequency == "weekly":
                delta = timedelta(weeks=1)
            elif frequency == "monthly":
                delta = timedelta(days=30)
            else:
                delta = timedelta(weeks=1)  # 默认每周
                
            while current_date <= end_dt:
                dates.append(current_date.strftime("%Y%m%d"))
                current_date += delta
                
            return dates
            
        except Exception as e:
            self.logger.error(f"生成再平衡日期失败: {e}")
            return []
    
    async def _get_sector_data_at_date(self, date: str) -> Optional[Dict[str, pd.DataFrame]]:
        """获取指定日期的板块数据"""
        try:
            # 获取以该日期为结束的30天数据
            end_dt = datetime.strptime(date, "%Y%m%d")
            start_dt = end_dt - timedelta(days=60)  # 获取更多数据确保有足够的历史
            
            start_str = start_dt.strftime("%Y%m%d")
            end_str = end_dt.strftime("%Y%m%d")
            
            sectors_data = await self.sector_fetcher.get_all_sectors_data((start_str, end_str))
            
            if not sectors_data:
                self.logger.warning(f"无法获取{date}的板块数据")
                return None
                
            return sectors_data
            
        except Exception as e:
            self.logger.error(f"获取{date}板块数据失败: {e}")
            return None
    
    async def _run_sector_screening_at_date(self, sectors_data: Dict[str, pd.DataFrame], top_n: int) -> Optional[Dict[str, Any]]:
        """在指定日期运行板块筛选"""
        try:
            # 直接使用传入的数据进行筛选，而不是重新获取
            if not sectors_data:
                return None
            
            # 计算各板块的综合评分
            sectors_scores = {}
            for sector_name, sector_df in sectors_data.items():
                if sector_df.empty or len(sector_df) < 20:  # 至少需要20天数据
                    continue
                
                # 使用技术计算器计算评分
                scores = self.tech_calculator.calculate_sector_strength_score(sector_df)
                if scores:
                    sectors_scores[sector_name] = {
                        'sector_name': sector_name,
                        'comprehensive_score': scores.get('comprehensive_score', 0),
                        'technical_score': scores.get('technical_score', 0),
                        'money_flow_score': scores.get('money_flow_score', 0),
                        'fundamental_score': scores.get('fundamental_score', 0),
                        'rotation_score': scores.get('rotation_score', 0)
                    }
            
            if not sectors_scores:
                return None
            
            # 排序并筛选Top N
            sorted_sectors = sorted(
                sectors_scores.values(),
                key=lambda x: x.get('comprehensive_score', 0),
                reverse=True
            )
            
            top_sectors = sorted_sectors[:top_n]
            
            # 计算板块间相关性
            correlation_matrix = self.tech_calculator.calculate_sector_correlation(sectors_data)
            
            # 分析板块轮动模式
            rotation_analysis = self.tech_calculator.analyze_sector_rotation_pattern(sectors_data)
            
            return {
                'screening_time': datetime.now().isoformat(),
                'period': '30d',
                'total_sectors_analyzed': len(sectors_scores),
                'top_sectors': top_sectors,
                'correlation_analysis': correlation_matrix,
                'rotation_analysis': rotation_analysis
            }
            
        except Exception as e:
            self.logger.error(f"板块筛选失败: {e}")
            return None
    
    async def _rebalance_portfolio(self, backtest_state: Dict[str, Any], 
                                 screening_result: Dict[str, Any], 
                                 rebalance_date: str) -> None:
        """执行投资组合再平衡"""
        try:
            top_sectors = screening_result.get('top_sectors', [])
            if not top_sectors:
                return
                
            # 计算每个板块的权重（等权重）
            sector_weight = 1.0 / len(top_sectors)
            current_capital = backtest_state['current_capital']
            
            # 清空当前持仓
            old_portfolio = backtest_state['portfolio'].copy()
            backtest_state['portfolio'] = {}
            
            # 计算新持仓
            for sector_info in top_sectors:
                sector_name = sector_info.get('sector_name', 'Unknown')
                allocation_amount = current_capital * sector_weight
                
                backtest_state['portfolio'][sector_name] = {
                    'allocation': allocation_amount,
                    'weight': sector_weight,
                    'sector_score': sector_info.get('comprehensive_score', 0)
                }
                
                # 记录交易
                if sector_name in old_portfolio:
                    old_allocation = old_portfolio[sector_name]['allocation']
                    if abs(allocation_amount - old_allocation) > 1000:  # 1000元以上的变化才记录
                        backtest_state['transactions'].append({
                            'date': rebalance_date,
                            'sector': sector_name,
                            'action': 'Rebalance',
                            'old_allocation': old_allocation,
                            'new_allocation': allocation_amount,
                            'change': allocation_amount - old_allocation
                        })
                else:
                    # 新增持仓
                    backtest_state['transactions'].append({
                        'date': rebalance_date,
                        'sector': sector_name,
                        'action': 'Buy',
                        'old_allocation': 0,
                        'new_allocation': allocation_amount,
                        'change': allocation_amount
                    })
            
            # 记录卖出交易
            for sector_name, old_info in old_portfolio.items():
                if sector_name not in backtest_state['portfolio']:
                    backtest_state['transactions'].append({
                        'date': rebalance_date,
                        'sector': sector_name,
                        'action': 'Sell',
                        'old_allocation': old_info['allocation'],
                        'new_allocation': 0,
                        'change': -old_info['allocation']
                    })
                    
        except Exception as e:
            self.logger.error(f"投资组合再平衡失败: {e}")
    
    async def _calculate_period_returns(self, backtest_state: Dict[str, Any], 
                                      start_date: str, end_date: str) -> None:
        """计算期间收益率"""
        try:
            # 获取期间数据
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            # 计算每日收益率
            current_date = start_dt
            while current_date <= end_dt:
                date_str = current_date.strftime("%Y%m%d")
                
                # 计算当日投资组合收益率
                daily_return = await self._calculate_daily_portfolio_return(backtest_state, date_str)
                # 如果_calculate_daily_portfolio_return抛出异常，这里会直接传播
                backtest_state['daily_returns'].append({
                    'date': date_str,
                    'return': daily_return,
                    'portfolio_value': backtest_state['current_capital']
                })
                
                # 更新投资组合价值
                backtest_state['current_capital'] *= (1 + daily_return / 100)
                
                current_date += timedelta(days=1)
                
        except Exception as e:
            self.logger.error(f"计算期间收益率失败: {e}")
            # 重新抛出异常，确保错误传播到上层
            raise
    
    async def _calculate_daily_portfolio_return(self, backtest_state: Dict[str, Any], date: str) -> Optional[float]:
        """计算单日投资组合收益率"""
        try:
            if not backtest_state['portfolio']:
                return 0.0
                
            total_return = 0.0
            total_weight = 0.0
            
            for sector_name, sector_info in backtest_state['portfolio'].items():
                weight = sector_info['weight']
                
                # 获取板块当日收益率
                sector_return = await self._get_sector_daily_return(sector_name, date)
                # 如果_get_sector_daily_return抛出异常，这里会直接传播
                total_return += sector_return * weight
                total_weight += weight
            
            if total_weight > 0:
                return total_return / total_weight
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"计算单日投资组合收益率失败: {e}")
            # 重新抛出异常，确保错误传播到上层
            raise
    
    async def _get_sector_daily_return(self, sector_name: str, date: str) -> Optional[float]:
        """获取板块单日收益率"""
        try:
            # 从缓存中获取板块历史数据
            cache_key = f"sector_history_{sector_name}"
            cached_data = await self.cache_manager.get(cache_key)
            
            # 检查缓存数据是否足够新
            target_date = datetime.strptime(date, "%Y%m%d")
            data_fresh_enough = False
            
            if cached_data is not None and not cached_data.empty:
                if 'date' in cached_data.columns:
                    try:
                        cached_data['date_dt'] = pd.to_datetime(cached_data['date'])
                        latest_date = cached_data['date_dt'].max()
                        # 检查缓存数据是否包含目标日期或更新的日期
                        if latest_date >= target_date:
                            data_fresh_enough = True
                            self.logger.debug(f"板块{sector_name}缓存数据足够新，最新日期: {latest_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        self.logger.warning(f"检查缓存数据新鲜度失败: {e}")
            
            if not data_fresh_enough:
                # 缓存数据不够新，获取更新的数据
                end_dt = target_date + timedelta(days=7)  # 获取比目标日期多7天的数据
                start_dt = end_dt - timedelta(days=120)   # 获取120天的数据
                
                start_str = start_dt.strftime("%Y%m%d")
                end_str = end_dt.strftime("%Y%m%d")
                
                self.logger.info(f"板块{sector_name}缓存数据不够新，重新获取数据: {start_str} - {end_str}")
                sectors_data = await self.sector_fetcher.get_all_sectors_data((start_str, end_str))
                if sector_name in sectors_data:
                    # 缓存数据
                    await self.cache_manager.set(cache_key, sectors_data[sector_name])
                    cached_data = sectors_data[sector_name]
                    self.logger.info(f"板块{sector_name}数据更新成功，最新日期: {sectors_data[sector_name]['date'].max()}")
                else:
                    error_msg = f"板块{sector_name}在{start_str}-{end_str}期间无数据"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            if cached_data is None or cached_data.empty:
                error_msg = f"板块{sector_name}缓存数据为空"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 尝试多种日期格式匹配
            date_formats = [
                target_date.strftime("%Y-%m-%d"),  # 2025-08-12
                target_date.strftime("%Y/%m/%d"),  # 2025/08/12
                target_date.strftime("%Y%m%d"),    # 20250812
                target_date.strftime("%Y-%m-%d %H:%M:%S"),  # 2025-08-12 00:00:00
            ]
            
            self.logger.debug(f"查找板块{sector_name}在{date}的收益率")
            self.logger.debug(f"数据列: {list(cached_data.columns)}")
            self.logger.debug(f"数据前5行: {cached_data.head()}")
            
            if 'date' in cached_data.columns:
                # 尝试多种日期格式匹配
                for date_format in date_formats:
                    exact_match = cached_data[cached_data['date'] == date_format]
                    if not exact_match.empty:
                        self.logger.debug(f"找到精确匹配，日期格式: {date_format}")
                        # 计算当日收益率
                        if 'pct_change' in cached_data.columns:
                            pct_change = float(exact_match['pct_change'].iloc[0])
                            self.logger.debug(f"板块{sector_name}在{date_format}的收益率: {pct_change}%")
                            return pct_change
                        else:
                            # 如果没有pct_change列，手动计算
                            idx = exact_match.index[0]
                            if idx > 0:
                                prev_close = cached_data.iloc[idx-1]['close']
                                curr_close = cached_data.iloc[idx]['close']
                                manual_return = ((curr_close / prev_close - 1) * 100) if prev_close > 0 else 0
                                self.logger.debug(f"板块{sector_name}在{date_format}的手动计算收益率: {manual_return}%")
                                return manual_return
                            return 0
                
                # 如果精确匹配失败，尝试找最近的交易日
                try:
                    cached_data['date_dt'] = pd.to_datetime(cached_data['date'])
                    target_dt = pd.Timestamp(target_date)
                    
                    # 找到最近的交易日
                    cached_data['date_diff'] = abs(cached_data['date_dt'] - target_dt)
                    nearest_idx = cached_data['date_diff'].idxmin()
                    nearest_date = cached_data.loc[nearest_idx, 'date_dt']
                    
                    self.logger.debug(f"板块{sector_name}在{date}未找到精确匹配，最近交易日: {nearest_date.strftime('%Y-%m-%d')}")
                    
                    # 如果最近的日期在3天内，使用该日期的收益率
                    if abs((nearest_date - target_dt).days) <= 3:
                        if 'pct_change' in cached_data.columns:
                            pct_change = float(cached_data.loc[nearest_idx, 'pct_change'])
                            self.logger.debug(f"板块{sector_name}在{nearest_date.strftime('%Y-%m-%d')}的收益率: {pct_change}%")
                            return pct_change
                        else:
                            # 手动计算收益率
                            if nearest_idx > 0:
                                prev_close = cached_data.iloc[nearest_idx-1]['close']
                                curr_close = cached_data.loc[nearest_idx, 'close']
                                manual_return = ((curr_close / prev_close - 1) * 100) if prev_close > 0 else 0
                                self.logger.debug(f"板块{sector_name}在{nearest_date.strftime('%Y-%m-%d')}的手动计算收益率: {manual_return}%")
                                return manual_return
                            return 0
                except Exception as e:
                    self.logger.warning(f"尝试找最近交易日失败: {e}")
            
            # 如果所有方法都失败，提供详细的错误信息
            error_msg = f"板块{sector_name}在{date}无法获取收益率，数据不可用。"
            error_msg += f" 可用日期范围: {cached_data['date'].min()} 到 {cached_data['date'].max()}"
            error_msg += f" 数据列: {list(cached_data.columns)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        except Exception as e:
            error_msg = f"获取板块{sector_name}在{date}的收益率失败: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _calculate_final_metrics(self, backtest_state: Dict[str, Any]) -> Dict[str, float]:
        """计算最终回测指标"""
        try:
            daily_returns = backtest_state['daily_returns']
            if not daily_returns:
                return {}
                
            # 计算总收益率
            initial_capital = backtest_state['initial_capital']
            final_capital = backtest_state['current_capital']
            total_return = (final_capital / initial_capital - 1) * 100
            
            # 计算年化收益率
            if len(daily_returns) > 0:
                start_date = datetime.strptime(daily_returns[0]['date'], "%Y%m%d")
                end_date = datetime.strptime(daily_returns[-1]['date'], "%Y%m%d")
                days = (end_date - start_date).days
                if days > 0:
                    annualized_return = ((final_capital / initial_capital) ** (365 / days) - 1) * 100
                else:
                    annualized_return = 0
            else:
                annualized_return = 0
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown(daily_returns)
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            
            # 计算波动率
            volatility = self._calculate_volatility(daily_returns)
            
            return {
                'total_return': round(total_return, 2),
                'annualized_return': round(annualized_return, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'volatility': round(volatility, 2),
                'total_trading_days': len(daily_returns),
                'win_rate': self._calculate_win_rate(daily_returns)
            }
            
        except Exception as e:
            self.logger.error(f"计算最终指标失败: {e}")
            return {}
    
    def _calculate_max_drawdown(self, daily_returns: List[Dict[str, Any]]) -> float:
        """计算最大回撤"""
        try:
            if not daily_returns:
                return 0.0
                
            portfolio_values = [ret['portfolio_value'] for ret in daily_returns]
            max_dd = 0.0
            peak = portfolio_values[0]
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
                    
            return max_dd
            
        except Exception as e:
            self.logger.error(f"计算最大回撤失败: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, daily_returns: List[Dict[str, Any]]) -> float:
        """计算夏普比率"""
        try:
            if len(daily_returns) < 2:
                return 0.0
                
            returns = [ret['return'] for ret in daily_returns]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
                
            # 假设无风险利率为3%
            risk_free_rate = 3.0 / 365  # 日化无风险利率
            sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(365)
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"计算夏普比率失败: {e}")
            return 0.0
    
    def _calculate_volatility(self, daily_returns: List[Dict[str, Any]]) -> float:
        """计算波动率"""
        try:
            if len(daily_returns) < 2:
                return 0.0
                
            returns = [ret['return'] for ret in daily_returns]
            volatility = np.std(returns) * np.sqrt(365)  # 年化波动率
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"计算波动率失败: {e}")
            return 0.0
    
    def _calculate_win_rate(self, daily_returns: List[Dict[str, Any]]) -> float:
        """计算胜率"""
        try:
            if not daily_returns:
                return 0.0
                
            positive_days = sum(1 for ret in daily_returns if ret['return'] > 0)
            win_rate = (positive_days / len(daily_returns)) * 100
            
            return round(win_rate, 1)
            
        except Exception as e:
            self.logger.error(f"计算胜率失败: {e}")
            return 0.0
    
    async def _generate_backtest_report(self, backtest_state: Dict[str, Any], 
                                      start_date: str, end_date: str) -> str:
        """生成回测报告"""
        try:
            # 创建报告目录
            import os
            report_dir = "reports/backtest"
            os.makedirs(report_dir, exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"sector_backtest_{start_date}_{end_date}_{timestamp}.md"
            report_path = os.path.join(report_dir, report_filename)
            
            # 生成报告内容
            report_content = self._format_backtest_report(backtest_state, start_date, end_date)
            
            # 保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            self.logger.info(f"回测报告已保存: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成回测报告失败: {e}")
            return ""
    
    def _format_backtest_report(self, backtest_state: Dict[str, Any], 
                               start_date: str, end_date: str) -> str:
        """格式化回测报告"""
        try:
            metrics = backtest_state['performance_metrics']
            
            report = f"""# 板块回测报告

## 回测概览
- **回测期间**: {start_date} - {end_date}
- **初始资金**: {backtest_state['initial_capital']:,.0f} 元
- **最终资金**: {backtest_state['current_capital']:,.0f} 元
- **总收益率**: {metrics.get('total_return', 0):.2f}%
- **年化收益率**: {metrics.get('annualized_return', 0):.2f}%

## 风险指标
- **最大回撤**: {metrics.get('max_drawdown', 0):.2f}%
- **夏普比率**: {metrics.get('sharpe_ratio', 0):.3f}
- **年化波动率**: {metrics.get('volatility', 0):.2f}%
- **胜率**: {metrics.get('win_rate', 0):.1f}%
- **总交易天数**: {metrics.get('total_trading_days', 0)} 天

## 投资组合配置记录
"""
            
            for allocation in backtest_state['sector_allocations']:
                report += f"\n### {allocation['date']}\n"
                report += "| 板块 | 权重 | 评分 |\n"
                report += "|------|------|------|\n"
                
                for sector_name, sector_info in allocation['portfolio'].items():
                    report += f"| {sector_name} | {sector_info['weight']*100:.1f}% | {sector_info['sector_score']:.1f} |\n"
            
            report += "\n## 交易记录\n"
            report += "| 日期 | 板块 | 操作 | 变化金额 |\n"
            report += "|------|------|------|----------|\n"
            
            for transaction in backtest_state['transactions']:
                report += f"| {transaction['date']} | {transaction['sector']} | {transaction['action']} | {transaction['change']:,.0f} |\n"
            
            report += f"\n## 生成时间\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"格式化回测报告失败: {e}")
            return f"# 回测报告生成失败\n\n错误信息: {str(e)}"
