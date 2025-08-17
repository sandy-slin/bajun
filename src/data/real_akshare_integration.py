#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKShare真实数据集成测试
验证与akshare库的数据获取效果，确保数据真实性和时效性

功能:
1. A股股票基础数据获取
2. 申万行业指数数据获取
3. 实时行情数据获取
4. 数据质量验证和异常处理
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logging.warning("AKShare未安装，请运行: pip install akshare")

from ..validation.real_data_validator import RealDataValidator


class RealAKShareIntegration:
    """AKShare真实数据集成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_validator = RealDataValidator()
        
        if not AKSHARE_AVAILABLE:
            raise ImportError("AKShare库未安装，无法获取真实数据")
        
        # 数据获取配置
        self.data_config = {
            'stock_info_timeout': 30,      # 股票信息获取超时
            'sector_data_timeout': 60,     # 板块数据获取超时
            'retry_attempts': 3,           # 重试次数
            'retry_delay': 2,              # 重试延迟秒数
            'max_daily_requests': 1000     # 每日最大请求数限制
        }
        
        # 申万一级行业代码映射
        self.sw_industry_mapping = {
            '银行': '801780',
            '非银金融': '801790', 
            '房地产': '801180',
            '食品饮料': '801120',
            '医药生物': '801150',
            '电子': '801080',
            '计算机': '801750',
            '通信': '801760',
            '电气设备': '801710',
            '机械设备': '801890',
            '汽车': '801880',
            '化工': '801130',
            '钢铁': '801040',
            '有色金属': '801050',
            '建筑材料': '801170',
            '建筑装饰': '801720',
            '电力设备': '801710',
            '国防军工': '801740',
            '农林牧渔': '801010',
            '采掘': '801030',
            '公用事业': '801160',
            '交通运输': '801200',
            '轻工制造': '801140',
            '纺织服装': '801110',
            '商业贸易': '801210',
            '休闲服务': '801230',
            '家用电器': '801770',
            '传媒': '801780',
            '综合': '801230'
        }
        
        # 请求计数器 (简单实现)
        self.request_count = 0
        self.last_reset_date = datetime.now().date()
    
    async def test_akshare_connectivity(self) -> Dict:
        """测试AKShare连接性和数据可用性"""
        try:
            self.logger.info("开始测试AKShare连接性...")
            
            test_results = {
                'connectivity_test': await self._test_basic_connectivity(),
                'stock_data_test': await self._test_stock_data_access(),
                'sector_data_test': await self._test_sector_data_access(),
                'data_quality_test': await self._test_data_quality(),
                'performance_test': await self._test_performance()
            }
            
            # 综合评估
            overall_success = all(
                result.get('success', False) for result in test_results.values()
            )
            
            return {
                'overall_success': overall_success,
                'test_timestamp': datetime.now().isoformat(),
                'detailed_results': test_results,
                'recommendations': self._generate_integration_recommendations(test_results)
            }
            
        except Exception as e:
            self.logger.error(f"AKShare连接性测试失败: {e}")
            return {
                'overall_success': False,
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }
    
    async def _test_basic_connectivity(self) -> Dict:
        """测试基础连接"""
        try:
            self.logger.info("测试AKShare基础连接...")
            
            # 测试获取A股基本信息
            stock_info = await self._safe_akshare_call(
                ak.stock_info_a_code_name
            )
            
            if stock_info is not None and len(stock_info) > 0:
                # 验证数据真实性
                validation_result = await self.data_validator.validate_and_ensure_real_data(
                    stock_info.to_dict('records'), 'akshare_stock_info'
                )
                
                return {
                    'success': True,
                    'stock_count': len(stock_info),
                    'data_quality_score': validation_result['quality_score'],
                    'sample_data': stock_info.head(3).to_dict('records')
                }
            else:
                return {
                    'success': False,
                    'error': '无法获取股票基本信息'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'基础连接测试失败: {str(e)}'
            }
    
    async def _test_stock_data_access(self) -> Dict:
        """测试股票数据获取"""
        try:
            self.logger.info("测试股票历史数据获取...")
            
            # 测试几个主要股票的数据获取
            test_stocks = ['000001', '000002', '600519', '000858', '300750']
            successful_stocks = []
            failed_stocks = []
            
            for stock_code in test_stocks:
                try:
                    # 获取近30天数据
                    end_date = datetime.now().strftime('%Y%m%d')
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                    
                    stock_data = await self._safe_akshare_call(
                        ak.stock_zh_a_hist,
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=""
                    )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        # 验证数据质量
                        validation_result = await self.data_validator.validate_and_ensure_real_data(
                            stock_data.to_dict('records'), f'akshare_stock_{stock_code}'
                        )
                        
                        successful_stocks.append({
                            'stock_code': stock_code,
                            'data_points': len(stock_data),
                            'quality_score': validation_result['quality_score'],
                            'latest_date': stock_data.iloc[-1]['日期'] if '日期' in stock_data.columns else 'unknown'
                        })
                    else:
                        failed_stocks.append(stock_code)
                        
                except Exception as e:
                    self.logger.warning(f"股票{stock_code}数据获取失败: {e}")
                    failed_stocks.append(stock_code)
            
            success_rate = len(successful_stocks) / len(test_stocks)
            
            return {
                'success': success_rate >= 0.6,  # 60%成功率及格
                'success_rate': success_rate,
                'successful_stocks': successful_stocks,
                'failed_stocks': failed_stocks,
                'average_quality_score': np.mean([s['quality_score'] for s in successful_stocks]) if successful_stocks else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'股票数据测试失败: {str(e)}'
            }
    
    async def _test_sector_data_access(self) -> Dict:
        """测试板块数据获取"""
        try:
            self.logger.info("测试申万行业数据获取...")
            
            # 测试主要行业指数
            test_sectors = ['银行', '食品饮料', '医药生物', '电子', '非银金融']
            successful_sectors = []
            failed_sectors = []
            
            for sector_name in test_sectors:
                try:
                    if sector_name in self.sw_industry_mapping:
                        sector_code = self.sw_industry_mapping[sector_name]
                        
                        # 获取申万行业指数数据
                        end_date = datetime.now().strftime('%Y%m%d')
                        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                        
                        sector_data = await self._safe_akshare_call(
                            ak.index_hist_sw,
                            symbol=sector_code,
                            period="day"
                        )
                        
                        if sector_data is not None and len(sector_data) > 0:
                            # 验证数据质量
                            validation_result = await self.data_validator.validate_and_ensure_real_data(
                                sector_data.to_dict('records'), f'akshare_sector_{sector_name}'
                            )
                            
                            successful_sectors.append({
                                'sector_name': sector_name,
                                'sector_code': sector_code,
                                'data_points': len(sector_data),
                                'quality_score': validation_result['quality_score'],
                                'latest_date': sector_data.iloc[-1]['日期'] if '日期' in sector_data.columns else 'unknown'
                            })
                        else:
                            failed_sectors.append(sector_name)
                    else:
                        failed_sectors.append(sector_name)
                        
                except Exception as e:
                    self.logger.warning(f"板块{sector_name}数据获取失败: {e}")
                    failed_sectors.append(sector_name)
            
            success_rate = len(successful_sectors) / len(test_sectors)
            
            return {
                'success': success_rate >= 0.6,
                'success_rate': success_rate,
                'successful_sectors': successful_sectors,
                'failed_sectors': failed_sectors,
                'average_quality_score': np.mean([s['quality_score'] for s in successful_sectors]) if successful_sectors else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'板块数据测试失败: {str(e)}'
            }
    
    async def _test_data_quality(self) -> Dict:
        """测试数据质量"""
        try:
            self.logger.info("测试数据质量...")
            
            # 获取测试数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            test_data = await self._safe_akshare_call(
                ak.stock_zh_a_hist,
                symbol='000001',  # 平安银行作为测试标的
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            
            if test_data is None or len(test_data) == 0:
                return {
                    'success': False,
                    'error': '无法获取测试数据'
                }
            
            # 质量检查项目
            quality_checks = {
                'data_completeness': self._check_data_completeness(test_data),
                'data_consistency': self._check_data_consistency(test_data),
                'data_timeliness': self._check_data_timeliness(test_data),
                'data_accuracy': self._check_data_accuracy(test_data)
            }
            
            overall_quality = np.mean([check['score'] for check in quality_checks.values()])
            
            return {
                'success': overall_quality >= 0.8,
                'overall_quality_score': overall_quality,
                'quality_checks': quality_checks,
                'data_sample_size': len(test_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'数据质量测试失败: {str(e)}'
            }
    
    async def _test_performance(self) -> Dict:
        """测试性能"""
        try:
            self.logger.info("测试数据获取性能...")
            
            # 测试单次请求耗时
            start_time = datetime.now()
            
            test_data = await self._safe_akshare_call(
                ak.stock_zh_a_hist,
                symbol='000001',
                period="daily",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d'),
                adjust=""
            )
            
            single_request_time = (datetime.now() - start_time).total_seconds()
            
            # 测试并发请求 (简化版)
            concurrent_start = datetime.now()
            
            concurrent_results = await asyncio.gather(
                self._safe_akshare_call(ak.stock_zh_a_hist, symbol='000001', period="daily"),
                self._safe_akshare_call(ak.stock_zh_a_hist, symbol='000002', period="daily"),
                self._safe_akshare_call(ak.stock_zh_a_hist, symbol='600519', period="daily"),
                return_exceptions=True
            )
            
            concurrent_time = (datetime.now() - concurrent_start).total_seconds()
            successful_concurrent = sum(1 for result in concurrent_results if not isinstance(result, Exception))
            
            return {
                'success': single_request_time < 10 and concurrent_time < 30,  # 性能标准
                'single_request_time': single_request_time,
                'concurrent_request_time': concurrent_time,
                'concurrent_success_rate': successful_concurrent / len(concurrent_results),
                'performance_grade': self._grade_performance(single_request_time, concurrent_time)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'性能测试失败: {str(e)}'
            }
    
    async def _safe_akshare_call(self, func, **kwargs):
        """安全的AKShare API调用，包含重试机制"""
        for attempt in range(self.data_config['retry_attempts']):
            try:
                # 检查请求限制
                if not self._check_request_limit():
                    raise Exception("达到每日请求限制")
                
                # 执行请求
                self.request_count += 1
                result = await asyncio.to_thread(func, **kwargs)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"AKShare请求失败 (尝试{attempt+1}/{self.data_config['retry_attempts']}): {e}")
                
                if attempt < self.data_config['retry_attempts'] - 1:
                    await asyncio.sleep(self.data_config['retry_delay'])
                else:
                    raise e
        
        return None
    
    def _check_request_limit(self) -> bool:
        """检查请求限制"""
        current_date = datetime.now().date()
        
        # 重置每日计数器
        if current_date != self.last_reset_date:
            self.request_count = 0
            self.last_reset_date = current_date
        
        return self.request_count < self.data_config['max_daily_requests']
    
    def _check_data_completeness(self, data: pd.DataFrame) -> Dict:
        """检查数据完整性"""
        try:
            required_columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return {
                    'score': 0.0,
                    'issue': f'缺少必要列: {missing_columns}'
                }
            
            # 检查缺失值
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            completeness_score = max(0, 1 - missing_ratio * 2)
            
            return {
                'score': completeness_score,
                'missing_ratio': missing_ratio,
                'total_data_points': len(data)
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _check_data_consistency(self, data: pd.DataFrame) -> Dict:
        """检查数据一致性"""
        try:
            consistency_issues = []
            
            # 检查价格逻辑一致性
            if all(col in data.columns for col in ['开盘', '收盘', '最高', '最低']):
                # 最高价应该 >= 开盘价、收盘价
                high_issues = ((data['最高'] < data['开盘']) | (data['最高'] < data['收盘'])).sum()
                if high_issues > 0:
                    consistency_issues.append(f'{high_issues}条记录最高价异常')
                
                # 最低价应该 <= 开盘价、收盘价
                low_issues = ((data['最低'] > data['开盘']) | (data['最低'] > data['收盘'])).sum()
                if low_issues > 0:
                    consistency_issues.append(f'{low_issues}条记录最低价异常')
            
            # 检查成交量合理性
            if '成交量' in data.columns:
                negative_volume = (data['成交量'] < 0).sum()
                if negative_volume > 0:
                    consistency_issues.append(f'{negative_volume}条记录成交量为负')
            
            consistency_score = max(0, 1 - len(consistency_issues) * 0.2)
            
            return {
                'score': consistency_score,
                'issues': consistency_issues
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _check_data_timeliness(self, data: pd.DataFrame) -> Dict:
        """检查数据时效性"""
        try:
            if '日期' not in data.columns:
                return {
                    'score': 0.0,
                    'error': '无日期列'
                }
            
            # 获取最新数据日期
            latest_date = pd.to_datetime(data['日期']).max()
            current_time = datetime.now()
            
            # 计算延迟时间
            delay_hours = (current_time - latest_date).total_seconds() / 3600
            
            # 工作日内24小时，周末72小时内为及格
            max_delay = 72 if current_time.weekday() >= 5 else 24
            timeliness_score = max(0, 1 - delay_hours / max_delay)
            
            return {
                'score': timeliness_score,
                'latest_date': latest_date.isoformat(),
                'delay_hours': delay_hours
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _check_data_accuracy(self, data: pd.DataFrame) -> Dict:
        """检查数据准确性"""
        try:
            accuracy_issues = []
            
            # 检查价格范围合理性
            if '收盘' in data.columns:
                # A股价格通常在0.01-1000元之间
                unreasonable_prices = ((data['收盘'] < 0.01) | (data['收盘'] > 1000)).sum()
                if unreasonable_prices > 0:
                    accuracy_issues.append(f'{unreasonable_prices}条记录价格异常')
            
            # 检查成交量范围
            if '成交量' in data.columns:
                # 成交量过大可能有问题
                high_volume = (data['成交量'] > 1e10).sum()  # 100亿股
                if high_volume > 0:
                    accuracy_issues.append(f'{high_volume}条记录成交量过大')
            
            accuracy_score = max(0, 1 - len(accuracy_issues) * 0.3)
            
            return {
                'score': accuracy_score,
                'issues': accuracy_issues
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _grade_performance(self, single_time: float, concurrent_time: float) -> str:
        """评估性能等级"""
        if single_time < 3 and concurrent_time < 10:
            return 'excellent'
        elif single_time < 5 and concurrent_time < 20:
            return 'good'
        elif single_time < 10 and concurrent_time < 30:
            return 'acceptable'
        else:
            return 'poor'
    
    def _generate_integration_recommendations(self, test_results: Dict) -> List[str]:
        """生成集成建议"""
        recommendations = []
        
        try:
            # 连接性建议
            if not test_results['connectivity_test']['success']:
                recommendations.append('建议检查网络连接和AKShare版本')
            
            # 数据访问建议
            stock_success = test_results['stock_data_test'].get('success_rate', 0)
            if stock_success < 0.8:
                recommendations.append('股票数据获取成功率偏低，建议增加重试机制')
            
            sector_success = test_results['sector_data_test'].get('success_rate', 0)
            if sector_success < 0.8:
                recommendations.append('板块数据获取成功率偏低，建议使用备用数据源')
            
            # 质量建议
            quality_score = test_results['data_quality_test'].get('overall_quality_score', 0)
            if quality_score < 0.8:
                recommendations.append('数据质量偏低，建议增强数据清洗和验证')
            
            # 性能建议
            perf_grade = test_results['performance_test'].get('performance_grade', 'poor')
            if perf_grade in ['poor', 'acceptable']:
                recommendations.append('数据获取性能可优化，建议实现缓存机制')
            
            if not recommendations:
                recommendations.append('AKShare集成测试通过，可以正式使用')
                
        except Exception as e:
            recommendations.append(f'建议生成异常: {str(e)}')
        
        return recommendations
    
    async def get_real_stock_data(self, stock_code: str, days: int = 30) -> Optional[List[Dict]]:
        """获取真实股票数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            stock_data = await self._safe_akshare_call(
                ak.stock_zh_a_hist,
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            
            if stock_data is not None and len(stock_data) > 0:
                # 验证数据真实性
                await self.data_validator.validate_and_ensure_real_data(
                    stock_data.to_dict('records'), f'akshare_real_stock_{stock_code}'
                )
                
                # 转换为标准格式
                standardized_data = []
                for _, row in stock_data.iterrows():
                    standardized_data.append({
                        'date': row['日期'],
                        'open': float(row['开盘']),
                        'high': float(row['最高']),
                        'low': float(row['最低']),
                        'close': float(row['收盘']),
                        'volume': int(row['成交量']) if pd.notna(row['成交量']) else 0,
                        'amount': float(row['成交额']) if '成交额' in row and pd.notna(row['成交额']) else 0
                    })
                
                return standardized_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取股票{stock_code}真实数据失败: {e}")
            return None
    
    async def get_real_sector_data(self, sector_name: str, days: int = 30) -> Optional[List[Dict]]:
        """获取真实板块数据"""
        try:
            if sector_name not in self.sw_industry_mapping:
                self.logger.warning(f"未知板块: {sector_name}")
                return None
            
            sector_code = self.sw_industry_mapping[sector_name]
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            sector_data = await self._safe_akshare_call(
                ak.index_hist_sw,
                symbol=sector_code,
                period="day"
            )
            
            if sector_data is not None and len(sector_data) > 0:
                # 验证数据真实性
                await self.data_validator.validate_and_ensure_real_data(
                    sector_data.to_dict('records'), f'akshare_real_sector_{sector_name}'
                )
                
                # 转换为标准格式
                standardized_data = []
                for _, row in sector_data.iterrows():
                    standardized_data.append({
                        'date': row['日期'],
                        'open': float(row['开盘']),
                        'high': float(row['最高']),
                        'low': float(row['最低']),
                        'close': float(row['收盘']),
                        'volume': int(row['成交量']) if pd.notna(row['成交量']) else 0,
                        'amount': float(row['成交额']) if '成交额' in row and pd.notna(row['成交额']) else 0
                    })
                
                return standardized_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取板块{sector_name}真实数据失败: {e}")
            return None