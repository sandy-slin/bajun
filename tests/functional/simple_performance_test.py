#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的历史性能基准测试
直接使用AKShare数据进行历史回测验证
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("❌ AKShare未安装")

class SimplePerformanceBaseline:
    """简化的性能基准测试器"""
    
    def __init__(self):
        self.validation_results = []
        
        # 申万行业代码映射
        self.sector_mapping = {
            '银行': '801780',
            '食品饮料': '801120',
            '医药生物': '801150',
            '电子': '801080',
            '非银金融': '801790'
        }
    
    async def run_baseline_test(self):
        """运行基准测试"""
        print("🎯 开始历史性能基准测试")
        print("=" * 50)
        
        # 验证时间点
        test_dates = [
            '20240701',
            '20240801', 
            '20240901',
            '20241001',
            '20241101'
        ]
        
        total_results = []
        
        for i, test_date in enumerate(test_dates, 1):
            print(f"\n📅 测试时间点 {i}/5: {test_date}")
            
            try:
                result = await self._test_single_period(test_date)
                if result:
                    total_results.append(result)
                    print(f"   ✅ 测试完成")
                else:
                    print(f"   ❌ 测试失败")
            except Exception as e:
                print(f"   ❌ 测试异常: {e}")
        
        # 分析结果
        if total_results:
            summary = self._analyze_results(total_results)
            self._display_summary(summary)
            
            # 保存结果
            self._save_results(total_results, summary)
            return True
        else:
            print("❌ 所有测试均失败")
            return False
    
    async def _test_single_period(self, test_date):
        """测试单个时间点"""
        try:
            # 1. 板块分析测试
            sector_results = await self._test_sector_prediction(test_date)
            
            # 2. 股票选择测试  
            stock_results = await self._test_stock_selection(test_date)
            
            # 3. 组合表现测试
            portfolio_results = await self._test_portfolio_performance(
                test_date, stock_results
            )
            
            return {
                'test_date': test_date,
                'sector_results': sector_results,
                'stock_results': stock_results,
                'portfolio_results': portfolio_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"   单期测试失败: {e}")
            return None
    
    async def _test_sector_prediction(self, test_date):
        """测试板块预测效果"""
        print("     🏢 测试板块预测...")
        
        sector_predictions = []
        actual_performance = []
        
        for sector_name, sector_code in self.sector_mapping.items():
            try:
                # 获取板块历史数据
                sector_data = ak.index_hist_sw(symbol=sector_code, period='day')
                
                if sector_data is not None and len(sector_data) > 100:
                    # 找到测试日期位置
                    sector_data['日期'] = pd.to_datetime(sector_data['日期'])
                    test_datetime = pd.to_datetime(test_date)
                    
                    # 找到测试日期之前的数据
                    historical_data = sector_data[sector_data['日期'] <= test_datetime]
                    future_data = sector_data[sector_data['日期'] > test_datetime]
                    
                    if len(historical_data) >= 20 and len(future_data) >= 5:
                        # 简单动量预测
                        recent_returns = historical_data['收盘'].pct_change().tail(20).mean()
                        prediction_score = 50 + recent_returns * 1000  # 转换为评分
                        
                        # 实际5日后表现
                        start_price = historical_data['收盘'].iloc[-1]
                        end_price = future_data['收盘'].iloc[4] if len(future_data) >= 5 else future_data['收盘'].iloc[-1]
                        actual_return = (end_price / start_price - 1) * 100
                        
                        sector_predictions.append(prediction_score)
                        actual_performance.append(actual_return)
                        
            except Exception as e:
                print(f"     板块{sector_name}测试失败: {e}")
                continue
        
        if len(sector_predictions) >= 3:
            # 计算预测准确率
            prediction_accuracy = self._calculate_prediction_accuracy(
                sector_predictions, actual_performance
            )
            
            return {
                'predictions': sector_predictions,
                'actual': actual_performance, 
                'accuracy': prediction_accuracy,
                'sectors_tested': len(sector_predictions)
            }
        
        return {'accuracy': 0, 'sectors_tested': 0}
    
    async def _test_stock_selection(self, test_date):
        """测试股票选择效果"""
        print("     📈 测试股票选择...")
        
        # 测试股票池
        test_stocks = ['000001', '000002', '600519', '000858', '300750']
        
        selected_stocks = []
        actual_returns = []
        
        for stock_code in test_stocks:
            try:
                # 获取股票历史数据
                end_date = (pd.to_datetime(test_date) + timedelta(days=10)).strftime('%Y%m%d')
                start_date = (pd.to_datetime(test_date) - timedelta(days=30)).strftime('%Y%m%d')
                
                stock_data = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period='daily',
                    start_date=start_date,
                    end_date=end_date,
                    adjust=''
                )
                
                if stock_data is not None and len(stock_data) > 10:
                    stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                    test_datetime = pd.to_datetime(test_date)
                    
                    # 分离历史和未来数据
                    historical = stock_data[stock_data['日期'] <= test_datetime]
                    future = stock_data[stock_data['日期'] > test_datetime]
                    
                    if len(historical) >= 5 and len(future) >= 3:
                        # 简单选股评分 (基于动量)
                        recent_momentum = historical['收盘'].pct_change().tail(5).mean()
                        selection_score = 50 + recent_momentum * 1000
                        
                        # 实际表现
                        start_price = historical['收盘'].iloc[-1]
                        end_price = future['收盘'].iloc[2] if len(future) >= 3 else future['收盘'].iloc[-1]
                        actual_return = (end_price / start_price - 1) * 100
                        
                        selected_stocks.append({
                            'code': stock_code,
                            'score': selection_score,
                            'actual_return': actual_return
                        })
                        actual_returns.append(actual_return)
                        
            except Exception as e:
                print(f"     股票{stock_code}测试失败: {e}")
                continue
        
        if selected_stocks:
            # 选择前3只股票
            selected_stocks.sort(key=lambda x: x['score'], reverse=True)
            top_stocks = selected_stocks[:3]
            
            # 计算胜率
            positive_returns = sum(1 for stock in top_stocks if stock['actual_return'] > 0)
            win_rate = positive_returns / len(top_stocks) if top_stocks else 0
            
            return {
                'selected_stocks': top_stocks,
                'win_rate': win_rate,
                'average_return': np.mean([s['actual_return'] for s in top_stocks])
            }
        
        return {'win_rate': 0, 'average_return': 0}
    
    async def _test_portfolio_performance(self, test_date, stock_results):
        """测试组合表现"""
        print("     💼 测试组合表现...")
        
        if not stock_results or 'selected_stocks' not in stock_results:
            return {'total_return': 0, 'sharpe_ratio': 0}
        
        selected_stocks = stock_results['selected_stocks']
        if not selected_stocks:
            return {'total_return': 0, 'sharpe_ratio': 0}
        
        # 等权重组合
        weight = 1.0 / len(selected_stocks)
        portfolio_return = sum(stock['actual_return'] * weight for stock in selected_stocks)
        
        # 简化的夏普比率 (假设无风险利率为3%)
        returns = [stock['actual_return'] for stock in selected_stocks]
        portfolio_std = np.std(returns) if len(returns) > 1 else 10
        sharpe_ratio = (portfolio_return - 3) / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'total_return': portfolio_return,
            'sharpe_ratio': sharpe_ratio,
            'selected_count': len(selected_stocks)
        }
    
    def _calculate_prediction_accuracy(self, predictions, actual):
        """计算预测准确率"""
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0
        
        # 方向预测准确率
        correct_predictions = 0
        for pred, actual_val in zip(predictions, actual):
            pred_direction = 1 if pred > 50 else -1
            actual_direction = 1 if actual_val > 0 else -1
            
            if pred_direction == actual_direction:
                correct_predictions += 1
        
        return correct_predictions / len(predictions)
    
    def _analyze_results(self, results):
        """分析测试结果"""
        print("\n📊 分析测试结果...")
        
        sector_accuracies = []
        stock_win_rates = []
        portfolio_returns = []
        
        for result in results:
            if 'sector_results' in result:
                sector_accuracies.append(result['sector_results'].get('accuracy', 0))
            
            if 'stock_results' in result:
                stock_win_rates.append(result['stock_results'].get('win_rate', 0))
            
            if 'portfolio_results' in result:
                portfolio_returns.append(result['portfolio_results'].get('total_return', 0))
        
        # 计算统计指标
        summary = {
            'validation_periods': len(results),
            'sector_analysis': {
                'average_accuracy': np.mean(sector_accuracies) if sector_accuracies else 0,
                'best_accuracy': max(sector_accuracies) if sector_accuracies else 0,
                'worst_accuracy': min(sector_accuracies) if sector_accuracies else 0,
                'stability': np.std(sector_accuracies) if len(sector_accuracies) > 1 else 0
            },
            'stock_selection': {
                'average_win_rate': np.mean(stock_win_rates) if stock_win_rates else 0,
                'best_win_rate': max(stock_win_rates) if stock_win_rates else 0,
                'worst_win_rate': min(stock_win_rates) if stock_win_rates else 0,
                'stability': np.std(stock_win_rates) if len(stock_win_rates) > 1 else 0
            },
            'portfolio_performance': {
                'average_return': np.mean(portfolio_returns) if portfolio_returns else 0,
                'best_return': max(portfolio_returns) if portfolio_returns else 0,
                'worst_return': min(portfolio_returns) if portfolio_returns else 0,
                'positive_periods': sum(1 for r in portfolio_returns if r > 0),
                'volatility': np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0
            }
        }
        
        # 计算整体评级
        avg_sector_acc = summary['sector_analysis']['average_accuracy']
        avg_stock_win = summary['stock_selection']['average_win_rate']
        avg_portfolio_ret = summary['portfolio_performance']['average_return']
        
        overall_score = (avg_sector_acc * 40 + avg_stock_win * 40 + 
                        min(max(avg_portfolio_ret + 50, 0), 100) * 20) / 100 * 100
        
        summary['overall_assessment'] = {
            'score': overall_score,
            'grade': self._calculate_grade(overall_score),
            'readiness': self._assess_readiness(overall_score)
        }
        
        return summary
    
    def _calculate_grade(self, score):
        """计算等级"""
        if score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def _assess_readiness(self, score):
        """评估就绪状态"""
        if score >= 75:
            return 'production_ready'
        elif score >= 65:
            return 'beta_ready'
        elif score >= 55:
            return 'development_stage'
        else:
            return 'needs_improvement'
    
    def _display_summary(self, summary):
        """显示结果摘要"""
        print("\n🎯 历史性能基准测试结果")
        print("=" * 50)
        
        print(f"📊 验证期数: {summary['validation_periods']}")
        
        # 板块分析结果
        sector = summary['sector_analysis']
        print(f"\n🏢 板块分析表现:")
        print(f"   平均准确率: {sector['average_accuracy']:.1%}")
        print(f"   最佳准确率: {sector['best_accuracy']:.1%}")
        print(f"   准确率稳定性: {sector['stability']:.3f}")
        
        # 股票选择结果
        stock = summary['stock_selection'] 
        print(f"\n📈 股票选择表现:")
        print(f"   平均胜率: {stock['average_win_rate']:.1%}")
        print(f"   最佳胜率: {stock['best_win_rate']:.1%}")
        print(f"   胜率稳定性: {stock['stability']:.3f}")
        
        # 组合管理结果
        portfolio = summary['portfolio_performance']
        print(f"\n💼 组合管理表现:")
        print(f"   平均收益: {portfolio['average_return']:.2f}%")
        print(f"   最佳收益: {portfolio['best_return']:.2f}%")
        print(f"   盈利期数: {portfolio['positive_periods']}/{summary['validation_periods']}")
        print(f"   收益波动性: {portfolio['volatility']:.2f}%")
        
        # 整体评估
        overall = summary['overall_assessment']
        print(f"\n🏆 整体评估:")
        print(f"   综合得分: {overall['score']:.1f}")
        print(f"   系统等级: {overall['grade']}")
        print(f"   就绪状态: {overall['readiness']}")
        
        # 给出建议
        print(f"\n💡 基准建立结论:")
        if overall['score'] >= 70:
            print("   ✅ 系统性能基准达标，可进入算法优化阶段")
        elif overall['score'] >= 60:
            print("   ⚠️  系统性能基准基本可用，建议优化后再进行下一阶段")
        else:
            print("   ❌ 系统性能基准未达标，需要重新设计核心算法")
    
    def _save_results(self, results, summary):
        """保存结果"""
        try:
            # 创建报告目录
            reports_dir = Path('reports/baseline')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存详细结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with open(reports_dir / f'baseline_results_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'detailed_results': results,
                    'summary': summary
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 结果已保存至: reports/baseline/baseline_results_{timestamp}.json")
            
        except Exception as e:
            print(f"保存结果失败: {e}")


async def main():
    """主函数"""
    if not AKSHARE_AVAILABLE:
        print("❌ AKShare未安装，无法运行基准测试")
        return False
    
    baseline_tester = SimplePerformanceBaseline()
    return await baseline_tester.run_baseline_test()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)