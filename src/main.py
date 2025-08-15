#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股基本信息获取系统主程序
提供命令行接口和核心业务逻辑协调
"""

import argparse
import asyncio
import logging
from typing import Optional, List

from data.fetcher import DataFetcher
from data.sector_fetcher import SectorFetcher
from data.technical_calculator import TechnicalCalculator
from data.enhanced_data_fetcher import EnhancedDataFetcher
from cache.manager import CacheManager
from analysis.llm_analyzer import LLMAnalyzer
from analysis.sector_screener import SectorScreener
from analysis.sector_analyzer import SectorAnalyzer
from analysis.sector_backtester import SectorBacktester
from analysis.enhanced_predictor import EnhancedPredictor
from analysis.prediction_validator import PredictionValidator
from analysis.model_optimizer import ModelOptimizer
from analysis.ml_predictor import MLPredictor
from analysis.advanced_validator import AdvancedValidator
from analysis.real_data_analyzer import RealDataAnalyzer
from analysis.optimized_data_analyzer import OptimizedDataAnalyzer
from analysis.report_manager import ReportManager
from config.settings import Settings


class StockInfoSystem:
    """A股信息系统主类"""
    
    def __init__(self):
        self.settings = Settings()
        self.cache_manager = CacheManager(self.settings.cache_dir)
        self.data_fetcher = DataFetcher(self.cache_manager)
        self.llm_analyzer = LLMAnalyzer(self.settings.deepseek_api_key)
        
        # 新增板块相关组件
        self.sector_fetcher = SectorFetcher(self.cache_manager)
        self.tech_calculator = TechnicalCalculator()
        self.report_manager = ReportManager()
        
        # 增强预测系统组件
        self.enhanced_data_fetcher = EnhancedDataFetcher(self.cache_manager)
        self.enhanced_predictor = EnhancedPredictor(self.enhanced_data_fetcher, self.tech_calculator)
        
        self.sector_screener = SectorScreener(
            self.sector_fetcher, 
            self.tech_calculator,
            self.enhanced_data_fetcher,
            self.enhanced_predictor,
            self.report_manager
        )
        self.sector_analyzer = SectorAnalyzer(
            self.sector_fetcher,
            self.tech_calculator,
            self.report_manager
        )
        
        # 添加板块回测器
        self.sector_backtester = SectorBacktester(
            self.sector_fetcher,
            self.cache_manager
        )
        
        # 添加预测验证器
        self.prediction_validator = PredictionValidator(
            self.sector_fetcher,
            self.enhanced_data_fetcher,
            self.enhanced_predictor,
            self.tech_calculator
        )
        
        # 添加模型优化器
        self.model_optimizer = ModelOptimizer(
            self.sector_fetcher,
            self.enhanced_data_fetcher,
            self.tech_calculator
        )
        
        # 添加机器学习预测器
        self.ml_predictor = MLPredictor(
            self.sector_fetcher,
            self.tech_calculator
        )
        
        # 添加高级验证器
        self.advanced_validator = AdvancedValidator(
            self.data_fetcher,
            self.sector_analyzer
        )
        
        # 添加真实数据分析器
        self.real_data_analyzer = RealDataAnalyzer(
            self.sector_fetcher,
            self.enhanced_data_fetcher,
            self.tech_calculator
        )
        
        # 添加优化版数据分析器
        self.optimized_analyzer = OptimizedDataAnalyzer(
            self.sector_fetcher,
            self.enhanced_data_fetcher,
            self.tech_calculator
        )
        
    async def run_analysis(self, stock_code: Optional[str] = None) -> dict:
        """运行完整的股票分析流程"""
        try:
            # 获取交易数据
            trading_data = await self.data_fetcher.get_trading_data(stock_code)
            
            # 运行LLM分析
            analysis_result = await self.llm_analyzer.analyze_comprehensive(
                trading_data, stock_code
            )
            
            return {
                'status': 'success',
                'data': trading_data,
                'analysis': analysis_result
            }
            
        except Exception as e:
            logging.error(f"分析失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_sector_screening(self, period: str = "1-2weeks", top_n: int = 5) -> dict:
        """运行板块筛选流程"""
        try:
            result = await self.sector_screener.screen_top_sectors(top_n=top_n, period=period)
            return result
            
        except Exception as e:
            logging.error(f"板块筛选失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_sector_list(self) -> List[str]:
        """获取支持的板块列表"""
        return self.sector_fetcher.get_supported_sectors()
        
    async def get_sector_summary(self, sector_name: str) -> dict:
        """获取板块分析摘要"""
        try:
            summary = await self.sector_screener.get_sector_analysis_summary(sector_name)
            return summary
            
        except Exception as e:
            logging.error(f"获取板块摘要失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_sector_analysis(self, sector_name: str, time_range: Optional[str] = None) -> dict:
        """运行单板块详细分析"""
        try:
            result = await self.sector_analyzer.analyze_sector(sector_name, time_range)
            return result
            
        except Exception as e:
            logging.error(f"板块分析失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def run_sector_backtest(self, start_date: str, end_date: str, 
                                rebalance_frequency: str = "weekly", 
                                top_n: int = 5, 
                                initial_capital: float = 1000000.0) -> dict:
        """运行板块回测"""
        try:
            result = await self.sector_backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency=rebalance_frequency,
                top_n=top_n,
                initial_capital=initial_capital
            )
            return result
            
        except Exception as e:
            logging.error(f"板块回测失败: {e}")
            return {'status': 'error', 'message': str(e)}
            
    async def run_prediction_validation(self, validation_periods: int = 10,
                                      prediction_days: int = 3) -> dict:
        """运行预测验证流程"""
        try:
            result = await self.prediction_validator.validate_predictions(
                validation_periods=validation_periods,
                prediction_days=prediction_days
            )
            
            # 保存验证报告
            if 'error' not in result:
                report_path = await self.prediction_validator.save_validation_report(result)
                result['report_path'] = report_path
                
            return result
            
        except Exception as e:
            logging.error(f"预测验证失败: {e}")
            return {'status': 'error', 'message': str(e)}
            
    async def run_model_optimization(self, optimization_cycles: int = 5,
                                   validation_periods: int = 8) -> dict:
        """运行模型优化流程"""
        try:
            result = await self.model_optimizer.optimize_model(
                optimization_cycles=optimization_cycles,
                validation_periods=validation_periods
            )
            
            # 保存优化报告
            if 'error' not in result:
                report_path = await self.model_optimizer.save_optimization_report(result)
                result['report_path'] = report_path
                
            return result
            
        except Exception as e:
            logging.error(f"模型优化失败: {e}")
            return {'status': 'error', 'message': str(e)}
            
    async def run_ml_training(self, training_months: int = 24, 
                            prediction_horizon: int = 5) -> dict:
        """运行机器学习模型训练流程"""
        try:
            result = await self.ml_predictor.train_models_on_real_data(
                training_months=training_months,
                prediction_horizon=prediction_horizon
            )
            return result
            
        except Exception as e:
            logging.error(f"ML模型训练失败: {e}")
            return {'error': str(e)}
            
    async def run_ml_evaluation(self, evaluation_months: int = 6) -> dict:
        """运行机器学习模型评估流程"""
        try:
            result = await self.ml_predictor.evaluate_model_performance_on_real_data(
                evaluation_months=evaluation_months
            )
            return result
            
        except Exception as e:
            logging.error(f"ML模型评估失败: {e}")
            return {'error': str(e)}
            
    async def run_real_data_analysis(self, analysis_months: int = 12,
                                   prediction_days: int = 5) -> dict:
        """运行基于真实数据的深度分析流程"""
        try:
            result = await self.real_data_analyzer.analyze_prediction_accuracy_by_real_patterns(
                analysis_months=analysis_months,
                prediction_days=prediction_days
            )
            
            # 保存分析报告
            if 'error' not in result:
                report_path = await self.real_data_analyzer.save_real_data_analysis_report(result)
                result['report_path'] = report_path
                
            return result
            
        except Exception as e:
            logging.error(f"真实数据分析失败: {e}")
            return {'error': str(e)}
            
    async def run_optimized_analysis(self, analysis_months: int = 2,
                                   prediction_days: int = 5) -> dict:
        """运行优化版高准确率分析流程"""
        try:
            result = await self.optimized_analyzer.analyze_prediction_accuracy_optimized(
                analysis_months=analysis_months,
                prediction_days=prediction_days
            )
            
            # 保存优化报告
            if 'error' not in result:
                report_path = await self.optimized_analyzer.save_optimization_report(result)
                result['report_path'] = report_path
                
            return result
            
        except Exception as e:
            logging.error(f"优化版分析失败: {e}")
            return {'error': str(e)}


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股基本信息获取系统')
    parser.add_argument('--stock', '-s', help='股票代码')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--list-reports', '-l', action='store_true', help='列出历史报告')
    parser.add_argument('--show-report', '-r', help='显示指定的报告文件')
    
    # 板块相关参数
    parser.add_argument('--sector-screening', action='store_true', help='板块筛选功能')
    parser.add_argument('--period', default='1-2weeks', help='预测周期 (default: 1-2weeks)')
    parser.add_argument('--top-n', type=int, default=5, help='返回前N个板块 (default: 5)')
    parser.add_argument('--list-sectors', action='store_true', help='列出支持的板块')
    parser.add_argument('--sector-summary', help='获取指定板块分析摘要')
    parser.add_argument('--sector-analysis', help='单板块详细分析')
    parser.add_argument('--time-range', help='时间范围，格式: "from YYMMDD to YYMMDD"')
    
    # 回测相关参数
    parser.add_argument('--sector-backtest', action='store_true', help='板块回测功能')
    parser.add_argument('--start-date', help='回测开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', help='回测结束日期 (YYYYMMDD)')
    parser.add_argument('--rebalance-freq', default='weekly', 
                       choices=['daily', 'weekly', 'monthly'], 
                       help='再平衡频率 (default: weekly)')
    parser.add_argument('--initial-capital', type=float, default=1000000.0, 
                       help='初始资金 (default: 1000000.0)')
    
    # 预测验证相关参数
    parser.add_argument('--prediction-validation', action='store_true', help='预测验证功能')
    parser.add_argument('--validation-periods', type=int, default=10, 
                       help='验证期数 (default: 10)')
    parser.add_argument('--prediction-days', type=int, default=3,
                       help='预测天数 (default: 3)')
    
    # 模型优化相关参数
    parser.add_argument('--model-optimization', action='store_true', help='模型参数优化功能')
    parser.add_argument('--optimization-cycles', type=int, default=5,
                       help='优化轮数 (default: 5)')
    parser.add_argument('--optimization-validation-periods', type=int, default=8,
                       help='优化验证期数 (default: 8)')
    
    # 机器学习相关参数
    parser.add_argument('--ml-training', action='store_true', help='机器学习模型训练功能')
    parser.add_argument('--training-months', type=int, default=24,
                       help='训练数据月数 (default: 24)')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                       help='预测时间窗口天数 (default: 5)')
    
    # 真实数据分析相关参数
    parser.add_argument('--real-data-analysis', action='store_true', help='基于真实数据的深度分析功能')
    parser.add_argument('--analysis-months', type=int, default=12,
                       help='分析历史数据月数 (default: 12)')
    parser.add_argument('--pattern-prediction-days', type=int, default=5,
                       help='模式分析预测天数 (default: 5)')
    
    # 优化版分析相关参数
    parser.add_argument('--optimized-analysis', action='store_true', help='优化版高准确率分析功能')
    parser.add_argument('--opt-analysis-months', type=int, default=2,
                       help='优化分析历史数据月数 (default: 2)')
    parser.add_argument('--opt-prediction-days', type=int, default=5,
                       help='优化分析预测天数 (default: 5)')
    
    # ML模型评估相关参数
    parser.add_argument('--ml-evaluation', action='store_true', help='机器学习模型性能评估功能')
    parser.add_argument('--evaluation-months', type=int, default=6,
                       help='评估数据月数 (default: 6)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    # 处理报告相关命令
    if args.list_reports:
        from analysis.report_manager import ReportManager
        report_manager = ReportManager()
        reports = report_manager.list_reports(args.stock)
        
        if reports:
            print("=== 历史分析报告 ===")
            for i, report in enumerate(reports[:10], 1):  # 显示最近10个
                print(f"{i}. {report}")
            if len(reports) > 10:
                print(f"... 还有 {len(reports) - 10} 个报告")
        else:
            target = args.stock or "所有股票"
            print(f"❌ 未找到 {target} 的历史报告")
        return
    
    if args.show_report:
        from pathlib import Path
        report_path = Path("reports") / args.show_report
        
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                print(f.read())
        else:
            print(f"❌ 报告文件不存在: {report_path}")
        return
    
    # 初始化系统
    system = StockInfoSystem()
    
    # 处理板块相关命令
    if args.list_sectors:
        sectors = await system.get_sector_list()
        print("=== 支持的板块列表 ===")
        for i, sector in enumerate(sectors, 1):
            print(f"{i:2d}. {sector}")
        print(f"\n共支持 {len(sectors)} 个板块")
        return
        
    if args.sector_summary:
        print(f"=== {args.sector_summary} 板块分析摘要 ===")
        summary = await system.get_sector_summary(args.sector_summary)
        
        if 'error' in summary:
            print(f"❌ 获取失败: {summary['error']}")
            return
            
        print(f"板块名称: {summary.get('sector_name', 'Unknown')}")
        print(f"综合评分: {summary.get('comprehensive_score', 0):.1f}/100")
        print(f"推荐等级: {summary.get('recommendation', 'Hold')}")
        print(f"最新价格: {summary.get('latest_price', 0):.2f}")
        print(f"5日涨跌幅: {summary.get('price_change_5d', 0):+.2f}%")
        print(f"成分股数量: {summary.get('stocks_count', 0)}")
        
        scores = summary.get('scores_breakdown', {})
        print(f"\n评分明细:")
        print(f"  技术面: {scores.get('technical', 0):.1f}/100")
        print(f"  资金流向: {scores.get('money_flow', 0):.1f}/100")
        print(f"  基本面: {scores.get('fundamental', 0):.1f}/100")
        print(f"  轮动周期: {scores.get('rotation', 0):.1f}/100")
        return
        
    if args.sector_screening:
        print(f"=== 开始板块筛选 (Top {args.top_n}, 周期: {args.period}) ===")
        result = await system.run_sector_screening(period=args.period, top_n=args.top_n)
        
        if 'status' in result and result['status'] == 'error':
            print(f"❌ 筛选失败: {result['message']}")
            return
            
        print(f"筛选时间: {result.get('screening_time', '')}")
        print(f"分析板块数: {result.get('total_sectors_analyzed', 0)}")
        
        top_sectors = result.get('top_sectors', [])
        print(f"\n=== Top {len(top_sectors)} 推荐板块 ===")
        
        for i, sector in enumerate(top_sectors, 1):
            name = sector.get('sector_name', 'Unknown')
            score = sector.get('comprehensive_score', 0)
            recommendation = sector.get('recommendation', 'Hold')
            
            print(f"{i}. {name}")
            print(f"   综合评分: {score:.1f}/100 | 推荐: {recommendation}")
            print(f"   技术面: {sector.get('technical_score', 0):.1f} | "
                  f"资金面: {sector.get('money_flow_score', 0):.1f} | "
                  f"基本面: {sector.get('fundamental_score', 0):.1f}")
            
        if result.get('report_path'):
            print(f"\n📝 详细报告已保存: {result['report_path']}")
            
        # 显示风险提示
        warnings = result.get('risk_warnings', [])
        if warnings:
            print(f"\n⚠️  风险提示:")
            for warning in warnings:
                print(f"   - {warning}")
                
        return
    
    if args.sector_analysis:
        print(f"=== {args.sector_analysis} 板块详细分析 ===")
        time_range_str = args.time_range if args.time_range else None
        if time_range_str:
            print(f"分析时间范围: {time_range_str}")
        else:
            print("分析时间范围: 默认最近一周")
            
        result = await system.run_sector_analysis(args.sector_analysis, time_range_str)
        
        if 'status' in result and result['status'] == 'error':
            print(f"❌ 分析失败: {result['message']}")
            return
            
        # 显示分析摘要
        print(f"\n板块名称: {result.get('sector_name', 'Unknown')}")
        print(f"综合评分: {result.get('comprehensive_score', 0):.1f}/100")
        print(f"推荐等级: {result.get('recommendation', 'Hold')}")
        print(f"分析周期: {result.get('time_range', {}).get('trading_days', 0)} 个交易日")
        
        # 价格分析
        price_analysis = result.get('price_analysis', {})
        print(f"\n=== 价格表现 ===")
        print(f"最新价格: {price_analysis.get('latest_price', 0):.2f}")
        print(f"今日涨跌: {price_analysis.get('price_change_1d', 0):+.2f}%")
        print(f"5日涨跌: {price_analysis.get('price_change_5d', 0):+.2f}%")
        print(f"区间涨跌: {price_analysis.get('price_change_period', 0):+.2f}%")
        
        # 个股分析
        stocks_analysis = result.get('stocks_analysis', {})
        print(f"\n=== 个股表现 ===")
        print(f"板块股数: {stocks_analysis.get('total_stocks', 0)}")
        print(f"平均涨跌幅: {stocks_analysis.get('average_return_5d', 0):+.2f}%")
        print(f"上涨股票占比: {stocks_analysis.get('positive_stocks_ratio', 0):.1f}%")
        
        # 领涨股
        leading_stocks = stocks_analysis.get('leading_stocks', [])[:3]
        if leading_stocks:
            print(f"领涨股TOP3:")
            for i, stock in enumerate(leading_stocks, 1):
                print(f"  {i}. {stock['name']}: {stock['price_change_5d']:+.2f}%")
        
        # 预测
        prediction = result.get('prediction', {})
        if prediction and 'trend_prediction' in prediction:
            print(f"\n=== 预测分析 ===")
            print(f"趋势预测: {prediction.get('trend_prediction', 'Unknown')}")
            print(f"预测概率: {prediction.get('probability', 0):.1f}%")
            price_range = prediction.get('price_range', {})
            print(f"价格区间: {price_range.get('lower', 0):.2f} - {price_range.get('upper', 0):.2f}")
        
        # 投资建议
        investment_advice = result.get('investment_advice', {})
        if investment_advice and 'overall_action' in investment_advice:
            print(f"\n=== 投资建议 ===")
            print(f"操作建议: {investment_advice.get('overall_action', 'Unknown')}")
            print(f"仓位建议: {investment_advice.get('position_ratio', 'Unknown')}")
            print(f"入场时机: {investment_advice.get('best_entry_timing', 'Unknown')}")
        
        # 风险提示
        warnings = result.get('risk_warnings', [])
        if warnings:
            print(f"\n⚠️  风险提示:")
            for warning in warnings[:3]:  # 只显示前3个
                print(f"   - {warning}")
        
        # 报告路径
        if result.get('report_path'):
            print(f"\n📝 详细报告已保存: {result['report_path']}")
            
        return
    
    # 处理板块回测命令
    if args.sector_backtest:
        if not args.start_date or not args.end_date:
            print("❌ 回测功能需要指定开始日期和结束日期")
            print("   使用 --start-date YYYYMMDD --end-date YYYYMMDD")
            return
            
        print(f"=== 开始板块回测 ===")
        print(f"回测期间: {args.start_date} - {args.end_date}")
        print(f"再平衡频率: {args.rebalance_freq}")
        print(f"Top N板块: {args.top_n}")
        print(f"初始资金: {args.initial_capital:,.0f} 元")
        
        result = await system.run_sector_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            rebalance_frequency=args.rebalance_freq,
            top_n=args.top_n,
            initial_capital=args.initial_capital
        )
        
        if 'error' in result:
            print(f"❌ 回测失败: {result['error']}")
            return
            
        # 显示回测摘要
        summary = result.get('backtest_summary', {})
        print(f"\n=== 回测结果摘要 ===")
        print(f"初始资金: {summary.get('initial_capital', 0):,.0f} 元")
        print(f"最终资金: {summary.get('final_capital', 0):,.0f} 元")
        print(f"总收益率: {summary.get('total_return', 0):+.2f}%")
        print(f"年化收益率: {summary.get('annualized_return', 0):+.2f}%")
        print(f"最大回撤: {summary.get('max_drawdown', 0):.2f}%")
        print(f"夏普比率: {summary.get('sharpe_ratio', 0):.3f}")
        
        # 显示详细结果
        detailed = result.get('detailed_results', {})
        metrics = detailed.get('performance_metrics', {})
        if metrics:
            print(f"\n=== 详细指标 ===")
            print(f"年化波动率: {metrics.get('volatility', 0):.2f}%")
            print(f"胜率: {metrics.get('win_rate', 0):.1f}%")
            print(f"总交易天数: {metrics.get('total_trading_days', 0)} 天")
        
        # 显示交易记录
        transactions = detailed.get('transactions', [])
        if transactions:
            print(f"\n=== 交易记录 (最近10笔) ===")
            for i, tx in enumerate(transactions[-10:], 1):
                print(f"{i:2d}. {tx['date']} | {tx['sector']} | {tx['action']} | "
                      f"{tx['change']:+,.0f}")
        
        # 显示报告路径
        if result.get('report_path'):
            print(f"\n📝 详细回测报告已保存: {result['report_path']}")
            
        return
    
    # 处理预测验证命令
    if args.prediction_validation:
        print(f"=== 开始预测验证 ===")
        print(f"验证期数: {args.validation_periods}")
        print(f"预测天数: {args.prediction_days}")
        print("正在分析历史数据...请耐心等待")
        
        result = await system.run_prediction_validation(
            validation_periods=args.validation_periods,
            prediction_days=args.prediction_days
        )
        
        if 'error' in result:
            print(f"❌ 验证失败: {result['error']}")
            return
            
        # 显示验证结果
        stats = result.get('overall_statistics', {})
        print(f"\n=== 验证结果 ===")
        print(f"验证期数: {result.get('validation_periods', 0)}")
        print(f"总预测次数: {result.get('total_predictions', 0)}")
        print(f"总体准确率: {stats.get('accuracy', 0):.1%}")
        print(f"方向准确率: {stats.get('avg_direction_accuracy', 0):.1f}%")
        print(f"收益率准确率: {stats.get('avg_return_accuracy', 0):.1f}%")
        print(f"预测稳定性: {stats.get('stability', 0):.1%}")
        print(f"平均误差: {stats.get('avg_return_error', 0):.2f}%")
        print(f"综合评级: {stats.get('grade', '未知')}")
        
        # 显示优化建议
        suggestions = result.get('optimization_suggestions', [])
        if suggestions:
            print(f"\n=== 优化建议 ===")
            for i, suggestion in enumerate(suggestions[:5], 1):
                print(f"{i}. {suggestion}")
                
        # 显示报告路径
        if result.get('report_path'):
            print(f"\n📝 详细验证报告已保存: {result['report_path']}")
            
        return
    
    # 处理模型优化命令
    if args.model_optimization:
        print(f"=== 开始模型优化 ===")
        print(f"优化轮数: {args.optimization_cycles}")
        print(f"验证期数: {args.optimization_validation_periods}")
        print("正在搜索最优参数配置...请耐心等待")
        
        result = await system.run_model_optimization(
            optimization_cycles=args.optimization_cycles,
            validation_periods=args.optimization_validation_periods
        )
        
        if 'error' in result:
            print(f"❌ 优化失败: {result['error']}")
            return
            
        # 显示优化结果
        print(f"\n=== 优化结果 ===")
        print(f"基线准确率: {result.get('baseline_accuracy', 0):.1%}")
        print(f"优化后准确率: {result.get('best_accuracy', 0):.1%}")
        print(f"绝对提升: {result.get('improvement', 0):+.3f}")
        print(f"相对提升: {result.get('improvement_percentage', 0):+.1f}%")
        
        # 显示最佳配置
        best_config = result.get('best_config', {})
        if best_config:
            print(f"\n=== 最佳参数配置 ===")
            for param, weight in best_config.items():
                print(f"{param}: {weight:.3f}")
        
        # 显示优化建议
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"\n=== 优化建议 ===")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. {rec}")
                
        # 显示报告路径
        if result.get('report_path'):
            print(f"\n📝 详细优化报告已保存: {result['report_path']}")
            
        return
    
    # 处理机器学习训练命令
    if args.ml_training:
        print(f"=== 开始机器学习模型训练 ===")
        print(f"训练数据月数: {args.training_months}")
        print(f"预测时间窗口: {args.prediction_horizon}天")
        print("正在训练多个机器学习模型...请耐心等待")
        
        result = await system.run_ml_training(
            training_months=args.training_months,
            prediction_horizon=args.prediction_horizon
        )
        
        if 'error' in result:
            print(f"❌ 训练失败: {result['error']}")
            return
            
        # 显示训练结果
        print(f"\n=== 训练完成 ===")
        print(f"训练模型数: {result.get('models_trained', 0)}")
        print(f"最佳模型: {result.get('best_model', 'Unknown')}")
        print(f"训练样本数: {result.get('training_data_summary', {}).get('total_samples', 0)}")
        
        # 显示模型性能
        performances = result.get('model_performances', {})
        if performances:
            print(f"\n=== 模型性能 ===")
            for model_name, perf in performances.items():
                if 'error' not in perf:
                    print(f"{model_name}:")
                    print(f"  方向准确率: {perf.get('direction_accuracy', 0):.1f}%")
                    print(f"  R²得分: {perf.get('train_r2', 0):.3f}")
                    print(f"  训练样本: {perf.get('training_samples', 0)}")
                    
        print(f"\n📝 模型已保存到 models/ 目录")
        return
    
    # 处理机器学习评估命令
    if args.ml_evaluation:
        print(f"=== 开始机器学习模型评估 ===")
        print(f"评估数据月数: {args.evaluation_months}")
        print("正在评估模型性能...请耐心等待")
        
        result = await system.run_ml_evaluation(
            evaluation_months=args.evaluation_months
        )
        
        if 'error' in result:
            print(f"❌ 评估失败: {result['error']}")
            return
            
        # 显示评估结果
        print(f"\n=== 评估结果 ===")
        print(f"评估板块数: {result.get('sectors_evaluated', 0)}")
        
        overall_perf = result.get('overall_performance', {})
        if overall_perf:
            print(f"平均方向准确率: {overall_perf.get('avg_direction_accuracy', 0):.1f}%")
            print(f"平均相关系数: {overall_perf.get('avg_correlation', 0):.3f}")
            print(f"最佳板块准确率: {overall_perf.get('best_sector_accuracy', 0):.1f}%")
            print(f"最差板块准确率: {overall_perf.get('worst_sector_accuracy', 0):.1f}%")
            print(f"总评估样本: {overall_perf.get('total_samples', 0)}")
            
        return
    
    # 处理真实数据分析命令
    if args.real_data_analysis:
        print(f"=== 开始基于真实数据的深度分析 ===")
        print(f"分析历史数据: {args.analysis_months}个月")
        print(f"模式预测天数: {args.pattern_prediction_days}天")
        print("正在分析真实市场数据模式...请耐心等待")
        
        result = await system.run_real_data_analysis(
            analysis_months=args.analysis_months,
            prediction_days=args.pattern_prediction_days
        )
        
        if 'error' in result:
            print(f"❌ 分析失败: {result['error']}")
            return
            
        # 显示分析结果
        print(f"\n=== 真实数据分析结果 ===")
        print(f"分析板块数: {result.get('sectors_analyzed', 0)}")
        
        # 有效模式
        effective_patterns = result.get('effective_patterns', {})
        if effective_patterns:
            print(f"\n=== 有效模式识别 ===")
            momentum_eff = effective_patterns.get('momentum_effectiveness', {})
            if momentum_eff.get('avg_accuracy', 0) > 0:
                print(f"动量模式平均准确率: {momentum_eff['avg_accuracy']:.1f}%")
                
            volume_eff = effective_patterns.get('volume_effectiveness', {})
            if volume_eff.get('avg_accuracy', 0) > 0:
                print(f"成交量模式平均准确率: {volume_eff['avg_accuracy']:.1f}%")
                
            trend_eff = effective_patterns.get('trend_effectiveness', {})
            if trend_eff.get('avg_accuracy', 0) > 0:
                print(f"趋势模式平均准确率: {trend_eff['avg_accuracy']:.1f}%")
        
        # 收益验证
        return_validation = result.get('return_validation', {}).get('overall_validation', {})
        if return_validation:
            print(f"\n=== 真实收益验证 ===")
            print(f"平均方向准确率: {return_validation.get('avg_direction_accuracy', 0):.1f}%")
            print(f"强信号准确率: {return_validation.get('avg_strong_signal_accuracy', 0):.1f}%")
            print(f"模式有效性: {return_validation.get('pattern_effectiveness', 'unknown')}")
        
        # 改进策略
        strategies = result.get('improvement_strategies', [])
        if strategies:
            print(f"\n=== 改进策略建议 ===")
            for i, strategy in enumerate(strategies[:5], 1):
                print(f"{i}. {strategy}")
        
        # 报告路径
        if result.get('report_path'):
            print(f"\n📝 详细分析报告已保存: {result['report_path']}")
            
        return
    
    # 处理优化版分析命令
    if args.optimized_analysis:
        print(f"=== 开始优化版高准确率分析 ===")
        print(f"分析历史数据: {args.opt_analysis_months}个月")
        print(f"预测时间窗口: {args.opt_prediction_days}天")
        print("正在进行高级特征工程和优化分析...请耐心等待")
        
        result = await system.run_optimized_analysis(
            analysis_months=args.opt_analysis_months,
            prediction_days=args.opt_prediction_days
        )
        
        if 'error' in result:
            print(f"❌ 优化分析失败: {result['error']}")
            return
            
        # 显示优化结果
        print(f"\n=== 优化分析结果 ===")
        print(f"分析板块数: {result.get('sectors_analyzed', 0)}")
        
        # 优化指标
        opt_metrics = result.get('optimization_metrics', {})
        if opt_metrics:
            print(f"\n=== 优化性能指标 ===")
            if 'overall_optimization_score' in opt_metrics:
                print(f"综合优化得分: {opt_metrics['overall_optimization_score']:.1f}/100")
                print(f"优化等级: {opt_metrics.get('optimization_grade', 'Unknown')}")
            
            # 显示Enhanced Momentum vs Baseline统一维度对比
            if 'optimization_performance' in opt_metrics:
                perf = opt_metrics['optimization_performance']
                
                # Enhanced Momentum (优化方法)
                enhanced_direction = perf.get('enhanced_direction_accuracy', 0)
                enhanced_strong = perf.get('enhanced_strong_signal_accuracy', 0)
                enhanced_best = perf.get('enhanced_best_performance', 0)
                
                # Baseline (传统方法)
                baseline_direction = perf.get('baseline_direction_accuracy', 0)
                baseline_strong = perf.get('baseline_strong_signal_accuracy', 0)
                baseline_best = perf.get('baseline_best_performance', 0)
                
                # 提升效果
                direction_improvement = perf.get('direction_improvement', 0)
                strong_improvement = perf.get('strong_signal_improvement', 0)
                improvement_ratio = perf.get('improvement_ratio', 0)
                
                print(f"\n【优化方法 vs 传统方法统一对比】")
                print(f"方向预测准确率: Enhanced {enhanced_direction:.1f}% vs Baseline {baseline_direction:.1f}% (提升{direction_improvement:+.1f}个百分点)")
                print(f"强信号准确率: Enhanced {enhanced_strong:.1f}% vs Baseline {baseline_strong:.1f}% (提升{strong_improvement:+.1f}个百分点)")  
                print(f"最佳表现准确率: Enhanced {enhanced_best:.1f}% vs Baseline {baseline_best:.1f}%")
                print(f"整体改进效果: {improvement_ratio:+.1f}%")
                
                # 额外的Enhanced指标
                enhanced_samples = perf.get('enhanced_samples', 0)
                enhanced_confidence = perf.get('enhanced_confidence', 0)
                if enhanced_samples > 0:
                    print(f"Enhanced样本数量: {enhanced_samples}")
                if enhanced_confidence > 0:
                    print(f"Enhanced平均置信度: {enhanced_confidence:.2f}")
            
            # 稳定性和可靠性指标
            if 'stability_metrics' in opt_metrics:
                stability = opt_metrics['stability_metrics']
                print(f"\n【稳定性与可靠性】")
                print(f"预测稳定性: {stability.get('prediction_stability', 0):.1f}%")
                if stability.get('baseline_std', 0) > 0:
                    print(f"Baseline波动性: {stability.get('baseline_std', 0):.2f}")
                sample_reliability = stability.get('sample_reliability', 0)
                if sample_reliability > 0:
                    print(f"样本可靠性: {sample_reliability:.1f}%")
        
        # 有效模式
        effective_patterns = result.get('effective_patterns', {})
        if effective_patterns:
            print(f"\n=== 优化模式识别 ===")
            for pattern_type, pattern_data in effective_patterns.items():
                if pattern_data.get('avg_accuracy', 0) > 50:
                    pattern_name = pattern_type.replace('_', ' ').title()
                    print(f"{pattern_name}: {pattern_data.get('avg_accuracy', 0):.1f}%")
        
        # 模式有效性评估
        return_validation = result.get('return_validation', {}).get('overall_validation', {})
        if return_validation:
            pattern_effectiveness = return_validation.get('pattern_effectiveness', 'unknown')
            effectiveness_map = {
                'very_high': '极高',
                'high': '高',
                'medium': '中等', 
                'low': '低',
                'very_low': '极低',
                'unknown': '未知'
            }
            print(f"\n=== 系统有效性评估 ===")
            print(f"整体模式有效性: {effectiveness_map.get(pattern_effectiveness, pattern_effectiveness)}")
        
        # 智能改进策略
        strategies = result.get('improvement_strategies', [])
        if strategies:
            print(f"\n=== 智能改进策略 ===")
            for i, strategy in enumerate(strategies[:5], 1):
                print(f"{i}. {strategy}")
        
        # 报告路径
        if result.get('report_path'):
            print(f"\n📝 优化分析报告已保存: {result['report_path']}")
            
        return
    
    # 执行股票分析
    if args.stock or not any([args.sector_screening, args.list_sectors, args.sector_summary, args.sector_analysis, args.sector_backtest, args.prediction_validation, args.model_optimization, args.ml_training, args.ml_evaluation, args.real_data_analysis, args.optimized_analysis]):
        result = await system.run_analysis(args.stock)
        
        if result['status'] == 'success':
            print("=== 分析完成 ===")
            print(f"数据记录数: {len(result['data'])}")
            
            # 显示报告保存路径
            if 'report_path' in result['analysis']:
                print(f"📝 分析报告已保存: {result['analysis']['report_path']}")
                print(f"📊 报告格式: Markdown")
            
            # 显示简要分析结果
            analysis = result['analysis']
            if args.verbose:
                print("\n=== 详细分析内容 ===")
                print(f"\n📊 事件分析:\n{analysis.get('events_analysis', '无')}")
                print(f"\n📈 技术分析:\n{analysis.get('trend_analysis', '无')}")
                print(f"\n💡 交易建议:\n{analysis.get('trading_advice', '无')}")
            else:
                print("\n=== 分析摘要 ===")
                events = analysis.get('events_analysis', '')
                if events:
                    # 显示事件分析的前100字符
                    summary = events[:100] + "..." if len(events) > 100 else events
                    print(f"📊 事件分析: {summary}")
                
                print(f"📈 分析时间: {analysis.get('analysis_time', 'Unknown')}")
                
                if analysis.get('note'):
                    print(f"ℹ️  说明: {analysis['note']}")
                
                print(f"\n💡 查看完整报告: cat {result['analysis'].get('report_path', '')}")
        else:
            print(f"❌ 分析失败: {result['message']}")


if __name__ == "__main__":
    asyncio.run(main())