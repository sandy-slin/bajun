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
from cache.manager import CacheManager
from analysis.llm_analyzer import LLMAnalyzer
from analysis.sector_screener import SectorScreener
from analysis.sector_analyzer import SectorAnalyzer
from analysis.sector_backtester import SectorBacktester
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
        self.sector_screener = SectorScreener(
            self.sector_fetcher, 
            self.tech_calculator, 
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
    
    # 执行股票分析
    if args.stock or not any([args.sector_screening, args.list_sectors, args.sector_summary, args.sector_analysis]):
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