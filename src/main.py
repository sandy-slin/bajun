#!/usr/bin/env python3
"""
A股基本信息获取系统主程序
提供命令行接口和核心业务逻辑协调
"""

import argparse
import asyncio
import logging
from typing import Optional

from data.fetcher import DataFetcher
from cache.manager import CacheManager
from analysis.llm_analyzer import LLMAnalyzer
from config.settings import Settings


class StockInfoSystem:
    """A股信息系统主类"""
    
    def __init__(self):
        self.settings = Settings()
        self.cache_manager = CacheManager(self.settings.cache_dir)
        self.data_fetcher = DataFetcher(self.cache_manager)
        self.llm_analyzer = LLMAnalyzer(self.settings.deepseek_api_key)
        
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
    
    # 执行股票分析
    system = StockInfoSystem()
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