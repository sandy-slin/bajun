#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能交易决策平台 - Phase 1 MVP主程序
效果优先的最小可用系统

三步交易决策流水线:
1. 板块分析与排名 (Sector Analysis & Ranking)
2. 板块内股票精选 (Stock Selection within Sectors)
3. 个人持仓分析与建议 (Portfolio Analysis & Recommendations)

包含基础反人性交易助手功能
"""

import argparse
import asyncio
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 导入核心引擎
from core.sector_engine import SectorEngine
from core.stock_engine import StockEngine  
from core.portfolio_engine import PortfolioEngine
from core.anti_human_nature_engine import AntiHumanNatureEngine

# 导入数据层
from data.sector_fetcher import SectorFetcher
from data.real_data_fetcher import RealDataFetcher
from data.technical_calculator import TechnicalCalculator

# 导入配置
from config.settings import Settings


class TradingDecisionPlatform:
    """A股智能交易决策平台 - MVP版本"""
    
    def __init__(self):
        """初始化平台组件"""
        self.settings = Settings()
        self.logger = self._setup_logging()
        
        # 初始化数据层
        self.sector_fetcher = SectorFetcher(None)  # 简化版无cache_manager
        self.data_fetcher = RealDataFetcher(None)
        self.tech_calculator = TechnicalCalculator()
        
        # 初始化核心引擎
        self.sector_engine = SectorEngine(self.sector_fetcher, self.tech_calculator)
        self.stock_engine = StockEngine(
            self.sector_fetcher, 
            self.data_fetcher, 
            self.tech_calculator
        )
        self.portfolio_engine = PortfolioEngine(
            self.data_fetcher,
            self.sector_fetcher,
            self.sector_engine,
            self.stock_engine
        )
        self.anti_human_nature_engine = AntiHumanNatureEngine()
        
        self.logger.info("A股智能交易决策平台 MVP 初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('.cache/mvp.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
    
    async def run_daily_workflow(self) -> Dict:
        """
        执行完整的每日交易决策工作流
        
        Returns:
            Dict: 完整的分析结果
        """
        try:
            start_time = datetime.now()
            self.logger.info("开始执行每日交易决策工作流")
            
            # Step 1: 板块分析与排名
            print("🔍 Step 1: 板块分析与排名...")
            sector_result = await self.sector_engine.analyze_top_sectors(
                lookback_months=6, top_n=5
            )
            
            if 'error' in sector_result:
                return {'error': f'板块分析失败: {sector_result["error"]}'}
            
            print(f"✅ 完成板块分析，识别出{len(sector_result['top_sectors'])}个优质板块")
            
            # Step 2: 板块内股票精选
            print("📈 Step 2: 板块内股票精选...")
            stock_result = await self.stock_engine.select_stocks_from_sectors(
                sector_result['top_sectors'], stocks_per_sector=5
            )
            
            if 'error' in stock_result:
                return {'error': f'股票筛选失败: {stock_result["error"]}'}
            
            print(f"✅ 完成股票筛选，共选出{stock_result['overall_summary']['total_stocks_selected']}只优质股票")
            
            # 生成每日推荐报告
            daily_report = {
                'timestamp': datetime.now().isoformat(),
                'workflow_type': 'daily_recommendation',
                'sector_analysis': sector_result,
                'stock_selection': stock_result,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            # 保存报告
            report_path = await self._save_daily_report(daily_report)
            daily_report['report_path'] = report_path
            
            self.logger.info(f"每日工作流完成，耗时{daily_report['processing_time_seconds']:.1f}秒")
            return daily_report
            
        except Exception as e:
            self.logger.error(f"每日工作流执行失败: {e}")
            return {'error': str(e)}
    
    async def analyze_portfolio(self, holdings_file: str) -> Dict:
        """
        分析个人投资组合
        
        Args:
            holdings_file: 持仓文件路径
            
        Returns:
            Dict: 投资组合分析结果
        """
        try:
            print(f"📊 分析投资组合: {holdings_file}")
            
            # Step 3: 个人持仓分析与建议
            portfolio_result = await self.portfolio_engine.analyze_portfolio(holdings_file)
            
            if 'error' in portfolio_result:
                return portfolio_result
            
            print("✅ 投资组合分析完成")
            return portfolio_result
            
        except Exception as e:
            self.logger.error(f"投资组合分析失败: {e}")
            return {'error': str(e)}
    
    async def evaluate_trading_decision(
        self,
        action: str,
        stock_code: str,
        current_price: float,
        position_info: Optional[Dict] = None
    ) -> Dict:
        """
        评估交易决策 (反人性检查)
        
        Args:
            action: 交易动作 BUY/SELL/HOLD
            stock_code: 股票代码
            current_price: 当前价格
            position_info: 持仓信息
            
        Returns:
            Dict: 交易决策评估结果
        """
        try:
            print(f"🛡️ 反人性交易检查: {action} {stock_code} @ {current_price}")
            
            evaluation_result = await self.anti_human_nature_engine.evaluate_trading_decision(
                action, stock_code, current_price, position_info
            )
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"交易决策评估失败: {e}")
            return {'error': str(e)}
    
    async def _save_daily_report(self, report: Dict) -> str:
        """保存每日报告"""
        try:
            # 创建报告目录
            reports_dir = Path('reports/daily')
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f'daily_recommendation_{timestamp}.json'
            report_path = reports_dir / report_filename
            
            # 保存JSON报告
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 生成Markdown摘要
            markdown_path = reports_dir / f'daily_summary_{timestamp}.md'
            await self._generate_markdown_summary(report, markdown_path)
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"保存每日报告失败: {e}")
            return ''
    
    async def _generate_markdown_summary(self, report: Dict, output_path: Path):
        """生成Markdown格式的摘要报告"""
        try:
            sector_analysis = report.get('sector_analysis', {})
            stock_selection = report.get('stock_selection', {})
            
            markdown_content = f"""# A股每日投资建议报告

## 报告信息
- **生成时间**: {report.get('timestamp', '')}
- **分析耗时**: {report.get('processing_time_seconds', 0):.1f}秒

## 📊 板块分析结果

### 市场概览
- **分析板块数**: {sector_analysis.get('total_sectors_analyzed', 0)}
- **市场情绪**: {sector_analysis.get('market_overview', {}).get('market_sentiment', 'unknown')}

### TOP 5 推荐板块
"""
            
            # 添加板块推荐
            top_sectors = sector_analysis.get('top_sectors', [])
            for i, sector in enumerate(top_sectors, 1):
                markdown_content += f"""
#### {i}. {sector.get('sector_name', '')}
- **综合评分**: {sector.get('composite_score', 0):.1f}/100
- **动量评分**: {sector.get('momentum_score', 0):.1f}/100  
- **相对强弱**: {sector.get('relative_strength_score', 0):.1f}/100
- **投资逻辑**: {sector.get('investment_logic', '')}
- **风险等级**: {sector.get('risk_level', 'unknown')}
"""
            
            # 添加股票推荐
            markdown_content += f"""
## 📈 股票精选结果

### 选股概览
- **处理板块数**: {stock_selection.get('sectors_processed', 0)}
- **分析股票总数**: {stock_selection.get('total_stocks_analyzed', 0)}
- **最终选出**: {stock_selection.get('overall_summary', {}).get('total_stocks_selected', 0)}只股票

### 分板块推荐股票
"""
            
            sector_selections = stock_selection.get('sector_selections', [])
            for sector_sel in sector_selections:
                sector_name = sector_sel.get('sector_name', '')
                selected_stocks = sector_sel.get('selected_stocks', [])
                
                markdown_content += f"""
#### {sector_name} ({len(selected_stocks)}只)
"""
                
                for j, stock in enumerate(selected_stocks, 1):
                    markdown_content += f"""
{j}. **{stock.get('stock_code', '')}** - {stock.get('stock_name', '')}
   - 综合评分: {stock.get('composite_score', 0):.1f}/100
   - 最新价格: ¥{stock.get('latest_price', 0):.2f}
   - 5日涨跌: {stock.get('price_change_5d', 0):+.2f}%
   - 风险等级: {stock.get('risk_level', 'unknown')}
"""
            
            # 添加风险提示
            markdown_content += f"""
## ⚠️ 重要提示

1. **投资有风险，入市需谨慎**
2. 本报告仅供参考，不构成投资建议
3. 请结合个人风险承受能力进行投资决策
4. 建议使用反人性交易助手进行决策检查

---
*报告由A股智能交易决策平台自动生成*
"""
            
            # 保存Markdown文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
        except Exception as e:
            self.logger.error(f"生成Markdown摘要失败: {e}")
    
    def print_sector_results(self, sector_result: Dict):
        """打印板块分析结果"""
        if 'error' in sector_result:
            print(f"❌ 板块分析失败: {sector_result['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 板块分析结果 (分析了{sector_result['total_sectors_analyzed']}个板块)")
        print(f"{'='*60}")
        
        top_sectors = sector_result['top_sectors']
        for i, sector in enumerate(top_sectors, 1):
            name = sector['sector_name']
            score = sector['composite_score']
            momentum = sector['momentum_score']
            strength = sector['relative_strength_score']
            logic = sector['investment_logic']
            
            print(f"\n{i}. 【{name}】- 综合评分: {score:.1f}/100")
            print(f"   动量评分: {momentum:.1f} | 相对强弱: {strength:.1f}")
            print(f"   投资逻辑: {logic}")
    
    def print_stock_results(self, stock_result: Dict):
        """打印股票筛选结果"""
        if 'error' in stock_result:
            print(f"❌ 股票筛选失败: {stock_result['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"📈 股票筛选结果")
        print(f"{'='*60}")
        
        overall_summary = stock_result['overall_summary']
        print(f"总计分析: {overall_summary['total_stocks_analyzed']}只股票")
        print(f"最终选出: {overall_summary['total_stocks_selected']}只股票")
        print(f"平均评分: {overall_summary['average_composite_score']:.1f}/100")
        print(f"选择质量: {overall_summary['selection_quality']}")
        
        sector_selections = stock_result['sector_selections']
        for sector_sel in sector_selections:
            sector_name = sector_sel['sector_name']
            selected_stocks = sector_sel['selected_stocks']
            
            print(f"\n🏢 【{sector_name}】({len(selected_stocks)}只)")
            
            for j, stock in enumerate(selected_stocks, 1):
                code = stock['stock_code']
                name = stock.get('stock_name', '')
                score = stock['composite_score']
                price = stock['latest_price']
                change = stock['price_change_5d']
                
                print(f"   {j}. {code} {name} - 评分:{score:.1f} 价格:¥{price:.2f} 5日:{change:+.1f}%")
    
    def print_portfolio_results(self, portfolio_result: Dict):
        """打印投资组合分析结果"""
        if 'error' in portfolio_result:
            print(f"❌ 投资组合分析失败: {portfolio_result['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 投资组合分析结果")
        print(f"{'='*60}")
        
        # 组合状态
        status = portfolio_result['portfolio_status']
        print(f"\n💰 组合状态:")
        print(f"   总资产: ¥{status['current_total_value']:,.0f}")
        print(f"   现金比例: {status['cash_ratio']:.1f}%")
        print(f"   总收益: ¥{status['total_return']:,.0f} ({status['total_return_rate']:+.1f}%)")
        print(f"   持仓数量: {status['position_count']}只")
        
        # 质量评估
        quality = portfolio_result['holdings_quality']
        print(f"\n📈 持仓质量:")
        print(f"   平均质量评分: {quality['average_quality_score']:.1f}/100")
        print(f"   质量等级: {quality['quality_grade']}")
        print(f"   高质量股票: {quality['high_quality_count']}只")
        print(f"   低质量股票: {quality['low_quality_count']}只")
        
        # 调仓建议
        rebalancing = portfolio_result['rebalancing_recommendations']
        print(f"\n🔄 调仓建议:")
        print(f"   总建议数: {rebalancing['total_recommendations']}")
        print(f"   卖出建议: {len(rebalancing['sell_recommendations'])}只")
        print(f"   减仓建议: {len(rebalancing['reduce_recommendations'])}只")
        print(f"   紧急程度: {rebalancing['rebalancing_urgency']}")
        
        # 整体评估
        assessment = portfolio_result['overall_assessment']
        print(f"\n⭐ 整体评估:")
        print(f"   综合评分: {assessment['overall_score']:.1f}/100")
        print(f"   评估等级: {assessment['grade']}")
        print(f"   评估意见: {assessment['assessment']}")
    
    def print_anti_human_nature_results(self, evaluation_result: Dict):
        """打印反人性交易评估结果"""
        if 'error' in evaluation_result:
            print(f"❌ 交易评估失败: {evaluation_result['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"🛡️ 反人性交易评估")
        print(f"{'='*60}")
        
        # 情绪分析
        emotion = evaluation_result['emotion_analysis']
        print(f"\n😊 情绪分析:")
        print(f"   情绪状态: {emotion['emotional_state']}")
        print(f"   风险等级: {emotion['risk_level']}")
        
        if emotion['detected_emotions']:
            print("   检测到的情绪:")
            for em in emotion['detected_emotions']:
                print(f"     - {em['emotion']}: 强度{em['intensity']}/10 ({em['trigger']})")
        
        # 行为分析
        behavior = evaluation_result['behavior_analysis']
        print(f"\n🎯 行为分析:")
        print(f"   行为评分: {behavior['behavior_score']:.1f}/100")
        print(f"   需要干预: {'是' if behavior['intervention_needed'] else '否'}")
        
        if behavior['detected_patterns']:
            print("   检测到的行为模式:")
            for pattern in behavior['detected_patterns']:
                print(f"     - {pattern['pattern']}: {pattern['severity']} ({pattern['description']})")
        
        # 最终建议
        recommendation = evaluation_result['final_recommendation']
        print(f"\n✅ 最终建议:")
        print(f"   建议动作: {recommendation['action']}")
        print(f"   建议原因: {recommendation['reason']}")
        
        if 'delay_minutes' in recommendation:
            print(f"   等待时间: {recommendation['delay_minutes']}分钟")
        
        # 干预建议
        intervention = evaluation_result['intervention_advice']
        if intervention['intervention_required']:
            print(f"\n⚠️ 干预措施:")
            for inter in intervention['interventions']:
                print(f"   类型: {inter['type']}")
                print(f"   措施: {inter.get('action', '无')}")
                print(f"   说明: {inter.get('message', '无')}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='A股智能交易决策平台 - Phase 1 MVP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  
  # 执行完整的每日工作流
  python src/mvp_main.py --daily-workflow
  
  # 仅分析板块
  python src/mvp_main.py --sector-analysis --top-n 5
  
  # 仅分析股票筛选
  python src/mvp_main.py --stock-selection
  
  # 分析个人持仓
  python src/mvp_main.py --portfolio-analysis --holdings holdings.json
  
  # 反人性交易检查
  python src/mvp_main.py --trading-check --action BUY --stock 000001 --price 15.50
        """
    )
    
    # 主要功能选项
    parser.add_argument('--daily-workflow', action='store_true',
                       help='执行完整的每日交易决策工作流')
    
    parser.add_argument('--sector-analysis', action='store_true',
                       help='仅执行板块分析')
    
    parser.add_argument('--stock-selection', action='store_true',
                       help='仅执行股票筛选 (需要先有板块分析结果)')
    
    parser.add_argument('--portfolio-analysis', action='store_true',
                       help='分析个人投资组合')
    
    parser.add_argument('--trading-check', action='store_true',
                       help='反人性交易决策检查')
    
    # 参数选项
    parser.add_argument('--top-n', type=int, default=5,
                       help='板块分析返回前N个板块 (default: 5)')
    
    parser.add_argument('--lookback-months', type=int, default=6,
                       help='板块分析回望月数 (default: 6)')
    
    parser.add_argument('--stocks-per-sector', type=int, default=5,
                       help='每个板块选择股票数量 (default: 5)')
    
    parser.add_argument('--holdings', type=str,
                       help='持仓文件路径 (JSON格式)')
    
    parser.add_argument('--action', type=str, choices=['BUY', 'SELL', 'HOLD'],
                       help='交易动作')
    
    parser.add_argument('--stock', type=str,
                       help='股票代码')
    
    parser.add_argument('--price', type=float,
                       help='当前价格')
    
    parser.add_argument('--cost-price', type=float,
                       help='成本价格 (用于反人性检查)')
    
    parser.add_argument('--shares', type=int,
                       help='持股数量 (用于反人性检查)')
    
    parser.add_argument('--output-format', choices=['console', 'json'], default='console',
                       help='输出格式 (default: console)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 初始化平台
    platform = TradingDecisionPlatform()
    
    # 检查参数
    if not any([args.daily_workflow, args.sector_analysis, args.stock_selection, 
               args.portfolio_analysis, args.trading_check]):
        # 默认执行每日工作流
        args.daily_workflow = True
    
    try:
        # 执行每日工作流
        if args.daily_workflow:
            print("🚀 启动A股智能交易决策平台 - 每日工作流")
            result = await platform.run_daily_workflow()
            
            if args.output_format == 'json':
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                if 'error' not in result:
                    platform.print_sector_results(result['sector_analysis'])
                    platform.print_stock_results(result['stock_selection'])
                    print(f"\n📝 详细报告已保存: {result.get('report_path', '')}")
                else:
                    print(f"❌ 工作流执行失败: {result['error']}")
        
        # 仅板块分析
        elif args.sector_analysis:
            print("🔍 执行板块分析...")
            result = await platform.sector_engine.analyze_top_sectors(
                lookback_months=args.lookback_months,
                top_n=args.top_n
            )
            
            if args.output_format == 'json':
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                platform.print_sector_results(result)
        
        # 仅股票筛选
        elif args.stock_selection:
            print("📈 执行股票筛选...")
            # 这里需要先获取板块分析结果，简化实现
            sector_result = await platform.sector_engine.analyze_top_sectors(
                lookback_months=args.lookback_months,
                top_n=args.top_n
            )
            
            if 'error' in sector_result:
                print(f"❌ 板块分析失败: {sector_result['error']}")
                return
            
            result = await platform.stock_engine.select_stocks_from_sectors(
                sector_result['top_sectors'],
                stocks_per_sector=args.stocks_per_sector
            )
            
            if args.output_format == 'json':
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                platform.print_stock_results(result)
        
        # 投资组合分析
        elif args.portfolio_analysis:
            if not args.holdings:
                print("❌ 请指定持仓文件路径 (--holdings)")
                return
            
            print(f"📊 分析投资组合: {args.holdings}")
            result = await platform.analyze_portfolio(args.holdings)
            
            if args.output_format == 'json':
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                platform.print_portfolio_results(result)
        
        # 反人性交易检查
        elif args.trading_check:
            if not all([args.action, args.stock, args.price]):
                print("❌ 交易检查需要指定: --action, --stock, --price")
                return
            
            position_info = None
            if args.cost_price and args.shares:
                position_info = {
                    'cost_price': args.cost_price,
                    'shares': args.shares
                }
            
            result = await platform.evaluate_trading_decision(
                args.action, args.stock, args.price, position_info
            )
            
            if args.output_format == 'json':
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                platform.print_anti_human_nature_results(result)
    
    except KeyboardInterrupt:
        print("\n👋 用户中断操作")
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 确保必要的目录存在
    Path('.cache').mkdir(exist_ok=True)
    Path('reports/daily').mkdir(parents=True, exist_ok=True)
    
    # 运行主程序
    asyncio.run(main())