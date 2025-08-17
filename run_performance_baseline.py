#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史性能基准建立 - Phase 2核心任务
运行5个时间点的回测验证，建立系统性能基准
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.sector_engine import SectorEngine
from src.core.stock_engine import StockEngine  
from src.core.portfolio_engine import PortfolioEngine
from src.validation.performance_validator import PerformanceValidator
from src.data.sector_fetcher import SectorFetcher
from src.data.trading_data_fetcher import TradingDataFetcher
from src.data.technical_calculator import TechnicalCalculator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """运行历史性能基准建立"""
    print("=" * 70)
    print("🎯 Phase 2: 历史性能基准建立")
    print("目标: 运行5个时间点回测验证，建立系统算法性能基准")
    print("=" * 70)
    
    try:
        # 初始化核心组件
        print("\n🔧 初始化系统组件...")
        
        # 数据层组件
        sector_fetcher = SectorFetcher()
        data_fetcher = TradingDataFetcher()
        tech_calculator = TechnicalCalculator()
        
        # 核心引擎
        sector_engine = SectorEngine(sector_fetcher, tech_calculator)
        stock_engine = StockEngine(data_fetcher, tech_calculator)
        portfolio_engine = PortfolioEngine(data_fetcher, tech_calculator)
        
        # 性能验证器
        performance_validator = PerformanceValidator(
            sector_engine, stock_engine, portfolio_engine
        )
        
        print("✅ 系统组件初始化完成")
        
        # 定义验证时间点 (过去5个时间点)
        validation_timepoints = [
            '20240701',  # 2024年7月1日
            '20240801',  # 2024年8月1日  
            '20240901',  # 2024年9月1日
            '20241001',  # 2024年10月1日
            '20241101'   # 2024年11月1日
        ]
        
        print(f"\n📅 设定验证时间点:")
        for i, timepoint in enumerate(validation_timepoints, 1):
            print(f"   {i}. {timepoint}")
        
        # 运行综合性能验证
        print(f"\n🚀 开始运行历史性能基准验证...")
        print(f"   预测天数: 5天")
        print(f"   验证期数: {len(validation_timepoints)}个")
        
        validation_start = datetime.now()
        
        validation_results = await performance_validator.run_comprehensive_validation(
            t_values=validation_timepoints,
            prediction_days=5,
            save_results=True
        )
        
        validation_duration = (datetime.now() - validation_start).total_seconds()
        
        # 显示验证结果
        print(f"\n✅ 历史性能验证完成！耗时: {validation_duration:.1f}秒")
        
        if 'error' in validation_results:
            print(f"❌ 验证失败: {validation_results['error']}")
            return False
        
        # 显示核心指标
        metadata = validation_results.get('validation_metadata', {})
        comprehensive = validation_results.get('comprehensive_analysis', {})
        summary = validation_results.get('performance_summary', {})
        
        print(f"\n📊 验证元数据:")
        print(f"   验证期数: {metadata.get('validation_periods', 0)}")
        print(f"   预测天数: {metadata.get('prediction_days', 0)}")
        print(f"   处理时间: {metadata.get('processing_time_seconds', 0):.1f}秒")
        
        if comprehensive:
            print(f"\n🎯 核心性能指标:")
            
            # 板块分析性能
            sector_perf = comprehensive.get('sector_analysis_performance', {})
            if sector_perf:
                print(f"   📈 板块分析:")
                print(f"      平均准确率: {sector_perf.get('average_accuracy', 0):.1%}")
                print(f"      最佳准确率: {sector_perf.get('best_accuracy', 0):.1%}")
                print(f"      稳定性: {sector_perf.get('accuracy_stability', 0):.3f}")
            
            # 股票选择性能
            stock_perf = comprehensive.get('stock_selection_performance', {})
            if stock_perf:
                print(f"   📊 股票选择:")
                print(f"      平均胜率: {stock_perf.get('average_win_rate', 0):.1%}")
                print(f"      最佳胜率: {stock_perf.get('best_win_rate', 0):.1%}")
                print(f"      稳定性: {stock_perf.get('win_rate_stability', 0):.3f}")
            
            # 组合管理性能
            portfolio_perf = comprehensive.get('portfolio_performance', {})
            if portfolio_perf:
                print(f"   💼 组合管理:")
                print(f"      平均收益: {portfolio_perf.get('average_return', 0):.2f}%")
                print(f"      最佳收益: {portfolio_perf.get('best_return', 0):.2f}%")
                print(f"      盈利期数: {portfolio_perf.get('positive_periods', 0)}")
            
            # 整体评估
            overall = comprehensive.get('overall_assessment', {})
            if overall:
                print(f"   🏆 整体评估:")
                print(f"      平均得分: {overall.get('average_score', 0):.1f}")
                print(f"      成功率: {overall.get('success_rate', 0):.1%}")
                print(f"      系统等级: {overall.get('grade', 'Unknown')}")
        
        if summary:
            print(f"\n📋 性能摘要:")
            print(f"   系统等级: {summary.get('overall_grade', 'Unknown')}")
            print(f"   平均得分: {summary.get('average_score', 0):.1f}")
            print(f"   成功率: {summary.get('success_rate', 0):.1f}%")
            print(f"   系统就绪状态: {summary.get('system_readiness', 'Unknown')}")
        
        # 显示优势和改进建议
        if 'key_strengths' in summary:
            print(f"\n✨ 系统优势:")
            for strength in summary['key_strengths']:
                print(f"   ✅ {strength}")
        
        if 'improvement_areas' in summary:
            print(f"\n🔧 改进方向:")
            for area in summary['improvement_areas']:
                print(f"   🎯 {area}")
        
        # 显示改进建议
        recommendations = validation_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # 保存位置
        if 'report_path' in validation_results:
            print(f"\n📄 详细报告已保存至: {validation_results['report_path']}")
        
        # 基准建立成功
        print(f"\n" + "=" * 70)
        if summary.get('system_readiness') in ['production_ready', 'beta_ready']:
            print("🎉 历史性能基准建立成功！")
            print("✅ 系统已准备好进行算法优化阶段")
        else:
            print("⚠️  历史性能基准建立完成，但系统需要进一步优化")
            print("🔧 建议在继续开发前解决发现的问题")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 历史性能基准建立失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 检查数据源连接")
        print("2. 确认历史数据可用性")
        print("3. 验证系统组件完整性")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)