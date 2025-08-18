#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKShare数据集成测试脚本
Phase 2: 验证真实数据源接入效果
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.real_akshare_integration import RealAKShareIntegration

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """运行AKShare集成测试"""
    print("=" * 60)
    print("🚀 Phase 2: AKShare真实数据集成测试")
    print("=" * 60)
    
    try:
        # 初始化集成器
        integration = RealAKShareIntegration()
        
        # 运行连接性测试
        print("\n📊 开始AKShare连接性和质量测试...")
        test_results = await integration.test_akshare_connectivity()
        
        # 显示测试结果
        print(f"\n✅ 测试完成状态: {'成功' if test_results.get('overall_success') else '失败'}")
        print(f"🕐 测试时间: {test_results.get('test_timestamp', 'unknown')}")
        
        if 'detailed_results' in test_results:
            detailed = test_results['detailed_results']
            
            # 连接性测试结果
            if 'connectivity_test' in detailed:
                conn = detailed['connectivity_test']
                print(f"\n🔗 连接性测试: {'✅' if conn.get('success') else '❌'}")
                if conn.get('success'):
                    print(f"   - 股票数量: {conn.get('stock_count', 0)}")
                    print(f"   - 数据质量评分: {conn.get('data_quality_score', 0):.2f}")
                
            # 股票数据测试结果
            if 'stock_data_test' in detailed:
                stock = detailed['stock_data_test']
                print(f"\n📈 股票数据测试: {'✅' if stock.get('success') else '❌'}")
                print(f"   - 成功率: {stock.get('success_rate', 0):.1%}")
                print(f"   - 平均质量评分: {stock.get('average_quality_score', 0):.2f}")
                if stock.get('successful_stocks'):
                    print(f"   - 成功获取股票: {len(stock['successful_stocks'])}只")
                
            # 板块数据测试结果
            if 'sector_data_test' in detailed:
                sector = detailed['sector_data_test']
                print(f"\n🏢 板块数据测试: {'✅' if sector.get('success') else '❌'}")
                print(f"   - 成功率: {sector.get('success_rate', 0):.1%}")
                print(f"   - 平均质量评分: {sector.get('average_quality_score', 0):.2f}")
                if sector.get('successful_sectors'):
                    print(f"   - 成功获取板块: {len(sector['successful_sectors'])}个")
                
            # 数据质量测试结果
            if 'data_quality_test' in detailed:
                quality = detailed['data_quality_test']
                print(f"\n🔍 数据质量测试: {'✅' if quality.get('success') else '❌'}")
                print(f"   - 综合质量评分: {quality.get('overall_quality_score', 0):.2f}")
                
            # 性能测试结果
            if 'performance_test' in detailed:
                perf = detailed['performance_test']
                print(f"\n⚡ 性能测试: {'✅' if perf.get('success') else '❌'}")
                print(f"   - 单次请求耗时: {perf.get('single_request_time', 0):.1f}秒")
                print(f"   - 并发请求耗时: {perf.get('concurrent_request_time', 0):.1f}秒")
                print(f"   - 性能等级: {perf.get('performance_grade', 'unknown')}")
        
        # 显示建议
        if 'recommendations' in test_results:
            print("\n💡 集成建议:")
            for i, rec in enumerate(test_results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # 测试实际数据获取
        print("\n" + "=" * 40)
        print("🧪 实际数据获取测试")
        print("=" * 40)
        
        # 测试股票数据获取
        print("\n📊 测试股票数据获取...")
        test_stocks = ['000001', '600519', '000858']
        
        for stock_code in test_stocks:
            print(f"\n正在测试股票 {stock_code}...")
            stock_data = await integration.get_real_stock_data(stock_code, days=10)
            
            if stock_data:
                print(f"   ✅ 成功获取 {len(stock_data)} 条数据")
                if stock_data:
                    latest = stock_data[-1]
                    print(f"   - 最新日期: {latest['date']}")
                    print(f"   - 收盘价: ¥{latest['close']:.2f}")
                    print(f"   - 成交量: {latest['volume']:,}")
            else:
                print(f"   ❌ 获取失败")
        
        # 测试板块数据获取
        print("\n🏢 测试板块数据获取...")
        test_sectors = ['银行', '食品饮料', '医药生物']
        
        for sector_name in test_sectors:
            print(f"\n正在测试板块 {sector_name}...")
            sector_data = await integration.get_real_sector_data(sector_name, days=10)
            
            if sector_data:
                print(f"   ✅ 成功获取 {len(sector_data)} 条数据")
                if sector_data:
                    latest = sector_data[-1]
                    print(f"   - 最新日期: {latest['date']}")
                    print(f"   - 收盘价: {latest['close']:.2f}")
                    print(f"   - 成交量: {latest['volume']:,}")
            else:
                print(f"   ❌ 获取失败")
        
        # 总结
        print("\n" + "=" * 60)
        if test_results.get('overall_success'):
            print("🎉 AKShare数据集成测试通过！")
            print("✅ 系统已准备好使用真实数据进行分析")
        else:
            print("⚠️  AKShare数据集成测试未完全通过")
            print("🔧 建议检查网络连接和AKShare配置")
        print("=" * 60)
        
        return test_results.get('overall_success', False)
        
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)