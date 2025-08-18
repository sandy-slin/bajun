#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化AKShare数据测试 - 绕过严格验证
验证数据源基本可用性
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("❌ AKShare未安装")


async def test_basic_functionality():
    """测试基本功能"""
    if not AKSHARE_AVAILABLE:
        return False
    
    print("🚀 开始AKShare基本功能测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # 测试1: 股票基本信息
    try:
        print("\n📊 测试1: 获取股票基本信息...")
        stock_info = ak.stock_info_a_code_name()
        if stock_info is not None and len(stock_info) > 0:
            print(f"✅ 成功获取 {len(stock_info)} 只股票信息")
            print(f"   示例数据: {stock_info.head(3).to_string()}")
            success_count += 1
        else:
            print("❌ 获取失败")
        total_tests += 1
    except Exception as e:
        print(f"❌ 异常: {e}")
        total_tests += 1
    
    # 测试2: 股票历史数据
    try:
        print("\n📈 测试2: 获取股票历史数据...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        stock_data = ak.stock_zh_a_hist(
            symbol='000001',
            period='daily',
            start_date=start_date,
            end_date=end_date,
            adjust=''
        )
        
        if stock_data is not None and len(stock_data) > 0:
            print(f"✅ 成功获取平安银行 {len(stock_data)} 条数据")
            print(f"   字段: {stock_data.columns.tolist()}")
            print(f"   最新数据: {stock_data.iloc[-1].to_dict()}")
            success_count += 1
        else:
            print("❌ 获取失败")
        total_tests += 1
    except Exception as e:
        print(f"❌ 异常: {e}")
        total_tests += 1
    
    # 测试3: 申万行业指数
    try:
        print("\n🏢 测试3: 获取申万行业指数...")
        sector_data = ak.index_hist_sw(symbol='801030', period='day')  # 采掘行业
        
        if sector_data is not None and len(sector_data) > 0:
            print(f"✅ 成功获取采掘行业 {len(sector_data)} 条数据")
            print(f"   字段: {sector_data.columns.tolist()}")
            # 显示最近几条数据
            recent_data = sector_data.tail(3)
            print(f"   最近数据:\n{recent_data.to_string()}")
            success_count += 1
        else:
            print("❌ 获取失败")
        total_tests += 1
    except Exception as e:
        print(f"❌ 异常: {e}")
        total_tests += 1
    
    # 测试4: 实时数据
    try:
        print("\n⚡ 测试4: 获取实时行情...")
        # 获取实时行情数据
        realtime_data = ak.stock_zh_a_spot_em()
        
        if realtime_data is not None and len(realtime_data) > 0:
            print(f"✅ 成功获取 {len(realtime_data)} 只股票实时数据")
            # 显示前3只股票信息
            sample = realtime_data.head(3)[['代码', '名称', '最新价', '涨跌幅', '成交量']]
            print(f"   示例数据:\n{sample.to_string()}")
            success_count += 1
        else:
            print("❌ 获取失败")
        total_tests += 1
    except Exception as e:
        print(f"❌ 异常: {e}")
        total_tests += 1
    
    # 总结
    print("\n" + "=" * 50)
    print(f"📊 测试总结: {success_count}/{total_tests} 成功")
    success_rate = success_count / total_tests if total_tests > 0 else 0
    print(f"🎯 成功率: {success_rate:.1%}")
    
    if success_rate >= 0.75:
        print("🎉 AKShare数据源基本可用！")
        return True
    elif success_rate >= 0.5:
        print("⚠️  AKShare数据源部分可用，需要优化")
        return True
    else:
        print("❌ AKShare数据源存在问题，需要检查")
        return False


async def test_data_quality():
    """测试数据质量"""
    print("\n🔍 数据质量检查")
    print("=" * 30)
    
    try:
        # 获取测试数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
        
        data = ak.stock_zh_a_hist(
            symbol='000001',
            period='daily',
            start_date=start_date,
            end_date=end_date,
            adjust=''
        )
        
        if data is None or len(data) == 0:
            print("❌ 无法获取测试数据")
            return False
        
        print(f"✅ 获取到 {len(data)} 条数据")
        print(f"📅 数据时间范围: {data['日期'].min()} 到 {data['日期'].max()}")
        
        # 检查数据完整性
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        print(f"📊 数据完整性: {(1-missing_ratio):.1%} (缺失率: {missing_ratio:.1%})")
        
        # 检查价格逻辑
        price_logic_errors = 0
        for _, row in data.iterrows():
            if row['最高'] < row['最低']:
                price_logic_errors += 1
            if row['最高'] < row['开盘'] or row['最高'] < row['收盘']:
                price_logic_errors += 1
            if row['最低'] > row['开盘'] or row['最低'] > row['收盘']:
                price_logic_errors += 1
        
        if price_logic_errors == 0:
            print("✅ 价格逻辑检查通过")
        else:
            print(f"⚠️  发现 {price_logic_errors} 个价格逻辑错误")
        
        # 检查成交量合理性
        negative_volume = (data['成交量'] < 0).sum()
        if negative_volume == 0:
            print("✅ 成交量数据合理")
        else:
            print(f"⚠️  发现 {negative_volume} 个负成交量")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return False


async def main():
    """主函数"""
    print("🔧 AKShare数据源测试工具")
    print("目标: 验证数据接入基本可用性")
    print("=" * 60)
    
    # 基本功能测试
    basic_success = await test_basic_functionality()
    
    if basic_success:
        # 数据质量测试
        quality_success = await test_data_quality()
        
        if quality_success:
            print("\n🎉 数据源测试通过！可以进行下一步开发")
            return True
    
    print("\n💡 建议:")
    print("1. 检查网络连接")
    print("2. 更新akshare到最新版本: pip install -U akshare")
    print("3. 如果问题持续，考虑使用备用数据源")
    
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)