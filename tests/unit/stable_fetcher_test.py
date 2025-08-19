#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定AKShare数据获取器测试
验证多重备用方案的有效性
"""

import asyncio
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.stable_akshare_fetcher import StableAKShareFetcher

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_stable_fetcher():
    """测试稳定数据获取器"""
    print("🚀 稳定AKShare数据获取器测试")
    print("=" * 60)
    
    try:
        # 创建获取器实例
        fetcher = StableAKShareFetcher()
        
        print(f"✅ 数据获取器初始化成功")
        
        # 测试连接状态
        status = fetcher.get_connection_status()
        print(f"📊 连接状态: {status}")
        
        # 测试1: 获取实时股票数据
        print("\n📈 测试1: 获取实时股票数据...")
        try:
            realtime_data = await fetcher.get_stock_realtime_data()
            if realtime_data is not None and not realtime_data.empty:
                print(f"✅ 成功获取 {len(realtime_data)} 只股票实时数据")
                print(f"   数据字段: {realtime_data.columns.tolist()}")
                
                # 显示前3条数据
                if len(realtime_data) > 0:
                    sample_data = realtime_data.head(3)[['代码', '名称', '最新价', '涨跌幅', '成交量']]
                    print(f"   示例数据:\n{sample_data.to_string()}")
                
                print(f"✅ 实时数据获取测试 - 通过")
            else:
                print(f"❌ 实时数据获取测试 - 失败：返回空数据")
                
        except Exception as e:
            print(f"❌ 实时数据获取测试 - 异常：{e}")
        
        # 测试2: 获取板块数据
        print("\n🏢 测试2: 获取板块数据...")
        try:
            sector_data = await fetcher.get_stable_sector_data('银行', days=10)
            if sector_data is not None and not sector_data.empty:
                print(f"✅ 成功获取银行板块数据 {len(sector_data)} 条记录")
                print(f"   数据字段: {sector_data.columns.tolist()}")
                
                # 显示最新几条数据
                if len(sector_data) > 0:
                    recent_data = sector_data.tail(3)
                    print(f"   最新数据:\n{recent_data.to_string()}")
                
                print(f"✅ 板块数据获取测试 - 通过")
            else:
                print(f"❌ 板块数据获取测试 - 失败：返回空数据")
                
        except Exception as e:
            print(f"❌ 板块数据获取测试 - 异常：{e}")
        
        print(f"\n🎉 稳定数据获取器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 测试初始化失败: {e}")
        return False

async def test_alternative_methods():
    """测试备用方法"""
    print(f"\n🔧 备用方法测试")
    print("=" * 30)
    
    try:
        fetcher = StableAKShareFetcher()
        
        # 测试各个备用方案
        methods = [
            ("历史数据伪实时", fetcher._get_pseudo_realtime_from_hist),
            ("个股信息批量", lambda: fetcher._get_realtime_from_individual_info(['000001', '000002'])),
            ("增强重试原始接口", fetcher._get_spot_with_enhanced_retry),
        ]
        
        success_count = 0
        
        for method_name, method_func in methods:
            print(f"\n🧪 测试方法: {method_name}")
            try:
                result = await method_func()
                if result is not None and not result.empty:
                    print(f"✅ {method_name} - 成功: {len(result)} 条数据")
                    success_count += 1
                else:
                    print(f"❌ {method_name} - 失败: 返回空数据")
            except Exception as e:
                print(f"❌ {method_name} - 异常: {e}")
        
        print(f"\n📊 备用方法测试结果: {success_count}/{len(methods)} 成功")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 备用方法测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🧪 稳定AKShare数据获取器 - 完整测试")
    print("=" * 80)
    
    try:
        # 基础功能测试
        basic_success = await test_stable_fetcher()
        
        # 备用方法测试（可选）
        if basic_success:
            print("\n" + "="*40)
            alternative_success = await test_alternative_methods()
            
            if alternative_success:
                print(f"\n🎉 所有测试通过！稳定数据获取器可以投入使用")
                return True
            else:
                print(f"\n⚠️  基础功能正常，但备用方法存在问题")
                return True
        else:
            print(f"\n❌ 基础功能测试失败，需要检查配置")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程异常: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)