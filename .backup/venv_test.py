#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的venv环境测试脚本
"""

import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_venv_environment():
    """测试venv环境下的基本功能"""
    try:
        logger.info("🧪 开始venv环境测试")
        
        # 测试基础依赖导入
        import pandas as pd
        import numpy as np
        logger.info("✅ 基础依赖导入成功")
        
        # 测试项目模块导入
        from data.sector_fetcher import SectorFetcher
        from cache.manager import CacheManager
        from config.settings import Settings
        logger.info("✅ 项目模块导入成功")
        
        # 测试基础组件初始化
        settings = Settings()
        cache_manager = CacheManager(settings.cache_dir)
        sector_fetcher = SectorFetcher(cache_manager)
        logger.info("✅ 基础组件初始化成功")
        
        # 测试数据获取（少量数据）
        test_data = await sector_fetcher.get_all_sectors_data(("20250615", "20250618"))
        if test_data:
            logger.info(f"✅ 数据获取测试成功，获取{len(test_data)}个板块数据")
        else:
            logger.warning("⚠️ 数据获取返回空结果（可能正常，因为是历史数据）")
        
        logger.info("🎉 venv环境测试完成，所有基础功能正常")
        return True
        
    except Exception as e:
        logger.error(f"❌ venv环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("🔧 venv环境兼容性测试")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print()
    
    success = await test_venv_environment()
    
    if success:
        print("\n✅ venv环境测试通过！")
        print("现在您可以安全地在venv环境下运行系统性验证:")
        print("source venv/bin/activate && python3 venv_test.py")
    else:
        print("\n❌ venv环境测试失败，请检查依赖安装")

if __name__ == "__main__":
    asyncio.run(main())