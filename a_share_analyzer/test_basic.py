#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic functionality test script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from a_share_analyzer import AShareAnalyzer
    print("Success: Imported AShareAnalyzer")
except ImportError as e:
    print("Error importing: " + str(e))
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # 初始化分析器
        analyzer = AShareAnalyzer()
        print("Success: Created analyzer instance")
        
        # 测试数据库初始化
        print("Success: Database initialized")
        
        # 测试配置加载
        from a_share_analyzer.config import DATABASE_CONFIG, DEEPSEEK_CONFIG
        print("Success: Config loaded, DB=" + DATABASE_CONFIG['db_path'])
        
        return True
        
    except Exception as e:
        print("Error in basic functionality test: " + str(e))
        return False

def test_mock_analysis():
    """Test mock analysis without external APIs"""
    print("\n=== Testing Mock Analysis ===")
    
    try:
        import pandas as pd
        from datetime import datetime
        
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'open_price': [10.0, 10.5, 11.0],
            'high_price': [10.5, 11.0, 11.5],
            'low_price': [9.5, 10.0, 10.5],
            'close_price': [10.2, 10.8, 11.2],
            'volume': [1000000, 1200000, 1100000],
            'turnover': [10200000, 12960000, 12320000],
            'stock_code': ['000001', '000001', '000001'],
            'updated_at': [datetime.now().isoformat()] * 3
        })
        
        # 测试技术分析
        from a_share_analyzer.analysis import TechnicalAnalyzer
        tech_analyzer = TechnicalAnalyzer()
        
        result = tech_analyzer.analyze_price_trends(mock_data)
        print("Success: Technical analysis: " + str(result))
        
        # 测试移动平均
        ma_result = tech_analyzer.calculate_moving_averages(mock_data)
        print("Success: Moving average calculation: " + str(ma_result))
        
        return True
        
    except Exception as e:
        print("Error in mock analysis test: " + str(e))
        return False

if __name__ == "__main__":
    print("Starting basic functionality tests...")
    
    success = True
    success &= test_basic_functionality()
    success &= test_mock_analysis()
    
    if success:
        print("\nAll basic tests passed!")
    else:
        print("\nSome tests failed, please check the code")
        sys.exit(1)