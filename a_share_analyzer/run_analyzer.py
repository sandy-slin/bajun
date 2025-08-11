#!/usr/bin/env python3
"""
A股分析程序入口脚本
"""

from a_share_analyzer import AShareAnalyzer


def main():
    """主程序入口"""
    analyzer = AShareAnalyzer()
    
    # 获取用户输入
    stock_codes = input("请输入股票代码（多个用逗号分隔，如: 000001,000002）: ").strip()
    if not stock_codes:
        stock_codes = "sz300061"
    
    # 解析股票代码
    codes = [code.strip() for code in stock_codes.split(',')]
    
    # 执行分析
    print(f"\n开始分析 {len(codes)} 只股票...")
    results = analyzer.analyze_multiple_stocks(codes)
    
    # 保存结果
    analyzer.save_analysis_results(results)
    
    print(f"\n分析完成！共分析了 {len(results)} 只股票。")


if __name__ == "__main__":
    main()