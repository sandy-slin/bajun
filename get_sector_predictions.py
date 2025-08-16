#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块预测输出脚本
专门用于获取各板块的预测结果和详细分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import logging
import json
from datetime import datetime
from data.sector_fetcher import SectorFetcher
from data.enhanced_data_fetcher import EnhancedDataFetcher
from data.technical_calculator import TechnicalCalculator
from analysis.optimized_data_analyzer import OptimizedDataAnalyzer
from cache.manager import CacheManager
from config.settings import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_sector_predictions(analysis_months=2, prediction_days=5, output_format='json'):
    """
    获取所有板块的预测输出
    
    Args:
        analysis_months: 分析月数，默认2个月
        prediction_days: 预测天数，默认5天
        output_format: 输出格式，'json' 或 'table'
    """
    logger.info("🚀 开始获取板块预测输出")
    
    try:
        # 初始化系统组件
        settings = Settings()
        cache_manager = CacheManager(settings.cache_dir)
        sector_fetcher = SectorFetcher(cache_manager)
        enhanced_data_fetcher = EnhancedDataFetcher(cache_manager)
        technical_calculator = TechnicalCalculator()
        
        # 初始化优化分析器
        analyzer = OptimizedDataAnalyzer(
            sector_fetcher=sector_fetcher,
            enhanced_data_fetcher=enhanced_data_fetcher,
            technical_calculator=technical_calculator,
            logger=logger
        )
        
        # 执行预测分析
        logger.info(f"📊 开始分析 {analysis_months} 个月数据，预测 {prediction_days} 天...")
        result = await analyzer.analyze_prediction_accuracy_optimized(
            analysis_months=analysis_months,
            prediction_days=prediction_days
        )
        
        if 'error' in result:
            logger.error(f"❌ 分析失败: {result['error']}")
            return None
        
        # 提取板块预测结果
        enhanced_momentum = result.get('effective_patterns', {}).get('enhanced_momentum', {})
        patterns = enhanced_momentum.get('patterns', [])
        
        if not patterns:
            logger.warning("⚠️ 未发现有效的板块预测模式")
            return None
        
        # 整理预测输出
        predictions = []
        for pattern in patterns:
            prediction = {
                'sector': pattern.get('sector', 'Unknown'),
                'accuracy': round(pattern.get('accuracy', 0), 2),
                'ensemble_accuracy': round(pattern.get('ensemble_accuracy', 0), 2),
                'ts_validated_accuracy': round(pattern.get('ts_validated_accuracy', 0), 2),
                'sample_size': pattern.get('sample_size', 0),
                'confidence': round(pattern.get('confidence', 0), 3),
                'stage': pattern.get('stage', 1),
                'selected_features': pattern.get('selected_features_count', 0),
                'validation_consistency': pattern.get('validation_consistency', 'unknown'),
                'ashare_features_enabled': pattern.get('ashare_features_enabled', False),
                'signal_quality': pattern.get('signal_quality', {}),
                'prediction_window': f"{prediction_days}天",
                'analysis_period': f"{analysis_months}个月"
            }
            
            # 计算预测方向
            if prediction['accuracy'] > 65:
                prediction['direction'] = '看涨'
                prediction['strength'] = '强'
            elif prediction['accuracy'] > 55:
                prediction['direction'] = '看涨'
                prediction['strength'] = '中'
            elif prediction['accuracy'] < 45:
                prediction['direction'] = '看跌'
                prediction['strength'] = '中'
            elif prediction['accuracy'] < 35:
                prediction['direction'] = '看跌'
                prediction['strength'] = '强'
            else:
                prediction['direction'] = '中性'
                prediction['strength'] = '弱'
            
            predictions.append(prediction)
        
        # 按准确率排序
        predictions.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # 输出结果
        if output_format == 'json':
            output_json_format(predictions)
        else:
            output_table_format(predictions)
        
        # 保存到文件
        save_predictions_to_file(predictions)
        
        return predictions
        
    except Exception as e:
        logger.error(f"❌ 获取板块预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def output_json_format(predictions):
    """输出JSON格式的预测结果"""
    print("\n📊 板块预测结果 (JSON格式):")
    print("=" * 80)
    print(json.dumps(predictions, ensure_ascii=False, indent=2))

def output_table_format(predictions):
    """输出表格格式的预测结果"""
    print("\n📊 板块预测结果 (表格格式):")
    print("=" * 120)
    print(f"{'板块名称':<15} {'准确率':<8} {'集成准确率':<10} {'时序准确率':<10} {'预测方向':<8} {'强度':<6} {'置信度':<8} {'样本数':<8} {'阶段':<6}")
    print("-" * 120)
    
    for pred in predictions:
        print(f"{pred['sector']:<15} {pred['accuracy']:<8.1f}% {pred['ensemble_accuracy']:<10.1f}% {pred['ts_validated_accuracy']:<10.1f}% {pred['direction']:<8} {pred['strength']:<6} {pred['confidence']:<8.3f} {pred['sample_size']:<8} {pred['stage']:<6}")
    
    # 统计摘要
    total_sectors = len(predictions)
    avg_accuracy = sum(p['accuracy'] for p in predictions) / total_sectors if total_sectors > 0 else 0
    high_accuracy_count = sum(1 for p in predictions if p['accuracy'] > 66.5)
    
    print("-" * 120)
    print(f"📈 统计摘要:")
    print(f"   总板块数: {total_sectors}")
    print(f"   平均准确率: {avg_accuracy:.1f}%")
    print(f"   高准确率板块数 (>66.5%): {high_accuracy_count}")
    print(f"   高准确率占比: {(high_accuracy_count/total_sectors*100):.1f}%" if total_sectors > 0 else "0%")

def save_predictions_to_file(predictions):
    """保存预测结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sector_predictions_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_sectors': len(predictions),
                'predictions': predictions
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 预测结果已保存到: {filename}")
        
    except Exception as e:
        print(f"⚠️ 保存文件失败: {e}")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='获取板块预测输出')
    parser.add_argument('--months', type=int, default=2, help='分析月数 (默认: 2)')
    parser.add_argument('--days', type=int, default=5, help='预测天数 (默认: 5)')
    parser.add_argument('--format', choices=['json', 'table'], default='table', help='输出格式 (默认: table)')
    
    args = parser.parse_args()
    
    predictions = await get_sector_predictions(
        analysis_months=args.months,
        prediction_days=args.days,
        output_format=args.format
    )
    
    if predictions:
        print(f"\n✅ 成功获取 {len(predictions)} 个板块的预测结果")
    else:
        print("\n❌ 未能获取板块预测结果")

if __name__ == "__main__":
    asyncio.run(main())