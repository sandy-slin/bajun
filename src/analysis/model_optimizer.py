# -*- coding: utf-8 -*-
"""
模型优化器
基于回测验证结果自动优化预测模型参数，提升预测准确性
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import json
import itertools
from copy import deepcopy

from analysis.prediction_validator import PredictionValidator
from analysis.enhanced_predictor import EnhancedPredictor
from data.enhanced_data_fetcher import EnhancedDataFetcher
from data.sector_fetcher import SectorFetcher
from data.technical_calculator import TechnicalCalculator


class ModelOptimizer:
    """模型优化器 - 自动调优预测模型参数"""
    
    def __init__(self, sector_fetcher: SectorFetcher, enhanced_data_fetcher: EnhancedDataFetcher,
                 tech_calculator: TechnicalCalculator):
        self.sector_fetcher = sector_fetcher
        self.enhanced_data_fetcher = enhanced_data_fetcher
        self.tech_calculator = tech_calculator
        self.logger = logging.getLogger(__name__)
        
        # 参数优化空间
        self.weight_search_space = {
            'technical': [0.25, 0.30, 0.35, 0.40, 0.45],
            'money_flow': [0.20, 0.25, 0.30, 0.35],
            'fundamental': [0.10, 0.15, 0.20, 0.25],
            'rotation': [0.05, 0.10, 0.15],
            'northbound': [0.05, 0.08, 0.10, 0.12],
            'margin': [0.02, 0.04, 0.06],
            'sentiment': [0.01, 0.03, 0.05]
        }
        
    async def optimize_model(self, optimization_cycles: int = 5, 
                           validation_periods: int = 8) -> Dict[str, Any]:
        """
        优化模型参数
        
        Args:
            optimization_cycles: 优化轮数
            validation_periods: 每轮验证期数
            
        Returns:
            Dict: 优化结果
        """
        try:
            self.logger.info(f"开始模型优化，优化轮数: {optimization_cycles}, 验证期数: {validation_periods}")
            
            # 获取基线性能
            baseline_result = await self._get_baseline_performance(validation_periods)
            baseline_accuracy = baseline_result.get('overall_statistics', {}).get('accuracy', 0)
            
            self.logger.info(f"基线准确率: {baseline_accuracy:.3f}")
            
            # 执行参数搜索优化
            optimization_results = []
            best_accuracy = baseline_accuracy
            best_config = None
            
            for cycle in range(optimization_cycles):
                self.logger.info(f"优化轮次: {cycle + 1}/{optimization_cycles}")
                
                # 生成候选参数配置
                candidate_configs = self._generate_candidate_configurations(
                    cycle, optimization_cycles, best_config
                )
                
                # 测试候选配置
                cycle_results = []
                for i, config in enumerate(candidate_configs):
                    self.logger.info(f"测试配置 {i+1}/{len(candidate_configs)}")
                    
                    result = await self._test_configuration(config, validation_periods)
                    accuracy = result.get('overall_statistics', {}).get('accuracy', 0)
                    
                    cycle_results.append({
                        'config': config,
                        'accuracy': accuracy,
                        'validation_result': result
                    })
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_config = config
                        self.logger.info(f"发现更优配置，准确率: {accuracy:.3f}")
                
                # 记录本轮最佳结果
                cycle_best = max(cycle_results, key=lambda x: x['accuracy'])
                optimization_results.append({
                    'cycle': cycle + 1,
                    'best_accuracy': cycle_best['accuracy'],
                    'best_config': cycle_best['config'],
                    'improvement': cycle_best['accuracy'] - baseline_accuracy,
                    'configs_tested': len(candidate_configs)
                })
                
                self.logger.info(f"轮次{cycle + 1}完成，最佳准确率: {cycle_best['accuracy']:.3f}")
            
            # 最终验证最佳配置
            if best_config:
                final_validation = await self._test_configuration(
                    best_config, validation_periods * 2  # 更长的验证期
                )
            else:
                final_validation = baseline_result
                
            optimization_summary = {
                'optimization_time': datetime.now().isoformat(),
                'optimization_cycles': optimization_cycles,
                'validation_periods': validation_periods,
                'baseline_accuracy': baseline_accuracy,
                'best_accuracy': best_accuracy,
                'improvement': best_accuracy - baseline_accuracy,
                'improvement_percentage': (best_accuracy - baseline_accuracy) / baseline_accuracy * 100 if baseline_accuracy > 0 else 0,
                'best_config': best_config,
                'optimization_results': optimization_results,
                'final_validation': final_validation,
                'recommendations': self._generate_recommendations(optimization_results, best_config)
            }
            
            self.logger.info(f"模型优化完成，准确率提升: {best_accuracy - baseline_accuracy:.3f}")
            
            return optimization_summary
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return {'error': str(e)}
            
    async def _get_baseline_performance(self, validation_periods: int) -> Dict[str, Any]:
        """获取基线性能"""
        try:
            # 使用默认配置创建预测器和验证器
            enhanced_predictor = EnhancedPredictor(self.enhanced_data_fetcher, self.tech_calculator)
            validator = PredictionValidator(
                self.sector_fetcher, self.enhanced_data_fetcher, 
                enhanced_predictor, self.tech_calculator
            )
            
            result = await validator.validate_predictions(
                validation_periods=validation_periods,
                prediction_days=3
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取基线性能失败: {e}")
            return {}
            
    def _generate_candidate_configurations(self, cycle: int, total_cycles: int, 
                                         best_config: Optional[Dict] = None) -> List[Dict]:
        """生成候选参数配置"""
        try:
            configs = []
            
            if cycle == 0:
                # 第一轮：网格搜索主要参数
                main_params = ['technical', 'money_flow', 'fundamental']
                combinations = list(itertools.product(
                    *[self.weight_search_space[param] for param in main_params]
                ))
                
                for combo in combinations[:15]:  # 限制数量
                    config = dict(zip(main_params, combo))
                    # 其他参数使用默认值
                    config.update({
                        'rotation': 0.10,
                        'northbound': 0.08,
                        'margin': 0.04,
                        'sentiment': 0.03
                    })
                    # 归一化权重
                    config = self._normalize_weights(config)
                    configs.append(config)
                    
            elif cycle < total_cycles - 1:
                # 中间轮：基于最佳配置的邻域搜索
                if best_config:
                    configs.extend(self._generate_neighbor_configs(best_config))
                else:
                    configs.append(self._get_default_config())
                    
            else:
                # 最后一轮：精细调优
                if best_config:
                    configs.extend(self._fine_tune_config(best_config))
                else:
                    configs.append(self._get_default_config())
                    
            return configs[:10]  # 每轮最多测试10个配置
            
        except Exception as e:
            self.logger.error(f"生成候选配置失败: {e}")
            return [self._get_default_config()]
            
    def _generate_neighbor_configs(self, base_config: Dict) -> List[Dict]:
        """生成邻域配置"""
        neighbors = []
        
        try:
            for param in base_config.keys():
                if param in self.weight_search_space:
                    current_value = base_config[param]
                    search_values = self.weight_search_space[param]
                    
                    # 找到当前值附近的值
                    for value in search_values:
                        if abs(value - current_value) <= 0.05 and value != current_value:
                            new_config = base_config.copy()
                            new_config[param] = value
                            neighbors.append(self._normalize_weights(new_config))
                            
            return neighbors[:8]
            
        except Exception as e:
            self.logger.error(f"生成邻域配置失败: {e}")
            return []
            
    def _fine_tune_config(self, base_config: Dict) -> List[Dict]:
        """精细调优配置"""
        fine_tuned = []
        
        try:
            # 对主要参数进行微调
            main_params = ['technical', 'money_flow', 'fundamental']
            
            for param in main_params:
                current_value = base_config[param]
                
                # 生成微调变化
                for delta in [-0.02, -0.01, 0.01, 0.02]:
                    new_value = current_value + delta
                    if 0.1 <= new_value <= 0.5:  # 确保在合理范围内
                        new_config = base_config.copy()
                        new_config[param] = new_value
                        fine_tuned.append(self._normalize_weights(new_config))
                        
            return fine_tuned[:6]
            
        except Exception as e:
            self.logger.error(f"精细调优失败: {e}")
            return []
            
    def _normalize_weights(self, config: Dict) -> Dict:
        """权重归一化"""
        try:
            total_weight = sum(config.values())
            if total_weight > 0:
                normalized = {k: round(v / total_weight, 3) for k, v in config.items()}
                return normalized
            else:
                return self._get_default_config()
                
        except Exception as e:
            self.logger.error(f"权重归一化失败: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'technical': 0.35,
            'money_flow': 0.25,
            'fundamental': 0.15,
            'rotation': 0.10,
            'northbound': 0.08,
            'margin': 0.04,
            'sentiment': 0.03
        }
        
    async def _test_configuration(self, config: Dict, validation_periods: int) -> Dict[str, Any]:
        """测试特定配置"""
        try:
            # 创建临时预测器使用指定配置
            enhanced_predictor = EnhancedPredictor(self.enhanced_data_fetcher, self.tech_calculator)
            enhanced_predictor.base_weights = config
            
            validator = PredictionValidator(
                self.sector_fetcher, self.enhanced_data_fetcher,
                enhanced_predictor, self.tech_calculator
            )
            
            result = await validator.validate_predictions(
                validation_periods=validation_periods,
                prediction_days=3
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"测试配置失败: {e}")
            return {'overall_statistics': {'accuracy': 0}}
            
    def _generate_recommendations(self, optimization_results: List[Dict], 
                                best_config: Optional[Dict]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        try:
            if not optimization_results:
                recommendations.append("优化过程中未获得有效结果，建议检查数据质量")
                return recommendations
                
            # 分析优化趋势
            improvements = [r.get('improvement', 0) for r in optimization_results]
            max_improvement = max(improvements) if improvements else 0
            
            if max_improvement > 0.05:  # 5%以上提升
                recommendations.append(f"模型优化效果显著，准确率提升了{max_improvement:.1%}")
            elif max_improvement > 0.02:  # 2-5%提升
                recommendations.append(f"模型优化有一定效果，准确率提升了{max_improvement:.1%}")
            else:
                recommendations.append("模型优化效果有限，可能需要更多数据或不同的优化策略")
                
            # 分析最佳配置
            if best_config:
                max_weight_param = max(best_config, key=best_config.get)
                recommendations.append(f"最重要的预测因子是{max_weight_param}，权重为{best_config[max_weight_param]:.3f}")
                
                # 检查权重平衡性
                weight_std = np.std(list(best_config.values()))
                if weight_std > 0.15:
                    recommendations.append("权重分布不均匀，考虑进一步平衡各因子权重")
                else:
                    recommendations.append("权重分布相对均匀，各因子贡献较为平衡")
                    
            # 分析优化稳定性
            if len(improvements) > 1:
                improvement_trend = np.polyfit(range(len(improvements)), improvements, 1)[0]
                if improvement_trend > 0:
                    recommendations.append("优化过程显示持续改进趋势，建议继续扩大搜索空间")
                else:
                    recommendations.append("优化过程显示收敛趋势，当前配置可能已接近最优")
                    
            if not recommendations:
                recommendations.append("优化完成，建议定期重新评估模型性能")
                
        except Exception as e:
            self.logger.error(f"生成建议失败: {e}")
            recommendations.append("无法生成优化建议")
            
        return recommendations
        
    async def save_optimization_report(self, optimization_result: Dict, 
                                     filename: str = None) -> str:
        """保存优化报告"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_optimization_{timestamp}.md"
                
            import os
            report_path = f"reports/optimization/{filename}"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            content = self._format_optimization_report(optimization_result)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.logger.info(f"优化报告已保存: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"保存优化报告失败: {e}")
            return ""
            
    def _format_optimization_report(self, result: Dict) -> str:
        """格式化优化报告"""
        try:
            content = f"""# 模型参数优化报告

## 优化概述
- **优化时间**: {result.get('optimization_time', 'Unknown')}
- **优化轮数**: {result.get('optimization_cycles', 0)}
- **验证期数**: {result.get('validation_periods', 0)}

## 优化结果

### 性能对比
- **基线准确率**: {result.get('baseline_accuracy', 0):.3f} ({result.get('baseline_accuracy', 0)*100:.1f}%)
- **优化后准确率**: {result.get('best_accuracy', 0):.3f} ({result.get('best_accuracy', 0)*100:.1f}%)
- **绝对提升**: {result.get('improvement', 0):+.3f}
- **相对提升**: {result.get('improvement_percentage', 0):+.1f}%

### 最佳参数配置
"""
            
            best_config = result.get('best_config', {})
            if best_config:
                for param, weight in best_config.items():
                    content += f"- **{param}**: {weight:.3f}\n"
            
            content += f"""
## 优化过程

"""
            
            optimization_results = result.get('optimization_results', [])
            for cycle_result in optimization_results:
                content += f"""### 轮次 {cycle_result.get('cycle', 0)}
- **测试配置数**: {cycle_result.get('configs_tested', 0)}
- **最佳准确率**: {cycle_result.get('best_accuracy', 0):.3f}
- **相比基线提升**: {cycle_result.get('improvement', 0):+.3f}

"""
            
            content += f"""## 优化建议

"""
            recommendations = result.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"
                
            # 添加最终验证结果
            final_validation = result.get('final_validation', {})
            if final_validation and 'overall_statistics' in final_validation:
                stats = final_validation['overall_statistics']
                content += f"""
## 最终验证结果

- **总体准确率**: {stats.get('accuracy', 0):.1%}
- **方向准确率**: {stats.get('avg_direction_accuracy', 0):.1f}%
- **收益率准确率**: {stats.get('avg_return_accuracy', 0):.1f}%
- **预测稳定性**: {stats.get('stability', 0):.1%}
- **综合评级**: {stats.get('grade', '未知')}
"""
            
            content += f"""
## 使用建议

1. **应用最佳配置**: 将优化后的参数配置应用到生产环境
2. **定期重新优化**: 建议每月重新评估和优化模型参数
3. **监控性能**: 持续监控预测准确率，如有下降及时调整

---
*报告生成时间: {result.get('optimization_time', 'Unknown')}*
*优化数据来源: 历史交易数据验证*
"""
            
            return content
            
        except Exception as e:
            self.logger.error(f"格式化优化报告失败: {e}")
            return f"报告生成失败: {str(e)}"