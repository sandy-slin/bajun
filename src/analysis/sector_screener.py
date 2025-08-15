"""
板块筛选模块
基于多维度评分体系筛选最值得跟踪的板块
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os

from data.sector_fetcher import SectorFetcher
from data.technical_calculator import TechnicalCalculator
from data.enhanced_data_fetcher import EnhancedDataFetcher
from analysis.enhanced_predictor import EnhancedPredictor
from analysis.report_manager import ReportManager


class SectorScreener:
    """板块筛选器"""
    
    def __init__(self, sector_fetcher: SectorFetcher, tech_calculator: TechnicalCalculator,
                 enhanced_data_fetcher: EnhancedDataFetcher, enhanced_predictor: EnhancedPredictor,
                 report_manager: ReportManager):
        self.sector_fetcher = sector_fetcher
        self.tech_calculator = tech_calculator
        self.enhanced_data_fetcher = enhanced_data_fetcher
        self.enhanced_predictor = enhanced_predictor
        self.report_manager = report_manager
        self.logger = logging.getLogger(__name__)
        # 是否使用增强预测系统
        self.use_enhanced_prediction = True
        
    async def screen_top_sectors(self, top_n: int = 5, period: str = "1-2weeks") -> Dict:
        """
        筛选Top N板块
        
        Args:
            top_n: 返回前N个板块
            period: 预测周期
            
        Returns:
            Dict: 筛选结果和详细分析
        """
        try:
            self.logger.info(f"开始筛选Top {top_n}板块，预测周期: {period}")
            
            # 1. 获取所有板块数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 获取30天数据用于分析
            date_range = (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            
            sectors_data = await self.sector_fetcher.get_all_sectors_data(date_range)
            
            if not sectors_data:
                raise ValueError("未能获取板块数据")
                
            # 2. 计算各板块评分 (使用增强预测系统)
            sectors_scores = {}
            for sector_name, sector_df in sectors_data.items():
                try:
                    if self.use_enhanced_prediction:
                        # 使用增强预测系统
                        enhanced_result = await self.enhanced_predictor.calculate_enhanced_sector_score(
                            sector_name, sector_df, date_range
                        )
                        scores = enhanced_result
                        scores['sector_name'] = sector_name
                        scores['data_quality'] = len(sector_df)
                        
                        # 保持兼容性，提取主要评分
                        if 'base_scores' in enhanced_result:
                            scores.update(enhanced_result['base_scores'])
                        
                        # 记录增强预测结果
                        prediction_info = enhanced_result.get('prediction', {})
                        if prediction_info and 'trend_prediction' in prediction_info:
                            scores['trend_prediction'] = prediction_info['trend_prediction']
                            scores['target_return'] = prediction_info.get('target_return', 0)
                            scores['confidence_level'] = prediction_info.get('confidence_level', '中等')
                    else:
                        # 使用基础评分系统
                        scores = self.tech_calculator.calculate_sector_strength_score(sector_df)
                        scores['sector_name'] = sector_name
                        scores['data_quality'] = len(sector_df)
                    
                    sectors_scores[sector_name] = scores
                    
                    self.logger.debug(f"{sector_name}: 综合评分 {scores.get('comprehensive_score', 0):.1f}")
                    
                except Exception as e:
                    self.logger.error(f"计算{sector_name}评分失败: {e}，回退到基础评分")
                    try:
                        # 回退到基础评分系统
                        scores = self.tech_calculator.calculate_sector_strength_score(sector_df)
                        scores['sector_name'] = sector_name
                        scores['data_quality'] = len(sector_df)
                        scores['fallback'] = True
                        sectors_scores[sector_name] = scores
                    except Exception as fallback_e:
                        self.logger.error(f"基础评分也失败: {fallback_e}")
                        continue
                    
            if not sectors_scores:
                raise ValueError("未能计算任何板块评分")
                
            # 3. 计算板块间相关性
            correlation_matrix = self.tech_calculator.calculate_sector_correlation(sectors_data)
            
            # 4. 分析板块轮动模式
            rotation_analysis = self.tech_calculator.analyze_sector_rotation_pattern(sectors_data)
            
            # 5. 排序并筛选Top N
            sorted_sectors = sorted(
                sectors_scores.values(),
                key=lambda x: x.get('comprehensive_score', 0),
                reverse=True
            )
            
            top_sectors = sorted_sectors[:top_n]
            
            # 6. 生成详细分析
            screening_result = {
                'screening_time': datetime.now().isoformat(),
                'period': period,
                'total_sectors_analyzed': len(sectors_scores),
                'top_sectors': top_sectors,
                'screening_criteria': self._get_screening_criteria(),
                'market_overview': self._generate_market_overview(sectors_scores),
                'risk_warnings': self._generate_risk_warnings(top_sectors),
                'correlation_analysis': correlation_matrix,
                'rotation_analysis': rotation_analysis
            }
            
            # 7. 生成报告
            report_path = await self._generate_screening_report(screening_result)
            screening_result['report_path'] = report_path
            
            self.logger.info(f"筛选完成，推荐板块: {[s['sector_name'] for s in top_sectors]}")
            
            return screening_result
            
        except Exception as e:
            self.logger.error(f"板块筛选失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def _get_screening_criteria(self) -> Dict:
        """获取筛选标准说明"""
        return {
            'technical_weight': 0.4,
            'money_flow_weight': 0.3,
            'fundamental_weight': 0.2,
            'rotation_weight': 0.1,
            'minimum_score': 50,
            'data_requirements': '至少5个交易日数据',
            'update_frequency': '每日更新评分'
        }
        
    def _generate_market_overview(self, sectors_scores: Dict) -> Dict:
        """生成市场概览"""
        try:
            all_scores = [scores.get('comprehensive_score', 0) for scores in sectors_scores.values()]
            
            if not all_scores:
                return {'status': '数据不足'}
                
            overview = {
                'average_score': sum(all_scores) / len(all_scores),
                'max_score': max(all_scores),
                'min_score': min(all_scores),
                'sectors_above_75': sum(1 for score in all_scores if score >= 75),
                'sectors_above_60': sum(1 for score in all_scores if score >= 60),
                'market_sentiment': self._assess_market_sentiment(all_scores)
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"生成市场概览失败: {e}")
            return {'status': 'error'}
            
    def _assess_market_sentiment(self, scores: List[float]) -> str:
        """评估市场情绪"""
        avg_score = sum(scores) / len(scores) if scores else 50
        
        if avg_score >= 70:
            return "乐观"
        elif avg_score >= 55:
            return "谨慎乐观"
        elif avg_score >= 45:
            return "中性"
        elif avg_score >= 30:
            return "谨慎悲观"
        else:
            return "悲观"
            
    def _generate_risk_warnings(self, top_sectors: List[Dict]) -> List[str]:
        """生成风险提示"""
        warnings = []
        
        try:
            # 检查推荐板块的风险特征
            high_score_count = sum(1 for sector in top_sectors 
                                 if sector.get('comprehensive_score', 0) >= 85)
                                 
            if high_score_count >= 3:
                warnings.append("多个板块评分过高，注意市场过热风险")
                
            # 检查技术指标风险
            high_rsi_count = sum(1 for sector in top_sectors 
                               if sector.get('rsi_score', 0) >= 80)
            if high_rsi_count >= 2:
                warnings.append("部分板块RSI过高，存在技术面调整风险")
                
            # 检查资金面风险
            low_volume_count = sum(1 for sector in top_sectors 
                                 if sector.get('volume_score', 0) <= 30)
            if low_volume_count >= 2:
                warnings.append("部分推荐板块成交量不足，流动性风险较高")
                
            if not warnings:
                warnings.append("当前推荐板块风险可控，建议密切关注市场变化")
                
        except Exception as e:
            self.logger.error(f"生成风险提示失败: {e}")
            warnings = ["风险评估出现异常，请谨慎投资"]
            
        return warnings
        
    async def _generate_screening_report(self, screening_result: Dict) -> str:
        """生成板块筛选报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sector_screening_{timestamp}.md"
            
            # 构建报告内容
            report_content = self._format_screening_report(screening_result)
            
            # 保存报告到文件
            report_path = f"reports/screening/{filename}"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"板块筛选报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成板块筛选报告失败: {e}")
            return ""
    
    def _format_screening_report(self, result: Dict) -> str:
        """格式化板块筛选报告"""
        try:
            # 基本信息
            content = f"""# A股板块筛选报告

## 筛选概述
- **筛选时间**: {result.get('screening_time', 'Unknown')}
- **预测周期**: {result.get('period', 'Unknown')}
- **分析板块数量**: {result.get('total_sectors_analyzed', 0)}
- **市场情绪**: {self._get_market_sentiment(result.get('market_overview', {}))}

## Top 5 推荐板块

"""
            
            # Top N板块详情
            top_sectors = result.get('top_sectors', [])
            for i, sector in enumerate(top_sectors[:5], 1):
                # 检查是否有增强预测结果
                has_enhanced = 'enhanced_scores' in sector and sector.get('enhanced_scores')
                prediction = sector.get('prediction', {})
                
                content += f"""### {i}. {sector.get('sector_name', 'Unknown')}

- **综合评分**: {sector.get('comprehensive_score', 0):.1f}/100
- **推荐等级**: {sector.get('recommendation', 'Hold')}"""
                
                # 如果有增强预测，显示预测结果
                if prediction and 'trend_prediction' in prediction:
                    content += f"""
- **趋势预测**: {prediction.get('trend_prediction', '震荡')}
- **预期收益**: {prediction.get('target_return', 0):+.1f}%
- **预测概率**: {prediction.get('probability', 50):.1f}%
- **置信水平**: {prediction.get('confidence_level', '中等')}
- **预测周期**: {prediction.get('prediction_horizon', '3个交易日')}"""

                content += f"""
- **技术面评分**: {sector.get('scores_breakdown', {}).get('technical', sector.get('technical_score', 0)):.1f}/100
  - RSI指标: {sector.get('scores_breakdown', {}).get('rsi_score', sector.get('rsi_score', 0)):.1f}
  - 均线突破: {sector.get('scores_breakdown', {}).get('ma_breakthrough_score', sector.get('ma_breakthrough_score', 0)):.1f}
  - MACD信号: {sector.get('scores_breakdown', {}).get('macd_score', sector.get('macd_score', 0)):.1f}
- **资金流向评分**: {sector.get('scores_breakdown', {}).get('money_flow', sector.get('money_flow_score', 0)):.1f}/100
  - 成交量: {sector.get('scores_breakdown', {}).get('volume_score', sector.get('volume_score', 0)):.1f}
  - 成交额: {sector.get('scores_breakdown', {}).get('amount_score', sector.get('amount_score', 0)):.1f}
  - 主力资金: {sector.get('scores_breakdown', {}).get('main_fund_score', sector.get('main_fund_score', 0)):.1f}
- **基本面评分**: {sector.get('scores_breakdown', {}).get('fundamental', sector.get('fundamental_score', 0)):.1f}/100
- **轮动周期评分**: {sector.get('scores_breakdown', {}).get('rotation', sector.get('rotation_score', 0)):.1f}/100"""
                
                # 如果有增强数据，显示增强指标
                if has_enhanced:
                    enhanced_scores = sector.get('enhanced_scores', {})
                    content += f"""

#### 增强指标
- **北向资金评分**: {enhanced_scores.get('northbound_score', 50):.1f}/100
- **融资融券评分**: {enhanced_scores.get('margin_score', 50):.1f}/100
- **市场情绪评分**: {enhanced_scores.get('sentiment_score', 50):.1f}/100
- **宏观环境评分**: {enhanced_scores.get('macro_score', 50):.1f}/100"""
                    
                    # 显示关键驱动因子
                    if prediction and 'key_drivers' in prediction:
                        key_drivers = prediction['key_drivers']
                        if key_drivers:
                            content += f"""
- **关键驱动因子**: {', '.join(key_drivers)}"""
                    
                    # 显示风险因子
                    if prediction and 'risk_factors' in prediction:
                        risk_factors = prediction['risk_factors']
                        if risk_factors and risk_factors != ['风险可控']:
                            content += f"""
- **风险因子**: {', '.join(risk_factors)}"""
                
                content += "\n\n"
            
            # 市场概览
            market_overview = result.get('market_overview', {})
            content += f"""## 市场概览

- **平均评分**: {market_overview.get('average_score', 0):.1f}
- **最高评分**: {market_overview.get('highest_score', 0):.1f}  
- **最低评分**: {market_overview.get('lowest_score', 0):.1f}
- **强势板块数量** (评分>75): {market_overview.get('strong_sectors_count', 0)}
- **关注板块数量** (评分>60): {market_overview.get('watch_sectors_count', 0)}

"""
            
            # 板块轮动分析
            rotation_analysis = result.get('rotation_analysis', {})
            if rotation_analysis:
                content += f"""## 板块轮动分析

### 市场阶段
- **当前阶段**: {self._get_market_phase_name(rotation_analysis.get('market_phase', 'unknown'))}

### 领涨板块
"""
                for sector in rotation_analysis.get('leading_sectors', [])[:5]:
                    content += f"- {sector.get('sector', 'Unknown')}: {sector.get('strength', 0):+.2f}%\n"
                
                content += f"""
### 领跌板块
"""
                for sector in rotation_analysis.get('lagging_sectors', [])[:5]:
                    content += f"- {sector.get('sector', 'Unknown')}: {sector.get('strength', 0):+.2f}%\n"
                
                content += f"""
### 轮动机会
"""
                for sector in rotation_analysis.get('rotation_opportunities', [])[:5]:
                    content += f"- {sector.get('sector', 'Unknown')}: {sector.get('strength', 0):+.2f}%\n"
                
                content += "\n"
            
            # 投资策略建议
            content += f"""## 投资策略建议

### 选股策略
1. **重点关注推荐板块内的龙头股票**
2. **优选技术面突破且有资金流入的个股**
3. **控制单一板块仓位，建议分散配置**

### 时机把握
1. **短期(1-2周)**: 关注技术面信号，适合波段操作
2. **中期(1-2月)**: 结合基本面判断，把握轮动机会
3. **风险控制**: 设置止损位，建议单个板块最大亏损不超过8%

## 风险提示

- 当前推荐板块风险可控，建议密切关注市场变化

## 下期关注

建议在{self._get_next_review_date()}重新评估板块表现，调整投资策略。

---
*报告生成时间: {result.get('screening_time', 'Unknown')}*
*免责声明: 本报告仅供参考，投资有风险，决策需谨慎*
"""
            
            return content
            
        except Exception as e:
            self.logger.error(f"格式化板块筛选报告失败: {e}")
            return f"报告生成失败: {str(e)}"
    
    def _get_market_sentiment(self, market_overview: Dict) -> str:
        """获取市场情绪描述"""
        avg_score = market_overview.get('average_score', 50)
        if avg_score > 70:
            return "乐观"
        elif avg_score > 55:
            return "谨慎乐观"
        elif avg_score > 45:
            return "中性"
        elif avg_score > 30:
            return "谨慎"
        else:
            return "悲观"
    
    def _get_market_phase_name(self, phase: str) -> str:
        """获取市场阶段名称"""
        phase_names = {
            'bull_market': '牛市',
            'bear_market': '熊市',
            'sideways_market': '震荡市',
            'unknown': '未知'
        }
        return phase_names.get(phase, '未知')
    
    def _get_next_review_date(self) -> str:
        """获取下次评估日期"""
        next_date = datetime.now() + timedelta(days=7)
        return next_date.strftime("%Y-%m-%d")
        
    async def get_sector_analysis_summary(self, sector_name: str) -> Dict:
        """获取单个板块的分析摘要"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            
            # 获取板块数据
            sectors_data = await self.sector_fetcher.get_all_sectors_data(date_range)
            sector_data = sectors_data.get(sector_name)
            
            if sector_data is None or sector_data.empty:
                return {'error': f'无法获取{sector_name}板块数据'}
                
            # 计算评分
            scores = self.tech_calculator.calculate_sector_strength_score(sector_data)
            
            # 获取板块成分股
            stocks = await self.sector_fetcher.get_sector_stocks(sector_name)
            
            summary = {
                'sector_name': sector_name,
                'analysis_time': datetime.now().isoformat(),
                'comprehensive_score': scores.get('comprehensive_score', 0),
                'recommendation': scores.get('recommendation', 'Hold'),
                'scores_breakdown': {
                    'technical': scores.get('technical_score', 0),
                    'money_flow': scores.get('money_flow_score', 0),
                    'fundamental': scores.get('fundamental_score', 0),
                    'rotation': scores.get('rotation_score', 0)
                },
                'stocks_count': len(stocks),
                'data_quality': len(sector_data),
                'latest_price': sector_data['close'].iloc[-1] if not sector_data.empty else 0,
                'price_change_5d': ((sector_data['close'].iloc[-1] / sector_data['close'].iloc[-5] - 1) * 100) if len(sector_data) >= 5 else 0
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取{sector_name}分析摘要失败: {e}")
            return {'error': str(e)}