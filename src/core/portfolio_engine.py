#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持仓分析引擎 - Phase 1 MVP核心组件
个人投资组合评估与建议系统

功能:
1. 持仓质量评估：基于当前时间点重新评分
2. 行业配置分析：板块集中度和分散性
3. 风险暴露评估：单只股票和行业风险
4. 调仓建议：买入/卖出/持有的具体建议
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..data.real_data_fetcher import RealDataFetcher
from ..data.sector_fetcher import SectorFetcher
from .sector_engine import SectorEngine
from .stock_engine import StockEngine


class PortfolioEngine:
    """持仓分析引擎 - 个人投资组合管理"""
    
    def __init__(
        self,
        data_fetcher: RealDataFetcher,
        sector_fetcher: SectorFetcher,
        sector_engine: SectorEngine,
        stock_engine: StockEngine
    ):
        self.data_fetcher = data_fetcher
        self.sector_fetcher = sector_fetcher
        self.sector_engine = sector_engine
        self.stock_engine = stock_engine
        self.logger = logging.getLogger(__name__)
        
        # 风险控制参数
        self.risk_limits = {
            'max_single_position': 0.15,     # 单只股票最大15%仓位
            'max_sector_allocation': 0.30,   # 单个板块最大30%配置
            'min_cash_position': 0.05,       # 最小5%现金仓位
            'transaction_cost_limit': 0.005  # 交易成本限制0.5%
        }
        
        # 评估维度权重
        self.evaluation_weights = {
            'quality_score': 0.4,      # 持仓质量评分权重
            'sector_allocation': 0.3,  # 板块配置评分权重
            'risk_exposure': 0.2,      # 风险暴露评分权重
            'opportunity_cost': 0.1    # 机会成本评分权重
        }
        
    async def analyze_portfolio(self, holdings_file: str) -> Dict:
        """
        分析个人投资组合
        
        Args:
            holdings_file: 持仓文件路径 (JSON格式)
            
        Returns:
            Dict: 完整的投资组合分析结果
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"开始投资组合分析，读取持仓文件: {holdings_file}")
            
            # 1. 加载持仓数据
            holdings = await self._load_holdings(holdings_file)
            if not holdings:
                return {'error': '无法加载持仓数据'}
            
            # 2. 获取当前市场数据
            market_data = await self._fetch_current_market_data(holdings)
            
            # 3. 计算投资组合当前状态
            portfolio_status = await self._calculate_portfolio_status(holdings, market_data)
            
            # 4. 持仓质量评估
            quality_analysis = await self._evaluate_holdings_quality(holdings, market_data)
            
            # 5. 行业配置分析
            sector_analysis = await self._analyze_sector_allocation(holdings, market_data)
            
            # 6. 风险暴露评估
            risk_analysis = await self._assess_risk_exposure(holdings, market_data)
            
            # 7. 生成调仓建议
            rebalancing_advice = await self._generate_rebalancing_advice(
                holdings, market_data, quality_analysis, sector_analysis, risk_analysis
            )
            
            # 8. 识别新投资机会
            new_opportunities = await self._identify_new_opportunities(holdings, sector_analysis)
            
            # 9. 综合评估报告
            overall_assessment = self._generate_overall_assessment(
                portfolio_status, quality_analysis, sector_analysis, 
                risk_analysis, rebalancing_advice
            )
            
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_status': portfolio_status,
                'holdings_quality': quality_analysis,
                'sector_allocation': sector_analysis,
                'risk_assessment': risk_analysis,
                'rebalancing_recommendations': rebalancing_advice,
                'new_opportunities': new_opportunities,
                'overall_assessment': overall_assessment,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            self.logger.info(f"投资组合分析完成，耗时{analysis_result['processing_time_seconds']:.1f}秒")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"投资组合分析失败: {e}")
            return {'error': str(e)}
    
    async def _load_holdings(self, holdings_file: str) -> Optional[Dict]:
        """加载持仓数据"""
        try:
            with open(holdings_file, 'r', encoding='utf-8') as f:
                holdings_data = json.load(f)
            
            # 验证数据格式
            required_fields = ['total_assets', 'cash_position', 'holdings']
            for field in required_fields:
                if field not in holdings_data:
                    self.logger.error(f"持仓文件缺少必要字段: {field}")
                    return None
            
            return holdings_data
            
        except Exception as e:
            self.logger.error(f"加载持仓文件失败: {e}")
            return None
    
    async def _fetch_current_market_data(self, holdings: Dict) -> Dict:
        """获取当前市场数据"""
        market_data = {}
        
        for holding in holdings['holdings']:
            stock_code = holding['stock_code']
            try:
                # 获取股票最新数据
                stock_data = await self.data_fetcher.get_trading_data(stock_code)
                if stock_data and len(stock_data) > 0:
                    latest_data = stock_data[-1]
                    market_data[stock_code] = {
                        'latest_price': latest_data['close'],
                        'volume': latest_data.get('volume', 0),
                        'change_1d': self._calculate_change(stock_data, 1),
                        'change_5d': self._calculate_change(stock_data, 5),
                        'change_20d': self._calculate_change(stock_data, 20),
                        'historical_data': stock_data[-30:]  # 最近30天数据
                    }
                    
                    # 获取股票所属板块
                    sector = await self._get_stock_sector(stock_code)
                    market_data[stock_code]['sector'] = sector
                    
            except Exception as e:
                self.logger.warning(f"获取股票{stock_code}数据失败: {e}")
                market_data[stock_code] = {'error': str(e)}
        
        return market_data
    
    async def _calculate_portfolio_status(self, holdings: Dict, market_data: Dict) -> Dict:
        """计算投资组合当前状态"""
        total_assets = holdings['total_assets']
        cash_position = holdings['cash_position']
        
        current_holdings_value = 0
        unrealized_pnl = 0
        
        position_details = []
        
        for holding in holdings['holdings']:
            stock_code = holding['stock_code']
            shares = holding['shares']
            cost_price = holding['cost_price']
            
            if stock_code in market_data and 'latest_price' in market_data[stock_code]:
                current_price = market_data[stock_code]['latest_price']
                current_value = shares * current_price
                cost_value = shares * cost_price
                pnl = current_value - cost_value
                
                current_holdings_value += current_value
                unrealized_pnl += pnl
                
                position_details.append({
                    'stock_code': stock_code,
                    'stock_name': holding.get('stock_name', ''),
                    'shares': shares,
                    'cost_price': cost_price,
                    'current_price': current_price,
                    'cost_value': cost_value,
                    'current_value': current_value,
                    'unrealized_pnl': pnl,
                    'return_rate': (pnl / cost_value * 100) if cost_value > 0 else 0,
                    'weight': (current_value / total_assets * 100) if total_assets > 0 else 0,
                    'sector': market_data[stock_code].get('sector', 'unknown')
                })
        
        current_total_value = current_holdings_value + cash_position
        total_return = current_total_value - total_assets
        
        return {
            'total_assets_original': total_assets,
            'current_total_value': current_total_value,
            'cash_position': cash_position,
            'cash_ratio': (cash_position / current_total_value * 100) if current_total_value > 0 else 0,
            'current_holdings_value': current_holdings_value,
            'holdings_ratio': (current_holdings_value / current_total_value * 100) if current_total_value > 0 else 0,
            'total_return': total_return,
            'total_return_rate': (total_return / total_assets * 100) if total_assets > 0 else 0,
            'unrealized_pnl': unrealized_pnl,
            'position_count': len(position_details),
            'position_details': position_details
        }
    
    async def _evaluate_holdings_quality(self, holdings: Dict, market_data: Dict) -> Dict:
        """评估持仓质量"""
        quality_scores = []
        detailed_evaluations = []
        
        for holding in holdings['holdings']:
            stock_code = holding['stock_code']
            
            if stock_code not in market_data or 'historical_data' in market_data[stock_code]:
                continue
                
            try:
                # 使用股票引擎重新评分
                df = pd.DataFrame(market_data[stock_code]['historical_data'])
                stock_score = await self.stock_engine._calculate_stock_score(stock_code)
                
                if stock_score:
                    quality_scores.append(stock_score['composite_score'])
                    
                    detailed_evaluations.append({
                        'stock_code': stock_code,
                        'current_quality_score': stock_score['composite_score'],
                        'return_expectation': stock_score['return_expectation_score'],
                        'volume_confirmation': stock_score['volume_confirmation_score'],
                        'technical_score': stock_score['technical_score'],
                        'risk_level': stock_score.get('risk_level', 'unknown'),
                        'recommendation': self._get_quality_recommendation(stock_score['composite_score'])
                    })
                    
            except Exception as e:
                self.logger.warning(f"评估股票{stock_code}质量失败: {e}")
                continue
        
        average_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            'average_quality_score': average_quality,
            'quality_grade': self._get_quality_grade(average_quality),
            'high_quality_count': len([s for s in quality_scores if s >= 70]),
            'low_quality_count': len([s for s in quality_scores if s <= 40]),
            'detailed_evaluations': detailed_evaluations,
            'quality_distribution': self._analyze_quality_distribution(quality_scores)
        }
    
    async def _analyze_sector_allocation(self, holdings: Dict, market_data: Dict) -> Dict:
        """分析行业配置"""
        sector_allocation = {}
        total_value = 0
        
        # 计算各板块配置
        for position in holdings['holdings']:
            stock_code = position['stock_code']
            
            if stock_code in market_data and 'latest_price' in market_data[stock_code]:
                sector = market_data[stock_code].get('sector', 'unknown')
                shares = position['shares']
                current_price = market_data[stock_code]['latest_price']
                value = shares * current_price
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = {
                        'value': 0,
                        'stocks': [],
                        'count': 0
                    }
                
                sector_allocation[sector]['value'] += value
                sector_allocation[sector]['stocks'].append(stock_code)
                sector_allocation[sector]['count'] += 1
                total_value += value
        
        # 计算比例和评估
        sector_analysis = []
        concentration_risk = 0
        
        for sector, data in sector_allocation.items():
            allocation_ratio = (data['value'] / total_value * 100) if total_value > 0 else 0
            
            # 检查是否超过风险限制
            risk_exceeded = allocation_ratio > (self.risk_limits['max_sector_allocation'] * 100)
            if risk_exceeded:
                concentration_risk += allocation_ratio - (self.risk_limits['max_sector_allocation'] * 100)
            
            sector_analysis.append({
                'sector': sector,
                'allocation_ratio': allocation_ratio,
                'value': data['value'],
                'stock_count': data['count'],
                'stocks': data['stocks'],
                'risk_exceeded': risk_exceeded,
                'diversification_score': self._calculate_sector_diversification_score(allocation_ratio)
            })
        
        # 排序
        sector_analysis.sort(key=lambda x: x['allocation_ratio'], reverse=True)
        
        return {
            'sector_count': len(sector_analysis),
            'sector_breakdown': sector_analysis,
            'largest_sector_ratio': sector_analysis[0]['allocation_ratio'] if sector_analysis else 0,
            'concentration_risk': concentration_risk,
            'diversification_score': self._calculate_overall_diversification_score(sector_analysis),
            'rebalancing_needed': concentration_risk > 0
        }
    
    async def _assess_risk_exposure(self, holdings: Dict, market_data: Dict) -> Dict:
        """评估风险暴露"""
        risk_metrics = {
            'position_concentration': [],
            'sector_concentration': [],
            'volatility_exposure': [],
            'liquidity_risk': []
        }
        
        total_value = sum(
            pos['shares'] * market_data[pos['stock_code']]['latest_price']
            for pos in holdings['holdings'] 
            if pos['stock_code'] in market_data and 'latest_price' in market_data[pos['stock_code']]
        )
        
        # 分析各种风险
        for position in holdings['holdings']:
            stock_code = position['stock_code']
            
            if stock_code not in market_data or 'latest_price' not in market_data[stock_code]:
                continue
                
            shares = position['shares']
            current_price = market_data[stock_code]['latest_price']
            position_value = shares * current_price
            weight = (position_value / total_value) if total_value > 0 else 0
            
            # 仓位集中度风险
            position_risk = weight > self.risk_limits['max_single_position']
            risk_metrics['position_concentration'].append({
                'stock_code': stock_code,
                'weight': weight * 100,
                'risk_exceeded': position_risk,
                'excess_weight': max(0, weight - self.risk_limits['max_single_position']) * 100
            })
            
            # 波动率风险评估
            if 'historical_data' in market_data[stock_code]:
                df = pd.DataFrame(market_data[stock_code]['historical_data'])
                volatility = self._calculate_volatility(df)
                risk_metrics['volatility_exposure'].append({
                    'stock_code': stock_code,
                    'weight': weight * 100,
                    'volatility': volatility,
                    'risk_contribution': weight * volatility
                })
        
        # 计算组合整体风险指标
        overall_risk = self._calculate_portfolio_risk_metrics(risk_metrics, total_value)
        
        return {
            'position_risks': risk_metrics['position_concentration'],
            'volatility_risks': risk_metrics['volatility_exposure'],
            'overall_risk_score': overall_risk['risk_score'],
            'risk_grade': overall_risk['risk_grade'],
            'max_position_weight': max([r['weight'] for r in risk_metrics['position_concentration']], default=0),
            'risk_warnings': self._generate_risk_warnings(risk_metrics, overall_risk)
        }
    
    async def _generate_rebalancing_advice(
        self, 
        holdings: Dict, 
        market_data: Dict,
        quality_analysis: Dict,
        sector_analysis: Dict,
        risk_analysis: Dict
    ) -> Dict:
        """生成调仓建议"""
        
        recommendations = []
        
        # 基于质量评估的建议
        for evaluation in quality_analysis['detailed_evaluations']:
            stock_code = evaluation['stock_code']
            quality_score = evaluation['current_quality_score']
            
            if quality_score >= 75:
                action = "HOLD"
                reason = f"质量评分优秀({quality_score:.1f}分)，建议继续持有"
            elif quality_score >= 60:
                action = "HOLD"
                reason = f"质量评分良好({quality_score:.1f}分)，维持现有仓位"
            elif quality_score >= 40:
                action = "REDUCE"
                reason = f"质量评分一般({quality_score:.1f}分)，建议适当减仓"
            else:
                action = "SELL"
                reason = f"质量评分偏低({quality_score:.1f}分)，建议考虑卖出"
            
            recommendations.append({
                'stock_code': stock_code,
                'action': action,
                'reason': reason,
                'priority': self._calculate_action_priority(quality_score, action),
                'urgency': 'high' if quality_score < 30 else 'medium' if quality_score < 50 else 'low'
            })
        
        # 基于风险控制的建议
        for position_risk in risk_analysis['position_risks']:
            if position_risk['risk_exceeded']:
                stock_code = position_risk['stock_code']
                
                # 查找是否已有建议
                existing_rec = next((r for r in recommendations if r['stock_code'] == stock_code), None)
                if existing_rec:
                    existing_rec['action'] = 'REDUCE'
                    existing_rec['reason'] += f"；仓位过重({position_risk['weight']:.1f}%)，需要减仓"
                    existing_rec['urgency'] = 'high'
                else:
                    recommendations.append({
                        'stock_code': stock_code,
                        'action': 'REDUCE',
                        'reason': f"仓位过重({position_risk['weight']:.1f}%)，超出单股15%限制",
                        'priority': 9,  # 高优先级
                        'urgency': 'high'
                    })
        
        # 排序建议（按优先级和紧急程度）
        recommendations.sort(key=lambda x: (-x['priority'], x['urgency'] != 'high'))
        
        return {
            'total_recommendations': len(recommendations),
            'sell_recommendations': [r for r in recommendations if r['action'] == 'SELL'],
            'reduce_recommendations': [r for r in recommendations if r['action'] == 'REDUCE'],
            'hold_recommendations': [r for r in recommendations if r['action'] == 'HOLD'],
            'all_recommendations': recommendations,
            'rebalancing_urgency': self._assess_rebalancing_urgency(recommendations)
        }
    
    async def _identify_new_opportunities(self, holdings: Dict, sector_analysis: Dict) -> Dict:
        """识别新投资机会"""
        try:
            # 获取当前最优板块
            top_sectors_result = await self.sector_engine.analyze_top_sectors(lookback_months=6, top_n=5)
            
            if 'error' in top_sectors_result:
                return {'error': '无法获取板块分析数据'}
            
            current_sectors = set()
            for position in holdings['holdings']:
                # 这里需要获取每只股票的板块信息
                # 简化实现，后续可增强
                current_sectors.add('unknown')  # 临时实现
            
            new_opportunities = []
            
            for sector in top_sectors_result['top_sectors']:
                sector_name = sector['sector_name']
                
                if sector_name not in current_sectors:
                    # 从该板块选择股票
                    stock_selection = await self.stock_engine.select_stocks_from_sectors(
                        [sector], stocks_per_sector=3
                    )
                    
                    if 'error' not in stock_selection and stock_selection['sector_selections']:
                        sector_stocks = stock_selection['sector_selections'][0]['selected_stocks']
                        
                        new_opportunities.append({
                            'sector': sector_name,
                            'sector_score': sector['composite_score'],
                            'investment_logic': sector['investment_logic'],
                            'recommended_stocks': sector_stocks[:2],  # 推荐前2只
                            'suggested_allocation': min(0.15, 1.0 / len(top_sectors_result['top_sectors']))  # 建议配置比例
                        })
            
            return {
                'new_sectors_count': len(new_opportunities),
                'opportunities': new_opportunities,
                'total_suggested_allocation': sum(opp['suggested_allocation'] for opp in new_opportunities)
            }
            
        except Exception as e:
            self.logger.warning(f"识别新投资机会失败: {e}")
            return {'error': str(e)}
    
    def _generate_overall_assessment(
        self,
        portfolio_status: Dict,
        quality_analysis: Dict,
        sector_analysis: Dict,
        risk_analysis: Dict,
        rebalancing_advice: Dict
    ) -> Dict:
        """生成综合评估"""
        
        # 计算综合评分
        quality_score = quality_analysis['average_quality_score']
        diversification_score = sector_analysis['diversification_score']
        risk_score = 100 - risk_analysis['overall_risk_score']  # 风险越低分数越高
        
        overall_score = (
            quality_score * self.evaluation_weights['quality_score'] +
            diversification_score * self.evaluation_weights['sector_allocation'] +
            risk_score * self.evaluation_weights['risk_exposure'] +
            50 * self.evaluation_weights['opportunity_cost']  # 机会成本简化为中性
        )
        
        # 生成评估等级
        if overall_score >= 80:
            grade = "excellent"
            assessment = "投资组合整体表现优秀，配置合理，风险可控"
        elif overall_score >= 65:
            grade = "good"
            assessment = "投资组合表现良好，存在一些优化空间"
        elif overall_score >= 50:
            grade = "fair"
            assessment = "投资组合表现一般，需要适当调整"
        else:
            grade = "poor"
            assessment = "投资组合存在较多问题，建议重新配置"
        
        # 关键改进建议
        improvement_suggestions = []
        
        if quality_analysis['low_quality_count'] > 0:
            improvement_suggestions.append(f"考虑清理{quality_analysis['low_quality_count']}只低质量股票")
        
        if sector_analysis['concentration_risk'] > 0:
            improvement_suggestions.append("降低板块集中度风险，增强分散化")
        
        if risk_analysis['max_position_weight'] > 15:
            improvement_suggestions.append("控制单只股票仓位，避免过度集中")
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'assessment': assessment,
            'key_strengths': self._identify_portfolio_strengths(portfolio_status, quality_analysis, sector_analysis),
            'improvement_suggestions': improvement_suggestions,
            'urgency_level': self._assess_overall_urgency(rebalancing_advice, risk_analysis)
        }
    
    # 辅助方法
    
    def _calculate_change(self, data: List, days: int) -> float:
        """计算价格变化"""
        if len(data) < days + 1:
            return 0.0
        current = data[-1]['close']
        past = data[-(days+1)]['close']
        return (current / past - 1) * 100
    
    async def _get_stock_sector(self, stock_code: str) -> str:
        """获取股票所属板块 (简化实现)"""
        # 这里应该实现真实的板块查询逻辑
        return "unknown"
    
    def _get_quality_recommendation(self, score: float) -> str:
        """获取质量建议"""
        if score >= 75:
            return "强烈推荐"
        elif score >= 60:
            return "推荐持有"
        elif score >= 40:
            return "谨慎持有"
        else:
            return "建议卖出"
    
    def _get_quality_grade(self, score: float) -> str:
        """获取质量等级"""
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
    
    def _analyze_quality_distribution(self, scores: List[float]) -> Dict:
        """分析质量分布"""
        if not scores:
            return {}
        
        return {
            'min_score': min(scores),
            'max_score': max(scores),
            'std_score': np.std(scores),
            'excellent_count': len([s for s in scores if s >= 80]),
            'good_count': len([s for s in scores if 65 <= s < 80]),
            'fair_count': len([s for s in scores if 50 <= s < 65]),
            'poor_count': len([s for s in scores if s < 50])
        }
    
    def _calculate_sector_diversification_score(self, allocation_ratio: float) -> float:
        """计算板块分散化评分"""
        # 理想的板块配置应该在5-25%之间
        if 5 <= allocation_ratio <= 25:
            return 100
        elif allocation_ratio < 5:
            return 80 - (5 - allocation_ratio) * 10
        else:  # > 25%
            return 80 - (allocation_ratio - 25) * 2
        
        return max(0, min(100, score))
    
    def _calculate_overall_diversification_score(self, sector_analysis: List[Dict]) -> float:
        """计算整体分散化评分"""
        if not sector_analysis:
            return 0
        
        # 基于板块数量和配置均匀度
        sector_count = len(sector_analysis)
        allocations = [s['allocation_ratio'] for s in sector_analysis]
        
        # 板块数量评分
        if sector_count >= 5:
            count_score = 100
        elif sector_count >= 3:
            count_score = 80
        else:
            count_score = 50
        
        # 配置均匀度评分 (基于方差)
        variance = np.var(allocations) if len(allocations) > 1 else 0
        evenness_score = max(0, 100 - variance)
        
        return (count_score + evenness_score) / 2
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """计算波动率"""
        if len(df) < 20:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # 年化波动率
    
    def _calculate_portfolio_risk_metrics(self, risk_metrics: Dict, total_value: float) -> Dict:
        """计算组合风险指标"""
        # 计算整体风险评分
        position_risks = len([r for r in risk_metrics['position_concentration'] if r['risk_exceeded']])
        volatility_risks = risk_metrics.get('volatility_exposure', [])
        
        avg_volatility = np.mean([v['risk_contribution'] for v in volatility_risks]) if volatility_risks else 0
        
        risk_score = min(100, max(0, 100 - position_risks * 20 - avg_volatility * 100))
        
        if risk_score >= 80:
            risk_grade = "low"
        elif risk_score >= 60:
            risk_grade = "medium"
        else:
            risk_grade = "high"
        
        return {
            'risk_score': risk_score,
            'risk_grade': risk_grade
        }
    
    def _generate_risk_warnings(self, risk_metrics: Dict, overall_risk: Dict) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        position_risks = risk_metrics['position_concentration']
        high_concentration = [r for r in position_risks if r['risk_exceeded']]
        
        if high_concentration:
            warnings.append(f"发现{len(high_concentration)}只股票仓位过重，建议减仓")
        
        if overall_risk['risk_grade'] == 'high':
            warnings.append("投资组合整体风险偏高，建议增强分散化")
        
        return warnings
    
    def _calculate_action_priority(self, quality_score: float, action: str) -> int:
        """计算操作优先级 (1-10, 10最高)"""
        if action == "SELL":
            return max(1, 10 - int(quality_score / 10))
        elif action == "REDUCE":
            return max(1, 8 - int(quality_score / 15))
        else:  # HOLD
            return max(1, int(quality_score / 20))
    
    def _assess_rebalancing_urgency(self, recommendations: List[Dict]) -> str:
        """评估调仓紧急程度"""
        high_urgency = len([r for r in recommendations if r['urgency'] == 'high'])
        
        if high_urgency >= 3:
            return "immediate"
        elif high_urgency >= 1:
            return "high"
        else:
            return "low"
    
    def _identify_portfolio_strengths(
        self, 
        portfolio_status: Dict, 
        quality_analysis: Dict, 
        sector_analysis: Dict
    ) -> List[str]:
        """识别投资组合优势"""
        strengths = []
        
        if quality_analysis['average_quality_score'] >= 70:
            strengths.append("持仓股票整体质量较高")
        
        if sector_analysis['diversification_score'] >= 70:
            strengths.append("板块配置相对分散")
        
        if portfolio_status['total_return_rate'] > 0:
            strengths.append(f"整体收益为正({portfolio_status['total_return_rate']:.1f}%)")
        
        return strengths
    
    def _assess_overall_urgency(self, rebalancing_advice: Dict, risk_analysis: Dict) -> str:
        """评估整体紧急程度"""
        rebalancing_urgency = rebalancing_advice.get('rebalancing_urgency', 'low')
        risk_grade = risk_analysis.get('risk_grade', 'low')
        
        if rebalancing_urgency == 'immediate' or risk_grade == 'high':
            return "immediate"
        elif rebalancing_urgency == 'high' or risk_grade == 'medium':
            return "high"
        else:
            return "low"