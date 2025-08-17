#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票筛选引擎 - Phase 1 MVP核心组件
板块内股票精选系统

功能:
1. 从TOP5板块中精选股票
2. 评分权重: 涨跌预期(50%) + 成交量确认(30%) + 技术指标(20%)
3. 每个板块精选5只股票
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..data.sector_fetcher import SectorFetcher
from ..data.real_data_fetcher import RealDataFetcher
from ..data.technical_calculator import TechnicalCalculator


class StockEngine:
    """股票筛选引擎 - 板块内精选逻辑"""
    
    def __init__(
        self, 
        sector_fetcher: SectorFetcher,
        data_fetcher: RealDataFetcher,
        tech_calculator: TechnicalCalculator
    ):
        self.sector_fetcher = sector_fetcher
        self.data_fetcher = data_fetcher
        self.tech_calculator = tech_calculator
        self.logger = logging.getLogger(__name__)
        
        # 评分权重配置 (符合requirement.md规定)
        self.scoring_weights = {
            'return_expectation': 0.5,   # 涨跌预期权重
            'volume_confirmation': 0.3,  # 成交量确认权重
            'technical_indicators': 0.2  # 技术指标权重
        }
        
        # 筛选标准
        self.selection_criteria = {
            'fundamental_filter': {
                'roe_min': 0.08,            # ROE > 8%
                'debt_equity_max': 0.6      # 债务权益比 < 0.6
            },
            'technical_filter': {
                'rsi_min': 30,              # RSI 30-70区间
                'rsi_max': 70,
                'momentum_positive': True    # 正向动量
            },
            'liquidity_filter': {
                'daily_volume_min': 10_000_000,  # 日成交量 > 1000万RMB
            },
            'size_constraint': {
                'market_cap_min': 1_000_000_000,   # 市值 > 10亿RMB
                'market_cap_max': 100_000_000_000  # 市值 < 1000亿RMB
            }
        }
        
    async def select_stocks_from_sectors(
        self,
        top_sectors: List[Dict],
        stocks_per_sector: int = 5
    ) -> Dict:
        """
        从TOP5板块中精选股票
        
        Args:
            top_sectors: 板块分析结果列表
            stocks_per_sector: 每个板块选择的股票数量
            
        Returns:
            Dict: 股票筛选结果
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"开始股票筛选，处理{len(top_sectors)}个板块")
            
            selection_results = []
            total_stocks_analyzed = 0
            
            for sector in top_sectors:
                sector_name = sector['sector_name']
                
                try:
                    sector_result = await self._select_stocks_in_sector(
                        sector_name, stocks_per_sector
                    )
                    
                    if sector_result and 'selected_stocks' in sector_result:
                        selection_results.append({
                            'sector_name': sector_name,
                            'sector_score': sector['composite_score'],
                            'stocks_analyzed': sector_result['stocks_analyzed'],
                            'selected_stocks': sector_result['selected_stocks'],
                            'selection_summary': sector_result['selection_summary']
                        })
                        total_stocks_analyzed += sector_result['stocks_analyzed']
                        
                except Exception as e:
                    self.logger.warning(f"板块{sector_name}股票筛选失败: {e}")
                    continue
            
            # 生成整体筛选结果
            result = {
                'timestamp': datetime.now().isoformat(),
                'selection_params': {
                    'stocks_per_sector': stocks_per_sector,
                    'scoring_weights': self.scoring_weights,
                    'selection_criteria': self.selection_criteria
                },
                'sectors_processed': len(selection_results),
                'total_stocks_analyzed': total_stocks_analyzed,
                'sector_selections': selection_results,
                'overall_summary': self._generate_overall_summary(selection_results),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            self.logger.info(f"股票筛选完成，耗时{result['processing_time_seconds']:.1f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"股票筛选失败: {e}")
            return {'error': str(e)}
    
    async def _select_stocks_in_sector(
        self,
        sector_name: str,
        target_count: int
    ) -> Optional[Dict]:
        """
        在单个板块内选择股票
        
        Args:
            sector_name: 板块名称
            target_count: 目标选择数量
            
        Returns:
            Dict: 板块内股票选择结果
        """
        try:
            # 1. 获取板块内所有股票
            sector_stocks = await self.sector_fetcher.get_sector_stocks(sector_name)
            
            if not sector_stocks:
                self.logger.warning(f"板块{sector_name}未找到股票数据")
                return None
            
            self.logger.info(f"板块{sector_name}共{len(sector_stocks)}只股票待分析")
            
            # 2. 对每只股票进行评分
            stock_scores = []
            
            for stock_code in sector_stocks[:50]:  # 限制前50只股票以提高效率
                try:
                    score_result = await self._calculate_stock_score(stock_code)
                    if score_result:
                        stock_scores.append(score_result)
                except Exception as e:
                    self.logger.debug(f"股票{stock_code}评分失败: {e}")
                    continue
            
            # 3. 应用筛选条件
            filtered_stocks = self._apply_selection_filters(stock_scores)
            
            # 4. 排序并选择TOP N
            filtered_stocks.sort(key=lambda x: x['composite_score'], reverse=True)
            selected_stocks = filtered_stocks[:target_count]
            
            return {
                'stocks_analyzed': len(stock_scores),
                'stocks_after_filter': len(filtered_stocks),
                'selected_stocks': selected_stocks,
                'selection_summary': self._generate_sector_summary(
                    sector_name, len(stock_scores), len(filtered_stocks), len(selected_stocks)
                )
            }
            
        except Exception as e:
            self.logger.warning(f"板块{sector_name}股票选择失败: {e}")
            return None
    
    async def _calculate_stock_score(self, stock_code: str) -> Optional[Dict]:
        """
        计算单只股票的综合评分
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict: 股票评分结果
        """
        try:
            # 获取股票历史数据
            stock_data = await self.data_fetcher.get_trading_data(stock_code)
            
            if not stock_data or len(stock_data) < 30:
                return None
            
            df = pd.DataFrame(stock_data)
            
            # 获取股票基本信息
            stock_info = await self._get_stock_basic_info(stock_code)
            
            # 1. 计算涨跌预期评分 (50%权重)
            return_expectation_score = self._calculate_return_expectation(df)
            
            # 2. 计算成交量确认评分 (30%权重)
            volume_confirmation_score = self._calculate_volume_confirmation(df)
            
            # 3. 计算技术指标评分 (20%权重)
            technical_score = self._calculate_technical_indicators(df)
            
            # 4. 计算综合评分
            composite_score = (
                return_expectation_score * self.scoring_weights['return_expectation'] +
                volume_confirmation_score * self.scoring_weights['volume_confirmation'] +
                technical_score * self.scoring_weights['technical_indicators']
            )
            
            return {
                'stock_code': stock_code,
                'stock_name': stock_info.get('name', ''),
                'composite_score': composite_score,
                'return_expectation_score': return_expectation_score,
                'volume_confirmation_score': volume_confirmation_score,
                'technical_score': technical_score,
                'latest_price': float(df.iloc[-1]['close']),
                'price_change_5d': self._calculate_price_change(df, 5),
                'market_cap': stock_info.get('market_cap', 0),
                'daily_volume_value': self._calculate_daily_volume_value(df),
                'rsi': self._calculate_rsi(df),
                'momentum': self._calculate_momentum(df),
                'liquidity_rank': self._assess_liquidity(df),
                'risk_level': self._assess_stock_risk(df)
            }
            
        except Exception as e:
            self.logger.debug(f"计算股票{stock_code}评分失败: {e}")
            return None
    
    def _calculate_return_expectation(self, df: pd.DataFrame) -> float:
        """
        计算涨跌预期评分 (0-100分)
        
        基于价格趋势、动量和技术形态
        """
        try:
            closes = df['close'].values
            
            # 短期趋势评分
            short_trend_score = self._calculate_short_trend(closes)
            
            # 动量评分
            momentum_score = self._calculate_momentum_score(closes)
            
            # 技术形态评分
            pattern_score = self._calculate_pattern_score(closes)
            
            # 组合评分
            expectation_score = (
                short_trend_score * 0.4 +   # 短期趋势 40%
                momentum_score * 0.4 +      # 动量 40%
                pattern_score * 0.2         # 技术形态 20%
            )
            
            return min(100, max(0, expectation_score))
            
        except Exception as e:
            self.logger.debug(f"计算涨跌预期评分失败: {e}")
            return 50.0
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """
        计算成交量确认评分 (0-100分)
        """
        try:
            if 'volume' not in df.columns:
                return 50.0
                
            closes = df['close'].values
            volumes = df['volume'].values
            
            # 量价配合度
            price_volume_correlation = self._calculate_price_volume_correlation(closes, volumes)
            
            # 成交量趋势
            volume_trend_score = self._calculate_volume_trend_score(volumes)
            
            # 流动性评分
            liquidity_score = self._calculate_liquidity_score(volumes)
            
            # 组合评分
            confirmation_score = (
                price_volume_correlation * 0.5 +  # 量价配合 50%
                volume_trend_score * 0.3 +        # 成交量趋势 30%
                liquidity_score * 0.2             # 流动性 20%
            )
            
            return min(100, max(0, confirmation_score))
            
        except Exception as e:
            self.logger.debug(f"计算成交量确认评分失败: {e}")
            return 50.0
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> float:
        """
        计算技术指标评分 (0-100分)
        """
        try:
            closes = df['close'].values
            
            # RSI评分
            rsi_score = self._calculate_rsi_score(closes)
            
            # 移动平均线评分
            ma_score = self._calculate_ma_score(closes)
            
            # MACD评分
            macd_score = self._calculate_macd_score(closes)
            
            # 组合评分
            technical_score = (
                rsi_score * 0.4 +     # RSI 40%
                ma_score * 0.4 +      # 移动平均线 40%
                macd_score * 0.2      # MACD 20%
            )
            
            return min(100, max(0, technical_score))
            
        except Exception as e:
            self.logger.debug(f"计算技术指标评分失败: {e}")
            return 50.0
    
    def _apply_selection_filters(self, stock_scores: List[Dict]) -> List[Dict]:
        """应用筛选条件"""
        filtered_stocks = []
        
        for stock in stock_scores:
            # 流动性筛选
            if stock['daily_volume_value'] < self.selection_criteria['liquidity_filter']['daily_volume_min']:
                continue
                
            # 市值筛选
            market_cap = stock['market_cap']
            if market_cap > 0:  # 有市值数据时才筛选
                min_cap = self.selection_criteria['size_constraint']['market_cap_min']
                max_cap = self.selection_criteria['size_constraint']['market_cap_max']
                if market_cap < min_cap or market_cap > max_cap:
                    continue
            
            # RSI筛选
            rsi = stock['rsi']
            if rsi > 0:  # 有RSI数据时才筛选
                rsi_min = self.selection_criteria['technical_filter']['rsi_min']
                rsi_max = self.selection_criteria['technical_filter']['rsi_max']
                if rsi < rsi_min or rsi > rsi_max:
                    continue
            
            # 动量筛选
            if self.selection_criteria['technical_filter']['momentum_positive']:
                if stock['momentum'] <= 0:
                    continue
            
            filtered_stocks.append(stock)
        
        return filtered_stocks
    
    async def _get_stock_basic_info(self, stock_code: str) -> Dict:
        """获取股票基本信息 (简化版)"""
        return {
            'name': f'股票{stock_code}',  # 简化实现
            'market_cap': 0  # 后续可通过API获取
        }
    
    def _calculate_short_trend(self, closes: np.ndarray) -> float:
        """计算短期趋势评分"""
        if len(closes) < 10:
            return 50.0
            
        # 5日与10日价格比较
        avg_5d = np.mean(closes[-5:])
        avg_10d = np.mean(closes[-10:])
        
        trend_strength = (avg_5d / avg_10d - 1) * 100
        score = 50 + trend_strength * 5  # 转换为评分
        
        return min(100, max(0, score))
    
    def _calculate_momentum_score(self, closes: np.ndarray) -> float:
        """计算动量评分"""
        if len(closes) < 5:
            return 50.0
            
        momentum = (closes[-1] / closes[-5] - 1) * 100
        score = 50 + momentum * 2.5
        
        return min(100, max(0, score))
    
    def _calculate_pattern_score(self, closes: np.ndarray) -> float:
        """计算技术形态评分 (简化版)"""
        if len(closes) < 20:
            return 50.0
            
        # 简化的形态识别: 突破形态
        recent_high = np.max(closes[-5:])
        historical_high = np.max(closes[-20:-5])
        
        if recent_high > historical_high:
            return 70.0  # 突破形态
        else:
            return 40.0  # 无明显形态
    
    def _calculate_price_volume_correlation(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """计算量价相关性评分"""
        if len(closes) < 10 or len(volumes) < 10:
            return 50.0
            
        try:
            price_changes = np.diff(closes[-10:])
            volume_changes = np.diff(volumes[-10:])
            
            correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
            if np.isnan(correlation):
                return 50.0
                
            score = 50 + correlation * 50
            return min(100, max(0, score))
            
        except Exception:
            return 50.0
    
    def _calculate_volume_trend_score(self, volumes: np.ndarray) -> float:
        """计算成交量趋势评分"""
        if len(volumes) < 10:
            return 50.0
            
        recent_avg = np.mean(volumes[-5:])
        historical_avg = np.mean(volumes[-15:-5]) if len(volumes) >= 15 else np.mean(volumes[:-5])
        
        if historical_avg > 0:
            trend = (recent_avg / historical_avg - 1) * 100
            score = 50 + trend * 2
            return min(100, max(0, score))
        
        return 50.0
    
    def _calculate_liquidity_score(self, volumes: np.ndarray) -> float:
        """计算流动性评分"""
        if len(volumes) < 5:
            return 50.0
            
        avg_volume = np.mean(volumes[-5:])
        
        # 简化的流动性评分
        if avg_volume > 50_000_000:  # 5000万股
            return 90.0
        elif avg_volume > 20_000_000:  # 2000万股
            return 70.0
        elif avg_volume > 5_000_000:   # 500万股
            return 50.0
        else:
            return 30.0
    
    def _calculate_rsi_score(self, closes: np.ndarray) -> float:
        """计算RSI评分"""
        if len(closes) < 15:
            return 50.0
            
        try:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # RSI转换为评分
            if 40 <= rsi <= 60:
                return 70.0  # 中性区间偏好
            elif 30 <= rsi < 40:
                return 80.0  # 超卖机会
            elif 60 < rsi <= 70:
                return 60.0  # 超买警告
            else:
                return 40.0  # 极端值
                
        except Exception:
            return 50.0
    
    def _calculate_ma_score(self, closes: np.ndarray) -> float:
        """计算移动平均线评分"""
        if len(closes) < 20:
            return 50.0
            
        current_price = closes[-1]
        ma_5 = np.mean(closes[-5:])
        ma_20 = np.mean(closes[-20:])
        
        # 多头排列检查
        if current_price > ma_5 > ma_20:
            return 80.0
        elif current_price > ma_20:
            return 60.0
        else:
            return 40.0
    
    def _calculate_macd_score(self, closes: np.ndarray) -> float:
        """计算MACD评分 (简化版)"""
        if len(closes) < 26:
            return 50.0
            
        # 简化的MACD计算
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        
        macd = ema_12 - ema_26
        
        if macd > 0:
            return 70.0
        else:
            return 40.0
    
    def _calculate_ema(self, data: np.ndarray, window: int) -> float:
        """计算指数移动平均"""
        if len(data) < window:
            return np.mean(data)
            
        alpha = 2 / (window + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return ema
    
    def _calculate_price_change(self, df: pd.DataFrame, days: int) -> float:
        """计算价格变化百分比"""
        if len(df) < days + 1:
            return 0.0
            
        current = df.iloc[-1]['close']
        past = df.iloc[-(days+1)]['close']
        
        return (current / past - 1) * 100
    
    def _calculate_daily_volume_value(self, df: pd.DataFrame) -> float:
        """计算日均成交金额"""
        if 'volume' not in df.columns or len(df) < 5:
            return 0.0
            
        recent_data = df.iloc[-5:]
        avg_volume = recent_data['volume'].mean()
        avg_price = recent_data['close'].mean()
        
        return avg_volume * avg_price
    
    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """计算RSI值"""
        closes = df['close'].values
        if len(closes) < 15:
            return 0.0
            
        try:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception:
            return 0.0
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """计算动量值"""
        if len(df) < 5:
            return 0.0
            
        current = df.iloc[-1]['close']
        past = df.iloc[-5]['close']
        
        return (current / past - 1) * 100
    
    def _assess_liquidity(self, df: pd.DataFrame) -> str:
        """评估流动性等级"""
        volume_value = self._calculate_daily_volume_value(df)
        
        if volume_value > 100_000_000:  # 1亿
            return "high"
        elif volume_value > 50_000_000:   # 5000万
            return "medium"
        else:
            return "low"
    
    def _assess_stock_risk(self, df: pd.DataFrame) -> str:
        """评估股票风险水平"""
        if len(df) < 20:
            return "unknown"
            
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        if volatility < 0.02:
            return "low"
        elif volatility < 0.04:
            return "medium"
        else:
            return "high"
    
    def _generate_sector_summary(
        self, 
        sector_name: str, 
        analyzed_count: int, 
        filtered_count: int, 
        selected_count: int
    ) -> str:
        """生成板块选择摘要"""
        filter_rate = (filtered_count / analyzed_count * 100) if analyzed_count > 0 else 0
        
        return (
            f"{sector_name}板块: 分析{analyzed_count}只股票，"
            f"通过筛选{filtered_count}只(通过率{filter_rate:.1f}%)，"
            f"最终精选{selected_count}只股票"
        )
    
    def _generate_overall_summary(self, selection_results: List[Dict]) -> Dict:
        """生成整体选择摘要"""
        if not selection_results:
            return {}
            
        total_analyzed = sum(r['stocks_analyzed'] for r in selection_results)
        total_selected = sum(len(r['selected_stocks']) for r in selection_results)
        
        avg_scores = []
        for result in selection_results:
            if result['selected_stocks']:
                sector_avg = np.mean([s['composite_score'] for s in result['selected_stocks']])
                avg_scores.append(sector_avg)
        
        return {
            'total_stocks_analyzed': total_analyzed,
            'total_stocks_selected': total_selected,
            'average_composite_score': np.mean(avg_scores) if avg_scores else 0,
            'selection_quality': self._assess_selection_quality(avg_scores)
        }
    
    def _assess_selection_quality(self, avg_scores: List[float]) -> str:
        """评估选择质量"""
        if not avg_scores:
            return "unknown"
            
        overall_avg = np.mean(avg_scores)
        
        if overall_avg >= 75:
            return "excellent"
        elif overall_avg >= 60:
            return "good"
        elif overall_avg >= 45:
            return "fair"
        else:
            return "poor"