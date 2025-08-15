"""
技术指标计算模块
提供各种技术分析指标的计算功能
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class TechnicalCalculator:
    """技术指标计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_sector_strength_score(self, sector_data: pd.DataFrame, 
                                      benchmark_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        计算板块强度评分
        
        Args:
            sector_data: 板块数据
            benchmark_data: 基准数据(如沪深300)
            
        Returns:
            Dict: 各项评分指标
        """
        if sector_data.empty:
            return self._get_default_scores()
            
        try:
            scores = {}
            
            # 1. 技术面强度评分 (40%权重)
            tech_scores = self._calculate_technical_scores(sector_data)
            scores.update(tech_scores)
            
            # 2. 资金流向评分 (30%权重) 
            money_flow_scores = self._calculate_money_flow_scores(sector_data)
            scores.update(money_flow_scores)
            
            # 3. 基本面景气度 (20%权重)
            fundamental_scores = self._calculate_fundamental_scores(sector_data)
            scores.update(fundamental_scores)
            
            # 4. 轮动周期判断 (10%权重)
            rotation_scores = self._calculate_rotation_scores(sector_data)
            scores.update(rotation_scores)
            
            # 综合评分计算
            comprehensive_score = (
                tech_scores.get('technical_score', 0) * 0.4 +
                money_flow_scores.get('money_flow_score', 0) * 0.3 +
                fundamental_scores.get('fundamental_score', 0) * 0.2 +
                rotation_scores.get('rotation_score', 0) * 0.1
            )
            
            scores['comprehensive_score'] = comprehensive_score
            scores['recommendation'] = self._get_recommendation(comprehensive_score)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"计算板块强度评分失败: {e}")
            return self._get_default_scores()
            
    def _calculate_technical_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算技术面评分"""
        scores = {}
        latest = df.iloc[-1]
        
        try:
            # RSI评分 (40-70区间为强势)
            rsi = latest.get('rsi', 50)
            if 40 <= rsi <= 70:
                rsi_score = min(100, (rsi - 40) / 30 * 100)
            else:
                rsi_score = max(0, 100 - abs(rsi - 55) * 2)
            scores['rsi_score'] = rsi_score
            
            # 均线突破评分
            ma_score = 0
            close_price = latest.get('close', 0)
            if close_price > latest.get('ma5', 0):
                ma_score += 25
            if close_price > latest.get('ma10', 0):
                ma_score += 25  
            if close_price > latest.get('ma20', 0):
                ma_score += 25
            if close_price > latest.get('ma60', 0):
                ma_score += 25
            scores['ma_breakthrough_score'] = ma_score
            
            # MACD金叉评分
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_score = 0
            if macd > macd_signal and macd > 0:
                macd_score = 100
            elif macd > macd_signal:
                macd_score = 70
            elif macd > 0:
                macd_score = 50
            else:
                macd_score = 20
            scores['macd_score'] = macd_score
            
            # 价格相对强弱
            if len(df) >= 5:
                recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                strength_score = min(100, max(0, 50 + recent_return * 10))
                scores['price_strength_score'] = strength_score
            else:
                scores['price_strength_score'] = 50
                
            # 技术面综合评分
            technical_score = (
                rsi_score * 0.25 +
                ma_score * 0.25 +
                macd_score * 0.25 +
                scores['price_strength_score'] * 0.25
            )
            scores['technical_score'] = technical_score
            
        except Exception as e:
            self.logger.error(f"计算技术面评分失败: {e}")
            scores = {
                'rsi_score': 50,
                'ma_breakthrough_score': 50,
                'macd_score': 50,
                'price_strength_score': 50,
                'technical_score': 50
            }
            
        return scores
        
    def _calculate_money_flow_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算资金流向评分"""
        scores = {}
        
        try:
            # 成交量变化评分
            if len(df) >= 5:
                recent_vol = df['volume'].iloc[-3:].mean()
                avg_vol = df['volume'].iloc[-10:-3].mean() if len(df) >= 10 else recent_vol
                
                if avg_vol > 0:
                    volume_ratio = recent_vol / avg_vol
                    volume_score = min(100, max(0, 50 + (volume_ratio - 1) * 50))
                else:
                    volume_score = 50
            else:
                volume_score = 50
                
            scores['volume_score'] = volume_score
            
            # 成交额增长评分  
            if len(df) >= 5:
                recent_amount = df['amount'].iloc[-3:].mean()
                avg_amount = df['amount'].iloc[-10:-3].mean() if len(df) >= 10 else recent_amount
                
                if avg_amount > 0:
                    amount_ratio = recent_amount / avg_amount
                    amount_score = min(100, max(0, 50 + (amount_ratio - 1) * 30))
                else:
                    amount_score = 50
            else:
                amount_score = 50
                
            scores['amount_score'] = amount_score
            
            # 主力资金评分（基于价量关系）
            main_fund_score = 50
            if len(df) >= 3:
                price_changes = df['pct_change'].iloc[-3:]
                volume_changes = df['volume'].pct_change().iloc[-3:]
                
                # 价涨量增为正面信号
                positive_signals = sum(1 for p, v in zip(price_changes, volume_changes) 
                                     if p > 0 and v > 0)
                main_fund_score = min(100, 30 + positive_signals * 23)
                
            scores['main_fund_score'] = main_fund_score
            
            # 资金流向综合评分
            money_flow_score = (
                volume_score * 0.4 +
                amount_score * 0.35 +
                main_fund_score * 0.25
            )
            scores['money_flow_score'] = money_flow_score
            
        except Exception as e:
            self.logger.error(f"计算资金流向评分失败: {e}")
            scores = {
                'volume_score': 50,
                'amount_score': 50,
                'main_fund_score': 50,
                'money_flow_score': 50
            }
            
        return scores
        
    def _calculate_fundamental_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算基本面评分"""
        scores = {}
        
        try:
            # 趋势稳定性评分
            if len(df) >= 10:
                returns = df['pct_change'].iloc[-10:]
                volatility = returns.std()
                trend_score = max(0, min(100, 100 - volatility * 5))
            else:
                trend_score = 50
                
            scores['trend_stability_score'] = trend_score
            
            # 估值修复评分（基于技术面代理）
            latest_close = df['close'].iloc[-1]
            if len(df) >= 60:
                ma60 = df['ma60'].iloc[-1]
                if ma60 > 0:
                    valuation_ratio = latest_close / ma60
                    # 接近或低于60日均线视为估值合理/低估
                    valuation_score = max(0, min(100, 150 - valuation_ratio * 50))
                else:
                    valuation_score = 50
            else:
                valuation_score = 50
                
            scores['valuation_score'] = valuation_score
            
            # 政策催化评分（基于随机模拟，实际应结合新闻分析）
            import random
            policy_score = random.uniform(30, 80)  # 模拟政策影响
            scores['policy_score'] = policy_score
            
            # 基本面综合评分
            fundamental_score = (
                trend_score * 0.4 +
                valuation_score * 0.4 +
                policy_score * 0.2
            )
            scores['fundamental_score'] = fundamental_score
            
        except Exception as e:
            self.logger.error(f"计算基本面评分失败: {e}")
            scores = {
                'trend_stability_score': 50,
                'valuation_score': 50, 
                'policy_score': 50,
                'fundamental_score': 50
            }
            
        return scores
        
    def _calculate_rotation_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算轮动周期评分"""
        scores = {}
        
        try:
            # 周期位置评分（基于价格波动周期）
            if len(df) >= 20:
                closes = df['close'].iloc[-20:]
                cycle_high = closes.max()
                cycle_low = closes.min()
                current_price = closes.iloc[-1]
                
                if cycle_high > cycle_low:
                    cycle_position = (current_price - cycle_low) / (cycle_high - cycle_low)
                    # 在周期底部到中部为较好买入时机
                    if cycle_position < 0.3:
                        position_score = 90
                    elif cycle_position < 0.6:
                        position_score = 75
                    elif cycle_position < 0.8:
                        position_score = 50
                    else:
                        position_score = 25
                else:
                    position_score = 50
            else:
                position_score = 50
                
            scores['cycle_position_score'] = position_score
            
            # 动量评分
            if len(df) >= 10:
                short_momentum = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
                long_momentum = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
                
                momentum_score = min(100, max(0, 50 + (short_momentum + long_momentum)))
            else:
                momentum_score = 50
                
            scores['momentum_score'] = momentum_score
            
            # 轮动强度评分（基于相对强弱）
            if len(df) >= 20:
                # 计算相对强弱指标
                recent_performance = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                if recent_performance > 5:
                    rotation_strength = 80
                elif recent_performance > 2:
                    rotation_strength = 65
                elif recent_performance > -2:
                    rotation_strength = 50
                elif recent_performance > -5:
                    rotation_strength = 35
                else:
                    rotation_strength = 20
            else:
                rotation_strength = 50
                
            scores['rotation_strength_score'] = rotation_strength
            
            # 轮动周期综合评分
            rotation_score = (
                position_score * 0.4 +
                momentum_score * 0.3 +
                rotation_strength * 0.3
            )
            scores['rotation_score'] = rotation_score
            
        except Exception as e:
            self.logger.error(f"计算轮动周期评分失败: {e}")
            scores = {
                'cycle_position_score': 50,
                'momentum_score': 50,
                'rotation_strength_score': 50,
                'rotation_score': 50
            }
            
        return scores
    
    def calculate_sector_correlation(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """计算板块间相关性"""
        try:
            correlation_matrix = {}
            
            # 提取各板块的收盘价数据
            sector_returns = {}
            for sector_name, df in sector_data.items():
                if not df.empty and 'close' in df.columns:
                    # 计算日收益率
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        sector_returns[sector_name] = returns
            
            # 计算相关性矩阵
            if len(sector_returns) > 1:
                # 对齐数据长度
                min_length = min(len(returns) for returns in sector_returns.values())
                aligned_returns = {}
                for sector_name, returns in sector_returns.items():
                    if len(returns) >= min_length:
                        aligned_returns[sector_name] = returns.iloc[-min_length:]
                
                if len(aligned_returns) > 1:
                    # 创建DataFrame并计算相关性
                    import pandas as pd
                    returns_df = pd.DataFrame(aligned_returns)
                    corr_matrix = returns_df.corr()
                    
                    # 转换为字典格式
                    for sector1 in corr_matrix.index:
                        correlation_matrix[sector1] = {}
                        for sector2 in corr_matrix.columns:
                            correlation_matrix[sector1][sector2] = float(corr_matrix.loc[sector1, sector2])
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"计算板块相关性失败: {e}")
            return {}
    
    def analyze_sector_rotation_pattern(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """分析板块轮动模式"""
        try:
            rotation_analysis = {
                'leading_sectors': [],
                'lagging_sectors': [],
                'rotation_opportunities': [],
                'market_phase': 'unknown'
            }
            
            # 计算各板块的相对强弱
            sector_strength = {}
            for sector_name, df in sector_data.items():
                if not df.empty and 'close' in df.columns and len(df) >= 20:
                    # 计算20日相对强弱
                    strength = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
                    sector_strength[sector_name] = strength
            
            if sector_strength:
                # 排序找出领涨和领跌板块
                sorted_sectors = sorted(sector_strength.items(), key=lambda x: x[1], reverse=True)
                
                # 领涨板块（前30%）
                top_count = max(1, len(sorted_sectors) // 3)
                rotation_analysis['leading_sectors'] = [
                    {'sector': name, 'strength': strength} 
                    for name, strength in sorted_sectors[:top_count]
                ]
                
                # 领跌板块（后30%）
                rotation_analysis['lagging_sectors'] = [
                    {'sector': name, 'strength': strength} 
                    for name, strength in sorted_sectors[-top_count:]
                ]
                
                # 轮动机会（中等强度板块）
                mid_start = top_count
                mid_end = len(sorted_sectors) - top_count
                if mid_end > mid_start:
                    rotation_analysis['rotation_opportunities'] = [
                        {'sector': name, 'strength': strength} 
                        for name, strength in sorted_sectors[mid_start:mid_end]
                    ]
                
                # 判断市场阶段
                avg_strength = sum(sector_strength.values()) / len(sector_strength)
                if avg_strength > 3:
                    rotation_analysis['market_phase'] = 'bull_market'
                elif avg_strength < -3:
                    rotation_analysis['market_phase'] = 'bear_market'
                else:
                    rotation_analysis['market_phase'] = 'sideways_market'
            
            return rotation_analysis
            
        except Exception as e:
            self.logger.error(f"分析板块轮动模式失败: {e}")
            return {
                'leading_sectors': [],
                'lagging_sectors': [],
                'rotation_opportunities': [],
                'market_phase': 'unknown'
            }
        
    def _get_default_scores(self) -> Dict[str, float]:
        """获取默认评分"""
        return {
            'technical_score': 50,
            'money_flow_score': 50,
            'fundamental_score': 50,
            'rotation_score': 50,
            'comprehensive_score': 50,
            'recommendation': 'Hold'
        }
        
    def _get_recommendation(self, score: float) -> str:
        """根据评分获取推荐等级"""
        if score >= 85:
            return 'Strong Buy'
        elif score >= 75:
            return 'Buy'
        elif score >= 60:
            return 'Hold+'
        elif score >= 40:
            return 'Hold'
        elif score >= 25:
            return 'Hold-'
        else:
            return 'Avoid'
            
    def calculate_relative_strength(self, sector_data: pd.DataFrame, 
                                  benchmark_data: pd.DataFrame) -> float:
        """计算相对强弱比率"""
        try:
            if len(sector_data) >= 5 and len(benchmark_data) >= 5:
                sector_return = (sector_data['close'].iloc[-1] / sector_data['close'].iloc[-5] - 1)
                benchmark_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[-5] - 1)
                
                if benchmark_return != 0:
                    relative_strength = sector_return / benchmark_return
                    return max(0, min(2, relative_strength))  # 限制在0-2之间
                    
        except Exception as e:
            self.logger.error(f"计算相对强弱失败: {e}")
            
        return 1.0  # 默认相对强弱为1
        
    def detect_breakthrough_signals(self, df: pd.DataFrame) -> List[Dict]:
        """检测突破信号"""
        signals = []
        
        if len(df) < 20:
            return signals
            
        try:
            latest = df.iloc[-1]
            
            # MA突破信号
            if (latest['close'] > latest['ma20'] and 
                df.iloc[-2]['close'] <= df.iloc[-2]['ma20']):
                signals.append({
                    'type': 'MA_BREAKTHROUGH',
                    'description': '突破20日均线',
                    'strength': 'Medium'
                })
                
            # 成交量放大信号  
            if len(df) >= 10:
                avg_vol = df['volume'].iloc[-10:-1].mean()
                if latest['volume'] > avg_vol * 2:
                    signals.append({
                        'type': 'VOLUME_SURGE',
                        'description': '成交量放大超过2倍',
                        'strength': 'High'
                    })
                    
            # MACD金叉信号
            if (latest['macd'] > latest['macd_signal'] and
                df.iloc[-2]['macd'] <= df.iloc[-2]['macd_signal']):
                signals.append({
                    'type': 'MACD_GOLDEN_CROSS',
                    'description': 'MACD金叉',
                    'strength': 'High'
                })
                
        except Exception as e:
            self.logger.error(f"检测突破信号失败: {e}")
            
        return signals