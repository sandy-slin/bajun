"""
阶段四：A股市场特征增强器 - 针对A股独特制度设计的优化系统
专门处理T+1交易、涨跌停限制、政策敏感性等A股特有因素
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AShareMarketEnhancer:
    """A股市场特征增强器，专门针对A股市场制度特征进行优化"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.price_limits = {
            'main_board': 0.10,    # 主板涨跌停10%
            'st_stock': 0.05,      # ST股票5%
            'new_stock': 0.44,     # 新股首日44%
            'kcb_board': 0.20,     # 科创板20%
            'gem_board': 0.20      # 创业板20%
        }
        self.policy_keywords = [
            '央行', '证监会', '银保监会', '国务院', '财政部',
            '降准', '降息', 'LPR', '货币政策', '监管',
            '减税', '刺激', '调控', '政策', '改革'
        ]
        
    def enhance_with_ashare_features(self, data: pd.DataFrame, 
                                   sector_name: str = "unknown") -> pd.DataFrame:
        """
        为数据添加A股市场特有特征
        
        Args:
            data: 原始股票数据
            sector_name: 板块名称，用于判断涨跌停限制
            
        Returns:
            增强后的数据
        """
        try:
            if data.empty or len(data) < 10:
                self.logger.warning("数据不足，无法进行A股特征增强")
                return data
                
            enhanced_data = data.copy()
            
            # 1. T+1交易制度特征
            enhanced_data = self._add_t_plus_1_features(enhanced_data)
            
            # 2. 涨跌停限制特征
            enhanced_data = self._add_price_limit_features(enhanced_data, sector_name)
            
            # 3. 政策敏感性特征
            enhanced_data = self._add_policy_sensitivity_features(enhanced_data)
            
            # 4. 板块轮动特征
            enhanced_data = self._add_sector_rotation_features(enhanced_data, sector_name)
            
            # 5. A股特有技术指标
            enhanced_data = self._add_ashare_technical_indicators(enhanced_data)
            
            # 6. 资金流向特征（模拟北向资金、融资融券）
            enhanced_data = self._add_capital_flow_features(enhanced_data)
            
            # 7. 市场情绪周期特征
            enhanced_data = self._add_market_sentiment_cycle_features(enhanced_data)
            
            self.logger.info(f"A股特征增强完成，新增{len(enhanced_data.columns) - len(data.columns)}个特征")
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"A股特征增强失败: {e}")
            return data
    
    def _add_t_plus_1_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加T+1交易制度相关特征"""
        try:
            # T+1制度下，当日买入无法当日卖出，影响交易策略
            
            # 1. 隔夜跳空特征（T+1制度下更重要）
            data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            data['overnight_gap_abs'] = abs(data['overnight_gap'])
            
            # 2. 收盘价相对当日均价的位置（T+1持仓过夜风险）
            data['close_vs_avg_price'] = (data['close'] - (data['high'] + data['low']) / 2) / data['close']
            
            # 3. 尾盘交易强度（T+1制度下尾盘更关键）
            # 用成交量分布模拟尾盘特征
            data['volume_tail_strength'] = data['volume'] / data['volume'].rolling(5).mean()
            
            # 4. 次日开盘预测信号（基于收盘价位置和成交量）
            data['next_day_gap_signal'] = np.where(
                (data['close'] > data['high'] * 0.95) & (data['volume'] > data['volume'].rolling(10).mean()),
                1,  # 预期次日高开
                np.where(
                    (data['close'] < data['low'] * 1.05) & (data['volume'] > data['volume'].rolling(10).mean()),
                    -1,  # 预期次日低开
                    0
                )
            )
            
            # 5. T+1风险评估（持仓过夜风险）
            price_volatility = data['close'].rolling(10).std() / data['close'].rolling(10).mean()
            volume_volatility = data['volume'].rolling(10).std() / data['volume'].rolling(10).mean()
            data['t_plus_1_risk'] = price_volatility * volume_volatility
            
            # 6. 隔夜持仓收益率（模拟T+1的隔夜效应）
            data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            data['overnight_return_ma5'] = data['overnight_return'].rolling(5).mean()
            data['overnight_return_volatility'] = data['overnight_return'].rolling(10).std()
            
            self.logger.info("T+1交易制度特征添加完成")
            return data
            
        except Exception as e:
            self.logger.error(f"T+1特征添加失败: {e}")
            return data
    
    def _add_price_limit_features(self, data: pd.DataFrame, sector_name: str) -> pd.DataFrame:
        """添加涨跌停限制相关特征"""
        try:
            # 根据板块确定涨跌停限制
            price_limit = self._get_price_limit(sector_name)
            
            # 1. 涨跌停距离
            daily_return = data['close'].pct_change()
            data['limit_up_distance'] = (price_limit - daily_return) / price_limit
            data['limit_down_distance'] = (daily_return + price_limit) / price_limit
            
            # 2. 涨跌停触及概率
            data['limit_up_probability'] = np.where(daily_return > price_limit * 0.8, 1, 0)
            data['limit_down_probability'] = np.where(daily_return < -price_limit * 0.8, 1, 0)
            
            # 3. 涨跌停封单强度（用成交量模拟）
            # 接近涨跌停时成交量的表现
            data['limit_volume_strength'] = np.where(
                abs(daily_return) > price_limit * 0.7,
                data['volume'] / data['volume'].rolling(20).mean(),
                1.0
            )
            
            # 4. 连续涨跌停特征
            limit_up_signal = (daily_return > price_limit * 0.95).astype(int)
            limit_down_signal = (daily_return < -price_limit * 0.95).astype(int)
            
            data['consecutive_limit_up'] = self._calculate_consecutive(limit_up_signal)
            data['consecutive_limit_down'] = self._calculate_consecutive(limit_down_signal)
            
            # 5. 涨跌停后的反转概率
            data['post_limit_up_reversal'] = np.where(
                limit_up_signal.shift(1) == 1,
                (data['close'] < data['open']).astype(int),
                0
            )
            
            data['post_limit_down_reversal'] = np.where(
                limit_down_signal.shift(1) == 1,
                (data['close'] > data['open']).astype(int),
                0
            )
            
            # 6. 接近涨跌停时的分时特征模拟
            # 用最高价、最低价模拟分时走势
            data['intraday_limit_approach'] = np.maximum(
                (data['high'] - data['open']) / data['open'] / price_limit,
                (data['open'] - data['low']) / data['open'] / price_limit
            )
            
            # 7. 涨跌停板块效应
            data['sector_limit_momentum'] = self._calculate_sector_limit_momentum(daily_return, price_limit)
            
            self.logger.info(f"涨跌停特征添加完成，限制比例: {price_limit*100}%")
            return data
            
        except Exception as e:
            self.logger.error(f"涨跌停特征添加失败: {e}")
            return data
    
    def _add_policy_sensitivity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加政策敏感性特征"""
        try:
            # A股对政策高度敏感，这里用技术指标模拟政策影响
            
            # 1. 政策敏感度指标（基于异常波动）
            returns = data['close'].pct_change()
            volatility = returns.rolling(10).std()
            
            # 异常波动可能表示政策影响
            data['policy_sensitivity'] = np.where(
                abs(returns) > volatility * 2,
                abs(returns) / volatility,
                0
            )
            
            # 2. 政策周期特征（月度、季度）
            if hasattr(data.index, 'month'):
                # 基于月份的政策周期
                data['policy_month_cycle'] = np.sin(2 * np.pi * data.index.month / 12)
                data['policy_quarter_cycle'] = np.sin(2 * np.pi * (data.index.month - 1) / 3)
            else:
                # 用简单的周期模拟
                data['policy_month_cycle'] = np.sin(2 * np.pi * np.arange(len(data)) / 22)  # 月度周期
                data['policy_quarter_cycle'] = np.sin(2 * np.pi * np.arange(len(data)) / 66)  # 季度周期
            
            # 3. 政策预期指标（基于趋势变化）
            price_trend = data['close'].rolling(20).mean().pct_change(5)
            volume_trend = data['volume'].rolling(20).mean().pct_change(5)
            
            data['policy_expectation'] = (price_trend + volume_trend) / 2
            
            # 4. 监管风险指标
            # 基于价格和成交量的异常组合判断监管风险
            price_spike = abs(returns) > returns.rolling(20).std() * 2.5
            volume_spike = data['volume'] > data['volume'].rolling(20).mean() * 2
            
            data['regulatory_risk'] = (price_spike & volume_spike).astype(int)
            
            # 5. 政策利好/利空信号
            # 基于价量配合度判断政策影响方向
            price_volume_corr = returns.rolling(10).corr(data['volume'].pct_change())
            
            data['policy_positive_signal'] = np.where(
                (returns > 0) & (price_volume_corr > 0.3) & (data['volume'] > data['volume'].rolling(10).mean()),
                1, 0
            )
            
            data['policy_negative_signal'] = np.where(
                (returns < 0) & (price_volume_corr > 0.3) & (data['volume'] > data['volume'].rolling(10).mean()),
                1, 0
            )
            
            # 6. 政策影响衰减
            data['policy_impact_decay'] = data['policy_sensitivity'].rolling(5).mean() * np.exp(-np.arange(len(data)) * 0.1)
            
            self.logger.info("政策敏感性特征添加完成")
            return data
            
        except Exception as e:
            self.logger.error(f"政策敏感性特征添加失败: {e}")
            return data
    
    def _add_sector_rotation_features(self, data: pd.DataFrame, sector_name: str) -> pd.DataFrame:
        """添加板块轮动相关特征"""
        try:
            # A股市场板块轮动明显，这里模拟板块轮动效应
            
            # 1. 板块相对强度
            returns = data['close'].pct_change()
            
            # 短期相对强度（5日）
            data['sector_relative_strength_5d'] = returns.rolling(5).mean()
            
            # 中期相对强度（20日）  
            data['sector_relative_strength_20d'] = returns.rolling(20).mean()
            
            # 2. 板块轮动信号
            short_strength = data['sector_relative_strength_5d']
            long_strength = data['sector_relative_strength_20d']
            
            # 强势转弱信号
            data['sector_strength_weakening'] = np.where(
                (short_strength < long_strength) & (short_strength.shift(1) >= long_strength.shift(1)),
                1, 0
            )
            
            # 弱势转强信号
            data['sector_strength_strengthening'] = np.where(
                (short_strength > long_strength) & (short_strength.shift(1) <= long_strength.shift(1)),
                1, 0
            )
            
            # 3. 板块资金流向
            # 用价量关系模拟资金流向
            volume_ma = data['volume'].rolling(10).mean()
            price_change = data['close'].pct_change()
            
            data['sector_money_flow'] = np.where(
                price_change > 0,
                data['volume'] / volume_ma * price_change,
                -data['volume'] / volume_ma * abs(price_change)
            )
            
            data['sector_money_flow_ma5'] = data['sector_money_flow'].rolling(5).mean()
            
            # 4. 板块热度指标
            # 基于成交量和波动率
            volatility = returns.rolling(10).std()
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            
            data['sector_heat_index'] = volatility * volume_ratio
            data['sector_heat_rank'] = data['sector_heat_index'].rolling(20).rank(pct=True)
            
            # 5. 板块周期位置
            # 模拟板块在轮动周期中的位置
            cumulative_return = (1 + returns).cumprod()
            rolling_max = cumulative_return.rolling(60, min_periods=1).max()
            rolling_min = cumulative_return.rolling(60, min_periods=1).min()
            
            data['sector_cycle_position'] = (cumulative_return - rolling_min) / (rolling_max - rolling_min)
            
            # 6. 板块领涨/领跌特征
            data['sector_leadership'] = np.where(
                data['sector_heat_rank'] > 0.8,
                1,  # 领涨
                np.where(data['sector_heat_rank'] < 0.2, -1, 0)  # 领跌
            )
            
            # 7. 板块轮动预测信号
            # 基于多个指标的综合判断
            rotation_signals = (
                data['sector_strength_strengthening'] + 
                (data['sector_money_flow_ma5'] > 0).astype(int) + 
                (data['sector_heat_rank'] > 0.6).astype(int)
            )
            
            data['sector_rotation_bull_signal'] = (rotation_signals >= 2).astype(int)
            
            self.logger.info(f"板块轮动特征添加完成，板块: {sector_name}")
            return data
            
        except Exception as e:
            self.logger.error(f"板块轮动特征添加失败: {e}")
            return data
    
    def _add_ashare_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加A股特有技术指标"""
        try:
            # 1. 量价背离指标（A股重要特征）
            price_change = data['close'].pct_change()
            volume_change = data['volume'].pct_change()
            
            # 5日量价相关性
            data['price_volume_corr_5d'] = price_change.rolling(5).corr(volume_change)
            
            # 量价背离信号
            data['price_volume_divergence'] = np.where(
                (price_change > 0) & (volume_change < 0) |
                (price_change < 0) & (volume_change > 0),
                1, 0
            )
            
            # 2. 换手率相关指标
            # 用成交量模拟换手率
            volume_ma20 = data['volume'].rolling(20).mean()
            data['turnover_ratio'] = data['volume'] / volume_ma20
            data['turnover_ratio_ma5'] = data['turnover_ratio'].rolling(5).mean()
            
            # 3. 主力控盘度
            # 基于成交量分布特征
            volume_std = data['volume'].rolling(20).std()
            volume_mean = data['volume'].rolling(20).mean()
            data['main_control_degree'] = 1 - (volume_std / volume_mean)
            
            # 4. 筹码集中度
            # 基于价格区间分布
            price_range = data['high'] - data['low']
            avg_price_range = price_range.rolling(20).mean()
            data['chip_concentration'] = 1 - (price_range / avg_price_range)
            
            # 5. 资金净流入
            # 基于价格和成交量的关系
            data['net_money_flow'] = np.where(
                data['close'] > (data['high'] + data['low']) / 2,
                data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low']),
                -data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'])
            )
            
            data['net_money_flow_ma5'] = data['net_money_flow'].rolling(5).mean()
            
            # 6. 强弱分界线
            # A股特有的技术分析方法
            ma5 = data['close'].rolling(5).mean()
            ma10 = data['close'].rolling(10).mean()
            ma20 = data['close'].rolling(20).mean()
            
            data['strength_line'] = (ma5 + ma10 + ma20) / 3
            data['above_strength_line'] = (data['close'] > data['strength_line']).astype(int)
            
            # 7. 多空平衡点
            data['bull_bear_balance'] = (data['high'] + data['low'] + data['close']) / 3
            data['price_vs_balance'] = (data['close'] - data['bull_bear_balance']) / data['bull_bear_balance']
            
            self.logger.info("A股技术指标添加完成")
            return data
            
        except Exception as e:
            self.logger.error(f"A股技术指标添加失败: {e}")
            return data
    
    def _add_capital_flow_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加资金流向特征（模拟北向资金、融资融券等）"""
        try:
            # 1. 北向资金流向模拟
            # 基于隔夜跳空和开盘表现
            overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            volume_surge = data['volume'] / data['volume'].rolling(5).mean()
            
            data['northbound_flow_proxy'] = overnight_gap * volume_surge
            data['northbound_flow_ma5'] = data['northbound_flow_proxy'].rolling(5).mean()
            
            # 2. 融资融券余额变化模拟
            price_change = data['close'].pct_change()
            
            # 融资余额变化（价格上涨时增加）
            data['margin_buy_change'] = np.where(
                price_change > 0,
                price_change * data['volume'] / data['volume'].rolling(10).mean(),
                0
            )
            
            # 融券余额变化（价格下跌时增加）
            data['margin_sell_change'] = np.where(
                price_change < 0,
                abs(price_change) * data['volume'] / data['volume'].rolling(10).mean(),
                0
            )
            
            # 融资融券净额
            data['margin_net_change'] = data['margin_buy_change'] - data['margin_sell_change']
            data['margin_net_ma5'] = data['margin_net_change'].rolling(5).mean()
            
            # 3. 大单净流入
            # 基于价格跳跃和成交量放大
            price_jump = abs(price_change) > price_change.rolling(20).std() * 1.5
            volume_spike = data['volume'] > data['volume'].rolling(20).mean() * 1.5
            
            data['large_order_net_inflow'] = np.where(
                price_jump & volume_spike,
                np.sign(price_change) * data['volume'],
                0
            )
            
            data['large_order_net_inflow_ma5'] = data['large_order_net_inflow'].rolling(5).mean()
            
            # 4. 机构资金流向
            # 基于价格稳定性和持续性
            price_stability = 1 / (1 + data['close'].rolling(10).std() / data['close'].rolling(10).mean())
            price_persistence = (data['close'] > data['close'].rolling(5).mean()).astype(int).rolling(5).mean()
            
            data['institutional_flow'] = price_stability * price_persistence * data['volume']
            data['institutional_flow_ma10'] = data['institutional_flow'].rolling(10).mean()
            
            # 5. 散户情绪指标
            # 基于换手率和波动率
            turnover = data['volume'] / data['volume'].rolling(20).mean()
            volatility = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
            
            data['retail_sentiment'] = turnover * volatility
            data['retail_sentiment_extreme'] = (
                data['retail_sentiment'] > data['retail_sentiment'].rolling(20).quantile(0.8)
            ).astype(int)
            
            # 6. 外资流向综合指标
            data['foreign_capital_flow'] = (
                data['northbound_flow_ma5'] * 0.4 +
                data['large_order_net_inflow_ma5'] * 0.3 +
                data['institutional_flow_ma10'] * 0.3
            )
            
            self.logger.info("资金流向特征添加完成")
            return data
            
        except Exception as e:
            self.logger.error(f"资金流向特征添加失败: {e}")
            return data
    
    def _add_market_sentiment_cycle_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加市场情绪周期特征"""
        try:
            # A股市场情绪周期性明显
            
            # 1. 恐慌贪婪指数
            returns = data['close'].pct_change()
            volatility = returns.rolling(10).std()
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            
            # 贪婪指数（价格上涨+成交量放大）
            data['greed_index'] = np.where(
                (returns > 0) & (volume_ratio > 1.2),
                returns * volume_ratio,
                0
            )
            
            # 恐慌指数（价格下跌+成交量放大）
            data['fear_index'] = np.where(
                (returns < 0) & (volume_ratio > 1.2),
                abs(returns) * volume_ratio,
                0
            )
            
            data['fear_greed_balance'] = data['greed_index'] - data['fear_index']
            data['fear_greed_ma10'] = data['fear_greed_balance'].rolling(10).mean()
            
            # 2. 市场周期阶段
            cumulative_return = (1 + returns).cumprod()
            
            # 牛市阶段
            bull_phase = (cumulative_return > cumulative_return.rolling(60).max() * 0.95).astype(int)
            
            # 熊市阶段
            bear_phase = (cumulative_return < cumulative_return.rolling(60).min() * 1.05).astype(int)
            
            # 震荡阶段
            sideways_phase = 1 - bull_phase - bear_phase
            
            data['market_phase_bull'] = bull_phase
            data['market_phase_bear'] = bear_phase
            data['market_phase_sideways'] = sideways_phase
            
            # 3. 情绪极值反转信号
            greed_extreme = data['greed_index'] > data['greed_index'].rolling(60).quantile(0.9)
            fear_extreme = data['fear_index'] > data['fear_index'].rolling(60).quantile(0.9)
            
            data['sentiment_reversal_signal'] = np.where(
                greed_extreme, -1,  # 极度贪婪，看空信号
                np.where(fear_extreme, 1, 0)  # 极度恐慌，看多信号
            )
            
            # 4. 市场共振指标
            # 价格、成交量、情绪的共振
            price_momentum = returns.rolling(5).mean()
            volume_momentum = data['volume'].pct_change().rolling(5).mean()
            sentiment_momentum = data['fear_greed_balance'].rolling(5).mean()
            
            data['market_resonance'] = np.sign(price_momentum) + np.sign(volume_momentum) + np.sign(sentiment_momentum)
            data['market_resonance_strength'] = abs(data['market_resonance']) / 3
            
            # 5. 情绪周期预测
            # 基于情绪的周期性变化
            sentiment_cycle = np.sin(2 * np.pi * np.arange(len(data)) / 30)  # 30日周期
            data['sentiment_cycle_prediction'] = sentiment_cycle * data['market_resonance_strength']
            
            self.logger.info("市场情绪周期特征添加完成")
            return data
            
        except Exception as e:
            self.logger.error(f"市场情绪周期特征添加失败: {e}")
            return data
    
    def _get_price_limit(self, sector_name: str) -> float:
        """根据板块名称确定涨跌停限制"""
        sector_lower = sector_name.lower()
        
        if 'st' in sector_lower:
            return self.price_limits['st_stock']
        elif '科创' in sector_name or 'kcb' in sector_lower:
            return self.price_limits['kcb_board']
        elif '创业' in sector_name or 'gem' in sector_lower:
            return self.price_limits['gem_board']
        else:
            return self.price_limits['main_board']
    
    def _calculate_consecutive(self, signal_series: pd.Series) -> pd.Series:
        """计算连续信号的长度"""
        try:
            consecutive = []
            current_count = 0
            
            for i, value in enumerate(signal_series):
                if value == 1:
                    current_count += 1
                else:
                    current_count = 0
                consecutive.append(current_count)
            
            return pd.Series(consecutive, index=signal_series.index)
            
        except Exception:
            return pd.Series(0, index=signal_series.index)
    
    def _calculate_sector_limit_momentum(self, daily_return: pd.Series, price_limit: float) -> pd.Series:
        """计算板块涨跌停动量"""
        try:
            # 接近涨跌停时的动量特征
            limit_momentum = np.where(
                daily_return > price_limit * 0.7,
                daily_return / price_limit,
                np.where(
                    daily_return < -price_limit * 0.7,
                    daily_return / price_limit,
                    0
                )
            )
            
            return pd.Series(limit_momentum, index=daily_return.index)
            
        except Exception:
            return pd.Series(0, index=daily_return.index)
    
    def get_ashare_market_insights(self, data: pd.DataFrame) -> Dict:
        """获取A股市场洞察分析"""
        try:
            insights = {
                't_plus_1_risk_level': 'unknown',
                'price_limit_pressure': 'unknown', 
                'policy_sensitivity': 'unknown',
                'sector_rotation_stage': 'unknown',
                'capital_flow_direction': 'unknown',
                'market_sentiment_phase': 'unknown'
            }
            
            if data.empty:
                return insights
            
            # T+1风险水平
            if 't_plus_1_risk' in data.columns:
                avg_risk = data['t_plus_1_risk'].tail(10).mean()
                if avg_risk > 0.15:
                    insights['t_plus_1_risk_level'] = 'high'
                elif avg_risk > 0.08:
                    insights['t_plus_1_risk_level'] = 'medium'
                else:
                    insights['t_plus_1_risk_level'] = 'low'
            
            # 涨跌停压力
            if 'limit_up_distance' in data.columns:
                recent_limit_distance = data['limit_up_distance'].tail(5).mean()
                if recent_limit_distance < 0.3:
                    insights['price_limit_pressure'] = 'high'
                elif recent_limit_distance < 0.6:
                    insights['price_limit_pressure'] = 'medium'
                else:
                    insights['price_limit_pressure'] = 'low'
            
            # 政策敏感性
            if 'policy_sensitivity' in data.columns:
                policy_events = (data['policy_sensitivity'] > 1).sum()
                if policy_events > len(data) * 0.1:
                    insights['policy_sensitivity'] = 'high'
                elif policy_events > len(data) * 0.05:
                    insights['policy_sensitivity'] = 'medium'
                else:
                    insights['policy_sensitivity'] = 'low'
            
            # 板块轮动阶段
            if 'sector_cycle_position' in data.columns:
                cycle_position = data['sector_cycle_position'].tail(5).mean()
                if cycle_position > 0.7:
                    insights['sector_rotation_stage'] = 'peak'
                elif cycle_position > 0.4:
                    insights['sector_rotation_stage'] = 'rising'
                elif cycle_position > 0.2:
                    insights['sector_rotation_stage'] = 'bottom'
                else:
                    insights['sector_rotation_stage'] = 'declining'
            
            # 资金流向
            if 'foreign_capital_flow' in data.columns:
                capital_flow = data['foreign_capital_flow'].tail(10).mean()
                if capital_flow > 0:
                    insights['capital_flow_direction'] = 'inflow'
                else:
                    insights['capital_flow_direction'] = 'outflow'
            
            # 市场情绪阶段
            if 'fear_greed_ma10' in data.columns:
                sentiment = data['fear_greed_ma10'].tail(5).mean()
                if sentiment > 0.1:
                    insights['market_sentiment_phase'] = 'greedy'
                elif sentiment < -0.1:
                    insights['market_sentiment_phase'] = 'fearful'
                else:
                    insights['market_sentiment_phase'] = 'neutral'
            
            return insights
            
        except Exception as e:
            self.logger.error(f"A股市场洞察分析失败: {e}")
            return insights