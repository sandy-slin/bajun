"""
高级特征工程模块 - 专门优化预测准确率
针对低准确率问题进行特征工程优化
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class AdvancedFeatureEngineer:
    """高级特征工程器，专门解决预测准确率低的问题"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def create_high_accuracy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建高准确率特征集，针对低准确率问题优化
        专注于最有效的预测特征
        """
        try:
            if data.empty or len(data) < 30:
                self.logger.warning("数据不足，无法创建高级特征")
                return data
                
            features = data.copy()
            
            # 确保基础列存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in features.columns]
            if missing_cols:
                self.logger.error(f"缺少必要列: {missing_cols}")
                return data
            
            # 1. 高效价格特征
            features = self._add_high_efficiency_price_features(features)
            
            # 2. 优化成交量特征
            features = self._add_optimized_volume_features(features)
            
            # 3. 强化技术指标
            features = self._add_enhanced_technical_indicators(features)
            
            # 4. 多维波动率特征
            features = self._add_multi_volatility_features(features)
            
            # 5. 智能趋势识别
            features = self._add_smart_trend_features(features)
            
            # 6. 高精度动量指标
            features = self._add_precision_momentum_features(features)
            
            # 7. 市场微观结构
            features = self._add_microstructure_features(features)
            
            # 8. 预测信号强度
            features = self._add_signal_strength_features(features)
            
            # 9. 交叉验证特征
            features = self._add_cross_validation_features(features)
            
            # 10. 市场情绪特征增强 (阶段一优化)
            features = self._add_market_sentiment_features(features)
            
            # 11. 时间序列增强特征 (阶段一优化) 
            features = self._add_enhanced_time_series_features(features)
            
            # 数据清洗和质量控制
            features = self._clean_and_validate_features(features)
            
            self.logger.info(f"高级特征工程完成，生成{len(features.columns)}个特征，数据行数: {len(features)}")
            return features
            
        except Exception as e:
            self.logger.error(f"高级特征工程失败: {e}")
            return data
    
    def _add_high_efficiency_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加高效价格特征"""
        try:
            # 真实收益率（对数收益率更稳定）
            data['log_return'] = np.log(data['close'] / data['close'].shift(1))
            
            # 多周期收益率
            for period in [1, 2, 3, 5, 10]:
                data[f'return_{period}d'] = data['close'].pct_change(period)
                data[f'log_return_{period}d'] = np.log(data['close'] / data['close'].shift(period))
            
            # 价格相对位置（在近期范围内的位置）
            for window in [5, 10, 20]:
                rolling_min = data['low'].rolling(window=window).min()
                rolling_max = data['high'].rolling(window=window).max()
                data[f'price_position_{window}'] = (data['close'] - rolling_min) / (rolling_max - rolling_min)
            
            # 缺口分析
            data['gap_up'] = (data['open'] > data['close'].shift(1)).astype(int)
            data['gap_down'] = (data['open'] < data['close'].shift(1)).astype(int)
            data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            
            # 价格变化率的变化率（加速度）
            data['price_acceleration'] = data['log_return'].diff()
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加价格特征失败: {e}")
            return data
    
    def _add_optimized_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加优化的成交量特征"""
        try:
            # 成交量归一化
            data['volume_normalized'] = data['volume'] / data['volume'].rolling(window=20).mean()
            
            # 成交量变化率
            data['volume_change'] = data['volume'].pct_change()
            data['volume_acceleration'] = data['volume_change'].diff()
            
            # 价量关系强度
            price_change = data['close'].pct_change()
            data['price_volume_correlation'] = price_change.rolling(window=10).corr(data['volume_change'])
            
            # 成交量分布特征
            for window in [5, 10, 20]:
                data[f'volume_std_{window}'] = data['volume'].rolling(window=window).std()
                data[f'volume_skew_{window}'] = data['volume'].rolling(window=window).skew()
                data[f'volume_kurt_{window}'] = data['volume'].rolling(window=window).kurt()
            
            # OBV改进版
            direction = np.where(data['close'] > data['close'].shift(1), 1, 
                        np.where(data['close'] < data['close'].shift(1), -1, 0))
            data['obv_improved'] = (direction * data['volume']).cumsum()
            data['obv_ma5'] = data['obv_improved'].rolling(window=5).mean()
            data['obv_divergence'] = data['obv_improved'] - data['obv_ma5']
            
            # 成交量价格趋势
            data['vpt'] = (data['close'].pct_change() * data['volume']).cumsum()
            data['vpt_signal'] = data['vpt'].rolling(window=10).mean()
            
            # 成交量突破信号
            volume_mean = data['volume'].rolling(window=20).mean()
            volume_std = data['volume'].rolling(window=20).std()
            data['volume_breakout'] = (data['volume'] > volume_mean + 2 * volume_std).astype(int)
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加成交量特征失败: {e}")
            return data
    
    def _add_enhanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加增强技术指标"""
        try:
            # 改进的RSI（多周期）
            for period in [6, 14, 21]:
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # RSI背离检测
            data['rsi_bullish_divergence'] = ((data['close'] < data['close'].shift(10)) & 
                                            (data['rsi_14'] > data['rsi_14'].shift(10))).astype(int)
            data['rsi_bearish_divergence'] = ((data['close'] > data['close'].shift(10)) & 
                                            (data['rsi_14'] < data['rsi_14'].shift(10))).astype(int)
            
            # 多重MACD系统
            for fast, slow in [(5, 10), (12, 26), (19, 39)]:
                ema_fast = data['close'].ewm(span=fast).mean()
                ema_slow = data['close'].ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                signal = macd.ewm(span=9).mean()
                data[f'macd_{fast}_{slow}'] = macd
                data[f'macd_signal_{fast}_{slow}'] = signal
                data[f'macd_histogram_{fast}_{slow}'] = macd - signal
                data[f'macd_cross_{fast}_{slow}'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
            
            # 自适应布林带
            for period in [10, 20, 30]:
                sma = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                data[f'bb_upper_{period}'] = sma + (std * 2)
                data[f'bb_lower_{period}'] = sma - (std * 2)
                data[f'bb_position_{period}'] = (data['close'] - data[f'bb_lower_{period}']) / \
                                               (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}'])
                data[f'bb_squeeze_{period}'] = (std < std.shift(1)).astype(int)
            
            # 动态支撑阻力
            for window in [10, 20, 50]:
                data[f'resistance_{window}'] = data['high'].rolling(window=window).max()
                data[f'support_{window}'] = data['low'].rolling(window=window).min()
                data[f'resistance_distance_{window}'] = (data[f'resistance_{window}'] - data['close']) / data['close']
                data[f'support_distance_{window}'] = (data['close'] - data[f'support_{window}']) / data['close']
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加技术指标失败: {e}")
            return data
    
    def _add_multi_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加多维波动率特征"""
        try:
            # 真实波动率(ATR)改进版
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            for period in [5, 14, 21]:
                data[f'atr_{period}'] = true_range.rolling(window=period).mean()
                data[f'atr_percent_{period}'] = data[f'atr_{period}'] / data['close']
            
            # 波动率突破
            data['volatility_breakout'] = (true_range > true_range.rolling(window=20).mean() + 
                                         2 * true_range.rolling(window=20).std()).astype(int)
            
            # 价格波动率（不同周期）
            for period in [5, 10, 20]:
                returns = data['close'].pct_change()
                data[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)
                data[f'volatility_rank_{period}'] = data[f'volatility_{period}'].rolling(window=252).rank(pct=True)
            
            # 波动率均值回归信号
            vol_mean = data['volatility_20'].rolling(window=60).mean()
            data['volatility_mean_reversion'] = (data['volatility_20'] - vol_mean) / vol_mean
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加波动率特征失败: {e}")
            return data
    
    def _add_smart_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加智能趋势特征"""
        try:
            # 多重移动平均线系统
            ma_periods = [5, 10, 20, 30, 60]
            for period in ma_periods:
                data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
                data[f'ma_slope_{period}'] = (data[f'ma_{period}'] - data[f'ma_{period}'].shift(5)) / 5
                data[f'price_vs_ma_{period}'] = (data['close'] - data[f'ma_{period}']) / data[f'ma_{period}']
            
            # 趋势强度指标
            for short, long in [(5, 20), (10, 30), (20, 60)]:
                data[f'trend_strength_{short}_{long}'] = (data[f'ma_{short}'] - data[f'ma_{long}']) / data[f'ma_{long}']
                data[f'ma_cross_{short}_{long}'] = ((data[f'ma_{short}'] > data[f'ma_{long}']) & 
                                                   (data[f'ma_{short}'].shift(1) <= data[f'ma_{long}'].shift(1))).astype(int)
            
            # ADX趋势指标
            # 先计算真实波动率
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            plus_dm = (data['high'].diff()).where((data['high'].diff() > data['low'].diff()) & 
                                                 (data['high'].diff() > 0), 0)
            minus_dm = (data['low'].diff() * -1).where((data['low'].diff() * -1 > data['high'].diff()) & 
                                                      (data['low'].diff() * -1 > 0), 0)
            
            atr_14 = true_range.rolling(window=14).mean()
            plus_di = (plus_dm.rolling(window=14).mean() / atr_14) * 100
            minus_di = (minus_dm.rolling(window=14).mean() / atr_14) * 100
            dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            data['adx'] = dx.rolling(window=14).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加趋势特征失败: {e}")
            return data
    
    def _add_precision_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加高精度动量特征"""
        try:
            # 改进动量指标
            for period in [5, 10, 20]:
                data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
                data[f'momentum_acceleration_{period}'] = data[f'momentum_{period}'].diff()
            
            # ROC (Rate of Change)
            for period in [5, 10, 20]:
                data[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period) * 100
            
            # 动量背离
            price_momentum = data['close'] / data['close'].shift(20) - 1
            rsi_momentum = data['rsi_14'] / data['rsi_14'].shift(20) - 1
            data['momentum_divergence'] = price_momentum - rsi_momentum
            
            # Stochastic改进版
            for period in [9, 14, 21]:
                low_min = data['low'].rolling(window=period).min()
                high_max = data['high'].rolling(window=period).max()
                k_percent = 100 * (data['close'] - low_min) / (high_max - low_min)
                data[f'stoch_k_{period}'] = k_percent.rolling(window=3).mean()
                data[f'stoch_d_{period}'] = data[f'stoch_k_{period}'].rolling(window=3).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加动量特征失败: {e}")
            return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加市场微观结构特征"""
        try:
            # 价格跳跃检测
            returns = data['close'].pct_change()
            return_std = returns.rolling(window=20).std()
            data['price_jump'] = (np.abs(returns) > 3 * return_std).astype(int)
            
            # 买卖压力指标
            data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
            
            # 价格分布特征
            for window in [10, 20]:
                data[f'price_skew_{window}'] = data['close'].rolling(window=window).skew()
                data[f'price_kurt_{window}'] = data['close'].rolling(window=window).kurt()
            
            # 市场效率指标
            for window in [10, 20]:
                random_walk = np.sqrt(window) * data['close'].rolling(window=window).std()
                actual_walk = np.abs(data['close'] - data['close'].shift(window))
                data[f'market_efficiency_{window}'] = actual_walk / random_walk
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加微观结构特征失败: {e}")
            return data
    
    def _add_signal_strength_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加信号强度特征"""
        try:
            # 多重确认信号
            signal_count = 0
            
            # 趋势确认
            if 'ma_cross_5_20' in data.columns:
                signal_count += data['ma_cross_5_20']
            
            # 动量确认
            if 'rsi_14' in data.columns:
                signal_count += (data['rsi_14'] > 50).astype(int)
            
            # 成交量确认
            if 'volume_breakout' in data.columns:
                signal_count += data['volume_breakout']
            
            data['signal_strength'] = signal_count
            
            # 信号一致性
            bullish_signals = 0
            bearish_signals = 0
            
            if 'momentum_5' in data.columns:
                bullish_signals += (data['momentum_5'] > 0).astype(int)
                bearish_signals += (data['momentum_5'] < 0).astype(int)
            
            if 'rsi_14' in data.columns:
                bullish_signals += (data['rsi_14'] > 50).astype(int)
                bearish_signals += (data['rsi_14'] < 50).astype(int)
            
            data['signal_consensus'] = bullish_signals - bearish_signals
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加信号强度特征失败: {e}")
            return data
    
    def _add_cross_validation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加交叉验证特征"""
        try:
            # 多时间框架一致性
            short_trend = (data['ma_5'] > data['ma_20']).astype(int)
            medium_trend = (data['ma_20'] > data['ma_60']).astype(int) if 'ma_60' in data.columns else short_trend
            data['trend_consistency'] = short_trend * medium_trend
            
            # 指标背离度
            if all(col in data.columns for col in ['rsi_14', 'momentum_10']):
                rsi_normalized = (data['rsi_14'] - 50) / 50
                momentum_normalized = data['momentum_10'] / data['momentum_10'].rolling(window=20).std()
                data['indicator_divergence'] = np.abs(rsi_normalized - momentum_normalized)
            
            return data
            
        except Exception as e:
            self.logger.error(f"添加交叉验证特征失败: {e}")
            return data
    
    def _clean_and_validate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和质量控制"""
        try:
            # 处理无穷值
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # 计算每列的缺失率
            missing_rates = data.isnull().sum() / len(data)
            
            # 只删除缺失率超过50%的列
            high_missing_cols = missing_rates[missing_rates > 0.5].index
            if len(high_missing_cols) > 0:
                data = data.drop(columns=high_missing_cols)
                self.logger.info(f"删除高缺失率列: {list(high_missing_cols)}")
            
            # 前向填充
            data = data.fillna(method='ffill')
            
            # 后向填充剩余的NaN
            data = data.fillna(method='bfill')
            
            # 如果还有NaN，用中位数填充
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].isnull().any():
                    median_val = data[col].median()
                    if not np.isnan(median_val):
                        data[col] = data[col].fillna(median_val)
                    else:
                        data[col] = data[col].fillna(0)
            
            # 温和的异常值处理（只处理极端异常值）
            for col in numeric_cols:
                if col not in ['open', 'high', 'low', 'close', 'volume']:  # 保留原始价格数据
                    q995 = data[col].quantile(0.995)
                    q005 = data[col].quantile(0.005)
                    if not np.isnan(q995) and not np.isnan(q005):
                        data[col] = data[col].clip(lower=q005, upper=q995)
            
            # 确保至少保留一些数据
            if len(data) == 0 and len(data.columns) > 5:
                self.logger.warning("数据清洗后无剩余数据，尝试放宽清洗条件")
                # 重新开始，使用更宽松的条件
                return data  # 返回原始数据
            
            self.logger.info(f"数据清洗完成，最终特征数: {len(data.columns)}, 数据行数: {len(data)}")
            return data
            
        except Exception as e:
            self.logger.error(f"数据清洗失败: {e}")
            return data
    
    def _add_market_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加市场情绪特征增强 - 阶段一优化"""
        try:
            # 1. 模拟北向资金流向指标（基于成交量和价格变化）
            data['northbound_flow_proxy'] = (
                data['volume'] * data['close'].pct_change() * 
                (data['close'] > data['close'].rolling(5).mean()).astype(int)
            )
            
            # 北向资金流向强度（5日和20日移动平均）
            data['northbound_flow_5d'] = data['northbound_flow_proxy'].rolling(5).mean()
            data['northbound_flow_20d'] = data['northbound_flow_proxy'].rolling(20).mean()
            data['northbound_flow_ratio'] = data['northbound_flow_5d'] / (data['northbound_flow_20d'] + 1e-8)
            
            # 2. 模拟融资融券指标（基于成交量放大和价格波动）
            volume_ma = data['volume'].rolling(20).mean()
            price_volatility = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
            
            # 融资买入强度（成交量放大 + 价格上涨）
            data['margin_buy_intensity'] = (
                (data['volume'] / volume_ma) * 
                (data['close'] > data['close'].shift(1)).astype(int) *
                (1 + price_volatility)
            )
            
            # 融券卖出强度（成交量放大 + 价格下跌）
            data['margin_sell_intensity'] = (
                (data['volume'] / volume_ma) * 
                (data['close'] < data['close'].shift(1)).astype(int) *
                (1 + price_volatility)
            )
            
            # 融资融券余额变化率
            data['margin_balance_change'] = (
                data['margin_buy_intensity'] - data['margin_sell_intensity']
            ).rolling(5).mean()
            
            # 3. 市场恐慌指数（基于波动率构建）
            # 短期波动率 vs 长期波动率
            short_vol = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
            long_vol = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
            data['fear_index'] = short_vol / (long_vol + 1e-8)
            
            # VIX式恐慌指数（基于价格跳跃）
            price_jumps = abs(data['close'].pct_change())
            data['vix_proxy'] = price_jumps.rolling(10).mean() * 100
            
            # 恐慌指数的移动平均和偏离度
            data['fear_index_ma'] = data['fear_index'].rolling(10).mean()
            data['fear_deviation'] = (data['fear_index'] - data['fear_index_ma']) / (data['fear_index_ma'] + 1e-8)
            
            # 4. 板块资金流向强度
            # 价量配合度（价格上涨时成交量放大程度）
            price_change = data['close'].pct_change()
            volume_change = data['volume'].pct_change()
            data['price_volume_sync'] = np.where(
                price_change > 0,
                volume_change * price_change,  # 上涨时的价量配合
                -volume_change * abs(price_change)  # 下跌时的价量背离
            )
            
            # 资金流向强度指标
            data['money_flow_strength'] = (
                data['price_volume_sync'].rolling(5).sum() / 
                abs(data['price_volume_sync']).rolling(5).sum()
            )
            
            # 大单流入强度（基于成交量分析）
            volume_percentile = data['volume'].rolling(20).quantile(0.8)
            data['big_order_flow'] = np.where(
                data['volume'] > volume_percentile,
                data['volume'] * (data['close'] > data['open']).astype(int),
                0
            )
            data['big_order_flow_ratio'] = (
                data['big_order_flow'].rolling(5).sum() / 
                (data['volume'].rolling(5).sum() + 1)
            )
            
            # 5. 市场情绪复合指标
            # 标准化各个情绪指标
            sentiment_features = ['northbound_flow_ratio', 'margin_balance_change', 
                                'fear_deviation', 'money_flow_strength']
            
            for feature in sentiment_features:
                if feature in data.columns:
                    # Z-score标准化
                    mean_val = data[feature].rolling(20).mean()
                    std_val = data[feature].rolling(20).std()
                    data[f'{feature}_zscore'] = (data[feature] - mean_val) / (std_val + 1e-8)
            
            # 综合市场情绪指数
            sentiment_cols = [f'{f}_zscore' for f in sentiment_features if f'{f}_zscore' in data.columns]
            if sentiment_cols:
                data['market_sentiment_composite'] = data[sentiment_cols].mean(axis=1)
                data['market_sentiment_extreme'] = (
                    abs(data['market_sentiment_composite']) > 1.5
                ).astype(int)
            
            self.logger.info("市场情绪特征增强完成")
            return data
            
        except Exception as e:
            self.logger.error(f"市场情绪特征创建失败: {e}")
            return data
    
    def _add_enhanced_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加时间序列增强特征 - 阶段一优化"""
        try:
            # 1. LSTM风格的滑动窗口特征
            windows = [3, 5, 8, 13, 21]  # 斐波那契数列窗口
            
            for window in windows:
                # 滑动窗口统计特征
                data[f'close_mean_{window}'] = data['close'].rolling(window).mean()
                data[f'close_std_{window}'] = data['close'].rolling(window).std()
                data[f'close_skew_{window}'] = data['close'].rolling(window).skew()
                data[f'close_kurt_{window}'] = data['close'].rolling(window).kurt()
                
                # 价格在窗口内的相对位置
                rolling_min = data['close'].rolling(window).min()
                rolling_max = data['close'].rolling(window).max()
                data[f'close_percentile_{window}'] = (
                    (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
                )
                
                # 趋势强度（线性回归斜率）
                def calculate_slope(series):
                    if len(series) < 2:
                        return 0
                    x = np.arange(len(series))
                    slope = np.polyfit(x, series, 1)[0] if not series.isna().all() else 0
                    return slope
                
                data[f'trend_slope_{window}'] = (
                    data['close'].rolling(window).apply(calculate_slope, raw=False)
                )
            
            # 2. 多时间框架技术指标（模拟不同级别）
            # 短期：3-5天
            data['short_momentum'] = data['close'] / data['close'].shift(3) - 1
            data['short_volatility'] = data['close'].rolling(3).std() / data['close'].rolling(3).mean()
            
            # 中期：8-13天  
            data['medium_momentum'] = data['close'] / data['close'].shift(8) - 1
            data['medium_volatility'] = data['close'].rolling(8).std() / data['close'].rolling(8).mean()
            
            # 长期：21天
            data['long_momentum'] = data['close'] / data['close'].shift(21) - 1
            data['long_volatility'] = data['close'].rolling(21).std() / data['close'].rolling(21).mean()
            
            # 动量强度比较
            data['momentum_acceleration'] = (
                data['short_momentum'] - data['medium_momentum']
            )
            data['momentum_consistency'] = np.where(
                (data['short_momentum'] > 0) & (data['medium_momentum'] > 0) & (data['long_momentum'] > 0), 1,
                np.where(
                    (data['short_momentum'] < 0) & (data['medium_momentum'] < 0) & (data['long_momentum'] < 0), -1,
                    0
                )
            )
            
            # 3. 趋势持续性特征
            # 连续上涨/下跌天数
            price_direction = np.where(data['close'] > data['close'].shift(1), 1, -1)
            
            # 计算连续趋势长度
            def count_consecutive(series):
                """计算连续相同值的长度"""
                consecutive = []
                current_count = 1
                current_value = series.iloc[0] if len(series) > 0 else 0
                
                for i in range(1, len(series)):
                    if series.iloc[i] == current_value:
                        current_count += 1
                    else:
                        current_count = 1
                        current_value = series.iloc[i]
                    consecutive.append(current_count)
                
                return consecutive[0] if consecutive else 0
            
            # 使用滑动窗口计算趋势持续性
            data['trend_persistence'] = pd.Series(price_direction).rolling(10).apply(
                lambda x: abs(x.sum()) / len(x), raw=False
            )
            
            # 趋势强度权重（考虑成交量）
            volume_normalized = data['volume'] / data['volume'].rolling(20).mean()
            data['weighted_trend_strength'] = (
                data['trend_persistence'] * np.log1p(volume_normalized)
            )
            
            # 4. 周期性特征（基于技术分析周期）
            # 短周期反转信号
            data['short_cycle_reversal'] = np.where(
                (data['close'] < data['close'].rolling(5).mean()) & 
                (data['close'].shift(1) > data['close'].shift(1).rolling(5).mean()), 1,
                np.where(
                    (data['close'] > data['close'].rolling(5).mean()) & 
                    (data['close'].shift(1) < data['close'].shift(1).rolling(5).mean()), -1,
                    0
                )
            )
            
            # 中周期趋势确认
            data['medium_cycle_confirmation'] = np.where(
                (data['close'] > data['close'].rolling(13).mean()) & 
                (data['close'].rolling(5).mean() > data['close'].rolling(13).mean()), 1,
                np.where(
                    (data['close'] < data['close'].rolling(13).mean()) & 
                    (data['close'].rolling(5).mean() < data['close'].rolling(13).mean()), -1,
                    0
                )
            )
            
            # 5. 序列记忆特征（模拟LSTM记忆机制）
            # 历史价格影响衰减
            decay_factors = [0.9, 0.8, 0.7, 0.6, 0.5]
            data['price_memory'] = 0
            
            for i, factor in enumerate(decay_factors, 1):
                if i < len(data):
                    data['price_memory'] += (
                        data['close'].pct_change(i) * factor
                    )
            
            # 成交量记忆
            data['volume_memory'] = 0
            for i, factor in enumerate(decay_factors, 1):
                if i < len(data):
                    volume_change = data['volume'].pct_change(i)
                    data['volume_memory'] += volume_change * factor
            
            # 波动率记忆
            volatility = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
            data['volatility_memory'] = 0
            for i, factor in enumerate(decay_factors, 1):
                if i < len(data):
                    vol_change = volatility.pct_change(i)
                    data['volatility_memory'] += vol_change * factor
            
            self.logger.info("时间序列增强特征完成")
            return data
            
        except Exception as e:
            self.logger.error(f"时间序列增强特征创建失败: {e}")
            return data