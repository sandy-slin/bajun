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