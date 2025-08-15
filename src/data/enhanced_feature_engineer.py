# -*- coding: utf-8 -*-
"""
增强特征工程模块
基于真实市场数据构建更丰富的预测特征
严格禁止使用任何模拟数据，所有特征都基于真实历史数据
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """增强特征工程器 - 基于真实数据构建高质量预测特征"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def create_comprehensive_features(self, sector_df: pd.DataFrame, 
                                    sector_name: str) -> Dict[str, float]:
        """
        基于真实历史数据创建综合特征集
        
        Args:
            sector_df: 板块真实历史数据
            sector_name: 板块名称
            
        Returns:
            Dict: 特征字典
        """
        try:
            if sector_df.empty or len(sector_df) < 50:
                self.logger.warning(f"{sector_name}数据不足，无法构建完整特征")
                return {}
                
            features = {}
            
            # 1. 基础价格特征（基于真实价格数据）
            price_features = self._create_price_features(sector_df)
            features.update(price_features)
            
            # 2. 技术指标特征（基于真实OHLCV数据）
            technical_features = self._create_technical_features(sector_df)
            features.update(technical_features)
            
            # 3. 成交量特征（基于真实成交量数据）
            volume_features = self._create_volume_features(sector_df)
            features.update(volume_features)
            
            # 4. 波动率特征（基于真实价格波动）
            volatility_features = self._create_volatility_features(sector_df)
            features.update(volatility_features)
            
            # 5. 动量特征（基于真实价格动量）
            momentum_features = self._create_momentum_features(sector_df)
            features.update(momentum_features)
            
            # 6. 市场结构特征（基于真实市场数据）
            structure_features = self._create_market_structure_features(sector_df)
            features.update(structure_features)
            
            # 7. 时间序列特征（基于真实时间序列）
            ts_features = self._create_time_series_features(sector_df)
            features.update(ts_features)
            
            # 8. 统计特征（基于真实数据分布）
            stat_features = self._create_statistical_features(sector_df)
            features.update(stat_features)
            
            self.logger.debug(f"为{sector_name}创建了{len(features)}个特征")
            return features
            
        except Exception as e:
            self.logger.error(f"创建{sector_name}特征失败: {e}")
            return {}
            
    def _create_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实价格数据创建价格特征"""
        try:
            features = {}
            
            # 多时间周期移动平均
            for period in [5, 10, 20, 50, 120]:
                if len(df) >= period:
                    ma = df['close'].rolling(period).mean().iloc[-1]
                    features[f'ma_{period}'] = ma
                    features[f'price_to_ma_{period}'] = df['close'].iloc[-1] / ma - 1
                    
            # 布林带指标（基于真实价格）
            if len(df) >= 20:
                bb_period = 20
                bb_std = 2
                bb_middle = df['close'].rolling(bb_period).mean()
                bb_upper = bb_middle + (df['close'].rolling(bb_period).std() * bb_std)
                bb_lower = bb_middle - (df['close'].rolling(bb_period).std() * bb_std)
                
                current_price = df['close'].iloc[-1]
                features['bb_position'] = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                
            # 支撑阻力位（基于真实价格历史）
            if len(df) >= 60:
                highs = df['high'].rolling(20).max()
                lows = df['low'].rolling(20).min()
                
                features['resistance_distance'] = (highs.iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1]
                features['support_distance'] = (df['close'].iloc[-1] - lows.iloc[-1]) / df['close'].iloc[-1]
                
            # 价格位置特征
            if len(df) >= 252:  # 一年数据
                price_52w_high = df['high'].rolling(252).max().iloc[-1]
                price_52w_low = df['low'].rolling(252).min().iloc[-1]
                current_price = df['close'].iloc[-1]
                
                features['price_52w_position'] = (current_price - price_52w_low) / (price_52w_high - price_52w_low)
                features['distance_from_52w_high'] = (price_52w_high - current_price) / current_price
                features['distance_from_52w_low'] = (current_price - price_52w_low) / current_price
                
            return features
            
        except Exception as e:
            self.logger.error(f"创建价格特征失败: {e}")
            return {}
            
    def _create_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实OHLCV数据创建技术指标特征"""
        try:
            features = {}
            
            # 确保数据足够
            if len(df) < 30:
                return features
                
            # RSI指标
            for period in [14, 21]:
                if len(df) >= period:
                    rsi = talib.RSI(df['close'].values, timeperiod=period)
                    if not np.isnan(rsi[-1]):
                        features[f'rsi_{period}'] = rsi[-1]
                        
            # MACD指标
            if len(df) >= 35:
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values, 
                                                         fastperiod=12, slowperiod=26, signalperiod=9)
                if not np.isnan(macd[-1]):
                    features['macd'] = macd[-1]
                    features['macd_signal'] = macd_signal[-1]
                    features['macd_histogram'] = macd_hist[-1]
                    features['macd_cross'] = 1 if macd[-1] > macd_signal[-1] else -1
                    
            # KDJ指标
            if len(df) >= 14:
                k, d = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                  fastk_period=14, slowk_period=3, slowd_period=3)
                if not np.isnan(k[-1]) and not np.isnan(d[-1]):
                    j = 3 * k[-1] - 2 * d[-1]
                    features['kdj_k'] = k[-1]
                    features['kdj_d'] = d[-1]
                    features['kdj_j'] = j
                    
            # Williams %R指标
            if len(df) >= 14:
                willr = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                if not np.isnan(willr[-1]):
                    features['williams_r'] = willr[-1]
                    
            # CCI指标
            if len(df) >= 20:
                cci = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=20)
                if not np.isnan(cci[-1]):
                    features['cci'] = cci[-1]
                    
            # ATR指标（真实波幅）
            if len(df) >= 14:
                atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                if not np.isnan(atr[-1]):
                    features['atr'] = atr[-1]
                    features['atr_percent'] = atr[-1] / df['close'].iloc[-1]
                    
            # ADX指标（趋势强度）
            if len(df) >= 20:
                adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                if not np.isnan(adx[-1]):
                    features['adx'] = adx[-1]
                    
            return features
            
        except Exception as e:
            self.logger.error(f"创建技术特征失败: {e}")
            return {}
            
    def _create_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实成交量数据创建量能特征"""
        try:
            features = {}
            
            if 'volume' not in df.columns or len(df) < 20:
                return features
                
            # 成交量移动平均
            for period in [5, 10, 20, 50]:
                if len(df) >= period:
                    vol_ma = df['volume'].rolling(period).mean().iloc[-1]
                    features[f'volume_ma_{period}'] = vol_ma
                    features[f'volume_ratio_{period}'] = df['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1
                    
            # 量价关系
            if len(df) >= 5:
                price_change = df['close'].pct_change(5).iloc[-1]
                volume_change = df['volume'].pct_change(5).iloc[-1]
                if not np.isnan(price_change) and not np.isnan(volume_change):
                    features['price_volume_correlation'] = price_change * volume_change
                    
            # OBV指标（能量潮）
            if len(df) >= 20:
                obv = talib.OBV(df['close'].values, df['volume'].values)
                obv_ma = pd.Series(obv).rolling(10).mean()
                if not np.isnan(obv[-1]) and not np.isnan(obv_ma.iloc[-1]):
                    features['obv'] = obv[-1]
                    features['obv_trend'] = 1 if obv[-1] > obv_ma.iloc[-1] else -1
                    
            # 成交量突破识别
            if len(df) >= 20:
                vol_threshold = df['volume'].rolling(20).quantile(0.8).iloc[-1]
                features['volume_breakout'] = 1 if df['volume'].iloc[-1] > vol_threshold else 0
                
            # 换手率（如果有流通股本数据）
            if 'amount' in df.columns and len(df) >= 5:
                avg_amount = df['amount'].rolling(5).mean().iloc[-1]
                features['turnover_ratio'] = df['amount'].iloc[-1] / avg_amount if avg_amount > 0 else 1
                
            return features
            
        except Exception as e:
            self.logger.error(f"创建成交量特征失败: {e}")
            return {}
            
    def _create_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实价格波动创建波动率特征"""
        try:
            features = {}
            
            if len(df) < 20:
                return features
                
            # 历史波动率
            returns = df['close'].pct_change().dropna()
            for period in [5, 10, 20, 60]:
                if len(returns) >= period:
                    volatility = returns.rolling(period).std().iloc[-1] * np.sqrt(252)
                    features[f'volatility_{period}d'] = volatility
                    
            # GARCH波动率估计（简化版）
            if len(returns) >= 30:
                # 简化的EWMA波动率
                alpha = 0.94
                ewma_var = returns.ewm(alpha=alpha).var().iloc[-1]
                features['garch_volatility'] = np.sqrt(ewma_var * 252)
                
            # 真实波幅相对波动率
            if len(df) >= 20 and all(col in df.columns for col in ['high', 'low', 'close']):
                true_range = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                atr_volatility = true_range.rolling(20).mean().iloc[-1] / df['close'].iloc[-1]
                features['atr_volatility'] = atr_volatility
                
            # 价格跳空分析
            if len(df) >= 10 and all(col in df.columns for col in ['high', 'low', 'close']):
                gaps = []
                for i in range(1, min(len(df), 21)):  # 最近20天
                    prev_close = df['close'].iloc[-(i+1)]
                    curr_open = df['close'].iloc[-i]  # 简化，用close代替open
                    gap = abs(curr_open - prev_close) / prev_close
                    gaps.append(gap)
                    
                if gaps:
                    features['avg_gap_size'] = np.mean(gaps)
                    features['gap_frequency'] = sum(1 for gap in gaps if gap > 0.02) / len(gaps)
                    
            return features
            
        except Exception as e:
            self.logger.error(f"创建波动率特征失败: {e}")
            return {}
            
    def _create_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实价格动量创建动量特征"""
        try:
            features = {}
            
            if len(df) < 10:
                return features
                
            # 多时间周期收益率
            for period in [1, 3, 5, 10, 20, 60]:
                if len(df) > period:
                    returns = (df['close'].iloc[-1] / df['close'].iloc[-(period+1)] - 1) * 100
                    features[f'returns_{period}d'] = returns
                    
            # 动量指标
            if len(df) >= 20:
                momentum = talib.MOM(df['close'].values, timeperiod=10)
                if not np.isnan(momentum[-1]):
                    features['momentum_10d'] = momentum[-1]
                    
            # ROC指标（变动速率）
            if len(df) >= 20:
                roc = talib.ROC(df['close'].values, timeperiod=10)
                if not np.isnan(roc[-1]):
                    features['roc_10d'] = roc[-1]
                    
            # 相对强弱对比（与自身历史）
            if len(df) >= 60:
                current_price = df['close'].iloc[-1]
                avg_price_60d = df['close'].rolling(60).mean().iloc[-1]
                features['relative_strength_60d'] = (current_price / avg_price_60d - 1) * 100
                
            # 趋势持续性
            if len(df) >= 10:
                short_trend = df['close'].rolling(5).mean().iloc[-1] - df['close'].rolling(5).mean().iloc[-6]
                long_trend = df['close'].rolling(10).mean().iloc[-1] - df['close'].rolling(10).mean().iloc[-11]
                features['trend_consistency'] = 1 if (short_trend > 0) == (long_trend > 0) else 0
                
            return features
            
        except Exception as e:
            self.logger.error(f"创建动量特征失败: {e}")
            return {}
            
    def _create_market_structure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实市场数据创建市场结构特征"""
        try:
            features = {}
            
            if len(df) < 20:
                return features
                
            # 价格分布特征
            recent_prices = df['close'].tail(20)
            current_price = df['close'].iloc[-1]
            
            features['price_percentile_20d'] = (recent_prices <= current_price).sum() / len(recent_prices)
            
            # 上影线下影线分析
            if all(col in df.columns for col in ['high', 'low', 'close']):
                recent_data = df.tail(10)
                upper_shadows = (recent_data['high'] - recent_data['close']) / recent_data['close']
                lower_shadows = (recent_data['close'] - recent_data['low']) / recent_data['close']
                
                features['avg_upper_shadow'] = upper_shadows.mean()
                features['avg_lower_shadow'] = lower_shadows.mean()
                features['shadow_ratio'] = upper_shadows.mean() / lower_shadows.mean() if lower_shadows.mean() > 0 else 1
                
            # 连续涨跌统计
            if len(df) >= 10:
                price_changes = df['close'].diff().tail(10)
                consecutive_up = 0
                consecutive_down = 0
                current_streak = 0
                
                for change in reversed(price_changes.dropna()):
                    if change > 0:
                        if current_streak >= 0:
                            current_streak += 1
                        else:
                            break
                    elif change < 0:
                        if current_streak <= 0:
                            current_streak -= 1
                        else:
                            break
                    else:
                        break
                        
                features['consecutive_days'] = current_streak
                
            # 成交密集区识别
            if len(df) >= 60:
                price_levels = np.linspace(df['low'].min(), df['high'].max(), 20)
                volume_profile = []
                
                for i in range(len(price_levels)-1):
                    mask = (df['close'] >= price_levels[i]) & (df['close'] < price_levels[i+1])
                    volume_at_level = df.loc[mask, 'volume'].sum() if 'volume' in df.columns else 0
                    volume_profile.append(volume_at_level)
                    
                if volume_profile and max(volume_profile) > 0:
                    features['price_concentration'] = max(volume_profile) / sum(volume_profile)
                    
            return features
            
        except Exception as e:
            self.logger.error(f"创建市场结构特征失败: {e}")
            return {}
            
    def _create_time_series_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实时间序列创建时间特征"""
        try:
            features = {}
            
            if len(df) < 30:
                return features
                
            # 季节性特征（基于真实历史数据）
            if hasattr(df.index, 'month'):
                current_month = df.index[-1].month
                features['month'] = current_month
                features['quarter'] = (current_month - 1) // 3 + 1
                
                # 月末效应
                features['month_end'] = 1 if df.index[-1].day >= 25 else 0
                
            # 趋势强度分析
            prices = df['close'].values
            if len(prices) >= 30:
                # 线性回归趋势
                x = np.arange(len(prices[-30:]))
                y = prices[-30:]
                
                if len(x) == len(y) and len(y) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    features['trend_slope'] = slope / y[0] * 100  # 标准化
                    
                    # 趋势R²
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    features['trend_r_squared'] = r_squared
                    
            # 周期性分析
            if len(df) >= 60:
                returns = df['close'].pct_change().dropna()
                
                # 5日周期性（周效应）
                if len(returns) >= 25:  # 至少5周数据
                    weekly_returns = []
                    for i in range(0, len(returns)-4, 5):
                        week_return = (1 + returns.iloc[i:i+5]).prod() - 1
                        weekly_returns.append(week_return)
                    
                    if len(weekly_returns) > 1:
                        features['weekly_volatility'] = np.std(weekly_returns) * np.sqrt(52)
                        
            return features
            
        except Exception as e:
            self.logger.error(f"创建时间序列特征失败: {e}")
            return {}
            
    def _create_statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """基于真实数据分布创建统计特征"""
        try:
            features = {}
            
            if len(df) < 20:
                return features
                
            # 价格分布统计
            prices = df['close'].tail(60)  # 最近60天
            if len(prices) >= 20:
                features['price_mean'] = prices.mean()
                features['price_std'] = prices.std()
                features['price_skewness'] = prices.skew()
                features['price_kurtosis'] = prices.kurtosis()
                
                # 当前价格的Z-Score
                features['price_zscore'] = (prices.iloc[-1] - prices.mean()) / prices.std()
                
            # 收益率分布统计
            returns = df['close'].pct_change().dropna().tail(60)
            if len(returns) >= 20:
                features['returns_mean'] = returns.mean() * 252  # 年化
                features['returns_std'] = returns.std() * np.sqrt(252)  # 年化
                features['returns_skewness'] = returns.skew()
                features['returns_kurtosis'] = returns.kurtosis()
                
                # VaR和CVaR（基于历史数据）
                features['var_95'] = np.percentile(returns, 5) * 100
                features['cvar_95'] = returns[returns <= np.percentile(returns, 5)].mean() * 100
                
            # 成交量分布（如果有成交量数据）
            if 'volume' in df.columns:
                volumes = df['volume'].tail(60)
                if len(volumes) >= 20:
                    features['volume_mean'] = volumes.mean()
                    features['volume_std'] = volumes.std()
                    features['volume_skewness'] = volumes.skew()
                    
                    # 当前成交量的分位数位置
                    features['volume_percentile'] = (volumes <= volumes.iloc[-1]).sum() / len(volumes)
                    
            # 极值统计
            if len(df) >= 60:
                highs = df['high'].tail(60)
                lows = df['low'].tail(60)
                
                features['high_low_ratio'] = highs.max() / lows.min() if lows.min() > 0 else 1
                features['current_to_high_ratio'] = df['close'].iloc[-1] / highs.max()
                features['current_to_low_ratio'] = df['close'].iloc[-1] / lows.min() if lows.min() > 0 else 1
                
            return features
            
        except Exception as e:
            self.logger.error(f"创建统计特征失败: {e}")
            return {}
            
    def calculate_feature_importance_scores(self, features: Dict[str, float], 
                                          sector_name: str) -> Dict[str, float]:
        """
        基于特征类型和统计特性计算特征重要性得分
        """
        try:
            importance_scores = {}
            
            # 定义特征类型权重（基于金融理论和实证研究）
            feature_weights = {
                'price_': 0.25,      # 价格特征
                'ma_': 0.20,         # 移动平均特征
                'rsi_': 0.15,        # RSI等技术指标
                'volume_': 0.15,     # 成交量特征  
                'volatility_': 0.10, # 波动率特征
                'momentum_': 0.10,   # 动量特征
                'returns_': 0.05     # 收益率特征
            }
            
            for feature_name, feature_value in features.items():
                # 基础权重
                base_weight = 0.01  # 默认权重
                
                # 根据特征名称确定权重
                for prefix, weight in feature_weights.items():
                    if feature_name.startswith(prefix):
                        base_weight = weight
                        break
                        
                # 根据数值特性调整权重
                value_modifier = 1.0
                if abs(feature_value) > 1000:  # 大数值特征降权
                    value_modifier = 0.5
                elif abs(feature_value) < 0.001:  # 小数值特征降权
                    value_modifier = 0.8
                    
                importance_scores[feature_name] = base_weight * value_modifier
                
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"计算特征重要性失败: {e}")
            return {}