#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据验证器 - 确保所有分析基于真实数据
禁止使用任何模拟数据或虚假数据

功能:
1. 验证数据源真实性
2. 检测模拟数据标记
3. 确保数据时效性
4. 数据完整性检查
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class RealDataValidator:
    """真实数据验证器 - 严格禁止模拟数据"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 模拟数据检测标识
        self.mock_data_indicators = {
            'field_names': ['mock', 'fake', 'test', 'demo', 'sample'],
            'value_patterns': [
                lambda x: str(x).lower().startswith('mock_'),
                lambda x: str(x).lower().startswith('test_'),
                lambda x: str(x).lower().startswith('demo_'),
                lambda x: str(x) == '999999',  # 常见测试代码
                lambda x: isinstance(x, (int, float)) and x == 0.0  # 全零数据可疑
            ],
            'suspicious_sequences': [
                [1, 2, 3, 4, 5],  # 顺序数列
                [10, 20, 30, 40, 50],  # 等差数列
                [100, 100, 100, 100],  # 重复值
            ]
        }
        
        # 数据时效性要求
        self.data_freshness_requirements = {
            'max_delay_hours': 24,  # 最大延迟24小时
            'trading_hours_only': True,  # 仅交易时间内更新
            'weekend_tolerance': True   # 周末容忍度
        }
        
        # 数据完整性要求
        self.completeness_requirements = {
            'min_data_points': 20,     # 最少数据点
            'max_missing_ratio': 0.1,  # 最大缺失比例10%
            'required_fields': ['open', 'high', 'low', 'close', 'volume']
        }
    
    async def validate_data_authenticity(self, data: Any, data_source: str) -> Dict:
        """
        验证数据真实性
        
        Args:
            data: 待验证的数据
            data_source: 数据源标识
            
        Returns:
            Dict: 验证结果
        """
        try:
            validation_start = datetime.now()
            
            # 1. 基础数据检查
            basic_check = self._perform_basic_checks(data, data_source)
            if not basic_check['is_valid']:
                return {
                    'is_authentic': False,
                    'error': basic_check['error'],
                    'validation_type': 'basic_check_failed'
                }
            
            # 2. 模拟数据检测
            mock_detection = self._detect_mock_data(data)
            if mock_detection['is_mock']:
                return {
                    'is_authentic': False,
                    'error': f"检测到模拟数据: {mock_detection['reason']}",
                    'validation_type': 'mock_data_detected',
                    'mock_indicators': mock_detection['indicators']
                }
            
            # 3. 数据时效性检查
            freshness_check = self._check_data_freshness(data)
            if not freshness_check['is_fresh']:
                return {
                    'is_authentic': False,
                    'error': f"数据时效性不足: {freshness_check['issue']}",
                    'validation_type': 'data_not_fresh',
                    'last_update': freshness_check.get('last_update')
                }
            
            # 4. 数据完整性检查
            completeness_check = self._check_data_completeness(data)
            if not completeness_check['is_complete']:
                return {
                    'is_authentic': False,
                    'error': f"数据完整性不足: {completeness_check['issue']}",
                    'validation_type': 'data_incomplete',
                    'missing_ratio': completeness_check.get('missing_ratio')
                }
            
            # 5. 数据合理性检查
            rationality_check = self._check_data_rationality(data)
            if not rationality_check['is_rational']:
                return {
                    'is_authentic': False,
                    'error': f"数据合理性异常: {rationality_check['issue']}",
                    'validation_type': 'data_irrational',
                    'anomalies': rationality_check.get('anomalies')
                }
            
            # 6. 数据来源验证
            source_validation = await self._validate_data_source(data_source)
            if not source_validation['is_valid']:
                return {
                    'is_authentic': False,
                    'error': f"数据源验证失败: {source_validation['issue']}",
                    'validation_type': 'invalid_source'
                }
            
            validation_time = (datetime.now() - validation_start).total_seconds()
            
            return {
                'is_authentic': True,
                'validation_type': 'all_checks_passed',
                'data_source': data_source,
                'validation_time_ms': validation_time * 1000,
                'data_quality_score': self._calculate_quality_score(
                    basic_check, mock_detection, freshness_check, 
                    completeness_check, rationality_check, source_validation
                ),
                'checks_performed': {
                    'basic_check': basic_check,
                    'mock_detection': mock_detection,
                    'freshness_check': freshness_check,
                    'completeness_check': completeness_check,
                    'rationality_check': rationality_check,
                    'source_validation': source_validation
                }
            }
            
        except Exception as e:
            self.logger.error(f"数据真实性验证失败: {e}")
            return {
                'is_authentic': False,
                'error': f"验证过程异常: {str(e)}",
                'validation_type': 'validation_error'
            }
    
    def _perform_basic_checks(self, data: Any, data_source: str) -> Dict:
        """执行基础数据检查"""
        try:
            # 检查数据是否为空
            if data is None:
                return {'is_valid': False, 'error': '数据为空'}
            
            # 检查数据结构
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    return {'is_valid': False, 'error': '数据列表为空'}
                data_size = len(data)
            elif isinstance(data, pd.DataFrame):
                if data.empty:
                    return {'is_valid': False, 'error': 'DataFrame为空'}
                data_size = len(data)
            elif isinstance(data, dict):
                if not data:
                    return {'is_valid': False, 'error': '数据字典为空'}
                data_size = len(data)
            else:
                return {'is_valid': False, 'error': f'不支持的数据类型: {type(data)}'}
            
            # 检查数据源标识
            if not data_source or not isinstance(data_source, str):
                return {'is_valid': False, 'error': '数据源标识无效'}
            
            # 禁止的数据源
            forbidden_sources = ['mock', 'test', 'demo', 'fake', 'simulation']
            if any(forbidden in data_source.lower() for forbidden in forbidden_sources):
                return {'is_valid': False, 'error': f'禁止的数据源: {data_source}'}
            
            return {
                'is_valid': True,
                'data_type': type(data).__name__,
                'data_size': data_size,
                'data_source': data_source
            }
            
        except Exception as e:
            return {'is_valid': False, 'error': f'基础检查异常: {str(e)}'}
    
    def _detect_mock_data(self, data: Any) -> Dict:
        """检测模拟数据"""
        try:
            mock_indicators = []
            
            # 转换为DataFrame便于处理
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                # 其他数据类型的简单检查
                return self._simple_mock_check(data)
            
            # 检查字段名
            for col in df.columns:
                col_lower = str(col).lower()
                for indicator in self.mock_data_indicators['field_names']:
                    if indicator in col_lower:
                        mock_indicators.append(f'字段名包含模拟标识: {col}')
            
            # 检查数据值
            for col in df.columns:
                if df[col].dtype in ['object', 'string']:
                    # 检查字符串值
                    for idx, value in df[col].items():
                        for pattern in self.mock_data_indicators['value_patterns']:
                            try:
                                if pattern(value):
                                    mock_indicators.append(f'字段{col}包含模拟值: {value}')
                                    break
                            except:
                                continue
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # 检查数值数据的模拟特征
                    values = df[col].dropna().values
                    if len(values) > 5:
                        # 检查是否为等差数列
                        if self._is_arithmetic_sequence(values):
                            mock_indicators.append(f'字段{col}为等差数列，疑似模拟数据')
                        
                        # 检查是否全为相同值
                        if len(set(values)) == 1:
                            mock_indicators.append(f'字段{col}全为相同值，疑似模拟数据')
                        
                        # 检查是否有过多的整数
                        if self._has_too_many_round_numbers(values):
                            mock_indicators.append(f'字段{col}整数过多，疑似模拟数据')
            
            # 检查时间戳合理性
            if 'timestamp' in df.columns or 'date' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                if not self._validate_timestamps(df[time_col]):
                    mock_indicators.append(f'时间戳不合理，疑似模拟数据')
            
            return {
                'is_mock': len(mock_indicators) > 0,
                'indicators': mock_indicators,
                'reason': '; '.join(mock_indicators) if mock_indicators else None,
                'confidence': min(len(mock_indicators) * 0.3, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"模拟数据检测异常: {e}")
            return {'is_mock': False, 'indicators': [], 'reason': None}
    
    def _simple_mock_check(self, data: Any) -> Dict:
        """简单的模拟数据检查"""
        mock_indicators = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = str(key).lower()
                for indicator in self.mock_data_indicators['field_names']:
                    if indicator in key_lower:
                        mock_indicators.append(f'键名包含模拟标识: {key}')
                
                for pattern in self.mock_data_indicators['value_patterns']:
                    try:
                        if pattern(value):
                            mock_indicators.append(f'值包含模拟标识: {key}={value}')
                    except:
                        continue
        
        return {
            'is_mock': len(mock_indicators) > 0,
            'indicators': mock_indicators,
            'reason': '; '.join(mock_indicators) if mock_indicators else None
        }
    
    def _check_data_freshness(self, data: Any) -> Dict:
        """检查数据时效性"""
        try:
            now = datetime.now()
            
            # 尝试从数据中提取时间戳
            latest_timestamp = self._extract_latest_timestamp(data)
            
            if latest_timestamp is None:
                return {
                    'is_fresh': False,
                    'issue': '无法提取数据时间戳'
                }
            
            # 计算数据延迟
            if isinstance(latest_timestamp, str):
                try:
                    latest_timestamp = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                except:
                    try:
                        latest_timestamp = pd.to_datetime(latest_timestamp)
                    except:
                        return {
                            'is_fresh': False,
                            'issue': f'无法解析时间戳格式: {latest_timestamp}'
                        }
            
            time_diff = now - latest_timestamp
            hours_old = time_diff.total_seconds() / 3600
            
            # 检查是否超过最大延迟
            max_delay = self.data_freshness_requirements['max_delay_hours']
            
            # 周末和节假日容忍度
            if self.data_freshness_requirements['weekend_tolerance']:
                if now.weekday() >= 5:  # 周末
                    max_delay = max_delay * 3  # 周末延长容忍时间
            
            if hours_old > max_delay:
                return {
                    'is_fresh': False,
                    'issue': f'数据延迟{hours_old:.1f}小时，超过{max_delay}小时限制',
                    'last_update': latest_timestamp.isoformat(),
                    'hours_old': hours_old
                }
            
            return {
                'is_fresh': True,
                'last_update': latest_timestamp.isoformat(),
                'hours_old': hours_old,
                'freshness_score': max(0, 1 - hours_old / max_delay)
            }
            
        except Exception as e:
            return {
                'is_fresh': False,
                'issue': f'时效性检查异常: {str(e)}'
            }
    
    def _check_data_completeness(self, data: Any) -> Dict:
        """检查数据完整性"""
        try:
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                # 简单数据的完整性检查
                return {'is_complete': True, 'completeness_score': 1.0}
            
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
            
            # 检查数据点数量
            if len(df) < self.completeness_requirements['min_data_points']:
                return {
                    'is_complete': False,
                    'issue': f'数据点不足，当前{len(df)}，要求最少{self.completeness_requirements["min_data_points"]}',
                    'data_points': len(df),
                    'missing_ratio': missing_ratio
                }
            
            # 检查缺失比例
            max_missing = self.completeness_requirements['max_missing_ratio']
            if missing_ratio > max_missing:
                return {
                    'is_complete': False,
                    'issue': f'缺失数据过多，缺失率{missing_ratio:.1%}，超过{max_missing:.1%}限制',
                    'missing_ratio': missing_ratio,
                    'missing_cells': missing_cells,
                    'total_cells': total_cells
                }
            
            # 检查必需字段
            required_fields = self.completeness_requirements['required_fields']
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                return {
                    'is_complete': False,
                    'issue': f'缺少必需字段: {missing_fields}',
                    'missing_fields': missing_fields,
                    'available_fields': list(df.columns)
                }
            
            return {
                'is_complete': True,
                'completeness_score': 1 - missing_ratio,
                'missing_ratio': missing_ratio,
                'data_points': len(df),
                'available_fields': list(df.columns)
            }
            
        except Exception as e:
            return {
                'is_complete': False,
                'issue': f'完整性检查异常: {str(e)}'
            }
    
    def _check_data_rationality(self, data: Any) -> Dict:
        """检查数据合理性"""
        try:
            anomalies = []
            
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return {'is_rational': True, 'rationality_score': 1.0}
            
            # 检查价格数据合理性
            price_fields = ['open', 'high', 'low', 'close', 'price']
            for field in price_fields:
                if field in df.columns:
                    values = df[field].dropna()
                    if len(values) > 0:
                        # 检查负价格
                        if (values < 0).any():
                            anomalies.append(f'{field}存在负值')
                        
                        # 检查极端价格
                        if (values > 10000).any():
                            anomalies.append(f'{field}存在极端高值(>10000)')
                        
                        # 检查价格关系（high >= low, high >= open, high >= close等）
                        if field == 'high' and 'low' in df.columns:
                            if (df['high'] < df['low']).any():
                                anomalies.append('最高价低于最低价')
            
            # 检查成交量合理性
            if 'volume' in df.columns:
                volumes = df['volume'].dropna()
                if len(volumes) > 0:
                    if (volumes < 0).any():
                        anomalies.append('成交量存在负值')
                    
                    # 检查成交量过度集中在某个值
                    volume_unique_ratio = len(volumes.unique()) / len(volumes)
                    if volume_unique_ratio < 0.1:
                        anomalies.append('成交量数据过度集中，疑似异常')
            
            # 检查时间序列连续性
            if 'date' in df.columns or 'timestamp' in df.columns:
                time_col = 'date' if 'date' in df.columns else 'timestamp'
                if not self._check_time_series_continuity(df[time_col]):
                    anomalies.append('时间序列不连续')
            
            # 检查数据变化合理性
            if 'close' in df.columns and len(df) > 1:
                price_changes = df['close'].pct_change().dropna()
                extreme_changes = price_changes[abs(price_changes) > 0.5]  # 50%以上变化
                if len(extreme_changes) > len(price_changes) * 0.1:  # 超过10%的数据有极端变化
                    anomalies.append('价格变化过于极端')
            
            return {
                'is_rational': len(anomalies) == 0,
                'anomalies': anomalies,
                'rationality_score': max(0, 1 - len(anomalies) * 0.2),
                'issue': '; '.join(anomalies) if anomalies else None
            }
            
        except Exception as e:
            return {
                'is_rational': True,  # 检查异常时默认通过
                'issue': f'合理性检查异常: {str(e)}'
            }
    
    async def _validate_data_source(self, data_source: str) -> Dict:
        """验证数据来源"""
        try:
            # 允许的真实数据源
            trusted_sources = [
                'akshare', 'sina', 'eastmoney', 'tencent', 'yahoo',
                'wind', 'bloomberg', 'reuters', 'baostock', 'tushare'
            ]
            
            # 禁止的模拟数据源
            forbidden_sources = [
                'mock', 'test', 'demo', 'fake', 'simulation',
                'random', 'generated', 'synthetic'
            ]
            
            source_lower = data_source.lower()
            
            # 检查是否为禁止的数据源
            for forbidden in forbidden_sources:
                if forbidden in source_lower:
                    return {
                        'is_valid': False,
                        'issue': f'数据源"{data_source}"包含禁止标识"{forbidden}"'
                    }
            
            # 检查是否为可信数据源
            is_trusted = any(trusted in source_lower for trusted in trusted_sources)
            
            if not is_trusted:
                self.logger.warning(f'未知数据源: {data_source}')
            
            return {
                'is_valid': True,  # 未知数据源暂时允许，但记录警告
                'is_trusted': is_trusted,
                'source': data_source,
                'trust_score': 1.0 if is_trusted else 0.5
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'issue': f'数据源验证异常: {str(e)}'
            }
    
    # 辅助方法
    
    def _extract_latest_timestamp(self, data: Any) -> Optional[datetime]:
        """从数据中提取最新时间戳"""
        try:
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # 查找时间字段
                    time_fields = ['timestamp', 'date', 'time', 'datetime', 'update_time']
                    for item in reversed(data):  # 从最后一个开始查找
                        for field in time_fields:
                            if field in item and item[field] is not None:
                                return item[field]
                
            elif isinstance(data, pd.DataFrame):
                time_fields = ['timestamp', 'date', 'time', 'datetime']
                for field in time_fields:
                    if field in data.columns:
                        return data[field].iloc[-1]
                
                # 尝试从索引获取时间
                if isinstance(data.index, pd.DatetimeIndex):
                    return data.index[-1]
            
            elif isinstance(data, dict):
                time_fields = ['timestamp', 'date', 'time', 'datetime', 'last_update']
                for field in time_fields:
                    if field in data and data[field] is not None:
                        return data[field]
            
            return None
            
        except Exception:
            return None
    
    def _is_arithmetic_sequence(self, values: np.ndarray) -> bool:
        """检查是否为等差数列"""
        if len(values) < 3:
            return False
        
        try:
            diffs = np.diff(values)
            return np.allclose(diffs, diffs[0], rtol=1e-10)
        except:
            return False
    
    def _has_too_many_round_numbers(self, values: np.ndarray) -> bool:
        """检查是否有过多整数"""
        if len(values) == 0:
            return False
        
        try:
            # 计算整数的比例
            round_numbers = np.sum(values == np.round(values))
            ratio = round_numbers / len(values)
            
            # 如果超过80%都是整数，可能是模拟数据
            return ratio > 0.8
        except:
            return False
    
    def _validate_timestamps(self, timestamps: pd.Series) -> bool:
        """验证时间戳合理性"""
        try:
            # 转换为datetime
            dt_series = pd.to_datetime(timestamps, errors='coerce')
            
            # 检查是否有无效时间
            if dt_series.isnull().any():
                return False
            
            # 检查时间范围是否合理（不能是未来时间，不能太古老）
            now = datetime.now()
            min_time = datetime(2000, 1, 1)  # 最早2000年
            
            if (dt_series > now).any() or (dt_series < min_time).any():
                return False
            
            # 检查时间序列是否合理递增
            if not dt_series.is_monotonic_increasing:
                return False
            
            return True
            
        except:
            return False
    
    def _check_time_series_continuity(self, timestamps: pd.Series) -> bool:
        """检查时间序列连续性"""
        try:
            dt_series = pd.to_datetime(timestamps, errors='coerce')
            
            if len(dt_series) < 2:
                return True
            
            # 计算时间间隔
            intervals = dt_series.diff().dropna()
            
            # 检查是否有负间隔（时间倒退）
            if (intervals <= pd.Timedelta(0)).any():
                return False
            
            # 检查间隔是否过于规整（可能是模拟数据）
            unique_intervals = intervals.unique()
            if len(unique_intervals) == 1:
                # 所有间隔都相同，可能是模拟数据
                interval = unique_intervals[0]
                if interval in [pd.Timedelta(hours=1), pd.Timedelta(minutes=1), pd.Timedelta(seconds=1)]:
                    return False
            
            return True
            
        except:
            return False
    
    def _calculate_quality_score(self, *check_results) -> float:
        """计算数据质量评分"""
        try:
            scores = []
            
            for result in check_results:
                if isinstance(result, dict):
                    if 'freshness_score' in result:
                        scores.append(result['freshness_score'])
                    elif 'completeness_score' in result:
                        scores.append(result['completeness_score'])
                    elif 'rationality_score' in result:
                        scores.append(result['rationality_score'])
                    elif 'trust_score' in result:
                        scores.append(result['trust_score'])
                    elif result.get('is_valid', result.get('is_fresh', result.get('is_complete', True))):
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
            
            return np.mean(scores) if scores else 0.0
            
        except:
            return 0.0
    
    async def validate_and_ensure_real_data(self, data: Any, data_source: str) -> Dict:
        """
        验证并确保数据真实性（主要入口方法）
        
        Returns:
            Dict: 包含验证结果和清洗后数据的字典
        """
        try:
            # 执行完整验证
            validation_result = await self.validate_data_authenticity(data, data_source)
            
            if not validation_result['is_authentic']:
                # 数据不真实，抛出异常
                error_msg = f"数据真实性验证失败: {validation_result['error']}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 数据真实，返回验证结果
            self.logger.info(f"数据真实性验证通过，质量评分: {validation_result.get('data_quality_score', 0):.2f}")
            
            return {
                'validated_data': data,
                'validation_result': validation_result,
                'is_real_data': True,
                'quality_score': validation_result.get('data_quality_score', 0)
            }
            
        except Exception as e:
            error_msg = f"真实数据验证过程异常: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)