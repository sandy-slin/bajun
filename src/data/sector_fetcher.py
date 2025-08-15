"""
板块数据获取模块
负责获取A股板块相关数据，支持申万行业分类
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False

from cache.manager import CacheManager


class SectorFetcher:
    """板块数据获取器"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        self.sw_sectors = self._get_sw_sector_mapping()
        
    def _get_sw_sector_mapping(self) -> Dict[str, str]:
        """申万一级行业分类映射"""
        return {
            "银行": "801780",
            "非银金融": "801790", 
            "房地产": "801180",
            "食品饮料": "801120",
            "家用电器": "801110",
            "医药生物": "801150",
            "电子": "801080",
            "计算机": "801750",
            "通信": "801160",
            "机械设备": "801890",
            "电力设备": "801710",
            "汽车": "801880",
            "化工": "801130",
            "钢铁": "801040",
            "有色金属": "801050",
            "建筑材料": "801200",
            "建筑装饰": "801170",
            "轻工制造": "801140",
            "纺织服装": "801210",
            "商业贸易": "801200",
            "休闲服务": "801230",
            "交通运输": "801100",
            "公用事业": "801050",
            "农林牧渔": "801010",
            "采掘": "801030",
            "石油石化": "801020",
            "煤炭": "801032",
            "国防军工": "801740",
            "综合": "801890",
            "传媒": "801760",
            "环保": "801720"
        }
        
    async def get_all_sectors_data(self, date_range: Tuple[str, str] = None) -> Dict[str, pd.DataFrame]:
        """
        获取所有板块的数据
        
        Args:
            date_range: 时间范围 (start_date, end_date)，格式YYYYMMDD
            
        Returns:
            Dict[sector_name, DataFrame]: 各板块数据
        """
        if not date_range:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            date_range = (start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            
        self.logger.info(f"开始获取板块数据，时间范围: {date_range[0]} 到 {date_range[1]}")
        cache_key = f"all_sectors_data_{date_range[0]}_{date_range[1]}"
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            self.logger.info("从缓存获取所有板块数据")
            return cached_data
            
        sectors_data = {}
        
        if AKSHARE_AVAILABLE:
            try:
                self.logger.info("开始使用AKShare获取真实板块数据...")
                
                # 一次性获取所有申万一级行业数据
                all_sectors_df = await self._fetch_all_sectors_data_akshare(date_range)
                
                if all_sectors_df is not None and not all_sectors_df.empty:
                    self.logger.info(f"成功获取原始数据，开始处理各板块...")
                    # 按板块名称分组处理数据
                    for sector_name in self.sw_sectors.keys():
                        try:
                            self.logger.debug(f"处理板块: {sector_name}")
                            sector_data = self._process_sector_data(all_sectors_df, sector_name, date_range)
                            if sector_data is not None and not sector_data.empty:
                                sectors_data[sector_name] = sector_data
                                self.logger.debug(f"成功处理{sector_name}板块数据: {len(sector_data)}条记录")
                            else:
                                self.logger.warning(f"未获取到{sector_name}板块数据")
                        except Exception as e:
                            self.logger.error(f"处理{sector_name}板块数据失败: {e}")
                            continue
                        
                    if sectors_data:
                        self.logger.info(f"成功获取{len(sectors_data)}个板块的真实数据")
                        # 缓存真实数据
                        try:
                            await self.cache_manager.set(cache_key, sectors_data)
                        except Exception as e:
                            self.logger.warning(f"缓存板块数据失败: {e}")
                        return sectors_data
                    else:
                        self.logger.warning("未能获取任何板块的真实数据")
                        
                else:
                    self.logger.warning("API返回空数据")
                    
            except Exception as e:
                self.logger.error(f"AKShare获取板块数据失败: {e}")
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
        else:
            self.logger.warning("AKShare不可用")
        
        # 只有在完全无法获取真实数据时才使用模拟数据
        self.logger.warning("真实数据获取失败，使用模拟数据作为备选方案")
        sectors_data = await self._get_mock_sectors_data(date_range)
        
        # 缓存模拟数据
        try:
            await self.cache_manager.set(cache_key, sectors_data)
        except Exception as e:
            self.logger.warning(f"缓存模拟数据失败: {e}")
            
        return sectors_data
    
    async def _fetch_all_sectors_data_akshare(self, date_range: Tuple[str, str]) -> Optional[pd.DataFrame]:
        """一次性获取所有申万一级行业数据"""
        max_retries = 3
        retry_delay = 1  # 秒
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"尝试获取所有板块数据 (第{attempt + 1}次)")
                
                # 使用更稳定的index_analysis_daily_sw API获取申万一级行业数据
                df = ak.index_analysis_daily_sw(symbol='一级行业', start_date=date_range[0], end_date=date_range[1])
                
                if df is not None and not df.empty:
                    self.logger.info(f"成功获取原始数据: {len(df)}条记录，包含{df['指数名称'].nunique()}个板块")
                    return df
                else:
                    self.logger.warning(f"API返回空数据")
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"获取所有板块数据失败 (第{attempt + 1}次): {e}, {retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    self.logger.error(f"获取所有板块数据最终失败: {e}")
                    
        return None
    
    def _process_sector_data(self, all_sectors_df: pd.DataFrame, sector_name: str, date_range: Tuple[str, str]) -> Optional[pd.DataFrame]:
        """处理单个板块的数据"""
        try:
            # 过滤指定板块的数据
            sector_df = all_sectors_df[all_sectors_df['指数名称'] == sector_name].copy()
            
            if sector_df.empty:
                self.logger.warning(f"未找到{sector_name}板块数据")
                return None
            
            # 检查数据质量
            if len(sector_df) < 3:  # 至少需要3天数据
                self.logger.warning(f"{sector_name}板块数据不足: 仅{len(sector_df)}条记录")
                return None
            
            # 标准化列名和数据结构
            sector_df = sector_df.rename(columns={
                '发布日期': 'date',
                '收盘指数': 'close',
                '成交量': 'volume',
                '涨跌幅': 'pct_change',
                '均价': 'avg_price'
            })
            
            # 数据类型转换和清理
            sector_df['date'] = pd.to_datetime(sector_df['date'])
            sector_df['close'] = pd.to_numeric(sector_df['close'], errors='coerce')
            sector_df['volume'] = pd.to_numeric(sector_df['volume'], errors='coerce')
            sector_df['pct_change'] = pd.to_numeric(sector_df['pct_change'], errors='coerce')
            sector_df['avg_price'] = pd.to_numeric(sector_df['avg_price'], errors='coerce')
            
            # 移除无效数据
            sector_df = sector_df.dropna(subset=['close'])
            if sector_df.empty:
                self.logger.error(f"{sector_name}板块数据清理后为空")
                return None
            
            # 按日期排序
            sector_df = sector_df.sort_values('date')
            
            # 计算缺失的技术指标（基于收盘价）
            sector_df = self._calculate_basic_indicators_from_close(sector_df)
            
            return sector_df
            
        except Exception as e:
            self.logger.error(f"处理{sector_name}板块数据失败: {e}")
            return None
        
    def _calculate_basic_indicators_from_close(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于收盘价计算基础技术指标"""
        try:
            # 计算移动平均线
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean()
            
            # 计算相对强弱指数RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 添加缺失的列以保持兼容性
            df['open'] = df['close']  # 使用收盘价作为开盘价（近似）
            df['high'] = df['close'] * 1.02  # 模拟最高价
            df['low'] = df['close'] * 0.98   # 模拟最低价
            df['amount'] = df['volume'] * df['avg_price'] if 'avg_price' in df.columns else df['volume'] * df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return df
        
    async def _get_mock_sectors_data(self, date_range: Tuple[str, str]) -> Dict[str, pd.DataFrame]:
        """生成模拟板块数据"""
        self.logger.info("生成模拟板块数据用于测试")
        
        import numpy as np
        
        start_date = datetime.strptime(date_range[0], "%Y%m%d")
        end_date = datetime.strptime(date_range[1], "%Y%m%d")
        date_range_obj = pd.date_range(start=start_date, end=end_date, freq='D')
        
        sectors_data = {}
        
        for sector_name in list(self.sw_sectors.keys())[:10]:  # 限制为前10个板块
            # 生成随机走势数据
            np.random.seed(hash(sector_name) % 1000)  # 使用板块名生成固定种子
            
            n_days = len(date_range_obj)
            returns = np.random.normal(0.002, 0.03, n_days)  # 日收益率
            prices = [100]  # 起始价格100
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
                
            volumes = np.random.uniform(1000000, 10000000, n_days)
            
            df = pd.DataFrame({
                'date': date_range_obj,
                'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'high': [p * np.random.uniform(1.01, 1.05) for p in prices], 
                'low': [p * np.random.uniform(0.95, 0.99) for p in prices],
                'close': prices,
                'volume': volumes,
                'amount': [p * v for p, v in zip(prices, volumes)]
            })
            
            # 计算涨跌幅
            df['pct_change'] = df['close'].pct_change() * 100
            
            df = self._calculate_basic_indicators_from_close(df)
            sectors_data[sector_name] = df
            
        return sectors_data
        
    async def get_sector_stocks(self, sector_name: str) -> List[Dict[str, str]]:
        """获取板块内个股列表"""
        cache_key = f"sector_stocks_{sector_name}"
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data
            
        stocks = []
        
        if AKSHARE_AVAILABLE:
            try:
                self.logger.info(f"开始获取{sector_name}板块的真实成分股数据...")
                # 获取板块成分股
                df = ak.index_stock_cons_sw(symbol=sector_name)
                if df is not None and not df.empty:
                    stocks = [
                        {
                            'code': row['品种代码'],
                            'name': row['品种名称'],
                            'sector': sector_name
                        }
                        for _, row in df.iterrows()
                    ]
                    self.logger.info(f"成功获取{sector_name}板块成分股: {len(stocks)}只股票")
                    
                    # 缓存真实数据
                    try:
                        await self.cache_manager.set(cache_key, stocks)
                    except Exception as e:
                        self.logger.warning(f"缓存成分股数据失败: {e}")
                    return stocks
                else:
                    self.logger.warning(f"未获取到{sector_name}板块成分股数据")
                    
            except Exception as e:
                self.logger.error(f"获取{sector_name}成分股失败: {e}")
                
        # 只有在完全无法获取真实数据时才使用模拟数据
        self.logger.warning(f"真实成分股数据获取失败，为{sector_name}板块生成模拟数据")
        mock_stocks = [
            f"{sector_name}_股票{i:02d}" for i in range(1, 21)
        ]
        stocks = [
            {
                'code': f"00{i:04d}",
                'name': name,
                'sector': sector_name
            }
            for i, name in enumerate(mock_stocks, 1)
        ]
        
        # 缓存模拟数据
        try:
            await self.cache_manager.set(cache_key, stocks)
        except Exception as e:
            self.logger.warning(f"缓存模拟成分股数据失败: {e}")
            
        return stocks
        
    def get_supported_sectors(self) -> List[str]:
        """获取支持的板块列表"""
        return list(self.sw_sectors.keys())