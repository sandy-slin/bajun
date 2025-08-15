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
    import baostock as bs
    AKSHARE_AVAILABLE = True
    BAOSTOCK_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
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
            
        cache_key = f"all_sectors_data_{date_range[0]}_{date_range[1]}"
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            self.logger.info("从缓存获取所有板块数据")
            return cached_data
            
        sectors_data = {}
        
        if AKSHARE_AVAILABLE:
            try:
                # 获取申万行业数据
                for sector_name, sector_code in self.sw_sectors.items():
                    sector_data = await self._fetch_sector_data_akshare(sector_name, date_range)
                    if sector_data is not None:
                        sectors_data[sector_name] = sector_data
                        
                self.logger.info(f"成功获取{len(sectors_data)}个板块数据")
                
            except Exception as e:
                self.logger.error(f"AKShare获取板块数据失败: {e}")
                sectors_data = await self._get_mock_sectors_data(date_range)
        else:
            self.logger.warning("AKShare不可用，使用模拟数据")
            sectors_data = await self._get_mock_sectors_data(date_range)
            
        # 缓存数据
        await self.cache_manager.set(cache_key, sectors_data)
        return sectors_data
        
    async def _fetch_sector_data_akshare(self, sector_name: str, date_range: Tuple[str, str]) -> Optional[pd.DataFrame]:
        """使用AKShare获取单个板块数据"""
        try:
            # 获取申万行业指数历史数据
            df = ak.index_hist_sw(symbol=sector_name, start_date=date_range[0], end_date=date_range[1])
            
            if df is not None and not df.empty:
                # 标准化列名
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open', 
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                })
                
                # 计算基础技术指标
                df = self._calculate_basic_indicators(df)
                return df
                
        except Exception as e:
            self.logger.error(f"获取{sector_name}板块数据失败: {e}")
            
        return None
        
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        try:
            # 计算涨跌幅
            df['pct_change'] = df['close'].pct_change() * 100
            
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
            
            df = self._calculate_basic_indicators(df)
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
                    
            except Exception as e:
                self.logger.error(f"获取{sector_name}成分股失败: {e}")
                
        if not stocks:
            # 返回模拟数据
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
            
        await self.cache_manager.set(cache_key, stocks)
        return stocks
        
    def get_supported_sectors(self) -> List[str]:
        """获取支持的板块列表"""
        return list(self.sw_sectors.keys())