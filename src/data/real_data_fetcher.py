"""
真实股票数据获取模块
使用akshare库获取真实的A股数据
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd


class RealDataFetcher:
    """真实数据获取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_stock_data(self, stock_code: str, days: int = 180) -> List[Dict]:
        """
        获取真实股票数据
        
        Args:
            stock_code: 股票代码，如 'sz300061' 或 '000001'
            days: 获取天数，默认180天（约6个月）
        
        Returns:
            股票数据列表
        """
        try:
            # 标准化股票代码
            symbol = self._normalize_stock_code(stock_code)
            
            self.logger.info(f"获取真实股票数据: {stock_code} -> {symbol}")
            
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 使用akshare获取历史行情数据
            # 在异步环境中运行同步函数
            df = await asyncio.get_event_loop().run_in_executor(
                None, 
                ak.stock_zh_a_hist,
                symbol,
                "daily",
                start_date.strftime("%Y%m%d"),
                end_date.strftime("%Y%m%d"),
                "qfq"  # 前复权
            )
            
            if df is None or df.empty:
                self.logger.warning(f"未获取到股票数据: {stock_code}")
                return []
            
            # 转换为标准格式
            data_list = []
            for _, row in df.iterrows():
                data_list.append({
                    'date': row['日期'].strftime('%Y-%m-%d') if hasattr(row['日期'], 'strftime') else str(row['日期']),
                    'code': stock_code,
                    'open': float(row['开盘']),
                    'high': float(row['最高']),
                    'low': float(row['最低']),
                    'close': float(row['收盘']),
                    'volume': int(row['成交量'])
                })
            
            self.logger.info(f"成功获取 {len(data_list)} 条数据")
            return data_list
            
        except Exception as e:
            self.logger.error(f"获取真实股票数据失败 {stock_code}: {e}")
            return []
    
    async def get_current_price(self, stock_code: str) -> Optional[float]:
        """获取当前股价"""
        try:
            symbol = self._normalize_stock_code(stock_code)
            
            # 获取实时行情
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                ak.stock_zh_a_spot_em
            )
            
            if df is None or df.empty:
                return None
            
            # 查找对应股票
            stock_info = df[df['代码'] == symbol]
            if stock_info.empty:
                return None
            
            current_price = float(stock_info.iloc[0]['最新价'])
            self.logger.info(f"{stock_code} 当前价格: {current_price}")
            return current_price
            
        except Exception as e:
            self.logger.error(f"获取当前股价失败 {stock_code}: {e}")
            return None
    
    async def get_market_indices(self, days: int = 180) -> List[Dict]:
        """获取主要市场指数数据"""
        indices = {
            '000001': '上证指数',
            '399001': '深证成指',
            '399006': '创业板指'
        }
        
        all_data = []
        for index_code, index_name in indices.items():
            try:
                data = await self.get_stock_data(index_code, days)
                # 标记为指数数据
                for item in data:
                    item['name'] = index_name
                    item['type'] = 'index'
                all_data.extend(data)
            except Exception as e:
                self.logger.warning(f"获取指数数据失败 {index_code}: {e}")
                continue
        
        return all_data
    
    async def get_stock_info(self, stock_code: str) -> Dict:
        """获取股票基本信息"""
        try:
            symbol = self._normalize_stock_code(stock_code)
            
            # 获取股票基本信息
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                ak.stock_individual_info_em,
                symbol
            )
            
            if df is None or df.empty:
                return {}
            
            # 转换为字典格式
            info_dict = {}
            for _, row in df.iterrows():
                info_dict[row['item']] = row['value']
            
            return info_dict
            
        except Exception as e:
            self.logger.error(f"获取股票信息失败 {stock_code}: {e}")
            return {}
    
    def _normalize_stock_code(self, stock_code: str) -> str:
        """
        标准化股票代码格式
        将各种格式统一为akshare需要的格式
        """
        if not stock_code:
            return stock_code
        
        # 移除前缀和转换
        code = stock_code.upper()
        
        if code.startswith('SZ'):
            # SZ300061 -> 300061
            return code[2:]
        elif code.startswith('SH'):
            # SH000001 -> 000001
            return code[2:]
        elif len(code) == 6 and code.isdigit():
            # 已经是6位数字格式
            return code
        else:
            # 其他格式，尝试提取数字部分
            import re
            numbers = re.findall(r'\d+', code)
            if numbers:
                num_code = numbers[0]
                # 补齐到6位
                return num_code.zfill(6)
        
        return stock_code
    
    async def validate_stock_code(self, stock_code: str) -> bool:
        """验证股票代码是否有效"""
        try:
            symbol = self._normalize_stock_code(stock_code)
            
            # 尝试获取少量数据来验证
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                ak.stock_zh_a_hist,
                symbol,
                "daily",
                (datetime.now() - timedelta(days=5)).strftime("%Y%m%d"),
                datetime.now().strftime("%Y%m%d"),
                "qfq"
            )
            
            return df is not None and not df.empty
            
        except Exception:
            return False