# -*- coding: utf-8 -*-
"""
Database cache management module
"""

import sqlite3
import json
from datetime import datetime, timedelta
try:
    from typing import Optional, Dict, Any
except ImportError:
    Optional = None
    Dict = dict
    Any = object
import pandas as pd

from ..config import DATABASE_CONFIG, DB_SCHEMAS, CACHE_CONFIG


class DatabaseCache:
    """数据库缓存管理器"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_CONFIG["db_path"]
        self.init_database()
    
    def init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for table_name, schema in DB_SCHEMAS.items():
            cursor.execute(schema)
        
        conn.commit()
        conn.close()
    
    def get_stock_data(self, stock_code, start_date, end_date):
        """从缓存获取股票数据"""
        conn = sqlite3.connect(self.db_path)
        
        cached_data = pd.read_sql_query("""
            SELECT * FROM stock_data 
            WHERE stock_code = ? AND date >= ? AND date <= ?
            ORDER BY date
        """, conn, params=(stock_code, start_date, end_date))
        
        conn.close()
        return cached_data
    
    def save_stock_data(self, stock_data):
        """保存股票数据到缓存"""
        conn = sqlite3.connect(self.db_path)
        stock_data.to_sql('stock_data', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
    
    def is_stock_data_fresh(self, stock_code):
        """检查股票数据是否新鲜"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(updated_at) FROM stock_data WHERE stock_code = ?
        """, (stock_code,))
        
        result = cursor.fetchone()[0]
        conn.close()
        
        if not result:
            return False
            
        last_update = datetime.fromisoformat(result)
        expire_threshold = datetime.now() - timedelta(days=CACHE_CONFIG["stock_data_expire_days"])
        
        return last_update > expire_threshold
    
    def get_events_cache(self, stock_code, event_type):
        """获取事件分析缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        expire_time = datetime.now() - timedelta(days=CACHE_CONFIG["events_cache_expire_days"])
        
        cursor.execute("""
            SELECT content FROM events_cache 
            WHERE stock_code = ? AND event_type = ? 
            AND datetime(cached_at) > datetime(?)
        """, (stock_code, event_type, expire_time.isoformat()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                return None
        
        return None
    
    def save_events_cache(self, stock_code, event_type, content):
        """保存事件分析缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO events_cache 
            (stock_code, event_type, content, cached_at)
            VALUES (?, ?, ?, ?)
        """, (stock_code, event_type, json.dumps(content, ensure_ascii=False), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()