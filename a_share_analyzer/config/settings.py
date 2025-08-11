# -*- coding: utf-8 -*-
"""
Configuration file
"""

from datetime import timedelta

# 数据库配置
DATABASE_CONFIG = {
    "db_path": "a_share_cache.db"
}

# DeepSeek API配置
DEEPSEEK_CONFIG = {
    "api_key": "sk-f4affcb7b78243f5a138e7c9bdbbd6ee",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
}

# 缓存配置
CACHE_CONFIG = {
    "stock_data_expire_days": 1,
    "events_cache_expire_days": 1
}

# 数据获取配置
DATA_CONFIG = {
    "default_months": 6,
    "sector_analysis_days": 7,
    "recent_data_points": 10,
    "max_concepts_per_stock": 3
}

# 数据库表结构
DB_SCHEMAS = {
    "stock_data": """
        CREATE TABLE IF NOT EXISTS stock_data (
            stock_code TEXT,
            date TEXT,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            turnover REAL,
            updated_at TEXT,
            PRIMARY KEY (stock_code, date)
        )
    """,
    "events_cache": """
        CREATE TABLE IF NOT EXISTS events_cache (
            stock_code TEXT,
            event_type TEXT,
            content TEXT,
            cached_at TEXT,
            PRIMARY KEY (stock_code, event_type)
        )
    """
}