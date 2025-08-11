# -*- coding: utf-8 -*-
"""
Configuration module
"""

from .settings import (
    DATABASE_CONFIG,
    DEEPSEEK_CONFIG, 
    CACHE_CONFIG,
    DATA_CONFIG,
    DB_SCHEMAS
)

__all__ = [
    "DATABASE_CONFIG",
    "DEEPSEEK_CONFIG",
    "CACHE_CONFIG", 
    "DATA_CONFIG",
    "DB_SCHEMAS"
]