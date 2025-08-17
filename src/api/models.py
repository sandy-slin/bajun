#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API数据模型定义
使用Pydantic进行数据验证和序列化
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum

# 基础响应模型
class BaseResponse(BaseModel):
    success: bool = True
    message: str = "操作成功"
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Any] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict] = None

# 枚举类型
class TradingAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class MarketSentiment(str, Enum):
    OPTIMISTIC = "optimistic"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"
    PESSIMISTIC = "pessimistic"

# 板块分析模型
class SectorInfo(BaseModel):
    sector_name: str = Field(..., description="板块名称")
    sector_code: str = Field(..., description="板块代码")
    composite_score: float = Field(..., ge=0, le=100, description="综合评分")
    momentum_score: float = Field(..., ge=0, le=100, description="动量评分")
    relative_strength_score: float = Field(..., ge=0, le=100, description="相对强弱评分")
    investment_logic: str = Field(..., description="投资逻辑")
    latest_price: float = Field(..., gt=0, description="最新价格")
    price_change_5d: float = Field(..., description="5日涨跌幅(%)")
    volume_trend: str = Field(..., description="成交量趋势")
    risk_level: RiskLevel = Field(..., description="风险水平")
    confidence_level: float = Field(..., ge=0, le=1, description="预测置信度")

class SectorAnalysisRequest(BaseModel):
    lookback_months: int = Field(default=6, ge=1, le=12, description="回望月数")
    top_n: int = Field(default=5, ge=1, le=10, description="返回前N个板块")

class SectorAnalysisResponse(BaseResponse):
    data: Optional[Dict] = Field(None, description="分析结果")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "板块分析完成",
                "data": {
                    "analysis_params": {
                        "lookback_months": 6,
                        "top_n": 5
                    },
                    "top_sectors": [
                        {
                            "sector_name": "医药生物",
                            "composite_score": 78.5,
                            "investment_logic": "强烈推荐，技术面强劲"
                        }
                    ],
                    "market_overview": {
                        "average_score": 65.2,
                        "market_sentiment": "optimistic"
                    }
                }
            }
        }

# 股票分析模型
class StockInfo(BaseModel):
    stock_code: str = Field(..., description="股票代码")
    stock_name: str = Field(..., description="股票名称")
    current_price: float = Field(..., gt=0, description="当前价格")
    selection_score: float = Field(..., ge=0, le=100, description="选股评分")
    sector_name: str = Field(..., description="所属板块")
    expected_return: float = Field(..., description="预期收益率(%)")
    risk_assessment: RiskLevel = Field(..., description="风险评估")
    recommendation: TradingAction = Field(..., description="操作建议")
    confidence: float = Field(..., ge=0, le=1, description="推荐置信度")

class StockSelectionRequest(BaseModel):
    sectors: List[str] = Field(..., min_items=1, description="目标板块列表")
    stocks_per_sector: int = Field(default=5, ge=1, le=10, description="每个板块选股数量")
    min_score: float = Field(default=60, ge=0, le=100, description="最低评分要求")

class StockSelectionResponse(BaseResponse):
    data: Optional[Dict] = Field(None, description="选股结果")

class StockAnalysisRequest(BaseModel):
    stock_code: str = Field(..., pattern=r"^\d{6}$", description="6位股票代码")
    analysis_days: int = Field(default=30, ge=1, le=90, description="分析天数")

class StockAnalysisResponse(BaseResponse):
    data: Optional[Dict] = Field(None, description="股票分析结果")

# 投资组合模型
class HoldingInfo(BaseModel):
    stock_code: str = Field(..., description="股票代码")
    stock_name: str = Field(..., description="股票名称")
    shares: int = Field(..., gt=0, description="持股数量")
    cost_price: float = Field(..., gt=0, description="成本价格")
    current_price: float = Field(..., gt=0, description="当前价格")
    market_value: float = Field(..., gt=0, description="市值")
    profit_loss: float = Field(..., description="盈亏金额")
    profit_loss_pct: float = Field(..., description="盈亏比例(%)")
    weight: float = Field(..., ge=0, le=1, description="仓位权重")
    
    @validator('market_value', always=True)
    def calculate_market_value(cls, v, values):
        if 'shares' in values and 'current_price' in values:
            return values['shares'] * values['current_price']
        return v
    
    @validator('profit_loss', always=True)
    def calculate_profit_loss(cls, v, values):
        if 'shares' in values and 'current_price' in values and 'cost_price' in values:
            return values['shares'] * (values['current_price'] - values['cost_price'])
        return v

class PortfolioAnalysisRequest(BaseModel):
    holdings: List[HoldingInfo] = Field(..., min_items=1, description="持仓信息")
    analysis_type: str = Field(default="comprehensive", description="分析类型")

class PortfolioSummary(BaseModel):
    total_market_value: float = Field(..., description="总市值")
    total_cost: float = Field(..., description="总成本")
    total_profit_loss: float = Field(..., description="总盈亏")
    total_profit_loss_pct: float = Field(..., description="总盈亏比例(%)")
    sector_distribution: Dict[str, float] = Field(..., description="板块分布")
    risk_metrics: Dict[str, float] = Field(..., description="风险指标")

class RebalanceRecommendation(BaseModel):
    action: TradingAction = Field(..., description="操作类型")
    stock_code: str = Field(..., description="股票代码")
    current_weight: float = Field(..., description="当前权重")
    target_weight: float = Field(..., description="目标权重")
    reason: str = Field(..., description="调整原因")
    priority: int = Field(..., ge=1, le=5, description="优先级(1-5)")

class PortfolioAnalysisResponse(BaseResponse):
    data: Optional[Dict] = Field(None, description="组合分析结果")

class PortfolioOptimizeRequest(BaseModel):
    holdings: List[HoldingInfo] = Field(..., min_items=1, description="持仓信息")
    optimization_target: str = Field(default="risk_return", description="优化目标")
    constraints: Optional[Dict[str, Any]] = Field(None, description="约束条件")

# 交易助手模型
class TradingCheckRequest(BaseModel):
    action: TradingAction = Field(..., description="交易动作")
    stock_code: str = Field(..., pattern=r"^\d{6}$", description="股票代码")
    price: float = Field(..., gt=0, description="交易价格")
    quantity: int = Field(..., gt=0, description="交易数量")
    reason: Optional[str] = Field(None, description="交易原因")

class EmotionSignal(BaseModel):
    emotion_type: str = Field(..., description="情绪类型")
    intensity: float = Field(..., ge=0, le=1, description="情绪强度")
    detected_at: datetime = Field(default_factory=datetime.now, description="检测时间")

class TradingCheckResponse(BaseResponse):
    data: Optional[Dict] = Field(None, description="交易检查结果")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "交易检查完成",
                "data": {
                    "allowed": True,
                    "risk_level": "medium",
                    "emotion_signals": [],
                    "recommendations": ["建议设置止损位"],
                    "cooldown_remaining": 0
                }
            }
        }

class EmotionControlRequest(BaseModel):
    current_action: TradingAction = Field(..., description="当前操作")
    market_conditions: Dict[str, Any] = Field(..., description="市场状况")
    personal_context: Dict[str, Any] = Field(default_factory=dict, description="个人情况")

class EmotionControlResponse(BaseResponse):
    data: Optional[Dict] = Field(None, description="情绪控制建议")

# WebSocket消息模型
class WSMessage(BaseModel):
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")

class WSSubscription(BaseModel):
    type: str = Field(..., description="订阅类型")
    symbols: List[str] = Field(..., description="订阅标的")
    interval: str = Field(default="1s", description="推送间隔")

# 系统管理模型
class SystemStatus(BaseModel):
    service_status: Dict[str, str] = Field(..., description="服务状态")
    performance_metrics: Dict[str, float] = Field(..., description="性能指标")
    last_update: datetime = Field(default_factory=datetime.now, description="最后更新时间")

class OptimizationResult(BaseModel):
    optimization_id: str = Field(..., description="优化ID")
    parameters: Dict[str, Any] = Field(..., description="优化参数")
    performance_improvement: float = Field(..., description="性能改善")
    applied: bool = Field(default=False, description="是否已应用")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

# 分页模型
class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1, description="页码")
    size: int = Field(default=20, ge=1, le=100, description="每页数量")

class PaginatedResponse(BaseResponse):
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页")
    size: int = Field(..., description="每页数量")
    pages: int = Field(..., description="总页数")
    data: List[Any] = Field(..., description="数据列表")