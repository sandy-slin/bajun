#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反人性交易助手API路由
提供交易决策检查、情绪控制、纪律执行等功能 - 禁止模拟数据
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum

from ..models import *

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

class TradingAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradingCheckRequest(BaseModel):
    action: TradingAction
    stock_code: str
    stock_name: str
    price: float
    quantity: int
    reason: Optional[str] = None

class AntiHumanNatureTradingService:
    def __init__(self):
        # 反人性交易规则
        self.trading_rules = {
            'panic_sell_threshold': -0.10,  # 10%亏损启动恐慌检查
            'greed_profit_threshold': 0.15,  # 15%盈利启动贪婪检查
            'max_single_position': 0.15,    # 单股最大15%仓位
            'cooling_period_minutes': 30,   # 30分钟冷静期
            'max_daily_trades': 5,          # 每日最大交易次数
            'risk_control_enabled': True    # 风险控制开关
        }
    
    async def check_trading_decision(self, request: TradingCheckRequest) -> Dict:
        """检查交易决策的合理性"""
        if not self.trading_rules['risk_control_enabled']:
            return {
                'check_type': 'disabled',
                'status': 'passed',
                'passed': True,
                'message': '风险控制已禁用，交易检查跳过'
            }
        
        try:
            # 获取股票当前状态
            stock_status = await self._get_stock_current_status(request.stock_code)
            
            if request.action == TradingAction.SELL:
                return await self._check_panic_selling(request, stock_status)
            elif request.action == TradingAction.BUY:
                buy_checks = []
                buy_checks.append(await self._check_greed_control(request, stock_status))
                buy_checks.append(await self._check_position_limits(request, stock_status))
                
                # 如果任何检查失败，返回最严重的警告
                for check in buy_checks:
                    if check['status'] == 'blocked':
                        return check
                    elif check['status'] == 'warning':
                        return check
                
                return {
                    'check_type': 'comprehensive_buy_check',
                    'status': 'passed',
                    'passed': True,
                    'message': '买入检查通过',
                    'all_checks': buy_checks
                }
            else:
                return {
                    'check_type': 'hold_action',
                    'status': 'passed', 
                    'passed': True,
                    'message': '持有决策无需特殊检查'
                }
                
        except Exception as e:
            logger.error(f"交易决策检查失败: {e}")
            return {
                'check_type': 'error',
                'status': 'error',
                'passed': False,
                'message': f'交易检查系统错误: {e}',
                'suggestion': '建议暂停交易，检查系统状态'
            }
    
    async def _get_stock_current_status(self, stock_code: str) -> Dict:
        """获取股票当前状态"""
        if not AKSHARE_AVAILABLE:
            return {
                'error': 'AKShare不可用，无法获取股票状态',
                'price': None,
                'change_pct': None
            }
        
        try:
            # 获取股票实时数据
            stock_zh_a_spot = ak.stock_zh_a_spot_em()
            stock_data = stock_zh_a_spot[stock_zh_a_spot['代码'] == stock_code]
            
            if stock_data.empty:
                return {
                    'error': f'无法找到股票{stock_code}',
                    'price': None,
                    'change_pct': None
                }
            
            stock_row = stock_data.iloc[0]
            return {
                'price': float(stock_row['最新价']),
                'change_pct': float(stock_row['涨跌幅']),
                'volume': int(stock_row['成交量']),
                'turnover': float(stock_row['成交额']),
                'high': float(stock_row['最高']),
                'low': float(stock_row['最低']),
                'data_source': 'akshare_realtime'
            }
            
        except Exception as e:
            logger.error(f"获取股票{stock_code}状态失败: {e}")
            return {
                'error': f'获取股票状态失败: {e}',
                'price': None,
                'change_pct': None
            }
    
    async def _check_panic_selling(self, request: TradingCheckRequest, stock_status: Dict) -> Dict:
        """恐慌卖出检查"""
        # 无法获取真实数据时，基于用户提供的价格判断
        if 'error' in stock_status:
            return {
                'check_type': 'panic_sell_prevention',
                'status': 'warning',
                'passed': True,  # 无法确定时允许交易
                'message': f'无法获取{request.stock_code}实时数据，请谨慎卖出',
                'data_error': stock_status['error'],
                'suggestion': '建议确认当前市场价格后再做决策'
            }
        
        current_change = stock_status.get('change_pct', 0)
        
        # 基于当日跌幅判断是否可能是恐慌性卖出
        if current_change <= self.trading_rules['panic_sell_threshold'] * 100:  # 转换为百分比
            return {
                'check_type': 'panic_sell_prevention',
                'status': 'warning',
                'passed': False,
                'message': f"当前跌幅{current_change:.1f}%，可能为恐慌性卖出",
                'current_price': stock_status['price'],
                'checklist': [
                    "基本面是否发生重大变化？",
                    "是否达到预设止损位？", 
                    "是否受短期情绪影响？",
                    "长期投资逻辑是否仍然成立？"
                ],
                'cooling_period': f"{self.trading_rules['cooling_period_minutes']}分钟",
                'advice': "建议冷静分析后再做决定"
            }
        
        return {
            'check_type': 'panic_sell_prevention',
            'status': 'passed',
            'passed': True,
            'message': '卖出决策检查通过',
            'current_change': current_change
        }
    
    async def _check_greed_control(self, request: TradingCheckRequest, stock_status: Dict) -> Dict:
        """贪婪控制检查"""
        # 由于需要用户历史成本价格，这里基于当日涨幅做基础判断
        if 'error' in stock_status:
            return {
                'check_type': 'greed_control',
                'status': 'passed',
                'passed': True,
                'message': '无法进行贪婪控制检查',
                'data_error': stock_status['error']
            }
        
        current_change = stock_status.get('change_pct', 0)
        
        # 如果当日涨幅过大，提醒注意贪婪心理
        if current_change >= self.trading_rules['greed_profit_threshold'] * 100:  # 转换为百分比
            return {
                'check_type': 'greed_control',
                'status': 'warning',
                'passed': True,  # 不阻止交易，但给出警告
                'message': f"当日涨幅{current_change:.1f}%，建议避免追涨",
                'current_price': stock_status['price'],
                'suggestions': [
                    "避免情绪化追涨",
                    "考虑分批建仓",
                    "关注基本面变化",
                    "设置合理止损位"
                ]
            }
        
        return {
            'check_type': 'greed_control',
            'status': 'passed',
            'passed': True,
            'message': '贪婪控制检查通过',
            'current_change': current_change
        }
    
    async def _check_position_limits(self, request: TradingCheckRequest, stock_status: Dict) -> Dict:
        """仓位控制检查"""
        # 由于无法获取用户真实仓位，仅做基础检查
        return {
            'check_type': 'position_limits', 
            'status': 'passed',
            'passed': True,
            'message': '仓位控制检查通过',
            'note': '无法获取实际仓位信息，请自行确认不超过15%单股仓位限制',
            'max_position_limit': f"{self.trading_rules['max_single_position']:.1%}",
            'reminder': f"建议单只股票仓位不超过{self.trading_rules['max_single_position']:.1%}"
        }

# 创建服务实例
trading_service = AntiHumanNatureTradingService()

@router.post("/check", summary="交易决策检查")
async def check_trading_decision(request: TradingCheckRequest):
    """
    反人性交易决策检查
    
    - **action**: 交易动作 (BUY/SELL/HOLD)
    - **stock_code**: 股票代码
    - **stock_name**: 股票名称
    - **price**: 交易价格
    - **quantity**: 交易数量
    - **reason**: 交易理由（可选）
    
    返回交易决策检查结果和建议
    """
    try:
        logger.info(f"交易决策检查: {request.action} {request.stock_code} {request.quantity}股")
        result = await trading_service.check_trading_decision(request)
        logger.info(f"交易决策检查完成: {result['status']}")
        return result
    except Exception as e:
        logger.error(f"交易决策检查API错误: {e}")
        raise HTTPException(status_code=500, detail=f"交易决策检查失败: {str(e)}")

@router.get("/rules", summary="获取交易规则")
async def get_trading_rules():
    """
    获取当前反人性交易规则配置
    """
    return {
        "trading_rules": trading_service.trading_rules,
        "description": {
            "panic_sell_threshold": "恐慌卖出阈值（亏损百分比）",
            "greed_profit_threshold": "贪婪控制阈值（盈利百分比）", 
            "max_single_position": "单股最大仓位限制",
            "cooling_period_minutes": "强制冷静期（分钟）",
            "max_daily_trades": "每日最大交易次数",
            "risk_control_enabled": "风险控制开关"
        }
    }

@router.get("/analysis/{days}", summary="获取交易行为分析")
async def get_trading_behavior_analysis(days: int):
    """
    获取交易行为分析报告
    
    - **days**: 分析天数 (7-365)
    
    注意：由于缺乏真实交易历史数据，此功能需要用户提供交易记录
    """
    return {
        "analysis_period": f"最近{days}天",
        "message": "交易行为分析需要真实交易历史数据",
        "requirement": "请提供包含以下字段的交易记录：日期、股票代码、动作、价格、数量、原因",
        "analysis_capabilities": [
            "交易频率分析",
            "情绪化交易识别",
            "盈亏比统计",
            "止损执行率",
            "追涨杀跌行为检测"
        ],
        "data_source": "user_provided_trading_history_required"
    }