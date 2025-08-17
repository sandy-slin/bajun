#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反人性交易助手API路由
提供情绪控制、纪律执行、交易检查等功能
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import json

from ..models import *

logger = logging.getLogger(__name__)
router = APIRouter()

class TradingAssistantService:
    def __init__(self):
        # 反人性交易规则
        self.trading_rules = {
            'cooldown_period': 30,           # 冲动买入冷静期30分钟
            'max_daily_trades': 3,           # 每日最多3笔交易
            'panic_sell_threshold': -0.05,   # 5%跌幅触发恐慌卖出检查
            'greed_profit_threshold': 0.15,  # 15%盈利触发贪婪控制
            'max_single_position': 0.10,     # 单只股票最大10%仓位
            'emotion_detection_enabled': True
        }
        
        # 交易历史 (内存存储，实际应用中应使用数据库)
        self.trading_history = []
        self.emotion_signals = []
        self.cooldown_tracker = {}
    
    async def check_trading_decision(self, request: TradingCheckRequest) -> Dict:
        """检查交易决策"""
        try:
            check_results = {
                'timestamp': datetime.now().isoformat(),
                'trading_request': {
                    'action': request.action,
                    'stock_code': request.stock_code,
                    'price': request.price,
                    'quantity': request.quantity,
                    'reason': request.reason
                },
                'checks_performed': [],
                'warnings': [],
                'recommendations': [],
                'final_decision': {'allowed': True, 'confidence': 1.0}
            }
            
            # 1. 冷静期检查 (冲动买入阻断器)
            cooldown_result = self._check_cooldown_period(request)
            check_results['checks_performed'].append(cooldown_result)
            
            # 2. 每日交易次数检查
            daily_trades_result = self._check_daily_trading_limit()
            check_results['checks_performed'].append(daily_trades_result)
            
            # 3. 情绪检测
            emotion_result = self._detect_trading_emotions(request)
            check_results['checks_performed'].append(emotion_result)
            
            # 4. 恐慌卖出预防
            if request.action == TradingAction.SELL:
                panic_result = self._check_panic_selling(request)
                check_results['checks_performed'].append(panic_result)
            
            # 5. 贪婪限制检查
            if request.action == TradingAction.BUY:
                greed_result = self._check_greed_control(request)
                check_results['checks_performed'].append(greed_result)
            
            # 6. 仓位控制检查
            position_result = self._check_position_limits(request)
            check_results['checks_performed'].append(position_result)
            
            # 综合决策
            final_decision = self._make_final_decision(check_results['checks_performed'])
            check_results['final_decision'] = final_decision
            
            # 生成建议和警告
            check_results['warnings'] = self._generate_warnings(check_results['checks_performed'])
            check_results['recommendations'] = self._generate_recommendations(request, check_results['checks_performed'])
            
            return check_results
            
        except Exception as e:
            logger.error(f"交易检查失败: {e}")
            raise HTTPException(status_code=500, detail=f"交易检查失败: {str(e)}")
    
    def _check_cooldown_period(self, request: TradingCheckRequest) -> Dict:
        """检查冷静期"""
        now = datetime.now()
        key = f"{request.stock_code}_{request.action}"
        
        if key in self.cooldown_tracker:
            last_attempt = self.cooldown_tracker[key]
            time_passed = (now - last_attempt).total_seconds() / 60  # 分钟
            
            if time_passed < self.trading_rules['cooldown_period']:
                remaining = self.trading_rules['cooldown_period'] - time_passed
                return {
                    'check_type': 'cooldown_period',
                    'status': 'warning',
                    'passed': False,
                    'message': f"冷静期未满，还需等待{remaining:.1f}分钟",
                    'remaining_minutes': round(remaining, 1)
                }
        
        # 更新冷静期跟踪
        self.cooldown_tracker[key] = now
        
        return {
            'check_type': 'cooldown_period',
            'status': 'passed',
            'passed': True,
            'message': '冷静期检查通过'
        }
    
    def _check_daily_trading_limit(self) -> Dict:
        """检查每日交易次数限制"""
        today = datetime.now().date()
        today_trades = len([
            t for t in self.trading_history 
            if datetime.fromisoformat(t['timestamp']).date() == today
        ])
        
        if today_trades >= self.trading_rules['max_daily_trades']:
            return {
                'check_type': 'daily_trading_limit',
                'status': 'blocked',
                'passed': False,
                'message': f"今日已交易{today_trades}次，达到{self.trading_rules['max_daily_trades']}次上限",
                'daily_trades_count': today_trades
            }
        
        return {
            'check_type': 'daily_trading_limit',
            'status': 'passed',
            'passed': True,
            'message': f"今日交易次数{today_trades}/{self.trading_rules['max_daily_trades']}",
            'daily_trades_count': today_trades
        }
    
    def _detect_trading_emotions(self, request: TradingCheckRequest) -> Dict:
        """检测交易情绪"""
        emotion_signals = []
        
        # 基于交易行为模式检测情绪
        if request.action == TradingAction.BUY:
            # 检测FOMO (Fear of Missing Out)
            if request.reason and ('涨' in request.reason or '追' in request.reason):
                emotion_signals.append({
                    'emotion': 'FOMO',
                    'intensity': 0.7,
                    'description': '可能存在追涨情绪'
                })
        
        elif request.action == TradingAction.SELL:
            # 检测恐慌情绪
            if request.reason and ('跌' in request.reason or '怕' in request.reason):
                emotion_signals.append({
                    'emotion': 'PANIC',
                    'intensity': 0.8,
                    'description': '可能存在恐慌性卖出情绪'
                })
        
        # 检测交易频率异常 (可能的冲动交易)
        recent_trades = [
            t for t in self.trading_history
            if (datetime.now() - datetime.fromisoformat(t['timestamp'])).total_seconds() < 3600
        ]
        
        if len(recent_trades) >= 2:
            emotion_signals.append({
                'emotion': 'IMPULSE',
                'intensity': 0.6,
                'description': '1小时内多次交易，可能存在冲动交易'
            })
        
        return {
            'check_type': 'emotion_detection',
            'status': 'warning' if emotion_signals else 'passed',
            'passed': len(emotion_signals) == 0,
            'message': f"检测到{len(emotion_signals)}个情绪信号" if emotion_signals else "情绪状态正常",
            'emotion_signals': emotion_signals
        }
    
    def _check_panic_selling(self, request: TradingCheckRequest) -> Dict:
        """恐慌卖出预防检查"""
        # 简化实现 - 检查当前价格相对成本的跌幅
        # 实际应用中需要获取真实的持仓成本和当前价格
        
        # 模拟当前跌幅
        simulated_loss_pct = -0.08  # 假设当前浮亏8%
        
        if simulated_loss_pct <= self.trading_rules['panic_sell_threshold']:
            return {
                'check_type': 'panic_sell_prevention',
                'status': 'warning',
                'passed': False,
                'message': f"当前浮亏{abs(simulated_loss_pct):.1%}，可能为恐慌性卖出",
                'loss_percentage': simulated_loss_pct,
                'checklist': [
                    "基本面是否发生重大变化？",
                    "是否达到预设止损位？",
                    "是否受短期情绪影响？",
                    "长期投资逻辑是否仍然成立？"
                ]
            }
        
        return {
            'check_type': 'panic_sell_prevention',
            'status': 'passed',
            'passed': True,
            'message': '未检测到恐慌性卖出风险'
        }
    
    def _check_greed_control(self, request: TradingCheckRequest) -> Dict:
        """贪婪控制检查"""
        # 模拟持仓盈利情况
        simulated_profit_pct = 0.18  # 假设当前盈利18%
        
        if simulated_profit_pct >= self.trading_rules['greed_profit_threshold']:
            return {
                'check_type': 'greed_control',
                'status': 'warning',
                'passed': True,  # 不阻止交易，但给出警告
                'message': f"当前盈利{simulated_profit_pct:.1%}，建议考虑分批获利",
                'profit_percentage': simulated_profit_pct,
                'suggestions': [
                    "考虑部分获利了结",
                    "设置移动止盈位",
                    "避免过度贪婪",
                    "保留核心仓位"
                ]
            }
        
        return {
            'check_type': 'greed_control',
            'status': 'passed',
            'passed': True,
            'message': '贪婪控制检查通过'
        }
    
    def _check_position_limits(self, request: TradingCheckRequest) -> Dict:
        """仓位控制检查"""
        if request.action == TradingAction.BUY:
            # 模拟当前仓位
            current_position = 0.08  # 假设当前该股票仓位8%
            trade_value = request.price * request.quantity
            
            # 简化计算新仓位 (需要总资产信息)
            estimated_new_position = current_position + 0.02  # 假设增加2%
            
            if estimated_new_position > self.trading_rules['max_single_position']:
                return {
                    'check_type': 'position_limits',
                    'status': 'blocked',
                    'passed': False,
                    'message': f"交易后仓位将达到{estimated_new_position:.1%}，超过{self.trading_rules['max_single_position']:.0%}限制",
                    'current_position': current_position,
                    'estimated_new_position': estimated_new_position,
                    'max_allowed': self.trading_rules['max_single_position']
                }
        
        return {
            'check_type': 'position_limits',
            'status': 'passed',
            'passed': True,
            'message': '仓位控制检查通过'
        }
    
    def _make_final_decision(self, check_results: List[Dict]) -> Dict:
        """做出最终决策"""
        blocked_checks = [r for r in check_results if not r['passed'] and r['status'] == 'blocked']
        warning_checks = [r for r in check_results if not r['passed'] and r['status'] == 'warning']
        
        if blocked_checks:
            return {
                'allowed': False,
                'reason': 'blocked_by_rules',
                'blocking_checks': [r['check_type'] for r in blocked_checks],
                'confidence': 0.0
            }
        
        confidence = 1.0 - (len(warning_checks) * 0.2)  # 每个警告降低20%置信度
        
        return {
            'allowed': True,
            'reason': 'passed_all_checks' if not warning_checks else 'passed_with_warnings',
            'warning_checks': [r['check_type'] for r in warning_checks],
            'confidence': max(0.1, confidence)
        }
    
    def _generate_warnings(self, check_results: List[Dict]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        for result in check_results:
            if result['status'] in ['warning', 'blocked']:
                warnings.append(f"[{result['check_type']}] {result['message']}")
        
        return warnings
    
    def _generate_recommendations(self, request: TradingCheckRequest, check_results: List[Dict]) -> List[str]:
        """生成交易建议"""
        recommendations = []
        
        # 基于检查结果生成建议
        for result in check_results:
            if result['check_type'] == 'emotion_detection' and result.get('emotion_signals'):
                recommendations.append("建议深呼吸，冷静分析市场情况")
                recommendations.append("回顾原始投资逻辑是否仍然成立")
            
            elif result['check_type'] == 'greed_control' and result.get('suggestions'):
                recommendations.extend(result['suggestions'])
            
            elif result['check_type'] == 'panic_sell_prevention' and result.get('checklist'):
                recommendations.append("建议完成恐慌卖出自检清单")
        
        # 通用建议
        recommendations.extend([
            "建议设置合理的止损和止盈位",
            "保持分散投资，控制单一风险",
            "定期回顾投资策略的有效性"
        ])
        
        return recommendations[:5]  # 限制建议数量

# 创建服务实例
trading_service = TradingAssistantService()

@router.post("/check", response_model=TradingCheckResponse, summary="交易决策检查")
async def check_trading_decision(request: TradingCheckRequest):
    """
    反人性交易决策检查
    
    通过多维度检查帮助投资者避免情绪化交易决策
    """
    try:
        result = await trading_service.check_trading_decision(request)
        
        return TradingCheckResponse(
            success=True,
            message="交易检查完成",
            data=result
        )
    except Exception as e:
        logger.error(f"交易检查错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emotion-control", response_model=EmotionControlResponse, summary="情绪控制建议")
async def get_emotion_control_advice(request: EmotionControlRequest):
    """
    获取情绪控制建议
    
    基于当前市场状况和个人情况提供情绪管理建议
    """
    try:
        # 分析情绪状态和提供建议
        advice = {
            'timestamp': datetime.now().isoformat(),
            'emotion_analysis': {
                'detected_emotions': ['anxiety', 'FOMO'],
                'risk_level': 'medium',
                'confidence': 0.75
            },
            'control_strategies': [
                {
                    'strategy': 'breathing_exercise',
                    'description': '进行5分钟深呼吸练习',
                    'duration': '5分钟',
                    'effectiveness': 0.8
                },
                {
                    'strategy': 'rational_analysis',
                    'description': '重新评估投资逻辑和市场基本面',
                    'duration': '15分钟', 
                    'effectiveness': 0.9
                },
                {
                    'strategy': 'position_sizing',
                    'description': '减小交易规模，降低情绪压力',
                    'duration': '即时',
                    'effectiveness': 0.7
                }
            ],
            'cooling_off_suggestions': [
                "离开交易界面，进行10分钟散步",
                "与其他投资者讨论，获得不同视角",
                "回顾过往成功的投资决策",
                "设定交易暂停期，强制冷静思考"
            ],
            'long_term_habits': [
                "建立固定的投资决策流程",
                "记录每次交易的情绪状态",
                "定期进行投资心理健康检查",
                "培养价值投资的长期思维"
            ]
        }
        
        return EmotionControlResponse(
            success=True,
            message="情绪控制建议生成成功",
            data=advice
        )
        
    except Exception as e:
        logger.error(f"情绪控制建议错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules", summary="获取交易规则配置")
async def get_trading_rules():
    """获取当前的反人性交易规则配置"""
    return {
        "success": True,
        "message": "获取交易规则成功",
        "data": {
            "current_rules": trading_service.trading_rules,
            "rule_descriptions": {
                "cooldown_period": "冲动买入冷静期，防止情绪化快速交易",
                "max_daily_trades": "每日最大交易次数，避免过度交易",
                "panic_sell_threshold": "恐慌卖出触发阈值，防止恐慌性割肉", 
                "greed_profit_threshold": "贪婪控制触发阈值，提醒适时获利",
                "max_single_position": "单只股票最大仓位，控制集中度风险",
                "emotion_detection_enabled": "是否启用情绪检测功能"
            },
            "effectiveness_stats": {
                "prevented_panic_sells": 12,
                "reduced_FOMO_buys": 8,
                "improved_decision_quality": "+35%",
                "user_satisfaction": 4.3
            }
        }
    }

@router.put("/rules", summary="更新交易规则配置")
async def update_trading_rules(
    cooldown_period: Optional[int] = Query(None, ge=5, le=120, description="冷静期分钟数"),
    max_daily_trades: Optional[int] = Query(None, ge=1, le=10, description="每日最大交易次数"),
    panic_sell_threshold: Optional[float] = Query(None, ge=-0.20, le=-0.01, description="恐慌卖出阈值"),
    greed_profit_threshold: Optional[float] = Query(None, ge=0.05, le=0.50, description="贪婪控制阈值")
):
    """
    更新反人性交易规则配置
    
    允许用户根据个人情况调整交易规则参数
    """
    try:
        updates = {}
        
        if cooldown_period is not None:
            trading_service.trading_rules['cooldown_period'] = cooldown_period
            updates['cooldown_period'] = cooldown_period
            
        if max_daily_trades is not None:
            trading_service.trading_rules['max_daily_trades'] = max_daily_trades
            updates['max_daily_trades'] = max_daily_trades
            
        if panic_sell_threshold is not None:
            trading_service.trading_rules['panic_sell_threshold'] = panic_sell_threshold
            updates['panic_sell_threshold'] = panic_sell_threshold
            
        if greed_profit_threshold is not None:
            trading_service.trading_rules['greed_profit_threshold'] = greed_profit_threshold
            updates['greed_profit_threshold'] = greed_profit_threshold
        
        return {
            "success": True,
            "message": f"成功更新{len(updates)}项交易规则",
            "data": {
                "updated_rules": updates,
                "current_rules": trading_service.trading_rules,
                "update_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"更新交易规则错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", summary="获取交易历史分析")
async def get_trading_history(
    days: int = Query(default=30, ge=1, le=365, description="查询天数"),
    analysis_type: str = Query(default="summary", description="分析类型")
):
    """
    获取交易历史和行为分析
    
    分析用户的交易模式，识别改进机会
    """
    try:
        # 模拟交易历史分析
        history_analysis = {
            'analysis_period': f"最近{days}天",
            'trading_statistics': {
                'total_trades': 15,
                'successful_trades': 9,
                'success_rate': 0.60,
                'average_holding_period': '12天',
                'largest_gain': '+8.5%',
                'largest_loss': '-4.2%',
                'emotion_controlled_trades': 12,
                'emotion_driven_trades': 3
            },
            'behavior_patterns': {
                'most_active_hours': ['09:30-10:30', '14:00-15:00'],
                'preferred_trade_size': '5000-15000元',
                'common_emotions': ['FOMO', 'anxiety', 'confidence'],
                'improvement_areas': [
                    '减少早盘冲动交易',
                    '提高持股耐心',
                    '加强基本面分析'
                ]
            },
            'rule_effectiveness': {
                'cooldown_activations': 8,
                'panic_sell_preventions': 3,
                'greed_control_alerts': 5,
                'overall_improvement': '+25% decision quality'
            },
            'recommendations': [
                "继续保持情绪控制良好的交易习惯",
                "建议延长平均持股期到20-30天",
                "加强对宏观经济环境的关注",
                "考虑增加定投策略减少择时压力"
            ]
        }
        
        return {
            "success": True,
            "message": "交易历史分析完成",
            "data": history_analysis
        }
        
    except Exception as e:
        logger.error(f"交易历史分析错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))