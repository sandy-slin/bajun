#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反人性交易助手 - Phase 1 MVP基础版本
情绪控制和纪律执行模块

功能:
1. 冲动交易阻断器：30分钟冷静期强制等待
2. 恐慌卖出预防器：下跌时理性检查清单
3. 贪婪限制器：盈利15%+时分批获利提醒
4. 基础行为分析：识别常见情绪化交易模式
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class AntiHumanNatureEngine:
    """反人性交易助手 - 行为控制与纪律执行"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 情绪控制参数
        self.emotion_control_settings = {
            'impulse_cooldown_minutes': 30,      # 冲动交易冷静期
            'panic_threshold': -0.05,            # 恐慌阈值 -5%
            'greed_threshold': 0.15,             # 贪婪阈值 +15%
            'max_daily_trades': 3,               # 每日最大交易次数
            'max_position_size': 0.10            # 单只股票最大10%仓位
        }
        
        # 纪律规则
        self.discipline_rules = {
            'stop_loss_threshold': -0.08,        # 止损阈值 -8%
            'profit_taking_levels': [0.15, 0.25, 0.40],  # 分批获利点位
            'position_size_limits': {
                'high_risk': 0.05,               # 高风险股票5%
                'medium_risk': 0.08,             # 中风险股票8%
                'low_risk': 0.10                 # 低风险股票10%
            },
            'trading_frequency_limit': {
                'daily': 3,                      # 日交易限制
                'weekly': 10,                    # 周交易限制
                'monthly': 30                    # 月交易限制
            }
        }
        
        # 行为模式识别
        self.behavior_patterns = {
            'revenge_trading': {
                'description': '报复性交易',
                'triggers': ['连续亏损', '单日大幅亏损', '高频交易'],
                'intervention': '强制休息24小时'
            },
            'fomo_buying': {
                'description': 'FOMO追高买入',
                'triggers': ['股价急涨后买入', '大幅高于成本价买入'],
                'intervention': '冷静期评估'
            },
            'panic_selling': {
                'description': '恐慌性卖出',
                'triggers': ['大幅下跌时卖出', '止损点附近卖出'],
                'intervention': '理性检查清单'
            },
            'overtrading': {
                'description': '过度交易',
                'triggers': ['交易频率过高', '仓位频繁调整'],
                'intervention': '交易频率限制'
            }
        }
        
        # 交易历史记录
        self.trading_history = []
        self.emotion_state_history = []
        
    async def evaluate_trading_decision(
        self, 
        action: str,                    # "BUY", "SELL", "HOLD"
        stock_code: str,
        current_price: float,
        position_info: Optional[Dict] = None,
        market_context: Optional[Dict] = None
    ) -> Dict:
        """
        评估交易决策，提供反人性建议
        
        Args:
            action: 交易动作
            stock_code: 股票代码  
            current_price: 当前价格
            position_info: 持仓信息
            market_context: 市场环境信息
            
        Returns:
            Dict: 包含评估结果和建议的字典
        """
        try:
            analysis_start = datetime.now()
            
            # 1. 情绪状态检测
            emotion_analysis = await self._detect_emotional_state(
                action, stock_code, current_price, position_info, market_context
            )
            
            # 2. 行为模式识别
            behavior_analysis = await self._analyze_behavior_patterns(
                action, stock_code, current_price, position_info
            )
            
            # 3. 纪律规则检查
            discipline_check = await self._check_discipline_rules(
                action, stock_code, current_price, position_info
            )
            
            # 4. 生成干预建议
            intervention_advice = await self._generate_intervention_advice(
                emotion_analysis, behavior_analysis, discipline_check
            )
            
            # 5. 记录分析结果
            self._record_analysis(action, stock_code, current_price, 
                                emotion_analysis, behavior_analysis, discipline_check)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'action_requested': action,
                'stock_code': stock_code,
                'emotion_analysis': emotion_analysis,
                'behavior_analysis': behavior_analysis,
                'discipline_check': discipline_check,
                'intervention_advice': intervention_advice,
                'final_recommendation': self._get_final_recommendation(
                    action, intervention_advice
                ),
                'processing_time_ms': (datetime.now() - analysis_start).total_seconds() * 1000
            }
            
        except Exception as e:
            self.logger.error(f"交易决策评估失败: {e}")
            return {'error': str(e)}
    
    async def _detect_emotional_state(
        self,
        action: str,
        stock_code: str,
        current_price: float,
        position_info: Optional[Dict],
        market_context: Optional[Dict]
    ) -> Dict:
        """检测当前情绪状态"""
        
        emotion_indicators = {
            'fear_level': 0,      # 恐惧程度 0-10
            'greed_level': 0,     # 贪婪程度 0-10
            'anxiety_level': 0,   # 焦虑程度 0-10
            'confidence_level': 5 # 信心程度 0-10
        }
        
        detected_emotions = []
        
        # 恐慌情绪检测
        if position_info and action == "SELL":
            cost_price = position_info.get('cost_price', current_price)
            current_return = (current_price / cost_price - 1) if cost_price > 0 else 0
            
            if current_return <= self.emotion_control_settings['panic_threshold']:
                emotion_indicators['fear_level'] = min(10, abs(current_return) * 50)
                detected_emotions.append({
                    'emotion': 'panic',
                    'intensity': emotion_indicators['fear_level'],
                    'trigger': f'当前亏损{current_return*100:.1f}%，可能触发恐慌性卖出'
                })
        
        # 贪婪情绪检测
        if position_info and action == "BUY":
            cost_price = position_info.get('cost_price', current_price)
            current_return = (current_price / cost_price - 1) if cost_price > 0 else 0
            
            if current_return >= self.emotion_control_settings['greed_threshold']:
                emotion_indicators['greed_level'] = min(10, current_return * 30)
                detected_emotions.append({
                    'emotion': 'greed',
                    'intensity': emotion_indicators['greed_level'],
                    'trigger': f'当前盈利{current_return*100:.1f}%，可能触发贪婪加仓'
                })
        
        # FOMO情绪检测
        if action == "BUY" and market_context:
            recent_change = market_context.get('price_change_1d', 0)
            if recent_change > 0.05:  # 单日涨幅超过5%
                emotion_indicators['anxiety_level'] = min(10, recent_change * 100)
                detected_emotions.append({
                    'emotion': 'fomo',
                    'intensity': emotion_indicators['anxiety_level'],
                    'trigger': f'股价今日上涨{recent_change*100:.1f}%，可能追高买入'
                })
        
        # 报复性交易检测
        if len(self.trading_history) >= 3:
            recent_trades = self.trading_history[-3:]
            if all(t.get('result', 'unknown') == 'loss' for t in recent_trades):
                emotion_indicators['anxiety_level'] = 8
                detected_emotions.append({
                    'emotion': 'revenge',
                    'intensity': 8,
                    'trigger': '连续亏损交易，可能触发报复性交易'
                })
        
        return {
            'emotion_indicators': emotion_indicators,
            'detected_emotions': detected_emotions,
            'dominant_emotion': self._identify_dominant_emotion(detected_emotions),
            'emotional_state': self._assess_emotional_state(emotion_indicators),
            'risk_level': self._calculate_emotional_risk(emotion_indicators)
        }
    
    async def _analyze_behavior_patterns(
        self,
        action: str,
        stock_code: str,
        current_price: float,
        position_info: Optional[Dict]
    ) -> Dict:
        """分析行为模式"""
        
        detected_patterns = []
        pattern_risks = []
        
        # 过度交易模式检测
        today_trades = self._count_today_trades()
        if today_trades >= self.discipline_rules['trading_frequency_limit']['daily']:
            detected_patterns.append({
                'pattern': 'overtrading',
                'severity': 'high',
                'description': f'今日已交易{today_trades}次，超出每日{self.discipline_rules["trading_frequency_limit"]["daily"]}次限制'
            })
            pattern_risks.append('过度交易风险')
        
        # 追涨杀跌模式检测
        if action == "BUY":
            recent_performance = self._get_recent_stock_performance(stock_code)
            if recent_performance and recent_performance > 0.1:  # 近期涨幅超过10%
                detected_patterns.append({
                    'pattern': 'chasing_momentum',
                    'severity': 'medium',
                    'description': f'股票近期上涨{recent_performance*100:.1f}%后买入，可能为追涨行为'
                })
                pattern_risks.append('追高风险')
        
        # 止损失败模式检测
        if action == "SELL" and position_info:
            cost_price = position_info.get('cost_price', current_price)
            current_return = (current_price / cost_price - 1) if cost_price > 0 else 0
            stop_loss_point = cost_price * (1 + self.discipline_rules['stop_loss_threshold'])
            
            if current_price < stop_loss_point and current_return < self.discipline_rules['stop_loss_threshold']:
                detected_patterns.append({
                    'pattern': 'delayed_stop_loss',
                    'severity': 'high',
                    'description': f'亏损{abs(current_return)*100:.1f}%未及时止损，超过-8%止损线'
                })
                pattern_risks.append('延迟止损风险')
        
        # 仓位过重模式检测
        if action == "BUY":
            current_position_ratio = self._calculate_position_ratio(stock_code)
            max_allowed = self.emotion_control_settings['max_position_size']
            
            if current_position_ratio > max_allowed:
                detected_patterns.append({
                    'pattern': 'position_overweight',
                    'severity': 'high',
                    'description': f'股票仓位{current_position_ratio*100:.1f}%，超过{max_allowed*100:.1f}%限制'
                })
                pattern_risks.append('仓位过重风险')
        
        return {
            'detected_patterns': detected_patterns,
            'pattern_risks': pattern_risks,
            'behavior_score': self._calculate_behavior_score(detected_patterns),
            'intervention_needed': len([p for p in detected_patterns if p['severity'] == 'high']) > 0
        }
    
    async def _check_discipline_rules(
        self,
        action: str,
        stock_code: str,
        current_price: float,
        position_info: Optional[Dict]
    ) -> Dict:
        """检查纪律规则"""
        
        rule_violations = []
        warnings = []
        
        # 检查交易频率限制
        today_trades = self._count_today_trades()
        if today_trades >= self.discipline_rules['trading_frequency_limit']['daily']:
            rule_violations.append({
                'rule': 'daily_trading_limit',
                'violation': f'今日交易次数({today_trades})超过限制({self.discipline_rules["trading_frequency_limit"]["daily"]})',
                'severity': 'high'
            })
        
        # 检查仓位限制
        if action == "BUY":
            position_ratio = self._calculate_position_ratio(stock_code)
            max_position = self.emotion_control_settings['max_position_size']
            
            if position_ratio > max_position:
                rule_violations.append({
                    'rule': 'position_size_limit',
                    'violation': f'仓位比例({position_ratio*100:.1f}%)超过限制({max_position*100:.1f}%)',
                    'severity': 'high'
                })
        
        # 检查止损规则
        if position_info and action != "SELL":
            cost_price = position_info.get('cost_price', current_price)
            current_return = (current_price / cost_price - 1) if cost_price > 0 else 0
            
            if current_return <= self.discipline_rules['stop_loss_threshold']:
                warnings.append({
                    'rule': 'stop_loss_rule',
                    'warning': f'当前亏损{abs(current_return)*100:.1f}%，已达到止损线',
                    'recommended_action': 'SELL'
                })
        
        # 检查获利了结规则
        if position_info and action != "SELL":
            cost_price = position_info.get('cost_price', current_price)
            current_return = (current_price / cost_price - 1) if cost_price > 0 else 0
            
            for level in self.discipline_rules['profit_taking_levels']:
                if current_return >= level:
                    warnings.append({
                        'rule': 'profit_taking_rule',
                        'warning': f'当前盈利{current_return*100:.1f}%，建议分批获利了结',
                        'recommended_action': f'SELL_{int(level*100)}%'
                    })
                    break
        
        return {
            'rule_violations': rule_violations,
            'warnings': warnings,
            'discipline_score': self._calculate_discipline_score(rule_violations, warnings),
            'action_blocked': len([v for v in rule_violations if v['severity'] == 'high']) > 0
        }
    
    async def _generate_intervention_advice(
        self,
        emotion_analysis: Dict,
        behavior_analysis: Dict,
        discipline_check: Dict
    ) -> Dict:
        """生成干预建议"""
        
        interventions = []
        
        # 基于情绪状态的干预
        if emotion_analysis['risk_level'] == 'high':
            dominant_emotion = emotion_analysis['dominant_emotion']
            
            if dominant_emotion == 'panic':
                interventions.append({
                    'type': 'emotional_intervention',
                    'action': 'cooldown_period',
                    'duration_minutes': self.emotion_control_settings['impulse_cooldown_minutes'],
                    'message': '检测到恐慌情绪，建议30分钟冷静期后再做决定',
                    'checklist': [
                        '这次亏损在可承受范围内吗？',
                        '卖出后是否有更好的投资机会？',
                        '是否只是短期波动？',
                        '基本面是否发生重大变化？'
                    ]
                })
            
            elif dominant_emotion == 'greed':
                interventions.append({
                    'type': 'emotional_intervention',
                    'action': 'greed_control',
                    'message': '检测到贪婪情绪，建议分批获利了结',
                    'suggestion': '考虑卖出30%仓位锁定部分利润'
                })
            
            elif dominant_emotion == 'fomo':
                interventions.append({
                    'type': 'emotional_intervention',
                    'action': 'fomo_prevention',
                    'duration_minutes': self.emotion_control_settings['impulse_cooldown_minutes'],
                    'message': '检测到FOMO情绪，建议冷静分析后再决定',
                    'checklist': [
                        '这只股票的基本面支撑当前价格吗？',
                        '是否错过了最佳买入时机？',
                        '有其他更好的投资机会吗？',
                        '这次买入是否过于冲动？'
                    ]
                })
        
        # 基于行为模式的干预
        if behavior_analysis['intervention_needed']:
            for pattern in behavior_analysis['detected_patterns']:
                if pattern['severity'] == 'high':
                    if pattern['pattern'] == 'overtrading':
                        interventions.append({
                            'type': 'behavior_intervention',
                            'action': 'trading_pause',
                            'duration_hours': 24,
                            'message': '检测到过度交易，建议暂停交易24小时'
                        })
                    
                    elif pattern['pattern'] == 'position_overweight':
                        interventions.append({
                            'type': 'behavior_intervention',
                            'action': 'position_reduction',
                            'message': '仓位过重，建议减仓至安全水平'
                        })
        
        # 基于纪律违规的干预
        if discipline_check['action_blocked']:
            interventions.append({
                'type': 'discipline_intervention',
                'action': 'action_blocked',
                'message': '当前操作违反交易纪律，已被系统阻止',
                'violations': discipline_check['rule_violations']
            })
        
        return {
            'intervention_required': len(interventions) > 0,
            'interventions': interventions,
            'severity_level': self._assess_intervention_severity(interventions),
            'estimated_cooldown_minutes': self._calculate_cooldown_time(interventions)
        }
    
    def _get_final_recommendation(self, requested_action: str, intervention_advice: Dict) -> Dict:
        """获取最终建议"""
        
        if intervention_advice['intervention_required']:
            severity = intervention_advice['severity_level']
            
            if severity == 'critical':
                return {
                    'action': 'BLOCKED',
                    'reason': '违反重要交易纪律，操作被阻止',
                    'alternative': '建议重新评估投资策略'
                }
            elif severity == 'high':
                return {
                    'action': 'DELAY',
                    'reason': '检测到高风险情绪或行为模式',
                    'delay_minutes': intervention_advice['estimated_cooldown_minutes'],
                    'alternative': f'{intervention_advice["estimated_cooldown_minutes"]}分钟后重新评估'
                }
            else:
                return {
                    'action': 'PROCEED_WITH_CAUTION',
                    'reason': '发现潜在风险，建议谨慎操作',
                    'conditions': '请仔细考虑风险提示后再操作'
                }
        else:
            return {
                'action': 'PROCEED',
                'reason': '未发现明显的情绪或纪律风险',
                'confidence': 'high'
            }
    
    # 辅助方法
    
    def _identify_dominant_emotion(self, detected_emotions: List[Dict]) -> str:
        """识别主导情绪"""
        if not detected_emotions:
            return 'neutral'
        
        # 按强度排序，返回最强的情绪
        sorted_emotions = sorted(detected_emotions, key=lambda x: x['intensity'], reverse=True)
        return sorted_emotions[0]['emotion']
    
    def _assess_emotional_state(self, emotion_indicators: Dict) -> str:
        """评估情绪状态"""
        max_emotion = max(emotion_indicators.values())
        
        if max_emotion >= 8:
            return 'highly_emotional'
        elif max_emotion >= 6:
            return 'emotional'
        elif max_emotion >= 4:
            return 'slightly_emotional'
        else:
            return 'calm'
    
    def _calculate_emotional_risk(self, emotion_indicators: Dict) -> str:
        """计算情绪风险等级"""
        avg_emotion = np.mean([emotion_indicators['fear_level'], 
                              emotion_indicators['greed_level'],
                              emotion_indicators['anxiety_level']])
        
        if avg_emotion >= 7:
            return 'high'
        elif avg_emotion >= 4:
            return 'medium'
        else:
            return 'low'
    
    def _count_today_trades(self) -> int:
        """统计今日交易次数"""
        today = datetime.now().date()
        return len([t for t in self.trading_history 
                   if datetime.fromisoformat(t.get('timestamp', '')).date() == today])
    
    def _get_recent_stock_performance(self, stock_code: str) -> Optional[float]:
        """获取股票近期表现 (简化实现)"""
        # 这里应该实现真实的股票表现查询
        return None
    
    def _calculate_position_ratio(self, stock_code: str) -> float:
        """计算持仓比例 (简化实现)"""
        # 这里应该实现真实的持仓比例计算
        return 0.05  # 默认5%
    
    def _calculate_behavior_score(self, detected_patterns: List[Dict]) -> float:
        """计算行为评分"""
        if not detected_patterns:
            return 100.0
        
        penalty = 0
        for pattern in detected_patterns:
            if pattern['severity'] == 'high':
                penalty += 30
            elif pattern['severity'] == 'medium':
                penalty += 15
            else:
                penalty += 5
        
        return max(0, 100 - penalty)
    
    def _calculate_discipline_score(self, violations: List[Dict], warnings: List[Dict]) -> float:
        """计算纪律评分"""
        penalty = len(violations) * 25 + len(warnings) * 10
        return max(0, 100 - penalty)
    
    def _assess_intervention_severity(self, interventions: List[Dict]) -> str:
        """评估干预严重程度"""
        if any(i['type'] == 'discipline_intervention' for i in interventions):
            return 'critical'
        elif any(i.get('action') == 'trading_pause' for i in interventions):
            return 'high'
        elif len(interventions) >= 2:
            return 'medium'
        elif len(interventions) == 1:
            return 'low'
        else:
            return 'none'
    
    def _calculate_cooldown_time(self, interventions: List[Dict]) -> int:
        """计算冷静期时间"""
        max_cooldown = 0
        
        for intervention in interventions:
            if 'duration_minutes' in intervention:
                max_cooldown = max(max_cooldown, intervention['duration_minutes'])
            elif 'duration_hours' in intervention:
                max_cooldown = max(max_cooldown, intervention['duration_hours'] * 60)
        
        return max_cooldown if max_cooldown > 0 else self.emotion_control_settings['impulse_cooldown_minutes']
    
    def _record_analysis(
        self,
        action: str,
        stock_code: str,
        current_price: float,
        emotion_analysis: Dict,
        behavior_analysis: Dict,
        discipline_check: Dict
    ):
        """记录分析结果"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'stock_code': stock_code,
            'price': current_price,
            'emotion_state': emotion_analysis['emotional_state'],
            'behavior_score': behavior_analysis['behavior_score'],
            'discipline_score': discipline_check['discipline_score'],
            'intervention_applied': len(emotion_analysis.get('detected_emotions', [])) > 0
        }
        
        self.trading_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.trading_history) > 1000:
            self.trading_history = self.trading_history[-500:]
    
    async def get_behavior_summary(self, days: int = 30) -> Dict:
        """获取行为分析摘要"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_records = [
            r for r in self.trading_history 
            if datetime.fromisoformat(r['timestamp']) >= cutoff_date
        ]
        
        if not recent_records:
            return {'message': f'过去{days}天无交易记录'}
        
        # 统计分析
        total_trades = len(recent_records)
        avg_behavior_score = np.mean([r['behavior_score'] for r in recent_records])
        avg_discipline_score = np.mean([r['discipline_score'] for r in recent_records])
        intervention_rate = len([r for r in recent_records if r['intervention_applied']]) / total_trades
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'average_behavior_score': avg_behavior_score,
            'average_discipline_score': avg_discipline_score,
            'intervention_rate': intervention_rate * 100,
            'improvement_trend': self._calculate_improvement_trend(recent_records),
            'recommendations': self._generate_behavior_recommendations(
                avg_behavior_score, avg_discipline_score, intervention_rate
            )
        }
    
    def _calculate_improvement_trend(self, records: List[Dict]) -> str:
        """计算改进趋势"""
        if len(records) < 10:
            return 'insufficient_data'
        
        # 简单的线性趋势分析
        recent_scores = [r['behavior_score'] for r in records[-10:]]
        early_scores = [r['behavior_score'] for r in records[:10]]
        
        recent_avg = np.mean(recent_scores)
        early_avg = np.mean(early_scores)
        
        if recent_avg > early_avg + 5:
            return 'improving'
        elif recent_avg < early_avg - 5:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_behavior_recommendations(
        self, 
        behavior_score: float, 
        discipline_score: float, 
        intervention_rate: float
    ) -> List[str]:
        """生成行为改进建议"""
        recommendations = []
        
        if behavior_score < 70:
            recommendations.append("建议减少情绪化交易，制定明确的交易计划")
        
        if discipline_score < 70:
            recommendations.append("建议严格执行止损和获利了结规则")
        
        if intervention_rate > 0.3:
            recommendations.append("干预率较高，建议加强自我控制训练")
        
        if not recommendations:
            recommendations.append("交易行为表现良好，继续保持")
        
        return recommendations