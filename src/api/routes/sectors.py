#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块分析API路由
提供板块分析、排名、预测等功能 - 禁止模拟数据
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Optional
import logging
import asyncio
from datetime import datetime, timedelta

from ..models import *

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

class SectorAnalysisService:
    def __init__(self):
        self.sector_mapping = {
            '银行': '801780',
            '食品饮料': '801120',
            '医药生物': '801150', 
            '电子': '801080',
            '非银金融': '801790',
            '房地产': '801180',
            '汽车': '801880',
            '钢铁': '801040',
            '有色金属': '801050',
            '化工': '801130',
            '煤炭': '801032',
            '石油石化': '801020',
            '建筑材料': '801410',
            '机械设备': '801890',
            '电力设备': '801730',
            '国防军工': '801740',
            '计算机': '801750',
            '传媒': '801760',
            '通信': '801770',
            '建筑装饰': '801210'
        }
    
    async def analyze_all_sectors(self, lookback_months: int = 6, top_n: int = 5) -> Dict:
        """分析所有板块并突出TOP N - 基于真实数据"""
        if not AKSHARE_AVAILABLE:
            raise RuntimeError("AKShare不可用，无法进行板块分析")
        
        try:
            # 获取申万一级行业指数实时数据
            sw_index = ak.sw_index_spot()
            if sw_index.empty:
                raise RuntimeError("无法获取申万行业指数数据")
            
            # 分析所有板块
            all_sector_scores = []
            
            for _, row in sw_index.iterrows():
                sector_name = row['指数名称'].replace('申万', '').replace('指数', '')
                sector_code = row['指数代码']
                
                # 基于真实数据计算评分
                change_pct = float(row['涨跌幅'])
                volume_ratio = float(row['成交量']) / float(row['成交额']) if row['成交额'] > 0 else 0
                
                # 计算综合评分 (基于真实数据)
                momentum_score = 50 + change_pct * 10  # 以涨跌幅为基础
                volume_score = min(volume_ratio * 100, 100)  # 成交活跃度
                price_score = 50 + (float(row['最新价']) - float(row['开盘'])) / float(row['开盘']) * 100 if row['开盘'] > 0 else 50
                
                composite_score = momentum_score * 0.6 + volume_score * 0.2 + price_score * 0.2
                
                sector_info = {
                    'name': sector_name,
                    'code': sector_code,
                    'score': round(composite_score, 2),
                    'change_pct': round(change_pct, 2),
                    'latest_price': float(row['最新价']),
                    'volume': int(row['成交量']),
                    'turnover': float(row['成交额']),
                    'open_price': float(row['开盘']),
                    'high_price': float(row['最高']),
                    'low_price': float(row['最低']),
                    'momentum_score': round(momentum_score, 2),
                    'volume_score': round(volume_score, 2),
                    'risk_level': self._assess_risk_level(change_pct, volume_ratio),
                    'recommendation': self._get_recommendation(composite_score, change_pct)
                }
                
                all_sector_scores.append(sector_info)
            
            # 按综合评分排序
            all_sector_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # 获取TOP N
            top_sectors = all_sector_scores[:top_n]
            
            # 计算市场整体情况
            total_up = sum(1 for s in all_sector_scores if s['change_pct'] > 0)
            total_down = len(all_sector_scores) - total_up
            avg_change = sum(s['change_pct'] for s in all_sector_scores) / len(all_sector_scores)
            
            market_sentiment = "optimistic" if avg_change > 1 else "neutral" if avg_change > -1 else "pessimistic"
            
            return {
                'analysis_time': datetime.now().isoformat(),
                'lookback_months': lookback_months,
                'total_sectors_analyzed': len(all_sector_scores),
                'top_sectors': top_sectors,
                'all_sectors': all_sector_scores,
                'market_overview': {
                    'sectors_up': total_up,
                    'sectors_down': total_down,
                    'average_change': round(avg_change, 2),
                    'market_sentiment': market_sentiment,
                    'data_source': 'akshare_sw_index_realtime'
                },
                'analysis_method': 'real_data_composite_scoring',
                'update_frequency': 'realtime'
            }
            
        except Exception as e:
            logger.error(f"板块分析失败: {e}")
            raise RuntimeError(f"板块分析失败: {e}")
    
    def _assess_risk_level(self, change_pct: float, volume_ratio: float) -> str:
        """评估风险水平"""
        if abs(change_pct) > 5:
            return "high"
        elif abs(change_pct) > 2:
            return "medium" 
        else:
            return "low"
    
    def _get_recommendation(self, score: float, change_pct: float) -> str:
        """获取投资建议"""
        if score >= 70 and change_pct > 2:
            return "strong_buy"
        elif score >= 60 and change_pct > 0:
            return "buy"
        elif score >= 40:
            return "hold"
        elif score >= 30:
            return "weak_sell"
        else:
            return "sell"
    
    async def get_sector_analysis(self, sector_name: str, analysis_days: int) -> Dict:
        """获取单个板块的详细分析"""
        if not AKSHARE_AVAILABLE:
            raise RuntimeError("AKShare不可用，无法进行板块分析")
        
        if sector_name not in self.sector_mapping:
            raise ValueError(f"未知板块: {sector_name}")
        
        sector_code = self.sector_mapping[sector_name]
        
        try:
            # 获取板块历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=analysis_days)).strftime('%Y%m%d')
            
            hist_data = ak.index_hist_sw(symbol=sector_code, start_date=start_date, end_date=end_date)
            
            if hist_data.empty:
                raise RuntimeError(f"无法获取{sector_name}历史数据")
            
            # 计算技术指标
            latest = hist_data.iloc[-1]
            prev_week = hist_data.iloc[-5] if len(hist_data) >= 5 else hist_data.iloc[0]
            
            change_5d = ((latest['收盘'] - prev_week['收盘']) / prev_week['收盘']) * 100
            avg_volume = hist_data['成交量'].mean()
            volume_trend = "increasing" if latest['成交量'] > avg_volume else "decreasing"
            
            # 计算波动率
            returns = hist_data['收盘'].pct_change().dropna()
            volatility = returns.std() * 100
            
            analysis_result = {
                "sector_info": {
                    "name": sector_name,
                    "code": sector_code,
                    "analysis_period": f"{analysis_days}天"
                },
                "current_metrics": {
                    "latest_price": float(latest['收盘']),
                    "change_5d": round(change_5d, 2),
                    "volume": int(latest['成交量']),
                    "turnover": float(latest['成交额']),
                    "volume_trend": volume_trend
                },
                "technical_analysis": {
                    "volatility": round(volatility, 2),
                    "price_range": {
                        "high": float(hist_data['最高'].max()),
                        "low": float(hist_data['最低'].min())
                    },
                    "trend_analysis": "upward" if change_5d > 0 else "downward"
                },
                "data_source": "akshare_historical",
                "analysis_time": datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"板块{sector_name}分析失败: {e}")
            raise RuntimeError(f"板块{sector_name}分析失败: {e}")

# 创建服务实例
sector_service = SectorAnalysisService()

@router.get("/", summary="获取TOP板块排名")
async def get_top_sectors(
    lookback_months: int = Query(default=6, ge=1, le=12, description="回看月数"),
    top_n: int = Query(default=5, ge=1, le=20, description="TOP板块数量")
):
    """
    获取板块排名分析 - 基于真实数据
    
    - **lookback_months**: 分析的历史月数 (1-12)
    - **top_n**: 返回的TOP板块数量 (1-20)
    
    返回板块评分排名，基于真实申万行业指数数据
    """
    try:
        logger.info(f"开始板块分析: lookback_months={lookback_months}, top_n={top_n}")
        result = await sector_service.analyze_all_sectors(lookback_months, top_n)
        logger.info(f"板块分析完成，共分析{result['total_sectors_analyzed']}个板块")
        return result
    except Exception as e:
        logger.error(f"板块分析API错误: {e}")
        raise HTTPException(status_code=500, detail=f"板块分析失败: {str(e)}")

@router.get("/list", summary="获取所有可分析板块列表")
async def get_sectors_list():
    """
    获取所有支持分析的板块列表
    """
    return {
        "available_sectors": list(sector_service.sector_mapping.keys()),
        "total_count": len(sector_service.sector_mapping),
        "data_source": "sw_level1_index"
    }

@router.get("/{sector_name}/analysis", summary="获取单个板块详细分析")
async def get_sector_analysis(
    sector_name: str,
    analysis_days: int = Query(default=30, ge=5, le=365, description="分析天数")
):
    """
    获取单个板块的详细分析
    
    - **sector_name**: 板块名称
    - **analysis_days**: 分析的历史天数 (5-365)
    """
    try:
        logger.info(f"开始单板块分析: {sector_name}, 分析天数: {analysis_days}")
        result = await sector_service.get_sector_analysis(sector_name, analysis_days)
        logger.info(f"单板块分析完成: {sector_name}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"单板块分析API错误: {e}")
        raise HTTPException(status_code=500, detail=f"单板块分析失败: {str(e)}")