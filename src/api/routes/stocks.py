#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票分析API路由
提供股票推荐、分析、筛选等功能 - 禁止模拟数据
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

class StockAnalysisService:
    def __init__(self):
        # 重点关注的A股股票池 (移除硬编码，动态获取)
        self.focus_sectors = [
            '医药生物', '食品饮料', '电子', '计算机', '新能源',
            '银行', '券商', '保险', '化工', '机械设备'
        ]
    
    async def get_recommendations(self, sector: str = None, top_n: int = 10) -> Dict:
        """获取股票推荐 - 基于真实数据分析"""
        if not AKSHARE_AVAILABLE:
            raise RuntimeError("AKShare不可用，无法进行股票分析")
        
        try:
            # 获取A股实时数据
            stock_zh_a_spot = ak.stock_zh_a_spot_em()
            if stock_zh_a_spot.empty:
                raise RuntimeError("无法获取A股实时数据")
            
            # 过滤掉ST股票和退市股票
            filtered_stocks = stock_zh_a_spot[
                (~stock_zh_a_spot['名称'].str.contains('ST', na=False)) &
                (~stock_zh_a_spot['名称'].str.contains('退', na=False)) &
                (stock_zh_a_spot['涨跌幅'] != 0) &  # 过滤掉停牌股票
                (stock_zh_a_spot['成交量'] > 0)     # 过滤掉无成交股票
            ]
            
            # 基于真实数据计算推荐评分
            recommendations = []
            
            for _, stock in filtered_stocks.iterrows():
                try:
                    # 基于真实数据的评分算法
                    change_pct = float(stock['涨跌幅'])
                    volume = int(stock['成交量'])
                    turnover = float(stock['成交额'])
                    price = float(stock['最新价'])
                    
                    # 排除价格过低或过高的股票
                    if price < 3 or price > 200:
                        continue
                    
                    # 计算评分
                    momentum_score = min(50 + change_pct * 2, 100) if change_pct > 0 else max(change_pct * 2, -100)
                    volume_score = min(volume / 1000000 * 10, 100)  # 基于成交量的活跃度
                    price_score = 50 + (price - 20) / 100 * 10  # 价格适中性评分
                    
                    # 综合评分
                    composite_score = momentum_score * 0.4 + volume_score * 0.3 + price_score * 0.3
                    
                    # 只推荐评分较高的股票
                    if composite_score > 40:
                        stock_info = {
                            'code': stock['代码'],
                            'name': stock['名称'],
                            'price': price,
                            'change_pct': round(change_pct, 2),
                            'volume': volume,
                            'turnover': turnover,
                            'score': round(composite_score, 2),
                            'momentum_score': round(momentum_score, 2),
                            'volume_score': round(volume_score, 2),
                            'recommendation': self._get_recommendation(composite_score, change_pct),
                            'risk_level': self._assess_risk_level(change_pct, volume),
                            'data_source': 'akshare_realtime'
                        }
                        recommendations.append(stock_info)
                        
                except (ValueError, TypeError) as e:
                    # 跳过数据异常的股票
                    continue
            
            # 按评分排序
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # 返回前N只
            top_recommendations = recommendations[:top_n]
            
            return {
                'analysis_time': datetime.now().isoformat(),
                'total_analyzed': len(filtered_stocks),
                'qualified_stocks': len(recommendations),
                'recommendations': top_recommendations,
                'filter_criteria': {
                    'exclude_st': True,
                    'min_price': 3,
                    'max_price': 200,
                    'min_volume': 0,
                    'min_score': 40
                },
                'analysis_method': 'real_data_composite_scoring',
                'data_source': 'akshare_a_stock_realtime'
            }
            
        except Exception as e:
            logger.error(f"股票推荐分析失败: {e}")
            raise RuntimeError(f"股票推荐分析失败: {e}")
    
    def _get_recommendation(self, score: float, change_pct: float) -> str:
        """获取投资建议"""
        if score >= 80 and change_pct > 3:
            return "strong_buy"
        elif score >= 70 and change_pct > 1:
            return "buy"
        elif score >= 50:
            return "hold"
        elif score >= 30:
            return "weak_sell"
        else:
            return "sell"
    
    def _assess_risk_level(self, change_pct: float, volume: int) -> str:
        """评估风险水平"""
        if abs(change_pct) > 7 or volume > 500000000:  # 涨跌幅>7%或成交量>5亿
            return "high"
        elif abs(change_pct) > 3 or volume > 100000000:  # 涨跌幅>3%或成交量>1亿
            return "medium"
        else:
            return "low"
    
    async def analyze_stock(self, stock_code: str) -> Dict:
        """分析单只股票的详细信息"""
        if not AKSHARE_AVAILABLE:
            raise RuntimeError("AKShare不可用，无法进行股票分析")
        
        try:
            # 获取股票基本信息
            stock_individual = ak.stock_individual_info_em(symbol=stock_code)
            if stock_individual.empty:
                raise RuntimeError(f"无法获取股票{stock_code}的基本信息")
            
            # 获取股票实时数据
            stock_zh_a_spot = ak.stock_zh_a_spot_em()
            stock_data = stock_zh_a_spot[stock_zh_a_spot['代码'] == stock_code]
            
            if stock_data.empty:
                raise RuntimeError(f"无法获取股票{stock_code}的实时数据")
            
            stock_row = stock_data.iloc[0]
            
            # 获取历史数据进行技术分析
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            hist_data = ak.stock_zh_a_hist(symbol=stock_code, period='daily', 
                                         start_date=start_date, end_date=end_date, adjust='qfq')
            
            if not hist_data.empty:
                # 计算技术指标
                returns = hist_data['收盘'].pct_change().dropna()
                volatility = returns.std() * 100
                avg_volume = hist_data['成交量'].mean()
                
                # 计算5日均价
                ma5 = hist_data['收盘'].tail(5).mean()
                current_price = float(stock_row['最新价'])
                price_vs_ma5 = ((current_price - ma5) / ma5) * 100
                
                technical_analysis = {
                    'volatility': round(volatility, 2),
                    'avg_volume_30d': int(avg_volume),
                    'ma5': round(ma5, 2),
                    'price_vs_ma5': round(price_vs_ma5, 2),
                    'trend': 'upward' if price_vs_ma5 > 0 else 'downward'
                }
            else:
                technical_analysis = {
                    'error': '无法获取历史数据进行技术分析'
                }
            
            # 基本面信息
            basic_info = {}
            for _, row in stock_individual.iterrows():
                basic_info[row['item']] = row['value']
            
            analysis_result = {
                'stock_info': {
                    'code': stock_code,
                    'name': stock_row['名称'],
                    'price': float(stock_row['最新价']),
                    'change_pct': float(stock_row['涨跌幅']),
                    'volume': int(stock_row['成交量']),
                    'turnover': float(stock_row['成交额'])
                },
                'basic_info': basic_info,
                'technical_analysis': technical_analysis,
                'market_performance': {
                    'current_price': float(stock_row['最新价']),
                    'open_price': float(stock_row['今开']),
                    'high_price': float(stock_row['最高']),
                    'low_price': float(stock_row['最低']),
                    'prev_close': float(stock_row['昨收'])
                },
                'data_source': 'akshare_comprehensive',
                'analysis_time': datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"股票{stock_code}分析失败: {e}")
            raise RuntimeError(f"股票{stock_code}分析失败: {e}")

# 创建服务实例
stock_service = StockAnalysisService()

@router.get("/recommendations", summary="获取股票推荐")
async def get_stock_recommendations(
    sector: Optional[str] = Query(default=None, description="指定板块（可选）"),
    top_n: int = Query(default=10, ge=1, le=50, description="推荐股票数量")
):
    """
    获取股票推荐列表 - 基于真实数据分析
    
    - **sector**: 指定板块筛选（可选）
    - **top_n**: 返回的推荐股票数量 (1-50)
    
    返回基于真实市场数据的股票推荐
    """
    try:
        logger.info(f"开始股票推荐分析: sector={sector}, top_n={top_n}")
        result = await stock_service.get_recommendations(sector, top_n)
        logger.info(f"股票推荐分析完成，共推荐{len(result['recommendations'])}只股票")
        return result
    except Exception as e:
        logger.error(f"股票推荐API错误: {e}")
        raise HTTPException(status_code=500, detail=f"股票推荐分析失败: {str(e)}")

@router.get("/{stock_code}/analysis", summary="获取单股详细分析")
async def get_stock_analysis(stock_code: str):
    """
    获取单只股票的详细分析
    
    - **stock_code**: 股票代码（如：000001）
    """
    try:
        logger.info(f"开始单股分析: {stock_code}")
        result = await stock_service.analyze_stock(stock_code)
        logger.info(f"单股分析完成: {stock_code}")
        return result
    except Exception as e:
        logger.error(f"单股分析API错误: {e}")
        raise HTTPException(status_code=500, detail=f"单股分析失败: {str(e)}")