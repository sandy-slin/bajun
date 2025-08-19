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

# Initialize logger first
logger = logging.getLogger(__name__)

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# Import the stable fetcher (defer initialization)
try:
    from ...data.stable_akshare_fetcher import StableAKShareFetcher
    STABLE_FETCHER_AVAILABLE = True
    logger.info("稳定数据获取器类导入成功")
except ImportError as e:
    STABLE_FETCHER_AVAILABLE = False
    StableAKShareFetcher = None
    logger.warning(f"稳定数据获取器不可用: {e}")

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
            # 使用稳定的数据获取器
            if STABLE_FETCHER_AVAILABLE:
                logger.info("使用稳定数据获取器获取股票数据...")
                stable_fetcher = StableAKShareFetcher()
                stock_zh_a_spot = await stable_fetcher.get_stock_realtime_data()
            else:
                # 原始重试机制作为备用
                stock_zh_a_spot = None
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"获取全市场股票数据，尝试第{attempt + 1}次...")
                        stock_zh_a_spot = ak.stock_zh_a_spot_em()
                        if not stock_zh_a_spot.empty:
                            logger.info(f"成功获取{len(stock_zh_a_spot)}只股票数据")
                            break
                        else:
                            logger.warning(f"第{attempt + 1}次获取到空数据")
                    except Exception as e:
                        logger.warning(f"第{attempt + 1}次获取失败: {e}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2)  # 等待2秒后重试
                        
                if stock_zh_a_spot is None or stock_zh_a_spot.empty:
                    raise RuntimeError("无法获取A股实时数据，请稍后重试")
            
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
    
    async def get_sector_stocks_prediction(self, sector_name: str, top_n: int = 10) -> Dict:
        """获取指定板块的个股预测 - 核心功能"""
        if not AKSHARE_AVAILABLE:
            raise RuntimeError("AKShare不可用，无法进行板块个股预测")
        
        # 验证板块名称
        if sector_name not in self.sector_stock_mapping:
            available_sectors = list(self.sector_stock_mapping.keys())
            raise ValueError(f"未知板块: {sector_name}。可用板块: {available_sectors}")
        
        try:
            # 获取申万行业成分股
            sector_code = self.sector_stock_mapping[sector_name]
            
            # 使用稳定的数据获取器
            if STABLE_FETCHER_AVAILABLE:
                logger.info("使用稳定数据获取器获取板块股票数据...")
                stable_fetcher = StableAKShareFetcher()
                stock_zh_a_spot = await stable_fetcher.get_stock_realtime_data()
            else:
                # 使用更稳定的股票数据获取方式，添加重试机制
                stock_zh_a_spot = None
                max_retries = 3
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"获取股票数据，尝试第{attempt + 1}次...")
                        stock_zh_a_spot = ak.stock_zh_a_spot_em()
                        if not stock_zh_a_spot.empty:
                            logger.info(f"成功获取{len(stock_zh_a_spot)}只股票数据")
                            break
                        else:
                            logger.warning(f"第{attempt + 1}次获取到空数据")
                    except Exception as e:
                        logger.warning(f"第{attempt + 1}次获取失败: {e}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2)  # 等待2秒后重试
                        
                if stock_zh_a_spot is None or stock_zh_a_spot.empty:
                    # 严格遵守：禁止模拟数据，获取不到真实数据时直接报错
                    raise RuntimeError("无法获取A股实时数据，请稍后重试")
            
            # 过滤股票
            filtered_stocks = stock_zh_a_spot[
                (~stock_zh_a_spot['名称'].str.contains('ST', na=False)) &
                (~stock_zh_a_spot['名称'].str.contains('退', na=False)) &
                (stock_zh_a_spot['涨跌幅'] != 0) &
                (stock_zh_a_spot['成交量'] > 0) &
                (stock_zh_a_spot['最新价'] >= 3) &
                (stock_zh_a_spot['最新价'] <= 200)
            ]
            
            # 基于板块特征的智能预测算法
            predictions = []
            
            for _, stock in filtered_stocks.iterrows():
                try:
                    # 如果股票名称包含板块相关关键词，优先考虑
                    sector_relevance = self._calculate_sector_relevance(stock['名称'], sector_name)
                    if sector_relevance == 0:
                        continue  # 跳过与板块无关的股票
                    
                    # 基于真实数据的预测评分
                    change_pct = float(stock['涨跌幅'])
                    volume = int(stock['成交量'])
                    turnover = float(stock['成交额'])
                    price = float(stock['最新价'])
                    market_cap = float(stock['流通市值']) if pd.notna(stock.get('流通市值')) else 0
                    
                    # 多维度评分算法
                    momentum_score = self._calculate_momentum_score(change_pct)
                    liquidity_score = self._calculate_liquidity_score(volume, turnover)
                    value_score = self._calculate_value_score(price, market_cap)
                    sector_score = sector_relevance * 20  # 板块相关性加权
                    
                    # 综合预测评分
                    prediction_score = (
                        momentum_score * 0.3 +    # 动量因子
                        liquidity_score * 0.25 +  # 流动性因子  
                        value_score * 0.25 +     # 估值因子
                        sector_score * 0.2       # 板块契合度
                    )
                    
                    # 只保留高质量预测
                    if prediction_score > 50:
                        stock_prediction = {
                            'code': stock['代码'],
                            'name': stock['名称'],
                            'sector': sector_name,
                            'current_price': price,
                            'change_pct': round(change_pct, 2),
                            'volume': volume,
                            'turnover': turnover,
                            'market_cap': market_cap,
                            'prediction_score': round(prediction_score, 2),
                            'momentum_score': round(momentum_score, 2),
                            'liquidity_score': round(liquidity_score, 2),
                            'value_score': round(value_score, 2),
                            'sector_relevance': round(sector_relevance, 2),
                            'prediction': self._get_prediction_label(prediction_score),
                            'risk_level': self._assess_prediction_risk(change_pct, volume, prediction_score),
                            'confidence': self._calculate_confidence(prediction_score, volume)
                        }
                        predictions.append(stock_prediction)
                        
                except (ValueError, TypeError) as e:
                    continue  # 跳过数据异常的股票
            
            # 按预测评分排序
            predictions.sort(key=lambda x: x['prediction_score'], reverse=True)
            
            # 返回TOP N
            top_predictions = predictions[:top_n]
            
            # 计算板块整体预测指标
            if predictions:
                avg_prediction_score = sum(p['prediction_score'] for p in predictions) / len(predictions)
                strong_buy_count = sum(1 for p in predictions if p['prediction'] == 'strong_buy')
                buy_count = sum(1 for p in predictions if p['prediction'] == 'buy')
                
                sector_outlook = "积极" if avg_prediction_score > 70 else "中性" if avg_prediction_score > 50 else "谨慎"
            else:
                avg_prediction_score = 0
                strong_buy_count = 0
                buy_count = 0
                sector_outlook = "无数据"
            
            return {
                'analysis_time': datetime.now().isoformat(),
                'sector_info': {
                    'name': sector_name,
                    'code': sector_code,
                    'outlook': sector_outlook
                },
                'prediction_summary': {
                    'total_analyzed': len(filtered_stocks),
                    'qualified_predictions': len(predictions),
                    'returned_count': len(top_predictions),
                    'avg_prediction_score': round(avg_prediction_score, 2),
                    'strong_buy_count': strong_buy_count,
                    'buy_count': buy_count
                },
                'top_predictions': top_predictions,
                'algorithm_info': {
                    'method': 'multi_factor_sector_prediction',
                    'factors': ['momentum', 'liquidity', 'value', 'sector_relevance'],
                    'weights': [0.3, 0.25, 0.25, 0.2],
                    'min_score_threshold': 50
                },
                'data_source': 'akshare_realtime_with_sector_analysis'
            }
            
        except Exception as e:
            logger.error(f"板块{sector_name}个股预测失败: {e}")
            raise RuntimeError(f"板块{sector_name}个股预测失败: {e}")
    
    def _calculate_sector_relevance(self, stock_name: str, sector_name: str) -> float:
        """计算股票与板块的相关性"""
        # 定义板块关键词映射
        sector_keywords = {
            '银行': ['银行', '农行', '工行', '建行', '中行', '交行', '招行', '民生', '光大', '兴业', '浦发', '平安银行'],
            '医药生物': ['药', '医', '生物', '健康', '康', '制药', '医疗', '基因', '疫苗', '诊断'],
            '食品饮料': ['食品', '饮料', '白酒', '啤酒', '乳业', '茶', '咖啡', '糖', '肉', '蒙牛', '伊利'],
            '电子': ['电子', '芯片', '半导体', '集成', '显示', '面板', 'PCB', '电路', '传感', '光电'],
            '计算机': ['科技', '软件', '信息', '网络', '云', '大数据', '人工智能', 'AI', '互联网', '通信'],
            '汽车': ['汽车', '车', '汽', '客车', '货车', '零部件', '轮胎', '发动机', '新能源车'],
            '房地产': ['地产', '房', '置业', '建设', '开发', '物业', '商业'],
            '化工': ['化工', '化学', '塑料', '橡胶', '涂料', '农药', '化肥', '石化'],
            '机械设备': ['机械', '设备', '工程', '制造', '重工', '机床', '泵', '阀门', '轴承'],
            '钢铁': ['钢铁', '钢', '铁', '特钢', '不锈钢'],
            '有色金属': ['有色', '金属', '铜', '铝', '锌', '铅', '镍', '钴', '锂'],
            '煤炭': ['煤炭', '煤', '焦炭', '焦煤'],
            '石油石化': ['石油', '石化', '化工', '燃气', '天然气'],
            '电力设备': ['电力', '电气', '电网', '输配电', '变压器', '开关'],
            '建筑材料': ['水泥', '玻璃', '陶瓷', '建材', '石膏', '保温'],
            '国防军工': ['军工', '航空', '航天', '船舶', '兵器', '核'],
            '传媒': ['传媒', '广告', '影视', '游戏', '出版', '教育', '文化'],
            '通信': ['通信', '5G', '光纤', '网络', '电信', '移动', '联通'],
            '建筑装饰': ['建筑', '装饰', '园林', '设计', '装修'],
            '非银金融': ['证券', '保险', '信托', '期货', '基金', '租赁', '担保']
        }
        
        keywords = sector_keywords.get(sector_name, [])
        if not keywords:
            return 0.1  # 默认低相关性
        
        # 计算匹配度
        matches = sum(1 for keyword in keywords if keyword in stock_name)
        if matches > 0:
            return min(matches / len(keywords) * 5, 5.0)  # 最高5分
        
        return 0.1  # 无匹配时给最低分
    
    def _calculate_momentum_score(self, change_pct: float) -> float:
        """计算动量评分"""
        if change_pct > 5:
            return 90
        elif change_pct > 3:
            return 80
        elif change_pct > 1:
            return 70
        elif change_pct > 0:
            return 60
        elif change_pct > -2:
            return 40
        elif change_pct > -5:
            return 20
        else:
            return 10
    
    def _calculate_liquidity_score(self, volume: int, turnover: float) -> float:
        """计算流动性评分"""
        # 基于成交量和成交额的流动性评分
        if volume > 100000000:  # 1亿股以上
            return 90
        elif volume > 50000000:  # 5000万股以上
            return 80
        elif volume > 10000000:  # 1000万股以上
            return 70
        elif volume > 5000000:   # 500万股以上
            return 60
        elif volume > 1000000:   # 100万股以上
            return 50
        else:
            return 30
    
    def _calculate_value_score(self, price: float, market_cap: float) -> float:
        """计算估值评分"""
        # 基于价格区间的估值评分
        if 10 <= price <= 50:
            base_score = 80
        elif 5 <= price < 10 or 50 < price <= 100:
            base_score = 70
        elif 3 <= price < 5 or 100 < price <= 150:
            base_score = 60
        else:
            base_score = 40
        
        # 市值调整
        if market_cap > 0:
            if 100e8 <= market_cap <= 1000e8:  # 100-1000亿合理区间
                return base_score + 10
            elif 50e8 <= market_cap < 100e8 or 1000e8 < market_cap <= 2000e8:
                return base_score
            else:
                return base_score - 10
        
        return base_score
    
    def _get_prediction_label(self, score: float) -> str:
        """获取预测标签"""
        if score >= 85:
            return "strong_buy"
        elif score >= 75:
            return "buy"
        elif score >= 65:
            return "hold"
        elif score >= 55:
            return "watch"
        else:
            return "avoid"
    
    def _assess_prediction_risk(self, change_pct: float, volume: int, score: float) -> str:
        """评估预测风险"""
        risk_factors = 0
        
        if abs(change_pct) > 8:
            risk_factors += 2
        elif abs(change_pct) > 5:
            risk_factors += 1
        
        if volume > 200000000:  # 成交量过大
            risk_factors += 1
        elif volume < 1000000:  # 成交量过小
            risk_factors += 1
        
        if score < 60:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, score: float, volume: int) -> float:
        """计算预测置信度"""
        base_confidence = min(score / 100, 0.95)
        
        # 成交量调整置信度
        if volume > 10000000:
            volume_boost = 0.1
        elif volume > 1000000:
            volume_boost = 0.05
        else:
            volume_boost = 0
        
        return min(base_confidence + volume_boost, 0.95)
    
    def _get_fallback_sector_predictions(self, sector_name: str, top_n: int) -> Dict:
        """降级方案：AKShare API不可用时的预设示例数据"""
        
        # 不同板块的示例股票数据
        sector_examples = {
            '医药生物': [
                {'code': '000001', 'name': '平安银行', 'price': 15.50, 'change_pct': 2.1, 'volume': 125000000, 'score': 75.5},
                {'code': '600519', 'name': '贵州茅台', 'price': 1680.00, 'change_pct': 1.8, 'volume': 8500000, 'score': 72.3},
                {'code': '000858', 'name': '五粮液', 'price': 145.20, 'change_pct': 1.5, 'volume': 45000000, 'score': 70.8},
                {'code': '300015', 'name': '爱尔眼科', 'price': 42.80, 'change_pct': 3.2, 'volume': 28000000, 'score': 68.9},
                {'code': '600276', 'name': '恒瑞医药', 'price': 58.90, 'change_pct': 2.8, 'volume': 35000000, 'score': 67.2}
            ],
            '银行': [
                {'code': '601398', 'name': '工商银行', 'price': 4.85, 'change_pct': 1.2, 'volume': 180000000, 'score': 73.1},
                {'code': '601939', 'name': '建设银行', 'price': 6.12, 'change_pct': 0.8, 'volume': 125000000, 'score': 71.5},
                {'code': '600036', 'name': '招商银行', 'price': 42.30, 'change_pct': 1.5, 'volume': 85000000, 'score': 69.8},
                {'code': '000001', 'name': '平安银行', 'price': 15.50, 'change_pct': 2.1, 'volume': 125000000, 'score': 68.2},
                {'code': '600000', 'name': '浦发银行', 'price': 9.85, 'change_pct': 1.8, 'volume': 95000000, 'score': 66.9}
            ],
            '计算机': [
                {'code': '002415', 'name': '海康威视', 'price': 38.50, 'change_pct': 2.5, 'volume': 68000000, 'score': 74.2},
                {'code': '000063', 'name': '中兴通讯', 'price': 28.90, 'change_pct': 3.1, 'volume': 95000000, 'score': 72.8},
                {'code': '002230', 'name': '科大讯飞', 'price': 45.20, 'change_pct': 4.2, 'volume': 52000000, 'score': 71.5},
                {'code': '600570', 'name': '恒生电子', 'price': 82.30, 'change_pct': 2.8, 'volume': 25000000, 'score': 70.1},
                {'code': '300059', 'name': '东方财富', 'price': 18.75, 'change_pct': 1.9, 'volume': 158000000, 'score': 69.3}
            ]
        }
        
        # 获取示例数据
        examples = sector_examples.get(sector_name, sector_examples['医药生物'])
        selected_stocks = examples[:min(top_n, len(examples))]
        
        # 构建标准格式的预测结果
        predictions = []
        for stock in selected_stocks:
            prediction = {
                'code': stock['code'],
                'name': stock['name'],
                'sector': sector_name,
                'current_price': stock['price'],
                'change_pct': stock['change_pct'],
                'volume': stock['volume'],
                'turnover': stock['volume'] * stock['price'] / 100,  # 估算成交额
                'market_cap': stock['volume'] * stock['price'] * 10,  # 估算市值
                'prediction_score': stock['score'],
                'momentum_score': 50 + stock['change_pct'] * 10,
                'liquidity_score': min(stock['volume'] / 1000000 * 2, 100),
                'value_score': 60.0,
                'sector_relevance': 5.0,
                'prediction': self._get_prediction_label(stock['score']),
                'risk_level': 'low' if stock['change_pct'] < 2 else 'medium',
                'confidence': min(stock['score'] / 100 + 0.1, 0.85)
            }
            predictions.append(prediction)
        
        # 计算汇总信息
        avg_score = sum(p['prediction_score'] for p in predictions) / len(predictions)
        strong_buy_count = sum(1 for p in predictions if p['prediction'] == 'strong_buy')
        buy_count = sum(1 for p in predictions if p['prediction'] == 'buy')
        
        return {
            'analysis_time': datetime.now().isoformat(),
            'sector_info': {
                'name': sector_name,
                'code': self.sector_stock_mapping.get(sector_name, '801150'),
                'outlook': '中性' if avg_score < 70 else '积极'
            },
            'prediction_summary': {
                'total_analyzed': 5000,  # 模拟分析数量
                'qualified_predictions': len(selected_stocks) * 20,  # 模拟合格数量
                'returned_count': len(predictions),
                'avg_prediction_score': round(avg_score, 2),
                'strong_buy_count': strong_buy_count,
                'buy_count': buy_count
            },
            'top_predictions': predictions,
            'algorithm_info': {
                'method': 'fallback_demo_data',
                'factors': ['momentum', 'liquidity', 'value', 'sector_relevance'],
                'weights': [0.3, 0.25, 0.25, 0.2],
                'min_score_threshold': 50,
                'note': 'AKShare API不可用，使用示例数据演示功能'
            },
            'data_source': 'fallback_demo_data_for_akshare_unavailable'
        }

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
