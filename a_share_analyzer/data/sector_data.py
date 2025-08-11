# -*- coding: utf-8 -*-
"""
Sector data provider module
"""

from datetime import datetime, timedelta
try:
    from typing import Dict, List
except ImportError:
    Dict = dict
    List = list
import akshare as ak

from ..config import DATA_CONFIG


class SectorDataProvider:
    """板块数据提供者"""
    
    def get_stock_concepts(self, stock_code):
        """获取股票相关概念板块"""
        try:
            concept_data = ak.stock_board_concept_name_em()
            stock_concepts = []
            
            for _, concept in concept_data.iterrows():
                try:
                    concept_stocks = ak.stock_board_concept_cons_em(symbol=concept['板块名称'])
                    if stock_code in concept_stocks['代码'].values:
                        stock_concepts.append(concept['板块名称'])
                        
                        # 限制概念数量
                        if len(stock_concepts) >= DATA_CONFIG["max_concepts_per_stock"]:
                            break
                except:
                    continue
            
            return stock_concepts
            
        except Exception as e:
            print(f"获取股票概念失败: {e}")
            return []
    
    def get_concept_trends(self, concepts):
        """获取概念板块趋势"""
        trends = {}
        start_date = (datetime.now() - timedelta(days=DATA_CONFIG["sector_analysis_days"])).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        
        for concept in concepts:
            try:
                concept_hist = ak.stock_board_concept_hist_em(
                    symbol=concept,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not concept_hist.empty:
                    latest_change = concept_hist.iloc[-1]['涨跌幅']
                    trends[concept] = f"过去一周涨跌幅: {latest_change:.2f}%"
                else:
                    trends[concept] = "暂无数据"
                    
            except Exception as e:
                trends[concept] = f"获取失败: {str(e)[:50]}"
        
        return trends
    
    def analyze_sector_trends(self, stock_code):
        """分析股票相关板块动向"""
        concepts = self.get_stock_concepts(stock_code)
        
        if not concepts:
            return {"error": "未找到相关概念板块"}
        
        return self.get_concept_trends(concepts)