#!/usr/bin/env python3
"""
A股智能交易决策平台 - 简化API服务
不依赖pandas等复杂依赖，专注核心功能展示
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List
import datetime

app = FastAPI(title="A股智能交易决策平台", version="1.6.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "A股智能交易决策平台API服务",
        "version": "1.6.0",
        "status": "运行正常",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "bajun-trading-platform"}

@app.get("/api/sectors")
async def get_sectors():
    """获取板块分析数据"""
    return {
        "sectors": [
            {"name": "医药生物", "score": 85.2, "change": "+2.3%", "rank": 1},
            {"name": "计算机", "score": 82.1, "change": "+1.8%", "rank": 2}, 
            {"name": "电子", "score": 78.9, "change": "+1.2%", "rank": 3},
            {"name": "新能源", "score": 75.4, "change": "-0.5%", "rank": 4},
            {"name": "军工", "score": 72.1, "change": "-1.1%", "rank": 5}
        ],
        "updated_at": datetime.datetime.now().isoformat()
    }

@app.get("/api/stocks/{sector}")
async def get_stocks(sector: str):
    """获取指定板块的股票推荐"""
    stocks = {
        "医药生物": [
            {"code": "300760", "name": "迈瑞医疗", "score": 92.1, "price": 315.50},
            {"code": "000661", "name": "长春高新", "score": 88.3, "price": 142.80},
            {"code": "300122", "name": "智飞生物", "score": 85.7, "price": 89.20}
        ],
        "计算机": [
            {"code": "002410", "name": "广联达", "score": 89.4, "price": 68.90},
            {"code": "300418", "name": "昆仑万维", "score": 87.2, "price": 45.60},
            {"code": "002230", "name": "科大讯飞", "score": 84.1, "price": 52.30}
        ]
    }
    
    return {
        "sector": sector,
        "stocks": stocks.get(sector, []),
        "updated_at": datetime.datetime.now().isoformat()
    }

@app.get("/api/portfolio")
async def get_portfolio():
    """获取投资组合分析"""
    return {
        "holdings": [
            {"code": "000858", "name": "五粮液", "position": 12.5, "profit": "+8.2%"},
            {"code": "300760", "name": "迈瑞医疗", "position": 15.0, "profit": "+12.3%"},
            {"code": "002410", "name": "广联达", "position": 10.0, "profit": "+5.1%"}
        ],
        "total_value": 285600.0,
        "total_profit": "+9.1%",
        "updated_at": datetime.datetime.now().isoformat()
    }

@app.post("/api/trading/check")
async def trading_check(data: Dict):
    """反人性交易检查"""
    action = data.get("action", "BUY")
    stock_code = data.get("stock_code", "000001")
    amount = data.get("amount", 1000)
    
    # 简单的反人性检查逻辑
    warnings = []
    if action == "BUY" and amount > 10000:
        warnings.append("建议分批买入，避免一次性重仓")
    
    if action == "SELL":
        warnings.append("是否基于恐慌情绪？建议冷静30分钟再决定")
        
    return {
        "action": action,
        "stock_code": stock_code,
        "amount": amount,
        "warnings": warnings,
        "recommendation": "谨慎操作" if warnings else "可以执行",
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 启动A股智能交易决策平台简化API服务...")
    print("📊 服务地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("⚛️  前端访问: http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=8000)