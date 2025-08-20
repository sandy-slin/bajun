#!/usr/bin/env python3
"""
简单的FastAPI测试服务
用于验证基础环境是否正常
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="A股智能交易决策平台 - 测试服务", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "A股智能交易决策平台测试服务运行正常", "status": "OK"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "bajun-test"}

if __name__ == "__main__":
    print("🚀 启动A股智能交易决策平台测试服务...")
    print("📊 服务地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)