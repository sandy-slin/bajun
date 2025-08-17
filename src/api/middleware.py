#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI中间件配置
处理CORS、认证、日志等横切关注点
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI):
    """配置所有中间件"""
    
    # CORS中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React开发服务器
            "http://localhost:3001", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    # 受信任主机中间件 (生产环境安全)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "localhost",
            "127.0.0.1",
            "*.localhost",
            "*"  # 开发环境允许所有主机
        ]
    )
    
    # 请求处理时间和日志中间件
    @app.middleware("http")
    async def process_time_middleware(request: Request, call_next: Callable):
        """记录请求处理时间和基本日志"""
        start_time = time.time()
        
        # 记录请求信息
        logger.info(f"收到请求: {request.method} {request.url}")
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 添加处理时间到响应头
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            # 记录响应信息
            logger.info(
                f"请求完成: {request.method} {request.url} - "
                f"状态码: {response.status_code} - "
                f"处理时间: {process_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            # 记录错误
            process_time = time.time() - start_time
            logger.error(
                f"请求处理失败: {request.method} {request.url} - "
                f"错误: {str(e)} - "
                f"处理时间: {process_time:.4f}s"
            )
            raise
    
    # 速率限制和安全中间件 (简化实现)
    @app.middleware("http") 
    async def security_middleware(request: Request, call_next: Callable):
        """基本安全检查中间件"""
        
        # 添加安全响应头
        response = await call_next(request)
        
        # 添加安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Server"] = "Bajun-Trading-Platform"
        
        return response
    
    logger.info("FastAPI中间件配置完成")

def setup_exception_handlers(app: FastAPI):
    """配置全局异常处理器"""
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """404错误处理"""
        return {
            "success": False,
            "error": "NOT_FOUND",
            "message": f"请求的资源不存在: {request.url.path}",
            "timestamp": time.time()
        }
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        """500错误处理"""
        logger.error(f"内部服务器错误: {request.url} - {str(exc)}")
        return {
            "success": False,
            "error": "INTERNAL_ERROR", 
            "message": "服务器内部错误，请稍后重试",
            "timestamp": time.time()
        }
    
    logger.info("全局异常处理器配置完成")