#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket连接管理器
处理实时数据推送和客户端连接管理
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import asyncio
import json
import logging
from datetime import datetime

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接管理
        self.active_connections: List[WebSocket] = []
        self.connection_subscriptions: Dict[WebSocket, Set[str]] = {}
        
        # 订阅管理
        self.subscriptions: Dict[str, Set[WebSocket]] = {
            'market_data': set(),      # 市场行情
            'sector_updates': set(),   # 板块更新  
            'portfolio_alerts': set(), # 组合预警
            'trading_signals': set(),  # 交易信号
            'system_status': set()     # 系统状态
        }
        
        # 数据推送任务
        self.push_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
    
    async def connect(self, websocket: WebSocket):
        """接受新的WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_subscriptions[websocket] = set()
        
        logger.info(f"新WebSocket连接，当前连接数: {len(self.active_connections)}")
        
        # 发送欢迎消息
        await websocket.send_json({
            "type": "welcome",
            "message": "连接到A股智能交易决策平台",
            "timestamp": datetime.now().isoformat(),
            "available_subscriptions": list(self.subscriptions.keys())
        })
        
        # 启动数据推送服务
        if not self.is_running:
            await self.start_data_push_service()
    
    async def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        try:
            # 从活跃连接中移除
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            # 清理订阅
            if websocket in self.connection_subscriptions:
                user_subscriptions = self.connection_subscriptions[websocket]
                for subscription_type in user_subscriptions:
                    if subscription_type in self.subscriptions:
                        self.subscriptions[subscription_type].discard(websocket)
                del self.connection_subscriptions[websocket]
            
            logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")
            
            # 如果没有活跃连接，停止数据推送
            if not self.active_connections:
                await self.stop_data_push_service()
                
        except Exception as e:
            logger.error(f"断开连接时发生错误: {e}")
    
    async def handle_subscription(self, websocket: WebSocket, message: Dict):
        """处理订阅请求"""
        try:
            subscription_type = message.get("subscription")
            
            if subscription_type not in self.subscriptions:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知的订阅类型: {subscription_type}",
                    "available_types": list(self.subscriptions.keys())
                })
                return
            
            # 添加订阅
            self.subscriptions[subscription_type].add(websocket)
            self.connection_subscriptions[websocket].add(subscription_type)
            
            await websocket.send_json({
                "type": "subscription_confirmed",
                "subscription": subscription_type,
                "message": f"成功订阅 {subscription_type}",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"WebSocket订阅: {subscription_type}")
            
        except Exception as e:
            logger.error(f"处理订阅请求失败: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "订阅请求处理失败"
            })
    
    async def handle_unsubscription(self, websocket: WebSocket, message: Dict):
        """处理取消订阅请求"""
        try:
            subscription_type = message.get("subscription")
            
            if subscription_type in self.subscriptions:
                self.subscriptions[subscription_type].discard(websocket)
                self.connection_subscriptions[websocket].discard(subscription_type)
                
                await websocket.send_json({
                    "type": "unsubscription_confirmed",
                    "subscription": subscription_type,
                    "message": f"成功取消订阅 {subscription_type}",
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"WebSocket取消订阅: {subscription_type}")
            
        except Exception as e:
            logger.error(f"处理取消订阅请求失败: {e}")
    
    async def broadcast_to_subscribers(self, subscription_type: str, data: Dict):
        """向订阅者广播数据"""
        if subscription_type not in self.subscriptions:
            return
        
        subscribers = self.subscriptions[subscription_type].copy()
        if not subscribers:
            return
        
        message = {
            "type": "data_update",
            "subscription": subscription_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # 并发发送给所有订阅者
        disconnected_clients = []
        
        for websocket in subscribers:
            try:
                # 检查连接状态，避免向已关闭的连接发送数据
                if websocket.client_state.name != "CONNECTED":
                    disconnected_clients.append(websocket)
                    continue
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"向客户端发送数据失败: {e}")
                disconnected_clients.append(websocket)
        
        # 清理断开的连接
        for websocket in disconnected_clients:
            await self.disconnect(websocket)
    
    async def start_data_push_service(self):
        """启动数据推送服务"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("启动WebSocket数据推送服务")
        
        # 启动各种数据推送任务
        self.push_tasks['market_data'] = asyncio.create_task(self.push_market_data())
        self.push_tasks['sector_updates'] = asyncio.create_task(self.push_sector_updates())
        self.push_tasks['portfolio_alerts'] = asyncio.create_task(self.push_portfolio_alerts())
        self.push_tasks['trading_signals'] = asyncio.create_task(self.push_trading_signals())
        self.push_tasks['system_status'] = asyncio.create_task(self.push_system_status())
    
    async def stop_data_push_service(self):
        """停止数据推送服务"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("停止WebSocket数据推送服务")
        
        # 取消所有推送任务
        for task_name, task in self.push_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"数据推送任务 {task_name} 已取消")
        
        self.push_tasks.clear()
    
    async def push_market_data(self):
        """推送市场行情数据 - 使用真实AKShare数据"""
        try:
            while self.is_running:
                # 获取真实市场数据
                market_data = await self._fetch_real_market_data()
                
                await self.broadcast_to_subscribers('market_data', market_data)
                await asyncio.sleep(5)  # 每5秒推送一次
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"市场数据推送错误: {e}")
    
    async def push_sector_updates(self):
        """推送板块更新 - 基于真实数据"""
        try:
            while self.is_running:
                # 获取真实板块数据
                sector_updates = await self._fetch_real_sector_data()
                
                await self.broadcast_to_subscribers('sector_updates', sector_updates)
                await asyncio.sleep(300)  # 每5分钟推送一次，减少API调用
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"板块更新推送错误: {e}")
            # 真实数据获取失败时不推送，避免模拟数据
    
    async def push_portfolio_alerts(self):
        """推送投资组合预警 - 基于真实持仓数据"""
        try:
            while self.is_running:
                # 获取真实投资组合预警
                portfolio_alerts = await self._fetch_real_portfolio_alerts()
                
                if portfolio_alerts:  # 只有真实预警时才推送
                    await self.broadcast_to_subscribers('portfolio_alerts', portfolio_alerts)
                
                await asyncio.sleep(600)  # 每10分钟检查一次
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"组合预警推送错误: {e}")
            # 真实数据获取失败时不推送
    
    async def push_trading_signals(self):
        """推送交易信号 - 基于真实分析"""
        try:
            while self.is_running:
                # 获取真实交易信号
                trading_signals = await self._fetch_real_trading_signals()
                
                if trading_signals:  # 只有真实信号时才推送
                    await self.broadcast_to_subscribers('trading_signals', trading_signals)
                
                await asyncio.sleep(1800)  # 每30分钟分析一次
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"交易信号推送错误: {e}")
            # 真实分析失败时不推送
    
    async def push_system_status(self):
        """推送系统状态 - 基于真实监控"""
        try:
            while self.is_running:
                # 获取真实系统状态
                system_status = await self._fetch_real_system_status()
                
                await self.broadcast_to_subscribers('system_status', system_status)
                await asyncio.sleep(300)  # 每5分钟推送一次
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"系统状态推送错误: {e}")
            # 系统状态获取失败时停止推送该服务
    
    async def _fetch_real_sector_data(self) -> Dict:
        """获取真实板块数据"""
        if not AKSHARE_AVAILABLE:
            logger.error("AKShare不可用，无法获取板块数据")
            raise RuntimeError("AKShare不可用，无法获取板块数据")
        
        try:
            # 获取申万一级行业指数
            sw_index = ak.sw_index_spot()
            if sw_index.empty:
                raise RuntimeError("无法获取申万行业指数数据")
            
            # 选择涨幅前3的板块
            top_sectors = sw_index.sort_values('涨跌幅', ascending=False).head(3)
            
            sector_list = []
            for _, sector in top_sectors.iterrows():
                sector_list.append({
                    'name': sector['指数名称'],
                    'code': sector['指数代码'],
                    'change_pct': float(sector['涨跌幅']),
                    'latest_price': float(sector['最新价']),
                    'trend': 'up' if sector['涨跌幅'] > 0 else 'down'
                })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'top_sectors': sector_list,
                'data_source': 'akshare_sw_index',
                'update_frequency': '5min'
            }
        except Exception as e:
            logger.error(f"获取真实板块数据失败: {e}")
            raise RuntimeError(f"无法获取真实板块数据: {e}")

    async def _fetch_real_portfolio_alerts(self) -> Dict:
        """获取真实投资组合预警"""
        # 由于需要用户持仓数据，这里返回空，避免模拟数据
        # 实际应用中需要从用户持仓数据库获取
        return None

    async def _fetch_real_trading_signals(self) -> Dict:
        """获取真实交易信号"""
        # 基于真实技术分析生成信号，暂时返回空避免模拟数据
        # 实际应用中需要基于真实技术指标计算
        return None

    async def _fetch_real_system_status(self) -> Dict:
        """获取真实系统状态"""
        try:
            import psutil
            import os
            
            # 获取真实系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 检查数据源连接状态
            akshare_status = "connected" if AKSHARE_AVAILABLE else "disconnected"
            
            return {
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'api_server': 'running',
                    'data_feeds': akshare_status,
                    'websocket': 'active',
                    'system_resources': 'normal'
                },
                'resource_usage': {
                    'cpu_percent': round(cpu_percent, 1),
                    'memory_percent': round(memory.percent, 1),
                    'disk_percent': round((disk.used / disk.total) * 100, 1),
                    'active_connections': len(self.active_connections)
                },
                'data_sources': {
                    'akshare': akshare_status,
                    'market_data_updated': datetime.now().strftime('%H:%M:%S')
                }
            }
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            # 返回基本状态信息
            return {
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'api_server': 'running',
                    'data_feeds': 'unknown',
                    'websocket': 'active'
                },
                'active_connections': len(self.active_connections),
                'status': 'limited_monitoring'
            }

    async def _fetch_real_market_data(self) -> Dict:
        """获取真实市场数据 - 禁止使用模拟数据"""
        if not AKSHARE_AVAILABLE:
            logger.error("AKShare不可用，无法获取真实市场数据")
            raise RuntimeError("AKShare不可用，无法获取真实市场数据")
        
        try:
            # 获取今日数据
            today = datetime.now().strftime('%Y%m%d')
            
            # 获取上证综指数据
            sh_hist = ak.index_zh_a_hist(symbol='000001', period='daily', start_date=today, end_date=today)
            
            if sh_hist.empty:
                logger.error("无法获取上证综指今日数据")
                raise RuntimeError("无法获取上证综指今日数据，请检查交易日或数据源")
            
            latest_sh = sh_hist.iloc[-1]
            sh_close = float(latest_sh['收盘'])
            sh_open = float(latest_sh['开盘'])
            sh_change = sh_close - sh_open
            sh_change_pct = (sh_change / sh_open) * 100 if sh_open > 0 else 0.0
            
            sh_data = {
                'value': round(sh_close, 2),
                'change': round(sh_change, 2),
                'change_pct': round(sh_change_pct, 2)
            }
            logger.info(f"获取上证综指数据成功: {sh_close}")
            
            # 获取深证成指数据
            sz_hist = ak.index_zh_a_hist(symbol='399001', period='daily', start_date=today, end_date=today)
            
            if sz_hist.empty:
                logger.error("无法获取深证成指今日数据")
                raise RuntimeError("无法获取深证成指今日数据，请检查交易日或数据源")
            
            latest_sz = sz_hist.iloc[-1]
            sz_close = float(latest_sz['收盘'])
            sz_open = float(latest_sz['开盘'])
            sz_change = sz_close - sz_open
            sz_change_pct = (sz_change / sz_open) * 100 if sz_open > 0 else 0.0
            
            sz_data = {
                'value': round(sz_close, 2),
                'change': round(sz_change, 2),
                'change_pct': round(sz_change_pct, 2)
            }
            logger.info(f"获取深证成指数据成功: {sz_close}")
            
            # 获取热门股票数据 (可选，如果失败不影响主要指数)
            hot_stocks = []
            try:
                hot_stock_codes = ['000001', '600519', '300750']
                for stock_code in hot_stock_codes:
                    stock_hist = ak.stock_zh_a_hist(symbol=stock_code, period='daily', start_date=today, end_date=today, adjust='qfq')
                    if not stock_hist.empty:
                        latest_stock = stock_hist.iloc[-1]
                        stock_close = float(latest_stock['收盘'])
                        stock_open = float(latest_stock['开盘'])
                        stock_change_pct = ((stock_close - stock_open) / stock_open) * 100 if stock_open > 0 else 0.0
                        
                        stock_name_map = {
                            '000001': '平安银行',
                            '600519': '贵州茅台', 
                            '300750': '宁德时代'
                        }
                        
                        hot_stocks.append({
                            'code': stock_code,
                            'name': stock_name_map.get(stock_code, f'股票{stock_code}'),
                            'price': round(stock_close, 2),
                            'change_pct': round(stock_change_pct, 2)
                        })
            except Exception as e:
                logger.warning(f"获取热门股票数据失败: {e}")
                # 热门股票获取失败不影响主要功能
                hot_stocks = []
            
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'market_indices': {
                    'sh_composite': sh_data,
                    'sz_component': sz_data
                },
                'hot_stocks': hot_stocks
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"获取真实市场数据失败: {e}")
            raise RuntimeError(f"无法获取真实市场数据: {e}")
    
    
    def get_connection_stats(self) -> Dict:
        """获取连接统计信息"""
        subscription_stats = {
            sub_type: len(subscribers) 
            for sub_type, subscribers in self.subscriptions.items()
        }
        
        return {
            'total_connections': len(self.active_connections),
            'subscription_stats': subscription_stats,
            'is_running': self.is_running,
            'active_push_tasks': len([t for t in self.push_tasks.values() if not t.done()])
        }