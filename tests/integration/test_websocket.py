#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket连接测试脚本
验证前后端WebSocket通信是否正常
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime

async def test_websocket_connection():
    """测试WebSocket连接和订阅功能"""
    uri = "ws://localhost:8000/ws/realtime"
    
    try:
        print("🔗 连接WebSocket服务器...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送订阅请求
            subscriptions = ['market_data', 'sector_updates', 'system_status']
            
            for subscription in subscriptions:
                subscribe_message = {
                    "type": "subscribe",
                    "subscription": subscription
                }
                
                await websocket.send(json.dumps(subscribe_message))
                print(f"📡 已订阅: {subscription}")
            
            # 监听消息
            print("\n🎧 开始监听实时数据 (按Ctrl+C停止)...")
            message_count = 0
            
            while message_count < 20:  # 接收20条消息后停止
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    
                    message_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"[{timestamp}] 收到消息 #{message_count}:")
                    print(f"  类型: {data.get('type', 'unknown')}")
                    
                    if data.get('subscription'):
                        print(f"  订阅: {data['subscription']}")
                    
                    if data.get('message'):
                        print(f"  内容: {data['message']}")
                    
                    print("")
                    
                except asyncio.TimeoutError:
                    print("⏰ 接收超时，继续等待...")
                    
            print("✅ WebSocket测试完成")
                    
    except ConnectionRefusedError:
        print("❌ 连接被拒绝，请确保后端服务正在运行")
        print("💡 提示: 运行 'python src/api/main.py' 启动后端服务")
        return False
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")
        return False
    
    return True

async def test_invalid_subscription():
    """测试无效订阅的错误处理"""
    uri = "ws://localhost:8000/ws/realtime"
    
    try:
        async with websockets.connect(uri) as websocket:
            # 等待欢迎消息
            welcome = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            
            # 发送无效订阅
            invalid_message = {
                "type": "subscribe",
                "subscription": "invalid_subscription_type"
            }
            
            await websocket.send(json.dumps(invalid_message))
            
            # 等待响应
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data.get('type') == 'error':
                print("✅ 错误处理测试通过: 无效订阅被正确拒绝")
                return True
            else:
                print(f"⚠️  错误处理测试: 收到消息类型 '{data.get('type')}' (期望 'error')")
                print(f"   消息内容: {data.get('message', 'N/A')}")
                return False
                
    except asyncio.TimeoutError:
        print("⚠️  错误处理测试: 服务器响应超时")
        return False
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 WebSocket功能测试")
    print("=" * 50)
    
    # 运行测试
    try:
        # 基本连接测试
        success = asyncio.run(test_websocket_connection())
        
        if success:
            print("\n🔍 测试错误处理...")
            asyncio.run(test_invalid_subscription())
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 WebSocket集成测试完成")
            print("💡 前端现在可以连接并接收实时数据")
        else:
            print("❌ WebSocket测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        sys.exit(1)