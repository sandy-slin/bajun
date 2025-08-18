#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能交易决策平台 - 增强版集成测试
改进的测试标准和流程，修复AsyncIO问题
"""

import asyncio
import aiohttp
import websockets
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000/ws/realtime"
        self.test_results = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def print_header(self, title: str):
        """打印测试标题"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print('='*60)
    
    def print_test(self, test_name: str, success: bool, details: str = ""):
        """打印测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results[test_name] = success
    
    async def test_api_health(self) -> bool:
        """测试API健康状态"""
        self.print_header("API健康状态测试")
        
        try:
            # 1. 基本健康检查
            async with self.session.get(f"{self.base_url}/health") as response:
                success = response.status == 200
                self.print_test("基本健康检查", success, f"状态码: {response.status}")
                
                if not success:
                    return False
            
            # 2. 系统信息接口
            async with self.session.get(f"{self.base_url}/api/v1/system/info") as response:
                success = response.status == 200
                self.print_test("系统信息接口", success, f"状态码: {response.status}")
                
                if success:
                    data = await response.json()
                    self.print_test("系统信息数据格式", 
                                    'success' in data and 'data' in data,
                                    f"包含必要字段: {list(data.keys())}")
            
            return True
            
        except Exception as e:
            self.print_test("健康检查异常", False, f"错误: {str(e)}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """测试核心API端点"""
        self.print_header("核心API端点测试")
        
        endpoints = [
            ("/api/v1/sectors/?top_n=3", "板块分析API"),
            ("/api/v1/stocks/", "股票列表API"),
            ("/api/v1/portfolio/preset", "投资组合预设API"),
            ("/api/v1/trading/rules", "交易规则API"),
            ("/api/v1/sectors/performance/validation", "板块性能验证API"),
            ("/api/v1/stocks/performance/validation", "股票性能验证API")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    success = response.status == 200
                    self.print_test(name, success, f"状态码: {response.status}")
                    
                    if success:
                        data = await response.json()
                        has_data = 'success' in data or 'data' in data
                        self.print_test(f"{name}数据格式", has_data, 
                                        f"响应结构: {type(data)}")
                    else:
                        all_passed = False
                        
            except Exception as e:
                self.print_test(f"{name}异常", False, f"错误: {str(e)}")
                all_passed = False
        
        return all_passed
    
    async def test_websocket_enhanced(self) -> bool:
        """增强版WebSocket测试"""
        self.print_header("WebSocket连接测试")
        
        try:
            # 1. 基本连接测试
            async with websockets.connect(self.ws_url, ping_interval=None) as websocket:
                self.print_test("WebSocket连接", True, "连接建立成功")
                
                # 2. 订阅测试
                subscriptions = ['market_data', 'system_status']
                subscription_confirmed = 0
                
                for subscription in subscriptions:
                    subscribe_message = {
                        "type": "subscribe",
                        "subscription": subscription
                    }
                    
                    await websocket.send(json.dumps(subscribe_message))
                    
                    # 等待确认消息
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(response)
                        
                        if data.get('type') == 'subscription_confirmed':
                            subscription_confirmed += 1
                            self.print_test(f"订阅确认: {subscription}", True)
                        
                    except asyncio.TimeoutError:
                        self.print_test(f"订阅确认: {subscription}", False, "超时")
                
                # 3. 数据接收测试
                data_received = 0
                for _ in range(5):  # 尝试接收5条消息
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                        data = json.loads(message)
                        
                        if data.get('type') == 'data_update':
                            data_received += 1
                            self.print_test(f"数据推送: {data.get('subscription', 'unknown')}", True)
                        
                    except asyncio.TimeoutError:
                        break
                
                # 4. 错误处理测试
                try:
                    error_message = {
                        "type": "subscribe",
                        "subscription": "invalid_subscription"
                    }
                    await websocket.send(json.dumps(error_message))
                    
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(response)
                    
                    error_handled = data.get('type') == 'error'
                    self.print_test("错误处理", error_handled, 
                                    f"返回类型: {data.get('type', 'unknown')}")
                    
                except asyncio.TimeoutError:
                    self.print_test("错误处理", False, "没有返回错误响应")
                
                # 总体评估
                self.print_test("实时数据接收", data_received >= 2, 
                                f"收到{data_received}条消息")
                
                return subscription_confirmed >= 1 and data_received >= 1
                
        except Exception as e:
            self.print_test("WebSocket测试异常", False, f"错误: {str(e)}")
            return False
    
    async def test_performance_validation(self) -> bool:
        """测试性能指标验证"""
        self.print_header("算法性能验证测试")
        
        try:
            # 1. 板块性能验证
            async with self.session.get(f"{self.base_url}/api/v1/sectors/performance/validation") as response:
                if response.status == 200:
                    data = await response.json()
                    accuracy = data.get('data', {}).get('sector_prediction_accuracy', 0)
                    
                    self.print_test("板块分析性能验证", True, f"状态码: {response.status}")
                    self.print_test("板块预测准确率", accuracy >= 65.0, 
                                    f"准确率: {accuracy}%")
                else:
                    self.print_test("板块分析性能验证", False, f"状态码: {response.status}")
                    return False
            
            # 2. 股票性能验证
            async with self.session.get(f"{self.base_url}/api/v1/stocks/performance/validation") as response:
                if response.status == 200:
                    data = await response.json()
                    win_rate = data.get('data', {}).get('stock_selection_win_rate', 0)
                    
                    self.print_test("股票选择性能验证", True, f"状态码: {response.status}")
                    self.print_test("股票选择胜率", win_rate >= 45.0, 
                                    f"胜率: {win_rate}%")
                else:
                    self.print_test("股票选择性能验证", False, f"状态码: {response.status}")
                    return False
            
            return True
            
        except Exception as e:
            self.print_test("性能验证异常", False, f"错误: {str(e)}")
            return False
    
    async def test_data_integrity(self) -> bool:
        """测试数据完整性"""
        self.print_header("数据完整性测试")
        
        try:
            # 1. 板块数据完整性
            async with self.session.get(f"{self.base_url}/api/v1/sectors/list") as response:
                if response.status == 200:
                    data = await response.json()
                    sectors = data.get('data', {}).get('sectors', [])
                    
                    self.print_test("板块数据获取", True, f"状态码: {response.status}")
                    self.print_test("板块数据完整性", len(sectors) >= 8, 
                                    f"获取到{len(sectors)}个板块")
                else:
                    self.print_test("板块数据获取", False, f"状态码: {response.status}")
                    return False
            
            # 2. 投资组合数据完整性
            async with self.session.get(f"{self.base_url}/api/v1/portfolio/preset") as response:
                if response.status == 200:
                    data = await response.json()
                    holdings = data.get('data', {}).get('preset_holdings', [])
                    
                    self.print_test("投资组合数据获取", True, f"状态码: {response.status}")
                    self.print_test("投资组合数据完整性", len(holdings) >= 10, 
                                    f"包含{len(holdings)}只股票")
                    
                    # 验证权重分布
                    total_weight = sum(h.get('weight', 0) for h in holdings)
                    self.print_test("权重分布合理性", 0.8 <= total_weight <= 1.2, 
                                    f"总权重: {total_weight:.2f}")
                else:
                    self.print_test("投资组合数据获取", False, f"状态码: {response.status}")
                    return False
            
            return True
            
        except Exception as e:
            self.print_test("数据完整性异常", False, f"错误: {str(e)}")
            return False
    
    def calculate_score(self) -> int:
        """计算测试总分"""
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        if total == 0:
            return 0
        
        return int((passed / total) * 100)
    
    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试结果总结")
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        score = self.calculate_score()
        
        print(f"📊 测试统计:")
        print(f"   总测试数: {total}")
        print(f"   通过: {passed}")
        print(f"   失败: {total - passed}")
        print(f"   成功率: {passed/total*100:.1f}%")
        print(f"   综合得分: {score}/100")
        
        if score >= 95:
            print("\n🏆 测试结果: 卓越 (≥95分)")
            print("🎉 系统运行完美，生产就绪！")
        elif score >= 90:
            print("\n🥇 测试结果: 优秀 (≥90分)")
            print("✅ 系统运行良好，可以部署")
        elif score >= 80:
            print("\n🥈 测试结果: 良好 (≥80分)")
            print("⚠️  系统基本正常，建议优化")
        elif score >= 70:
            print("\n🥉 测试结果: 及格 (≥70分)")
            print("⚠️  系统有问题，需要修复")
        else:
            print("\n❌ 测试结果: 不及格 (<70分)")
            print("🔧 系统存在严重问题，必须修复")
        
        # 显示失败的测试
        failed_tests = [name for name, result in self.test_results.items() if not result]
        if failed_tests:
            print(f"\n❌ 失败的测试:")
            for test in failed_tests:
                print(f"   - {test}")
        
        return score >= 90

async def main():
    """主测试函数"""
    print("🚀 A股智能交易决策平台 - 增强版系统测试")
    print(f"⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async with EnhancedTester() as tester:
        try:
            # 运行所有测试
            await tester.test_api_health()
            await tester.test_api_endpoints()
            await tester.test_websocket_enhanced()
            await tester.test_performance_validation()
            await tester.test_data_integrity()
            
            # 打印总结
            success = tester.print_summary()
            
            print(f"\n⏰ 测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return success
            
        except Exception as e:
            print(f"\n❌ 测试执行异常: {str(e)}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试运行失败: {str(e)}")
        sys.exit(1)