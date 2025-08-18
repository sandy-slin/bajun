#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能交易决策平台 - 系统集成测试
验证整个系统的端到端功能
"""

import asyncio
import requests
import websockets
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

class IntegrationTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000/ws/realtime"
        self.test_results = {}
        
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
    
    async def test_backend_health(self) -> bool:
        """测试后端健康状态"""
        self.print_header("后端服务健康检查")
        
        try:
            # 1. 基本健康检查
            response = requests.get(f"{self.base_url}/health", timeout=5)
            success = response.status_code == 200
            self.print_test("基本健康检查", success, f"状态码: {response.status_code}")
            
            if not success:
                return False
            
            # 2. 系统信息检查
            response = requests.get(f"{self.base_url}/api/v1/system/info", timeout=5)
            success = response.status_code == 200
            self.print_test("系统信息接口", success, f"状态码: {response.status_code}")
            
            return success
            
        except requests.exceptions.ConnectionError:
            self.print_test("后端连接", False, "无法连接到后端服务，请确保服务正在运行")
            return False
        except Exception as e:
            self.print_test("后端健康检查", False, f"异常: {str(e)}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """测试API端点功能"""
        self.print_header("API端点功能测试")
        
        all_success = True
        
        # 测试板块分析API
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/sectors/",
                params={"lookback_months": 6, "top_n": 3},
                timeout=10
            )
            success = response.status_code == 200
            self.print_test("板块分析API", success, f"状态码: {response.status_code}")
            all_success &= success
            
            if success:
                data = response.json()
                success = data.get('success', False)
                self.print_test("板块分析数据格式", success, f"返回数据: {type(data.get('data'))}")
                all_success &= success
        except Exception as e:
            self.print_test("板块分析API", False, f"异常: {str(e)}")
            all_success = False
        
        # 测试股票列表API
        try:
            response = requests.get(f"{self.base_url}/api/v1/stocks/", timeout=10)
            success = response.status_code == 200
            self.print_test("股票列表API", success, f"状态码: {response.status_code}")
            all_success &= success
        except Exception as e:
            self.print_test("股票列表API", False, f"异常: {str(e)}")
            all_success = False
        
        # 测试投资组合预设API
        try:
            response = requests.get(f"{self.base_url}/api/v1/portfolio/preset", timeout=10)
            success = response.status_code == 200
            self.print_test("投资组合预设API", success, f"状态码: {response.status_code}")
            all_success &= success
        except Exception as e:
            self.print_test("投资组合预设API", False, f"异常: {str(e)}")
            all_success = False
        
        # 测试交易规则API
        try:
            response = requests.get(f"{self.base_url}/api/v1/trading/rules", timeout=10)
            success = response.status_code == 200
            self.print_test("交易规则API", success, f"状态码: {response.status_code}")
            all_success &= success
        except Exception as e:
            self.print_test("交易规则API", False, f"异常: {str(e)}")
            all_success = False
        
        return all_success
    
    async def test_websocket_connection(self) -> bool:
        """测试WebSocket连接和实时数据推送"""
        self.print_header("WebSocket实时数据测试")
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.print_test("WebSocket连接", True, "连接建立成功")
                
                # 测试订阅功能
                subscriptions = ['market_data', 'system_status']
                for subscription in subscriptions:
                    subscribe_msg = {
                        "type": "subscribe",
                        "subscription": subscription
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                
                # 等待并接收消息
                messages_received = 0
                test_duration = 15  # 测试15秒
                start_time = time.time()
                
                while time.time() - start_time < test_duration and messages_received < 5:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        messages_received += 1
                        
                        if data.get('type') == 'subscription_confirmed':
                            self.print_test(f"订阅确认: {data.get('subscription')}", True)
                        elif data.get('type') == 'data_update':
                            self.print_test(f"数据推送: {data.get('subscription')}", True)
                    
                    except asyncio.TimeoutError:
                        continue
                
                success = messages_received > 0
                self.print_test("实时数据接收", success, f"收到{messages_received}条消息")
                return success
                
        except Exception as e:
            self.print_test("WebSocket连接", False, f"异常: {str(e)}")
            return False
    
    async def test_algorithm_performance(self) -> bool:
        """测试算法性能验证"""
        self.print_header("算法性能验证测试")
        
        try:
            # 板块分析性能验证
            response = requests.get(
                f"{self.base_url}/api/v1/sectors/performance/validation", 
                timeout=30
            )
            success = response.status_code == 200
            self.print_test("板块分析性能验证", success, f"状态码: {response.status_code}")
            
            if success:
                data = response.json()
                validation_data = data.get('data', {})
                accuracy = validation_data.get('performance_metrics', {}).get('average_accuracy', 0)
                self.print_test("板块预测准确率", accuracy >= 0.65, f"准确率: {accuracy:.1%}")
            
            # 股票选择性能验证
            response = requests.get(
                f"{self.base_url}/api/v1/stocks/performance/validation",
                timeout=30
            )
            success = response.status_code == 200
            self.print_test("股票选择性能验证", success, f"状态码: {response.status_code}")
            
            if success:
                data = response.json()
                validation_data = data.get('data', {})
                win_rate = validation_data.get('performance_metrics', {}).get('current_win_rate', 0)
                self.print_test("股票选择胜率", win_rate >= 0.45, f"胜率: {win_rate:.1%}")
            
            return True
            
        except Exception as e:
            self.print_test("算法性能验证", False, f"异常: {str(e)}")
            return False
    
    async def test_data_integration(self) -> bool:
        """测试数据集成功能"""
        self.print_header("数据集成测试")
        
        try:
            # 测试板块数据获取
            response = requests.get(f"{self.base_url}/api/v1/sectors/list", timeout=10)
            success = response.status_code == 200
            self.print_test("板块数据获取", success)
            
            if success:
                data = response.json()
                sectors = data.get('data', {}).get('sectors', [])
                self.print_test("板块数据完整性", len(sectors) > 0, f"获取到{len(sectors)}个板块")
            
            return success
            
        except Exception as e:
            self.print_test("数据集成", False, f"异常: {str(e)}")
            return False
    
    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试结果总结")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"📊 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for test_name, result in self.test_results.items():
                if not result:
                    print(f"   - {test_name}")
        
        overall_success = failed_tests == 0
        status = "🎉 全部通过" if overall_success else "⚠️  存在失败"
        print(f"\n{status}")
        
        return overall_success

async def main():
    """主测试函数"""
    print("🚀 A股智能交易决策平台 - 系统集成测试")
    print(f"⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = IntegrationTester()
    
    # 运行所有测试
    tests = [
        tester.test_backend_health(),
        tester.test_api_endpoints(),
        tester.test_websocket_connection(),
        tester.test_algorithm_performance(),
        tester.test_data_integration(),
    ]
    
    # 并发执行部分测试，串行执行其他测试
    try:
        # 先测试基础服务
        if not await tester.test_backend_health():
            print("❌ 后端服务不可用，跳过其他测试")
            tester.print_summary()
            return False
        
        # 测试API功能
        await tester.test_api_endpoints()
        
        # 测试WebSocket
        await tester.test_websocket_connection()
        
        # 测试算法性能
        await tester.test_algorithm_performance()
        
        # 测试数据集成
        await tester.test_data_integration()
        
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 测试执行异常: {e}")
        return False
    finally:
        # 打印测试总结
        overall_success = tester.print_summary()
        
        print(f"\n⏰ 测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if overall_success:
            print("\n🎉 系统集成测试全部通过！")
            print("💡 系统已准备就绪，可以进行部署")
        else:
            print("\n⚠️  系统存在问题，需要修复后再次测试")
        
        return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)