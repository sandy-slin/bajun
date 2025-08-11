"""
测试数据fixtures
提供各种测试场景的样本数据
"""

from datetime import datetime, timedelta


def get_sample_trading_data(days: int = 30, stock_code: str = '000001') -> list:
    """生成样本交易数据"""
    data = []
    start_date = datetime.now() - timedelta(days=days)
    base_price = 10.0
    
    for i in range(days):
        date = start_date + timedelta(days=i)
        if date.weekday() < 5:  # 工作日
            price = base_price + (i % 10) * 0.1
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'code': stock_code,
                'open': price,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price * 1.01,
                'volume': 100000 + (i % 100) * 1000
            })
    
    return data


def get_sample_market_events(days: int = 30) -> list:
    """生成样本市场事件"""
    events = []
    start_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        date = start_date + timedelta(days=i)
        events.append({
            'date': date.strftime('%Y-%m-%d'),
            'title': f'市场事件 {i+1}',
            'description': f'这是第{i+1}个重要市场事件的描述',
            'impact': 'high' if i % 5 == 0 else 'medium',
            'category': ['policy', 'economic', 'corporate'][i % 3]
        })
    
    return events


def get_sample_llm_response() -> dict:
    """生成样本LLM分析响应"""
    return {
        'events_analysis': """
        过去一个月市场主要事件分析：
        1. 政策层面：央行货币政策保持稳健
        2. 行业动态：科技板块表现活跃
        3. 公司公告：多家上市公司发布业绩预告
        
        未来一个月预期：
        1. 宏观经济数据发布
        2. 季度财报密集披露期
        3. 政策会议可能带来新变化
        """,
        'trend_analysis': """
        技术面分析：
        1. 价格趋势：呈现震荡上行格局
        2. 交易量：成交量温和放大
        3. 关键位置：支撑位在9.8元，阻力位在10.5元
        4. 技术指标：MACD金叉，RSI处于中性区域
        """,
        'trading_advice': """
        未来三天交易建议：
        1. 操作策略：逢低适量买入，控制仓位
        2. 风险控制：设置止损位在9.5元
        3. 关注要点：关注成交量变化和政策消息
        4. 时机把握：建议分批建仓，避免追高
        
        风险提示：股市有风险，投资需谨慎
        """
    }


def get_sample_cache_data() -> dict:
    """生成样本缓存数据"""
    return {
        'timestamp': datetime.now().isoformat(),
        'data': get_sample_trading_data(7)
    }


# 测试用的API响应模拟
MOCK_API_RESPONSES = {
    'trading_data_success': {
        'status': 'success',
        'data': get_sample_trading_data(30)
    },
    'trading_data_error': {
        'status': 'error',
        'message': 'API请求失败'
    },
    'llm_response_success': {
        'choices': [{
            'message': {
                'content': '这是一个模拟的LLM分析结果，包含了对股票走势的专业分析。'
            }
        }]
    },
    'llm_response_error': {
        'error': {
            'message': 'API quota exceeded',
            'type': 'quota_exceeded'
        }
    }
}


def get_mock_response(response_type: str):
    """获取模拟API响应"""
    return MOCK_API_RESPONSES.get(response_type, {})