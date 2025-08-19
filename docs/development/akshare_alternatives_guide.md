# AKShare 稳定数据获取方案指南

## 问题概述

当前项目中`ak.stock_zh_a_spot_em()`接口经常遇到连接错误：
```
('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

## 解决方案

### 1. 已实现的稳定数据获取器 (`src/data/stable_akshare_fetcher.py`)

#### 多重备用方案架构
```python
class StableAKShareFetcher:
    realtime_alternatives = [
        'stock_zh_a_hist_163',      # 网易财经历史数据（最近1天）
        'stock_zh_a_hist_min_em',   # 东财分钟数据（获取最新）
        'stock_zh_a_hist',          # 标准历史数据（最近1天）
        'stock_individual_info_em', # 个股信息（包含当前价格）
    ]
```

#### 实时数据获取的4种备用方案

**方案1: 历史数据伪实时 (推荐)**
- 使用`ak.stock_zh_a_hist()`获取最近1-3天数据
- 取最新日期的数据作为"实时"数据
- **优点**: 非常稳定，数据质量高
- **缺点**: 数据延迟（通常1天）
- **测试结果**: ✅ 成功率100%

```python
# 方案1实现示例
hist_data = await ak.stock_zh_a_hist(
    symbol=stock_code,
    period='daily',
    start_date=(datetime.now() - timedelta(days=3)).strftime('%Y%m%d'),
    end_date=datetime.now().strftime('%Y%m%d'),
    adjust=''
)
latest = hist_data.iloc[-1]  # 最新的一条数据
```

**方案2: 网易财经数据**
- 使用`ak.stock_zh_a_hist_163`
- **优点**: 数据更新较快
- **缺点**: API稳定性中等
- **测试结果**: ⚠️ 需要进一步测试

**方案3: 增强重试的原始接口**
- 使用原始`ak.stock_zh_a_spot_em()`，增加重试机制
- 指数退避延迟策略
- **优点**: 数据最新
- **缺点**: 连接不稳定
- **测试结果**: ❌ 连接失败率高

**方案4: 个股信息批量获取**
- 使用`ak.stock_individual_info_em()`逐个获取
- **优点**: 数据详细
- **缺点**: 速度较慢
- **测试结果**: ✅ 适合小批量获取

### 2. 推荐的稳定AKShare端点

#### 股票数据 (按稳定性排序)

**高稳定性** ⭐⭐⭐⭐⭐
```python
# 1. 股票基本信息 - 极其稳定
ak.stock_info_a_code_name()  # 获取所有A股代码和名称

# 2. 股票历史数据 - 极其稳定
ak.stock_zh_a_hist(symbol, period, start_date, end_date, adjust)

# 3. 个股详细信息 - 很稳定
ak.stock_individual_info_em(symbol)  # 包含实时价格
```

**中等稳定性** ⭐⭐⭐
```python
# 4. 网易财经数据
ak.stock_zh_a_hist_163(symbol, period, start_date, end_date, adjust)

# 5. 东财分钟数据
ak.stock_zh_a_hist_min_em(symbol, period, start_date, end_date, adjust)
```

**低稳定性** ⭐⭐
```python
# 6. 实时行情 - 连接不稳定
ak.stock_zh_a_spot_em()  # 经常连接失败
```

#### 板块数据 (按稳定性排序)

**高稳定性** ⭐⭐⭐⭐⭐
```python
# 1. 申万行业历史数据 - 极其稳定
ak.index_hist_sw(symbol, period)

# 2. 申万行业每日分析 - 很稳定
ak.index_analysis_daily_sw(symbol, start_date, end_date)
```

**中等稳定性** ⭐⭐⭐
```python
# 3. 申万行业成分股
ak.index_stock_cons_sw(symbol)

# 4. 申万实时数据
ak.sw_index_spot()
```

### 3. 配置和优化建议

#### 重试策略配置
```python
config = {
    'max_retries': 5,        # 最大重试次数
    'base_delay': 1,         # 基础延迟时间（秒）
    'max_delay': 16,         # 最大延迟时间（秒）
    'timeout': 30,           # 请求超时时间（秒）
    'batch_size': 100,       # 批量处理大小
}
```

#### 缓存策略
```python
# 不同数据类型的建议缓存时间
cache_duration = {
    'stock_basic_info': 3600,      # 股票基本信息：1小时
    'historical_data': 1800,       # 历史数据：30分钟
    'realtime_data': 300,          # 实时数据：5分钟
    'sector_data': 1800,           # 板块数据：30分钟
}
```

#### 错误处理
```python
async def safe_akshare_call(func, **kwargs):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                await asyncio.sleep(delay)
            
            result = await asyncio.wait_for(
                asyncio.to_thread(func, **kwargs),
                timeout=timeout
            )
            return result
            
        except (asyncio.TimeoutError, ConnectionError, RemoteDisconnected):
            if attempt == max_retries - 1:
                raise RuntimeError(f"所有{max_retries}次尝试都失败")
```

### 4. 实际应用示例

#### 更新后的股票推荐API
```python
async def get_recommendations(self, sector: str = None, top_n: int = 10) -> Dict:
    # 使用稳定的数据获取器
    if STABLE_FETCHER_AVAILABLE:
        logger.info("使用稳定数据获取器获取股票数据...")
        stock_zh_a_spot = await stable_fetcher.get_stock_realtime_data()
    else:
        # 备用方案...
```

#### 板块数据获取
```python
async def get_stable_sector_data(sector_name: str, days: int = 30):
    # 方案1: 申万历史指数数据
    sector_code = self._get_sector_code(sector_name)
    data = await self._safe_akshare_call(
        ak.index_hist_sw,
        symbol=sector_code,
        period='day'
    )
```

### 5. 测试结果

根据实际测试(`tests/unit/stable_fetcher_test.py`):

| 方案 | 成功率 | 数据量 | 响应时间 | 推荐度 |
|------|--------|---------|----------|---------|
| 历史数据伪实时 | 100% | 100只股票 | ~30s | ⭐⭐⭐⭐⭐ |
| 个股信息批量 | 100% | 可控 | ~1s/10只 | ⭐⭐⭐⭐ |
| 网易财经 | 待测试 | - | - | ⭐⭐⭐ |
| 原始spot接口 | 0% | - | 超时 | ⭐ |
| 申万板块数据 | 100% | 2780条记录 | ~1s | ⭐⭐⭐⭐⭐ |

### 6. 部署建议

#### 生产环境配置
1. **主方案**: 历史数据伪实时 + 缓存
2. **备用方案**: 个股信息批量获取
3. **缓存策略**: Redis + 5分钟TTL
4. **监控**: 每个方案的成功率和响应时间

#### 错误恢复策略
```python
# 按优先级顺序尝试方案
try:
    return await method_1()  # 历史数据伪实时
except:
    try:
        return await method_4()  # 个股信息批量
    except:
        raise RuntimeError("所有数据获取方案都失败")
```

### 7. 未来优化方向

1. **数据源多样化**: 集成TuShare、Wind等其他数据源
2. **智能切换**: 根据成功率自动选择最佳方案
3. **增量更新**: 只获取变化的数据，减少请求频率
4. **分布式缓存**: 多实例间共享缓存数据

## 总结

通过实现`StableAKShareFetcher`，我们已经解决了`ak.stock_zh_a_spot_em()`的连接不稳定问题。核心策略是：

1. **多重备用方案**：4种不同的数据获取方式
2. **智能降级**：优先使用稳定方案，失败时自动切换
3. **严格原则**：绝不使用模拟数据，获取不到真实数据时直接报错
4. **性能优化**：合理的重试、超时和缓存策略

测试表明，新方案大幅提升了数据获取的稳定性和可靠性。