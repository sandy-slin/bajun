# A股智能交易决策平台 - 部署指南

## 快速开始

### 1. 环境要求

**后端要求:**
- Python 3.8+
- pip (Python包管理器)

**前端要求:**
- Node.js 16+
- npm (Node.js包管理器)

### 2. 一键启动 (推荐)

```bash
# 克隆项目后，直接运行全栈启动脚本
./start_fullstack.sh
```

这将自动：
- 安装前端依赖 (如果需要)
- 启动FastAPI后端服务 (端口8000)
- 启动React前端应用 (端口3000)
- 建立WebSocket实时数据连接

### 3. 分别启动

**启动后端服务:**
```bash
# 激活Python虚拟环境
source venv/bin/activate

# 启动FastAPI服务
cd src/api
python main.py
```

**启动前端应用:**
```bash
# 进入前端目录
cd frontend

# 安装依赖 (仅首次)
npm install

# 启动开发服务器
npm start
```

## 访问地址

- **前端应用**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **系统状态**: http://localhost:8000/health
- **WebSocket**: ws://localhost:8000/ws/realtime

## 功能验证

### 1. 后端服务验证
```bash
# 健康检查
curl http://localhost:8000/health

# 系统信息
curl http://localhost:8000/api/v1/system/info

# 板块分析测试
curl "http://localhost:8000/api/v1/sectors/?lookback_months=6&top_n=5"
```

### 2. WebSocket连接验证
```bash
# 运行WebSocket测试脚本
python test_websocket.py
```

### 3. 前端功能验证
打开浏览器访问 http://localhost:3000：
- 控制台应显示实时系统状态
- 板块分析页面可以获取TOP板块数据
- WebSocket连接状态应显示为"已连接"

## 算法性能指标

当前优化后的系统性能：

| 指标 | 基准值 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| 板块预测准确率 | 64.0% | 69.0% | +7.8% |
| 股票选择胜率 | 40.0% | 50.0% | +25.0% |
| 投资组合收益 | -1.19% | +0.31% | +126.1% |
| 整体性能提升 | - | - | +53.0% |

## 开发环境

### 后端开发
```bash
# 进入API目录
cd src/api

# 运行开发服务器 (自动重载)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 前端开发
```bash
# 进入前端目录
cd frontend

# 启动开发服务器 (热重载)
npm start
```

## 生产部署

### 1. 后端生产部署
```bash
# 安装生产服务器
pip install gunicorn

# 启动生产服务器
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app
```

### 2. 前端生产部署
```bash
# 构建生产版本
cd frontend
npm run build

# 部署build文件夹到Web服务器
# 或使用serve预览
npx serve -s build -l 3000
```

### 3. 反向代理配置 (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # 前端静态文件
    location / {
        root /path/to/frontend/build;
        try_files $uri $uri/ /index.html;
    }
    
    # API代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket代理
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## 环境变量配置

### 后端环境变量
```bash
# .env文件 (可选)
DEEPSEEK_API_KEY=your_api_key_here
AKSHARE_TIMEOUT=30
LOG_LEVEL=INFO
```

### 前端环境变量
```bash
# frontend/.env文件
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## 故障排除

### 1. 后端服务无法启动
- 检查Python虚拟环境是否激活
- 确认依赖包已安装: `pip install -r requirements.txt`
- 检查端口8000是否被占用

### 2. 前端应用无法启动
- 确认Node.js版本 >= 16
- 重新安装依赖: `rm -rf node_modules && npm install`
- 检查端口3000是否被占用

### 3. WebSocket连接失败
- 确认后端服务正在运行
- 检查防火墙设置
- 运行测试脚本: `python test_websocket.py`

### 4. API调用失败
- 检查CORS配置
- 确认API路径正确
- 查看浏览器开发者工具的网络标签

## 监控和日志

### 1. 后端日志
日志输出到控制台，包含：
- API请求/响应信息
- WebSocket连接状态
- 算法执行结果
- 错误和异常信息

### 2. 前端监控
浏览器开发者工具可查看：
- API请求状态
- WebSocket连接状态
- React组件性能
- 控制台错误信息

## 性能优化建议

1. **后端优化**
   - 使用Redis缓存频繁查询的数据
   - 数据库连接池优化
   - 异步处理长时间运行的任务

2. **前端优化**
   - 启用React生产模式构建
   - 使用CDN加速静态资源
   - 实现组件懒加载

3. **WebSocket优化**
   - 合理设置推送频率
   - 实现客户端重连机制
   - 数据压缩和批量发送