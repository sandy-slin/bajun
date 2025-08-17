# A股智能交易决策平台 - 前端

基于React + TypeScript + Ant Design的现代化前端应用，为A股智能交易决策平台提供用户界面。

## 技术栈

- **React 18.2.0** - 用户界面框架
- **TypeScript** - 类型安全的JavaScript
- **Ant Design 5.6.3** - UI组件库
- **React Router Dom 6.12.1** - 路由管理
- **Axios** - HTTP客户端
- **Recharts 2.7.2** - 图表组件
- **Socket.io-client 4.7.1** - WebSocket连接
- **Day.js** - 日期处理库

## 项目结构

```
frontend/
├── public/                 # 静态资源
│   ├── index.html         # HTML模板
│   └── manifest.json      # PWA配置
├── src/
│   ├── components/        # 可复用组件
│   │   ├── Layout/       # 布局组件
│   │   └── Dashboard/    # 控制台组件
│   ├── contexts/         # React Context
│   │   ├── ApiContext.tsx      # API状态管理
│   │   └── WebSocketContext.tsx # WebSocket连接管理
│   ├── pages/            # 页面组件
│   │   ├── Dashboard.tsx        # 控制台页面
│   │   ├── SectorAnalysis.tsx   # 板块分析页面
│   │   ├── StockSelection.tsx   # 股票筛选页面
│   │   ├── Portfolio.tsx        # 投资组合页面
│   │   ├── TradingAssistant.tsx # 交易助手页面
│   │   └── Settings.tsx         # 系统设置页面
│   ├── services/         # 服务层
│   │   ├── api.ts        # REST API接口
│   │   └── websocket.ts  # WebSocket服务
│   ├── App.tsx           # 主应用组件
│   ├── App.css           # 应用样式
│   ├── index.tsx         # 入口文件
│   └── index.css         # 全局样式
├── package.json          # 项目配置
├── tsconfig.json         # TypeScript配置
└── README.md            # 项目说明
```

## 主要功能

### 1. 控制台 (Dashboard)
- 系统性能指标展示
- 实时市场数据
- 算法优化状态
- 快速操作入口

### 2. 板块分析 (Sector Analysis)
- TOP板块评分和排名
- 板块投资逻辑分析
- 历史性能验证
- 实时板块数据更新

### 3. 股票筛选 (Stock Selection)
- 智能股票筛选
- 个股详细分析
- 批量股票分析
- 选股性能验证

### 4. 投资组合 (Portfolio)
- 组合分析和优化
- 风险评估
- 回测分析
- 预设组合管理

### 5. 交易助手 (Trading Assistant)
- 反人性交易检查
- 情绪控制建议
- 交易规则管理
- 交易历史分析

### 6. 系统设置 (Settings)
- 算法参数配置
- 系统偏好设置
- 连接配置

## WebSocket实时数据

应用支持多种实时数据流：

- **market_data** - 市场行情数据
- **sector_updates** - 板块更新
- **portfolio_alerts** - 组合预警
- **trading_signals** - 交易信号
- **system_status** - 系统状态

## 开发命令

```bash
# 安装依赖
npm install

# 启动开发服务器
npm start

# 构建生产版本
npm run build

# 运行测试
npm test

# 代码检查
npm run lint

# 类型检查
npm run type-check
```

## 环境配置

创建 `.env` 文件配置环境变量：

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## API集成

前端通过以下方式与后端集成：

1. **REST API** - 使用axios进行HTTP请求
2. **WebSocket** - 使用原生WebSocket进行实时通信
3. **错误处理** - 统一的错误处理和用户提示
4. **加载状态** - 全局加载状态管理

## 响应式设计

- 支持桌面端 (1200px+)
- 支持平板端 (768px-1200px)
- 支持移动端 (<768px)
- 自适应布局和组件

## 性能优化

- React.memo优化组件渲染
- 懒加载路由组件
- 图片懒加载
- Bundle分割优化

## 部署说明

1. 构建生产版本：`npm run build`
2. 将build文件夹部署到Web服务器
3. 配置API代理或CORS
4. 确保WebSocket连接正常

## 浏览器支持

- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## 开发规范

- 使用TypeScript严格模式
- 遵循React Hooks最佳实践
- 使用Ant Design设计规范
- 组件化开发模式
- 统一的错误处理
- 完善的类型定义