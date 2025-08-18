# A股智能交易决策平台 (A-Share Intelligent Trading Decision Platform)

<div align="center">

**个人交易者的智能投资助手和反人性交易系统**

[![Version](https://img.shields.io/badge/version-v1.4.0-blue.svg)](https://github.com/your-repo/bajun)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![React](https://img.shields.io/badge/react-18+-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/fastapi-latest-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

</div>

## 📊 项目概述

A股智能交易决策平台是专为个人交易者设计的智能投资决策系统，结合了**智能投资顾问**和**反人性交易助手**双重功能，帮助个人投资者在A股市场中实现稳定盈利。

### 🎯 核心特色

- **个人投资者优势最大化**: 利用个人投资者的灵活性、速度和逆向思维优势
- **行为驱动交易**: 解决零售交易者亏损的根本原因——情绪化和无纪律交易
- **持续进化**: 基于用户行为和市场变化的自我改进系统
- **效果优先**: 优先考虑实际交易效果（准确率、收益率）而非UI/UX

## 🚀 快速开始

### 一键启动
```bash
# 1. 交互式启动菜单
./scripts/setup/quick_start.sh

# 2. 直接安装依赖
./scripts/setup/install_dependencies.sh

# 3. 启动服务
./scripts/deployment/start_services.sh

# 4. 访问应用
# 前端: http://localhost:3000
# API: http://localhost:8000/docs
```

详细启动指南请参考 [docs/development/QUICK_START.md](docs/development/QUICK_START.md)

## 🏗️ 系统架构

### 双核心架构
```
┌─────────────────────────────────────────────────────────────┐
│                个人交易决策平台                               │
├─────────────────────────┬───────────────────────────────────┤
│    智能投资顾问系统        │       反人性交易助手              │
│    (What & When)        │       (How & Discipline)         │
│    • 板块优先选择         │       • 情绪控制                 │
│    • 投资组合管理         │       • 纪律执行                 │
│    • 机会发现            │       • 行为分析                 │
│    • 多维度评分          │       • 习惯养成                 │
└─────────────────────────┴───────────────────────────────────┘
```

### 技术栈
- **后端**: FastAPI + SQLite + Redis
- **前端**: React + TypeScript + Ant Design
- **机器学习**: ONNX Runtime + scikit-learn + LightGBM
- **数据处理**: Polars (10x faster than pandas)
- **实时通信**: WebSocket + asyncio
- **缓存策略**: Redis + SQLite + Parquet files

## 📈 性能指标

| 指标 | 当前表现 | 基准比较 |
|------|----------|----------|
| 板块预测准确率 | 69.0% | +7.8% |
| 股票选择胜率 | 50.0% | +25.0% |
| 投资组合收益 | +0.31% | +126.1% |
| 算法整体提升 | +53.0% | 显著优化 |

## 💡 核心功能

### 🧠 智能投资顾问系统
- **板块优先选择**: 行业分析 → 个股筛选
- **投资组合管理**: 当前持仓分析和优化建议
- **机会发现**: 专注机构盲点（小盘股、主题轮动、逆向投资）
- **多维度评分**: 基本面 + 技术面 + 情绪面 + 机会面综合评分

### 🛡️ 反人性交易助手
- **情绪控制**: 通过技术约束防止冲动决策
- **纪律执行**: 自动化执行交易规则和风险管理
- **行为分析**: 个人交易模式识别和改进
- **习惯养成**: 游戏化系统培养良好交易习惯

### 📊 投资组合顾问
- **当前持仓分析**: 实时评估现有仓位
- **再平衡建议**: 基于市场条件和风险暴露
- **退出策略优化**: 动态止损和获利了结建议
- **未来投资规划**: 基于当前组合状态的下一步投资

## 🎮 使用示例

### 命令行模式
```bash
# 完整每日工作流
python src/main.py --daily-workflow

# 板块分析
python src/main.py --sector-analysis --lookback-months 6

# 股票筛选
python src/main.py --stock-selection --sectors "医药生物,计算机"

# 投资组合分析
python src/main.py --portfolio-analysis --holdings holdings.json

# 反人性交易检查
python src/main.py --trading-check --action BUY --stock 000001
```

### Web界面
- **仪表板**: 投资组合概览、市场监控、实时警报
- **分析视图**: 板块分析、股票筛选、组合优化
- **行为助手**: 情绪跟踪、纪律执行、习惯养成
- **设置配置**: 用户偏好、交易规则、风险参数

## 📁 项目结构

```
bajun/
├── src/                     # 源代码
│   ├── api/                 # FastAPI后端服务
│   ├── core/                # 核心业务引擎
│   ├── data/                # 数据获取模块
│   ├── analysis/            # 分析算法模块
│   └── validation/          # 数据验证模块
├── frontend/                # React前端应用
├── docs/                    # 项目文档
├── tests/                   # 测试代码
├── logs/                    # 日志文件
├── reports/                 # 分析报告
├── CLAUDE.md               # 系统架构和开发指南
├── QUICK_START.md          # 快速启动指南
└── start_services.sh       # 服务启动脚本
```

## 🧪 测试

### 功能测试
```bash
# 运行完整测试套件
./run_tests.sh

# 性能基准测试
python src/main.py --optimized-analysis

# 系统集成测试
python integration_test.py
```

### 质量检查
```bash
# 提交前检查
./pre_commit_check.sh

# 项目标准检查
./check_project_standards.sh
```

## 🔧 开发

### 环境要求
- Python 3.8+ (推荐3.10+)
- Node.js 16+ (推荐18+)
- 4GB+ 可用内存

### 开发模式
```bash
# 启动开发服务器
./dev.sh                                # 后端开发服务器
cd frontend && npm start                # 前端开发服务器

# 生产模式
python src/main.py --production
```

### Git工作流
系统配置了自动化Git钩子，确保每次提交都经过质量检查：
- 功能测试（11项检查）
- 项目标准检查（35项检查）
- 代码质量验证
- 性能基准保护

## 📖 文档

- [系统架构和开发指南](CLAUDE.md)
- [快速启动指南](QUICK_START.md)
- [项目需求文档](docs/user_input/requirement.md)
- [测试标准](docs/testing/test_standards.md)
- [项目标准规范](docs/development/project_standards.md)
- [完备迭代体系](docs/development/complete_iterative_system.md)

## 🎯 发展路线图

### Phase 1: MVP ✅
- 三步交易决策流水线
- 真实数据验证
- 基础反人性功能

### Phase 2: Beta ✅
- 机器学习模型优化
- 实时数据推送
- 历史性能验证

### Phase 3: Production ✅
- 完整全栈系统
- React前端界面
- 生产级部署

### Phase 4: 高级功能 🚧
- 高频数据处理
- 移动端适配
- 云端部署

## 🤝 贡献

欢迎贡献代码、报告bug或提出功能建议：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源。详情请参阅 [LICENSE](LICENSE) 文件。

## 📞 支持

如果您遇到问题或需要支持：

1. 查看 [QUICK_START.md](QUICK_START.md) 快速启动指南
2. 运行 `./run_tests.sh` 进行系统诊断
3. 检查 `logs/` 目录下的日志文件
4. 参考 [故障排除文档](docs/troubleshooting.md)

---

<div align="center">

**让智能辅助和纪律约束成为您在A股市场的致胜法宝**

[开始使用](QUICK_START.md) • [查看文档](CLAUDE.md) • [了解架构](docs/)

</div>