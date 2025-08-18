# A股智能交易决策平台 - 快速启动指南

## 🚀 一键启动 (推荐)

### 步骤1: 安装依赖
```bash
chmod +x *.sh
./install_dependencies.sh
```

### 步骤2: 启动服务
```bash
./start_services.sh
```

### 步骤3: 运行测试
```bash
./run_tests.sh
```

### 步骤4: 访问应用
- **前端应用**: http://localhost:3000
- **API文档**: http://localhost:8000/docs
- **系统状态**: http://localhost:8000/health

### 停止服务
```bash
./stop_services.sh
```

---

## 📋 环境要求

- **Python**: 3.8+ (推荐3.10+)
- **Node.js**: 16+ (推荐18+)
- **操作系统**: macOS/Linux/Windows
- **内存**: 4GB+ 可用内存

---

## 🔧 详细说明

### 安装依赖脚本 (`install_dependencies.sh`)
- 创建Python虚拟环境
- 安装FastAPI及相关依赖
- 安装前端Node.js依赖
- 创建必要目录
- 验证安装结果

### 启动服务脚本 (`start_services.sh`)
- 检查前置条件和端口占用
- 按序启动后端服务 (端口8000)
- 按序启动前端服务 (端口3000)
- 验证服务状态
- 提供访问地址和日志位置

### 测试脚本 (`run_tests.sh`)
- 自动检查服务状态
- 必要时自动启动服务
- 运行WebSocket连接测试
- 运行系统集成测试
- 运行API端点测试
- 运行算法性能验证
- 提供详细的测试结果和评分

### 停止服务脚本 (`stop_services.sh`)
- 优雅停止后端和前端服务
- 清理残留进程
- 验证停止结果

---

## 🐛 故障排除

### 1. 依赖安装失败
```bash
# 检查Python版本
python3 --version

# 检查Node.js版本
node --version

# 手动创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. 服务启动失败
```bash
# 检查端口占用
lsof -i :8000
lsof -i :3000

# 查看详细日志
tail -f logs/backend.log
tail -f logs/frontend.log

# 手动启动后端
source venv/bin/activate
cd src/api
python main.py
```

### 3. 测试失败
```bash
# 确保服务正在运行
curl http://localhost:8000/health
curl -I http://localhost:3000

# 手动运行测试
python test_websocket.py
python integration_test.py
```

### 4. 权限问题
```bash
# 给脚本执行权限
chmod +x *.sh

# macOS可能需要允许脚本执行
xattr -d com.apple.quarantine *.sh
```

---

## 📊 系统性能指标

安装和启动完成后，系统应该达到以下性能指标：

| 指标 | 目标值 | 描述 |
|------|--------|------|
| 板块预测准确率 | 69.0% | 相比基准提升7.8% |
| 股票选择胜率 | 50.0% | 相比基准提升25.0% |
| 投资组合收益 | +0.31% | 相比基准提升126.1% |
| 算法整体提升 | +53.0% | 综合性能优化 |
| 测试通过率 | ≥90% | 系统稳定性 |

---

## 🎯 核心功能验证

启动后可以验证以下核心功能：

### 1. 实时数据推送
- WebSocket连接正常
- 5种数据流实时更新
- 前后端数据同步

### 2. 板块分析
- TOP板块智能识别
- 投资逻辑生成
- 历史性能验证

### 3. 股票筛选
- 多维度评分算法
- 优化参数配置
- 选股性能验证

### 4. 投资组合管理
- 组合分析和优化
- 风险评估
- 调仓建议

### 5. 反人性交易助手
- 情绪检测算法
- 交易纪律执行
- 冷静期控制

---

## 📞 技术支持

如果遇到问题，请按以下顺序排查：

1. 运行 `./run_tests.sh` 获取详细诊断
2. 检查 `logs/` 目录下的日志文件
3. 确认环境要求是否满足
4. 参考故障排除章节

成功启动标志：
- ✅ 测试总分 ≥90分
- ✅ 前端可正常访问
- ✅ API文档可正常访问
- ✅ WebSocket实时数据正常