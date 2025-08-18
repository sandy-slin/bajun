# 项目标准规范 (Project Standards)

## 📁 标准目录结构

### 核心目录结构要求
```
bajun/
├── src/                          # 源代码目录
│   ├── api/                      # FastAPI后端服务
│   │   ├── main.py              # 主应用入口
│   │   ├── models.py            # Pydantic数据模型
│   │   ├── routes/              # API路由模块
│   │   │   ├── __init__.py
│   │   │   ├── sectors.py       # 板块分析API
│   │   │   ├── stocks.py        # 股票推荐API
│   │   │   ├── portfolio.py     # 投资组合API
│   │   │   └── trading.py       # 交易助手API
│   │   ├── middleware.py        # 中间件配置
│   │   └── websocket_manager.py # WebSocket管理
│   ├── core/                    # 核心业务逻辑
│   │   ├── sector_engine.py     # 板块分析引擎
│   │   ├── stock_engine.py      # 股票选择引擎
│   │   ├── portfolio_engine.py  # 投资组合引擎
│   │   └── anti_human_nature_engine.py # 反人性交易引擎
│   ├── data/                    # 数据获取模块
│   │   ├── sector_fetcher.py    # 板块数据获取
│   │   ├── stock_fetcher.py     # 股票数据获取
│   │   └── technical_calculator.py # 技术指标计算
│   ├── analysis/                # 分析算法模块
│   ├── validation/              # 数据验证模块
│   └── config/                  # 配置管理
├── frontend/                    # React前端应用
│   ├── src/
│   │   ├── pages/               # 页面组件
│   │   ├── components/          # 通用组件
│   │   ├── contexts/            # React Context
│   │   └── services/            # API服务
│   ├── public/                  # 静态资源
│   ├── package.json            # 前端依赖
│   └── tsconfig.json           # TypeScript配置
├── docs/                       # 项目文档
│   ├── development/            # 开发文档
│   │   ├── commit_checklist.md # 提交检查清单
│   │   ├── project_standards.md # 项目标准 (本文件)
│   │   └── api_documentation.md # API文档
│   ├── user_input/            # 用户需求文档
│   │   └── requirement.md     # 需求文档
│   ├── testing/               # 测试文档
│   │   └── test_standards.md  # 测试标准
│   └── architecture/          # 架构文档
├── tests/                     # 测试代码
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   └── framework/            # 测试框架
├── logs/                     # 日志文件
├── cache/                    # 缓存目录
├── reports/                  # 报告输出
├── .git/                     # Git版本控制
├── CLAUDE.md                 # Claude指导文档 (核心)
├── README.md                 # 项目说明
├── requirements.txt          # Python依赖
├── start_services.sh         # 服务启动脚本
├── stop_services.sh          # 服务停止脚本
├── pre_commit_check.sh       # 提交前检查
└── check_performance.sh      # 性能检查
```

### 必要文件清单

#### 🔥 核心必需文件
- `CLAUDE.md` - Claude指导文档，必须与实际代码保持同步
- `README.md` - 项目说明和使用指南
- `requirements.txt` - Python依赖管理
- `src/api/main.py` - 后端服务入口
- `frontend/package.json` - 前端依赖管理

#### 📋 标准配置文件
- `start_services.sh` / `stop_services.sh` - 服务管理
- `pre_commit_check.sh` / `check_performance.sh` - 质量检查
- `.gitignore` - Git忽略规则
- `frontend/tsconfig.json` - TypeScript配置

#### 📚 文档完整性要求
- `docs/user_input/requirement.md` - 需求文档
- `docs/testing/test_standards.md` - 测试标准
- `docs/development/` - 开发相关文档

## 📝 CLAUDE.md 同步要求

### 强制同步场景
当以下情况发生时，**必须**更新CLAUDE.md：

#### 1. 架构变更
- 新增/删除核心模块
- API接口结构调整
- 数据库schema变更
- 服务架构调整

#### 2. 核心功能变更
- 新增核心业务功能
- 算法逻辑重大调整
- 性能基准值变更
- 工作流程调整

#### 3. 技术栈变更
- 新增/移除重要依赖
- 框架版本升级
- 部署方式调整
- 开发工具链变更

#### 4. 接口变更
- API端点增删改
- 请求/响应格式调整
- 错误处理机制变更
- 认证授权调整

### CLAUDE.md 内容检查项

#### ✅ 必须包含的核心信息
- [ ] 项目概述和核心哲学
- [ ] 系统架构图和技术栈
- [ ] 开发命令和脚本使用
- [ ] API接口列表和文档
- [ ] 数据模型和业务逻辑
- [ ] 性能要求和基准值
- [ ] 测试标准和质量要求
- [ ] 部署和运维指南

#### ✅ 版本同步检查
- [ ] 版本号与实际代码一致
- [ ] 技术栈版本信息准确
- [ ] API接口与实际实现一致
- [ ] 配置文件路径正确
- [ ] 性能基准值为最新
- [ ] 测试命令有效可执行

## 🏗️ 代码规范要求

### Python代码规范
```python
# 文件头部注释 (必需)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块功能描述
详细说明模块的用途和主要功能
"""

# 导入顺序 (必需)
# 1. 标准库
import asyncio
import logging
from datetime import datetime

# 2. 第三方库
from fastapi import FastAPI
from pydantic import BaseModel

# 3. 项目内部模块
from ..models import SectorInfo
```

### TypeScript/React规范
```typescript
// 文件头部注释 (推荐)
/**
 * 组件功能描述
 * @author Claude
 * @version 1.3.0
 */

// 导入顺序
import React from 'react';
import { Card, Button } from 'antd';
import { useApi } from '../contexts/ApiContext';
```

### API路由规范
```python
# 路由定义规范
@router.get("/", response_model=Response, summary="简短描述")
async def endpoint_name(
    param: int = Query(..., description="参数说明")
):
    """
    详细的API文档说明
    
    Args:
        param: 参数详细说明
        
    Returns:
        Response: 返回值说明
    """
```

## 📄 文档规范要求

### README.md 必需内容
- [ ] 项目简介和核心功能
- [ ] 快速开始指南
- [ ] 安装和依赖要求
- [ ] 基本使用示例
- [ ] 目录结构说明
- [ ] 贡献指南
- [ ] 许可证信息

### API文档要求
- [ ] 所有API端点有完整文档
- [ ] 请求/响应示例
- [ ] 错误码说明
- [ ] 认证要求说明
- [ ] 限流和配额说明

### 业务文档要求
- [ ] 核心业务流程图
- [ ] 算法逻辑说明
- [ ] 数据流向图
- [ ] 用户操作指南

## 🔄 版本管理规范

### 版本号命名规则
- 格式: `vX.Y.Z` (如 v1.3.0)
- X: 主版本号 (重大架构变更)
- Y: 次版本号 (新功能添加)
- Z: 补丁版本号 (错误修复)

### 版本发布要求
- [ ] 版本号更新 (package.json, CLAUDE.md等)
- [ ] CHANGELOG.md 更新
- [ ] 标签创建: `git tag vX.Y.Z`
- [ ] 文档同步更新
- [ ] 性能基准验证

### 分支管理规范
- `main`: 主分支，稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 紧急修复分支

## 🎯 质量标准

### 代码质量要求
- [ ] 无语法错误
- [ ] 无未使用的导入
- [ ] 函数/类有合适的文档字符串
- [ ] 变量命名规范清晰
- [ ] 代码结构清晰合理

### 性能要求
- [ ] API响应时间 < 2秒
- [ ] 算法准确率达到基准
- [ ] 内存使用合理
- [ ] 无明显性能回退

### 测试覆盖要求
- [ ] 核心功能有单元测试
- [ ] API有集成测试
- [ ] 关键路径有端到端测试
- [ ] 性能测试定期执行

## 🚀 持续改进

### 定期检查项
- [ ] 月度文档同步检查
- [ ] 季度代码质量审查
- [ ] 半年度架构优化评估
- [ ] 年度技术栈升级评估

### 改进机制
- 发现问题立即记录
- 定期召开改进会议
- 制定改进计划和时间表
- 跟踪改进效果

这些标准将确保项目的长期可维护性和高质量发展。