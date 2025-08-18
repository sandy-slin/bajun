# 完备迭代体系使用指南 (Complete Iterative System Guide)

## 🎯 系统概述

完备迭代体系是一个多维度、多层次的质量保证框架，确保每次代码提交都经过全面的功能测试、性能验证和项目标准检查。

## 🏗️ 系统架构

```
完备迭代体系 (Complete Iterative System)
├── 功能质量检查 (pre_commit_check.sh)
│   ├── Phase 1: 基础环境检查
│   ├── Phase 2: 服务重启检查  
│   ├── Phase 3: 后端服务检查
│   ├── Phase 4: 核心API功能检查
│   ├── Phase 5: 前端服务检查
│   ├── Phase 6: 代码质量检查
│   ├── Phase 7: 项目标准规范检查 ⭐ 新增
│   └── Phase 8: 性能基准检查
└── 项目标准检查 (check_project_standards.sh)
    ├── Phase 1: 核心目录结构检查
    ├── Phase 2: 核心文件存在性检查
    ├── Phase 3: CLAUDE.md 内容一致性检查
    ├── Phase 4: 文档完整性检查
    ├── Phase 5: 代码规范检查
    └── Phase 6: 版本管理和Git规范
```

## 🚀 使用方法

### 1. 手动执行完整检查

```bash
# 执行完整的迭代体系检查
./pre_commit_check.sh

# 单独执行项目标准检查
./check_project_standards.sh
```

### 2. 自动化Git钩子

系统已配置Git pre-commit钩子，每次提交时自动执行：

```bash
# 正常提交（自动执行检查）
git commit -m "feat: 新功能实现"

# 如需紧急跳过检查（不推荐）
git commit --no-verify -m "hotfix: 紧急修复"
```

### 3. 集成开发工作流

```bash
# 开发完成后的标准工作流
git add .                    # 暂存变更
./pre_commit_check.sh       # 手动预检查（可选）
git commit -m "提交信息"     # 自动执行完整检查
```

## 📊 检查项目详情

### 功能质量检查项 (11项)

| 检查项目 | 类型 | 说明 |
|---------|------|------|
| 项目目录结构检查 | 必需 | 验证核心文件和目录存在 |
| 后端服务健康检查 | 必需 | 验证FastAPI服务正常运行 |
| 后端服务响应结构 | 必需 | 验证API响应格式正确 |
| API文档可访问 | 必需 | 验证Swagger文档可访问 |
| 板块分析API基础功能 | 必需 | 验证核心业务逻辑正常 |
| 板块分析API数据完整性 | 必需 | 验证数据结构完整 |
| 板块分析API评分数据 | 必需 | 验证算法输出正确 |
| TOP5板块数据完整性 | 必需 | 验证推荐结果有效 |
| 前端服务HTTP状态 | 必需 | 验证React应用可访问 |
| 前端页面内容检查 | 必需 | 验证页面内容加载正常 |
| 前端静态资源 | 必需 | 验证静态资源加载正常 |

### 项目标准检查项 (27项)

| 检查类别 | 检查项数量 | 说明 |
|---------|------------|------|
| 核心目录结构 | 7项 | src/, frontend/, docs/等关键目录 |
| 核心文件存在性 | 8项 | CLAUDE.md, README.md等关键文件 |
| CLAUDE.md一致性 | 6项 | 文档内容与实际代码同步性 |
| 文档完整性 | 4项 | 文档覆盖度和同步性 |
| 代码规范 | 4项 | 编码标准和文档规范 |
| 版本管理 | 4项 | Git工作流和分支管理 |

## 📈 评分标准

### 功能质量评分
- **通过**: 所有11项检查都成功
- **失败**: 任何1项检查失败则整体失败

### 项目标准评分
- **必需项**: 必须全部通过，否则检查失败
- **推荐项**: 允许警告，但会给出改进建议
- **智能建议**: 基于检查结果自动生成改进建议

### 综合评价
```
🎉 完备迭代体系检查通过
├── ✅ 功能质量: 11/11 通过
├── ✅ 项目标准: 通过 (可能有改进建议)
└── 💡 改进建议: N项 (可选)
```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 服务启动失败
```bash
# 检查端口占用
lsof -i :8000
lsof -i :3000

# 手动重启服务
./stop_services.sh
./start_services.sh
```

#### 2. API检查失败
```bash
# 检查后端服务状态
curl http://localhost:8000/health

# 查看后端日志
tail -f logs/backend.log
```

#### 3. 前端检查失败
```bash
# 检查前端服务状态
curl -I http://localhost:3000

# 查看前端日志
tail -f logs/frontend.log
```

#### 4. 项目标准检查失败
```bash
# 查看详细标准检查结果
cat logs/project_standards_output.log

# 手动执行标准检查
./check_project_standards.sh
```

## 📝 日志文件说明

| 日志文件 | 说明 |
|---------|------|
| `logs/pre_commit_check.log` | 功能质量检查详细结果 |
| `logs/project_standards_output.log` | 项目标准检查详细结果 |
| `logs/project_standards_check.log` | 项目标准检查原始日志 |
| `logs/backend.log` | 后端服务运行日志 |
| `logs/frontend.log` | 前端服务运行日志 |
| `logs/startup.log` | 服务启动过程日志 |

## 🚫 绕过检查（不推荐）

在紧急情况下可以绕过检查，但不推荐：

```bash
# 跳过Git钩子
git commit --no-verify -m "紧急提交"

# 临时禁用钩子
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled
git commit -m "提交信息"
mv .git/hooks/pre-commit.disabled .git/hooks/pre-commit
```

## 🔄 系统维护

### 定期维护任务

1. **每周检查**: 验证所有检查项仍然有效
2. **每月更新**: 根据项目发展更新检查标准
3. **每季度审查**: 评估检查项目的必要性和有效性

### 检查项目更新

当项目架构发生重大变化时，需要更新：

1. `pre_commit_check.sh` - 功能检查项目
2. `check_project_standards.sh` - 标准检查项目  
3. `docs/development/project_standards.md` - 标准文档
4. 本文档 - 使用指南

## 💡 最佳实践

1. **提交前预检**: 在git commit前手动运行检查，提前发现问题
2. **增量修复**: 不要一次性修复所有警告，逐步改进项目质量
3. **文档同步**: 每次架构变更后及时更新CLAUDE.md和相关文档
4. **日志查看**: 检查失败时，优先查看日志文件了解具体问题
5. **标准遵循**: 严格遵循项目标准，提高代码质量和维护性

这个完备迭代体系确保了每次提交都经过多维度验证，保障项目的长期健康发展。