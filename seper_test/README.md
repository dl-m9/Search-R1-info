# SePer 集成测试套件

本目录包含 SePer 与 Search-R1 集成的完整测试套件。

## 📁 测试文件

### 核心测试
- **`test_core.py`** - 核心功能测试 ✅ 推荐首先运行
- **`test_fixed.py`** - 修复版本测试（避免导入冲突）
- **`test_simple.py`** - 简化的结构测试

### 完整测试
- **`test_seper_integration.py`** - 完整集成测试（需要模型）
- **`test_seper_minimal.py`** - 最小结构测试
- **`test_seper_quick.py`** - 快速功能测试

### 自动化运行
- **`run_tests.sh`** - 自动化测试运行器

## 🚀 推荐测试流程

### 1. 快速验证（推荐）
```bash
cd Search-R1-info/seper_test
python test_core.py
```

### 2. 完整测试套件（可选）
```bash
cd Search-R1-info/seper_test
./run_tests.sh
```

### 3. 单独运行特定测试
```bash
# 核心功能测试
python test_core.py

# 基本结构测试
python test_simple.py

# 完整集成测试（需要GPU）
python test_seper_integration.py
```

## ✅ 测试状态

### 当前状态
- ✅ **文件结构完整** - 所有必需文件存在
- ✅ **基本功能可用** - SePer 计算器可以创建
- ✅ **配置正确** - 训练脚本包含 SePer 设置
- ✅ **模块可加载** - 奖励函数可以导入

### 已知问题
- ⚠️ 导入警告（不影响功能）
- ⚠️ 完整测试需要 GPU 环境
- ⚠️ 部分测试需要依赖包安装

## 📊 测试结果解读

### 成功输出示例
```
🧪 SePer Core Integration Test
==================================================
✅ All essential files present
✅ SePer integration structure is complete
✅ Training scripts are configured
✅ Basic functionality is available
✅ SUCCESS
```

### 如果测试失败
1. **文件缺失** - 检查路径和文件存在性
2. **导入错误** - 安装缺少的依赖包
3. **权限问题** - 检查文件权限

## 🔧 故障排除

### 常见问题

#### 1. 文件路径错误
```bash
# 检查是否在正确目录
pwd
# 应该显示: .../Search-R1-info/seper_test
```

#### 2. Python 路径问题
```bash
# 确保可以访问模块
python -c "import sys; print(sys.path)"
```

#### 3. 依赖包缺失
```bash
# 安装基本依赖
pip install torch transformers numpy
```

## 📝 测试覆盖范围

### ✅ 已覆盖
- 文件结构完整性
- 基本模块导入
- SePer 计算器创建
- 训练脚本配置
- 模型自动检测逻辑

### ⚠️ 部分覆盖
- 完整模型加载（需要 GPU）
- 端到端奖励计算（需要依赖）
- 实际训练运行（需要数据和环境）

## 🚀 测试通过后的步骤

如果 `test_core.py` 通过，说明 SePer 集成已经就绪：

1. **启动检索服务器**
   ```bash
   conda activate retriever
   bash retrieval_launch.sh
   ```

2. **运行 SePer 增强训练**
   ```bash
   conda activate searchr1
   bash train_grpo_seper.sh
   ```

3. **监控训练日志**
   ```bash
   tail -f nq-search-r1-grpo-seper-qwen2.5-3b-em.log
   ```

4. **查看 SePer 分数**
   在训练日志中查找：
   - `seper/delta_mean`: 平均 ΔSePer 分数
   - `seper/retrieval_mean`: 平均检索后 SePer
   - `critic/rewards/mean`: 综合奖励

## 📋 测试清单

运行测试前确认：

- [ ] 在 `Search-R1-info/seper_test/` 目录下
- [ ] Python 环境已激活
- [ ] 所有必需文件存在
- [ ] 路径配置正确

运行测试后检查：

- [ ] 核心测试通过
- [ ] 无关键错误信息
- [ ] 可以创建 SePer 计算器
- [ ] 训练配置正确

## 🔗 相关文件

- **集成代码**: `../verl/utils/reward_score/seper_reward.py`
- **训练脚本**: `../train_grpo_seper.sh`
- **配置文件**: `../verl/trainer/config/ppo_trainer_seper.yaml`
- **主文档**: `../SEPER_INTEGRATION.md`