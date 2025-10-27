# SePer 集成测试使用指南

## 🚀 快速开始

### 第一步：运行核心测试（推荐）
```bash
cd Search-R1-info/seper_test
python test_core.py
```

如果看到 `✅ SUCCESS`，说明 SePer 集成已经准备就绪！

### 第二步：验证完整功能（可选）
```bash
./run_tests.sh
```

## 📊 测试文件说明

| 测试文件 | 用途 | 推荐优先级 | 需要GPU |
|---------|------|-----------|--------|
| `test_core.py` | 核心功能验证 | ⭐⭐⭐ 最高 | ❌ |
| `test_simple.py` | 基本结构检查 | ⭐⭐ 中等 | ❌ |
| `test_fixed.py` | 修复版本测试 | ⭐ 中等 | ❌ |
| `test_seper_quick.py` | 快速功能测试 | ⭐ 中等 | ❌ |
| `test_seper_minimal.py` | 最小结构测试 | ⭐ 低 | ❌ |
| `test_seper_integration.py` | 完整集成测试 | ⭐ 最低 | ✅ |

## ✅ 测试成功后的步骤

### 1. 环境准备
```bash
# 确保在正确环境
conda activate searchr1

# 启动检索服务器（新终端）
conda activate retriever
bash retrieval_launch.sh
```

### 2. 运行训练
```bash
# 在主目录运行
cd Search-R1-info
bash train_grpo_seper.sh
```

### 3. 监控训练日志
```bash
# 查看 SePer 相关日志
tail -f nq-search-r1-grpo-seper-qwen2.5-3b-em.log | grep -i seper
```

## 🔧 常见问题解决

### 问题1：ImportError
```
❌ Import test failed: No module named 'transformers'
```
**解决方案**：
```bash
pip install transformers torch numpy
```

### 问题2：文件路径错误
```
❌ File not found: verl/utils/reward_score/seper_reward.py
```
**解决方案**：
```bash
# 确保在正确目录
pwd
# 应该显示: .../Search-R1-info/seper_test
```

### 问题3：权限错误
```
❌ Permission denied
```
**解决方案**：
```bash
chmod +x *.sh *.py
```

## 📈 测试结果解读

### 成功标志
```
🎯 Status:
✅ SePer integration structure is complete
✅ Training scripts are configured
✅ Basic functionality is available
✅ SUCCESS
```

### 失败标志
- 任何 `❌` 文件缺失报告
- ImportError 或 ModuleNotFoundError
- 权限拒绝错误

## 🎯 关键验证点

运行 `test_core.py` 时，确认以下测试通过：

1. **文件结构测试**
   - ✅ `verl/utils/reward_score/seper_reward.py`
   - ✅ `verl/trainer/main_ppo_seper.py`
   - ✅ `train_grpo_seper.sh`
   - ✅ `seper/seper/calculate.py`

2. **基本功能测试**
   - ✅ SePer create_collate_fn works
   - ✅ SePer calculator (disabled) created

3. **配置测试**
   - ✅ SePer weight configured
   - ✅ Enhanced trainer specified
   - ✅ Auto model detection enabled

## 🚀 进阶测试

如果核心测试通过，可以尝试更详细的测试：

### 完整功能测试
```bash
python test_seper_integration.py
```
*注意：需要GPU和完整的依赖安装*

### 手动验证配置
```bash
# 检查训练脚本配置
grep -n "seper_weight" ../train_grpo_seper.sh

# 检查模型自动检测
grep -n "model_path=null" ../train_grpo_seper.sh

# 检查增强训练器
grep -n "main_ppo_seper" ../train_grpo_seper.sh
```

## 📝 测试日志示例

### 正常输出
```
🧪 SePer Core Integration Test
==================================================
Project root: /path/to/Search-R1-info

📁 Test 1: File Structure
  ✅ verl/utils/reward_score/seper_reward.py
  ✅ verl/trainer/main_ppo_seper.py
  ✅ train_grpo_seper.sh
  ✅ seper/seper/calculate.py

🧮 Test 2: Basic Functions
  ✅ SePer create_collate_fn works
  ✅ SePer reward module loadable
SePer reward calculation disabled
  ✅ SePer calculator (disabled) created

✅ SUCCESS
```

### SePer 训练日志片段
```
[INFO] ✓ SePer calculator loaded: Qwen/Qwen2.5-3B
[INFO] ✓ SePer entailment model loaded: Deberta-v2-xlarge-mnli
[INFO] SePer models initialized successfully
[INFO] Using reward: EnhancedRewardManager with SePer weight: 0.7
[INFO] Combined score: EM score: 1.0, SePer score: 0.4521
[INFO] ✅ SePer integration working correctly
```

## 🎉 训练开始

一旦所有测试通过，你就可以开始训练了：

```bash
# 确保检索服务器运行（终端1）
conda activate retriever
bash retrieval_launch.sh

# 启动 SePer 增强训练（终端2）
conda activate searchr1
bash train_grpo_seper.sh

# 监控训练进度（终端3）
tail -f nq-search-r1-grpo-seper-qwen2.5-3b-em.log
```

## 📊 预期 SePer 输出

在训练日志中，你应该看到：

- `seper/delta_mean`: 平均 ΔSePer 分数
- `seper/retrieval_mean`: 平均检索后 SePer
- `seper/baseline_mean`: 平均基线 SePer
- `critic/rewards/mean`: 综合奖励分数

## 🔗 相关文档

- `README.md` - 测试套件概览
- `USAGE_GUIDE.md` - 本使用指南
- `../SEPER_INTEGRATION.md` - 完整集成指南

---

**💡 提示**: 如果遇到问题，先运行 `test_core.py` 确保基本集成正确，然后再尝试更复杂的测试。