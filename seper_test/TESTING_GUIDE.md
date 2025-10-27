# SePer 集成测试指南

本指南提供完整的测试流程来验证 SePer 与 Search-R1 的集成是否正确工作。

## 测试套件概览

### 🚀 快速测试 (test_seper_quick.py)
**目的**: 验证基本功能和接口
**特点**: 轻量级，不需要模型加载
**用时**: < 30秒

```bash
python test_seper_quick.py
```

**测试内容**:
- ✅ SePer 奖励函数导入
- ✅ 基本奖励计算逻辑
- ✅ Search-R1 格式解析
- ✅ SePer 禁用模式

### 🔧 结构测试 (test_seper_minimal.py)
**目的**: 验证文件结构和依赖
**特点**: 检查所有必需文件是否存在
**用时**: < 1分钟

```bash
python test_seper_minimal.py
```

**检查项目**:
- ✅ 目录结构完整性
- ✅ 关键文件存在性
- ✅ Python 包依赖可用性
- ✅ 配置文件正确性

### 🧪 完整集成测试 (test_seper_integration.py)
**目的**: 验证端到端功能
**特点**: 包含模型加载和计算
**用时**: 5-10分钟

```bash
python test_seper_integration.py
```

**测试内容**:
- ✅ SePer 模块完整导入
- ✅ SePer 计算器初始化
- ✅ 问题/上下文提取准确性
- ✅ 奖励计算正确性
- ✅ 增强奖励管理器功能
- ✅ 配置解析逻辑

### 🏃 运行所有测试
使用自动化脚本按顺序运行所有测试：

```bash
bash run_tests.sh
```

**测试流程**:
1. 快速功能测试 (必须通过)
2. 结构完整性测试 (必须通过)
3. 完整集成测试 (可选，需要模型)

## 预期输出示例

### 成功情况
```
🚀 SePer Integration Test Suite

==========================================================
Running: Quick Functionality Test
Command: python test_seper_quick.py
==========================================================

🚀 SePer Quick Integration Test
...
📊 QUICK TEST SUMMARY
Basic Imports: ✅ PASSED
Path Structure: ✅ PASSED
SePer Structure: ✅ PASSED
Reward Files: ✅ PASSED
Config Files: ✅ PASSED
Dependencies: ✅ PASSED

Overall: 6/6 tests passed
🎉 All minimal tests passed!
Basic SePer integration setup is correct.

✅ Quick Functionality Test PASSED
```

### 常见错误和解决方案

#### 1. Import Errors
```
❌ SePer import failed: No module named 'seper'
```
**解决方案**:
```bash
# 确保 SePer 路径正确
export PYTHONPATH="/path/to/seper:$PYTHONPATH"

# 检查文件是否存在
ls -la separ/seper/__init__.py
```

#### 2. Model Loading Errors
```
❌ Failed to load Qwen/Qwen2.5-3B, falling back to Llama-2-7b-chat-hf
```
**解决方案**:
- 检查网络连接
- 使用较小的模型
- 检查 HuggingFace token

#### 3. GPU Memory Errors
```
❌ CUDA out of memory
```
**解决方案**:
- 减少 `num_generations`: 从 10 → 5
- 减小 `computation_chunk_size`: 从 8 → 4
- 使用 CPU: 设置 `device='cpu'`

#### 4. Configuration Errors
```
❌ Configuration parsing test failed
```
**解决方案**:
- 检查 YAML 语法
- 验证配置参数类型
- 使用默认配置文件

## 测试数据样例

### 基本测试用例
```python
# 正确的 Search-R1 格式
solution_text = """Answer the given question.
Question: Is Paris the capital of France?
<search>capital of France</search>
<information>Paris is the capital of France.</information>
<answer>Yes, Paris is the capital of France.</answer>"""

ground_truth = {'target': ['Yes']}
```

### 边界情况测试
- ✅ 无搜索标签的答案
- ✅ 多轮搜索的答案
- ✅ 错误答案的处理
- ✅ 格式错误的答案

## 性能基准

### 计算时间期望
- **快速测试**: < 30秒
- **结构测试**: < 1分钟
- **完整测试**: 5-10分钟

### 内存使用期望
- **SePer 禁用**: 额外 0MB
- **SePer 启用**: 额外 4-6GB (生成模型 + 蕴含模型)

### 成功率指标
- **必须通过**: 快速测试 + 结构测试
- **推荐通过**: 完整集成测试
- **可接受的失败**: 仅内存相关的错误

## 故障排除步骤

### 1. 快速诊断
```bash
# 运行最轻量级的测试
python test_seper_quick.py

# 检查基本导入
python -c "from verl.utils.reward_score.seper_reward import compute_seper_reward; print('OK')"
```

### 2. 环境检查
```bash
# 检查 Python 路径
echo $PYTHONPATH

# 检查 SePer 文件
find . -name "*.py" | grep -E "(seper|reward)"

# 检查 GPU 状态
nvidia-smi
```

### 3. 逐步验证
```bash
# 1. 测试基本结构
python test_seper_minimal.py

# 2. 测试奖励函数 (无 SePer)
python test_seper_quick.py

# 3. 测试完整集成 (需要模型)
python test_seper_integration.py
```

## 集成验证清单

运行完所有测试后，检查以下项目：

### ✅ 文件集成
- [ ] `verl/utils/reward_score/seper_reward.py` 存在
- [ ] `verl/utils/reward_score/qa_em_seper.py` 存在
- [ ] `verl/trainer/main_ppo_seper.py` 存在
- [ ] `train_grpo_seper.sh` 存在

### ✅ 功能验证
- [ ] 快速测试通过
- [ ] 结构测试通过
- [ ] 奖励计算正确
- [ ] 配置解析正确

### ✅ 模型一致性
- [ ] SePer 自动检测 Search-R1 模型
- [ ] 路径配置正确
- [ ] 环境变量传递成功

### ✅ 训练准备
- [ ] 检索服务器可访问
- [ ] 数据文件存在
- [ ] GPU 内存充足
- [ ] 依赖包版本兼容

完成所有检查项后，就可以开始训练了：

```bash
# 启动训练
bash train_grpo_seper.sh
```

---

如果遇到问题，请参考测试输出中的错误信息，或查看相关的日志文件进行调试。