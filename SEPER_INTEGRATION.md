# SePer 集成到 Search-R1 指南

本指南详细说明如何将 SePer (Semantic Perplexity) 集成到 Search-R1 的 GRPO 训练流程中，使用 ΔSePer 作为检索效用的评估指标。

## 核心概念

### SePer 指标
- **SePer**: 语义困惑度，衡量模型对答案的不确定性
- **ΔSePer**: `SePer_with_retrieval - SePer_baseline`，衡量检索带来的语义不确定性减少
- **基线**: 没有检索上下文时的 SePer 分数
- **检索后**: 有检索上下文时的 SePer 分数

### 重要改进：模型一致性
**🎯 核心优化**: SePer generation model 现在自动匹配 Search-R1 训练模型
- **评估一致性**: 避免不同模型架构带来的偏差
- **纯ΔSePer**: 更准确地反映检索带来的改进（而非模型能力差异）
- **训练对齐**: SePer 评估与 Search-R1 学习目标完全一致

### 集成原理
1. **模型匹配**: SePer 自动使用与 Search-R1 相同的基础模型
2. **奖励组合**: `最终奖励 = (1-w) * EM_score + w * ΔSePer`
3. **权重控制**: `w` (seper_weight) 控制 SePer 在奖励中的权重
4. **评估策略**: 训练时使用 SePer，验证时仅用 EM (提高效率)

## 使用方法

### 1. 环境准备

确保 SePer 代码在正确的路径：
```bash
# 确保 separ/ 目录在 Search-R1-info/ 下
ls separ/  # 应该包含 seper/__init__.py 等文件
```

### 2. 启动检索服务器

```bash
# 在检索环境中启动
conda activate retriever
bash retrieval_launch.sh
```

### 3. 运行训练

#### 方法一：使用专门的 GRPO+SePer 脚本
```bash
# 在主环境中启动训练
conda activate searchr1
bash train_grpo_seper.sh
```

#### 方法二：使用配置文件
```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_seper \
    --config-name=ppo_trainer_seper \
    data.train_files=data/nq_search/train.parquet \
    data.val_files=data/nq_search/test.parquet \
    # ... 其他参数
```

## 关键配置参数

### SePer 相关参数
- `reward_model.seper_weight`: SePer 权重 (0-1)，默认 0.7
- `reward_model.seper_config.enabled`: 是否启用 SePer
- `reward_model.seper_config.model_path`: SePer 生成模型路径
- `reward_model.seper_config.num_generations`: SePer 生成次数 (建议 5-10)
- `reward_model.seper_config.computation_chunk_size`: 计算块大小 (内存/速度权衡)

### 效率优化参数
- `reward_model.val_seper_weight`: 验证时 SePer 权重 (建议 0.0，节省时间)
- `actor_rollout_ref.rollout.n_agent`: GRPO 智能体数量 (建议 5)

## 文件说明

### 新增文件
1. **`verl/utils/reward_score/seper_reward.py`**: SePer 核心计算模块
2. **`verl/utils/reward_score/qa_em_seper.py`**: EM+SePer 组合奖励函数
3. **`verl/trainer/main_ppo_seper.py`**: 集成 SePer 的 PPO 训练器
4. **`train_grpo_seper.sh`**: GRPO+SePer 训练脚本
5. **`verl/trainer/config/ppo_trainer_seper.yaml`**: SePer 集成配置文件

### 修改说明
- 保持与原有代码的兼容性
- SePer 作为可选功能，不影响原有训练流程
- 提供详细的错误处理和日志输出

## 性能考虑

### 计算开销
- **训练阶段**: 每个 batch 需要 ~10 秒 (5 个生成 + 蕴含检查)
- **验证阶段**: 建议禁用 SePer (`val_seper_weight=0.0`)
- **内存需求**: 额外需要 ~2GB (Deberta 模型) + ~2GB (生成模型)

### 优化建议
1. **减少生成次数**: `num_generations=5` (而非 10)
2. **使用较小模型**: `Llama-2-7b` 而非更大模型
3. **批量处理**: `computation_chunk_size=8` 平衡内存和速度
4. **验证禁用**: 仅在训练时使用 SePer

## 示例输出

```
Data source: nq_search
Solution: Question: Is Elon Musk older than Sam Altman?
Context: Elon Musk was born on June 28, 1971. Sam Altman was born on April 22, 1985.
<reasoning>I need to compare the birth dates.</reasoning>
<search>Elon Musk birth date Sam Altman birth date</search>
<information>Elon Musk was born on June 28, 1971. Sam Altman was born on April 22, 1985.</information>
<answer>Yes, Elon Musk is older than Sam Altman.</answer>
Ground truth: {'target': ['Yes']}
EM score: 1.0, SePer score: 0.4521
Combined score: 1.0 * 0.3 + 0.4521 * 0.7 = 0.6165
--------------------------------------------------
```

## 调试和监控

### Wandb 指标
- `critic/rewards/mean`: 平均奖励
- `seper/delta_mean`: 平均 ΔSePer 分数
- `seper/retrieval_mean`: 平均检索后 SePer
- `seper/baseline_mean`: 平均基线 SePer

### 日志输出
- 启用详细日志 (`num_examine > 0`)
- 监控 SePer 计算错误
- 检查奖励组合是否符合预期

## 故障排除

### 常见问题
1. **ImportError**: 检查 SePer 路径和依赖
2. **CUDA 内存不足**: 减少 `num_generations` 或 `computation_chunk_size`
3. **训练速度慢**: 考虑降低 `seper_weight` 或禁用验证 SePer

### 依赖检查
```python
# 验证 SePer 可用性
try:
    from seper.calculate import gen_answers_batch
    print("SePer modules loaded successfully")
except ImportError as e:
    print(f"SePer import failed: {e}")
```

## 扩展和定制

### 自定义奖励组合
在 `qa_em_seper.py` 中修改奖励组合逻辑：
```python
# 线性组合
final_score = (1 - seper_weight) * em_score + seper_score

# 或非线性组合
if em_score == 1.0:
    final_score = max(em_score, seper_score)
else:
    final_score = seper_score
```

### 支持更多数据源
在 `main_ppo_seper.py` 的 `_select_rm_score_fn` 中添加新的数据源映射。

## 性能基准

建议的配置组合：
- **高精度**: `seper_weight=0.8`, `num_generations=10`
- **平衡**: `seper_weight=0.5`, `num_generations=5` (默认)
- **高效**: `seper_weight=0.3`, `num_generations=3`

通过调整这些参数，可以在计算开销和训练效果之间找到最佳平衡点。