#!/bin/bash

# SePer 集成快速启动脚本
# 自动验证集成并准备训练

echo "🚀 SePer 集成快速启动"
echo "========================"

# 检查当前目录
if [[ ! -f "test_core.py" ]]; then
    echo "❌ 错误：请在 Search-R1-info/seper_test/ 目录下运行此脚本"
    exit 1
fi

echo "✅ 在正确的测试目录中"

# 步骤1：运行核心测试
echo ""
echo "🧪 步骤1：运行核心集成测试..."
python test_core.py

if [[ $? -ne 0 ]]; then
    echo "❌ 核心测试失败，请检查错误信息"
    exit 1
fi

echo ""
echo "✅ 核心测试通过！SePer 集成已就绪"

# 步骤2：检查环境
echo ""
echo "🔍 步骤2：检查训练环境..."

# 检查主训练脚本
if [[ ! -f "../train_grpo_seper.sh" ]]; then
    echo "❌ 训练脚本不存在：../train_grpo_seper.sh"
    exit 1
fi

# 检查检索服务器端口
if netstat -tuln | grep -q ":8000 "; then
    echo "✅ 检索服务器已在 8000 端口运行"
    RETRIEVAL_RUNNING=true
else
    echo "⚠️  检索服务器未运行，需要启动"
    RETRIEVAL_RUNNING=false
fi

# 检查环境
if command -v conda &> /dev/null; then
    echo "✅ Conda 可用"
    if [[ "$CONDA_DEFAULT_ENV" == "searchr1" ]]; then
        echo "✅ 在 searchr1 环境中"
    else
        echo "⚠️  当前环境：$CONDA_DEFAULT_ENV，建议激活 searchr1"
    fi
else
    echo "⚠️  Conda 不可用"
fi

# 步骤3：提供下一步指令
echo ""
echo "🎯 步骤3：下一步操作"
echo "========================"

if [[ "$RETRIEVAL_RUNNING" == "false" ]]; then
    echo "1. 启动检索服务器（新终端）："
    echo "   conda activate retriever"
    echo "   bash retrieval_launch.sh"
    echo ""
fi

echo "2. 运行 SePer 增强训练："
echo "   conda activate searchr1"
echo "   cd .."
echo "   bash train_grpo_seper.sh"
echo ""

echo "3. 监控训练日志："
echo "   tail -f nq-search-r1-grpo-seper-qwen2.5-3b-em.log"
echo ""

echo "📊 预期 SePer 输出："
echo "   - seper/delta_mean: ΔSePer 分数"
echo "   - seper/retrieval_mean: 检索后 SePer"
echo "   - seper/baseline_mean: 基线 SePer"
echo "   - critic/rewards/mean: 综合奖励"
echo ""

# 步骤4：提供快速验证命令
echo "🔧 步骤4：快速验证命令"
echo "========================"

echo "检查配置："
echo "grep -n \"seper_weight=0.7\" ../train_grpo_seper.sh"
echo ""
echo "检查训练器："
echo "grep -n \"main_ppo_seper\" ../train_grpo_seper.sh"
echo ""
echo "检查自动模型："
echo "grep -n \"model_path=null\" ../train_grpo_seper.sh"
echo ""

echo "✅ SePer 集成验证完成！"
echo "可以开始训练了。如果遇到问题，请查看 USAGE_GUIDE.md"