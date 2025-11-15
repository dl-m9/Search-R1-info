#!/bin/bash
# simple_auto_sync.sh - 简化的W&B自动同步
export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

# 配置（只需修改这里）
PROJECT_PATH="wandb/run-20251103_012042-1x9afn4j"  # 替换为你的项目路径
SYNC_EVERY=300  # 每5分钟同步一次（单位：秒）

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 切换到项目目录
cd "$SCRIPT_DIR/$PROJECT_PATH" || {
    echo "错误: 无法切换到目录 $SCRIPT_DIR/$PROJECT_PATH"
    exit 1
}

echo "$(date) - 开始自动同步: $(pwd)"

while true; do
    echo "$(date) - 检查并同步W&B离线数据..."
    
    # 直接同步当前项目目录（离线运行目录就是项目目录本身）
    if [ -d "." ]; then
        echo "$(date) - 同步: $(pwd)"
        wandb sync .
    fi
    
    echo "$(date) - 同步完成，等待${SYNC_EVERY}秒..."
    sleep $SYNC_EVERY
done