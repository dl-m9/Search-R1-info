#!/bin/bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Launch script for SEPER info gain service
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT="https://hf-mirror.com"
set -e

# Default values
HOST="${SEPER_HOST:-0.0.0.0}"
PORT="${SEPER_PORT:-0310}"
MODEL_PATH="${SEPER_MODEL_PATH:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3}"
DEVICE="${SEPER_DEVICE:-cuda:0}"
NUM_GENERATIONS="${SEPER_NUM_GENERATIONS:-10}"
TEMPERATURE="${SEPER_TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${SEPER_MAX_NEW_TOKENS:-128}"
MAX_CONTEXT_WORDS="${SEPER_MAX_CONTEXT_WORDS:-4096}"
SUB_BATCH_SIZE="${SEPER_SUB_BATCH_SIZE:-10}"
COMPUTATION_CHUNK_SIZE="${SEPER_COMPUTATION_CHUNK_SIZE:-8}"
# Recommended: MAX_CONCURRENT_REQUESTS = NUM_GPUS * 5-10
# For 4 GPUs: 20-40 is reasonable. Higher values may cause OOM or slow response.
MAX_CONCURRENT_REQUESTS="${SEPER_MAX_CONCURRENT_REQUESTS:-512}"
NUM_GPUS="${SEPER_NUM_GPUS:-4}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --max-context-words)
            MAX_CONTEXT_WORDS="$2"
            shift 2
            ;;
        --sub-batch-size)
            SUB_BATCH_SIZE="$2"
            shift 2
            ;;
        --computation-chunk-size)
            COMPUTATION_CHUNK_SIZE="$2"
            shift 2
            ;;
        --max-concurrent-requests)
            MAX_CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST                  Host to bind to (default: $HOST)"
            echo "  --port PORT                   Port to bind to (default: $PORT)"
            echo "  --model-path PATH             Path to generator model (default: $MODEL_PATH)"
            echo "  --device DEVICE              Device (e.g., cuda:0) (default: $DEVICE)"
            echo "  --num-generations N           Number of generations (default: $NUM_GENERATIONS)"
            echo "  --temperature T              Generation temperature (default: $TEMPERATURE)"
            echo "  --max-new-tokens N            Max new tokens (default: $MAX_NEW_TOKENS)"
            echo "  --max-context-words N         Max context words (default: $MAX_CONTEXT_WORDS)"
            echo "  --sub-batch-size N            Sub batch size for generation (default: $SUB_BATCH_SIZE)"
            echo "  --computation-chunk-size N    Computation chunk size (default: $COMPUTATION_CHUNK_SIZE)"
            echo "  --max-concurrent-requests N    Max concurrent requests (default: $MAX_CONCURRENT_REQUESTS)"
            echo "  --num-gpus N                  Number of GPUs to use (default: $NUM_GPUS, multi-GPU mode if > 1)"
            echo ""
            echo "Environment variables:"
            echo "  SEPER_HOST                   Host to bind to"
            echo "  SEPER_PORT                   Port to bind to"
            echo "  SEPER_MODEL_PATH             Path to generator model"
            echo "  SEPER_DEVICE                 Device (e.g., cuda:0)"
            echo "  SEPER_NUM_GENERATIONS        Number of generations"
            echo "  SEPER_TEMPERATURE            Generation temperature"
            echo "  SEPER_MAX_NEW_TOKENS         Max new tokens"
            echo "  SEPER_MAX_CONTEXT_WORDS      Max context words"
            echo "  SEPER_SUB_BATCH_SIZE         Sub batch size for generation"
            echo "  SEPER_COMPUTATION_CHUNK_SIZE  Computation chunk size"
            echo "  SEPER_MAX_CONCURRENT_REQUESTS Max concurrent requests"
            echo "  SEPER_NUM_GPUS               Number of GPUs to use"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

echo "Starting SEPER Info Gain Service..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL_PATH"
echo "  Device: $DEVICE"
echo "  Num generations: $NUM_GENERATIONS"
echo ""

# Export environment variables
export SEPER_HOST="$HOST"
export SEPER_PORT="$PORT"
export SEPER_MODEL_PATH="$MODEL_PATH"
export SEPER_DEVICE="$DEVICE"
export SEPER_NUM_GENERATIONS="$NUM_GENERATIONS"
export SEPER_TEMPERATURE="$TEMPERATURE"
export SEPER_MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export SEPER_MAX_CONTEXT_WORDS="$MAX_CONTEXT_WORDS"
export SEPER_SUB_BATCH_SIZE="$SUB_BATCH_SIZE"
export SEPER_COMPUTATION_CHUNK_SIZE="$COMPUTATION_CHUNK_SIZE"
export SEPER_MAX_CONCURRENT_REQUESTS="$MAX_CONCURRENT_REQUESTS"
export SEPER_NUM_GPUS="$NUM_GPUS"

# Run the server
nohup python3 -m seper.service.seper_server \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --num-generations "$NUM_GENERATIONS" \
    --sub-batch-size "$SUB_BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-context-words "$MAX_CONTEXT_WORDS" \
    --computation-chunk-size "$COMPUTATION_CHUNK_SIZE" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --num-gpus "$NUM_GPUS" > seper_service_${PORT}.log 2>&1 &
