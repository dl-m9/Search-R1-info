# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Search-R1 is a reinforcement learning framework for training reasoning-and-searching interleaved Large Language Models (LLMs). Built upon veRL (Volcano Engine Reinforcement Learning), it enables LLMs to learn to reason and make tool calls (e.g., search engines) in a coordinated manner through RL methods like PPO, GRPO, and reinforce.

## Development Commands

### Environment Setup

```bash
# Main Search-R1 environment
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb

# Optional retriever environment (separate environment recommended)
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

### Core Workflow Commands

#### Data Preparation
```bash
# Download Wikipedia corpus and indexes
python scripts/download.py --save_path /path/to/save
cat /path/to/save/part_* > /path/to/save/e5_Flat.index
gzip -d /path/to/save/wiki-18.jsonl.gz

# Process Natural Questions dataset
python scripts/data_process/nq_search.py
```

#### Training Commands
```bash
# Launch retrieval server (required before training)
conda activate retriever
bash retrieval_launch.sh

# PPO Training
conda activate searchr1
bash train_ppo.sh

# GRPO Training
conda activate searchr1
bash train_grpo.sh
```

#### Inference
```bash
# Launch retrieval server first if not running
conda activate retriever
bash retrieval_launch.sh

# Run inference with trained model
conda activate searchr1
python infer.py
```

## Architecture Overview

### Core Components

**Search-R1 Framework Architecture:**
- **veRL Integration**: Built on top of veRL's RL training framework with support for FSDP, Megatron-LM training and vLLM inference
- **Multi-Turn Search Integration**: LLMs learn to interleave reasoning with search engine calls using `<search>query</search>` syntax
- **Reward System**: Rule-based outcome rewards for correct answers, enabling autonomous learning of search behavior

**Key Directories:**
- `search_r1/`: Core Search-R1 functionality
  - `search/`: Retrieval and search engine implementations (local retrievers, web search APIs, rerankers)
  - `llm_agent/`: LLM generation and tensor handling utilities
- `verl/`: Modified veRL framework for RL training with search integration
  - `trainer/`: PPO trainer implementation and configurations
  - `models/`: LLM model wrappers (Llama, Qwen2.5) with Megatron and FSDP support
  - `third_party/vllm/`: Multiple vLLM version compatibility layers
- `scripts/`: Data processing utilities and download scripts

### Training Pipeline

**Data Flow:**
1. **Input**: Questions (NQ dataset) with ground truth answers
2. **LLM Generation**: Model generates reasoning and search queries
3. **Search Integration**: Retrieval server returns relevant documents via HTTP API
4. **Response**: Model incorporates search results and provides final answer
5. **Reward Calculation**: Rule-based reward based on answer correctness
6. **RL Update**: PPO/GRPO updates model policy

**Retrieval Server:**
- HTTP endpoint at `http://127.0.0.1:8000/retrieve`
- Supports multiple retriever types: BM25, dense retrieval (e5), web search APIs
- Returns top-k documents with scoring

### Model Behavior

**Search Template Format:**
```
<reasoning>internal reasoning process</reasoning>
<search>query terms</search>
<information>retrieved documents</information>
<reasoning>continued reasoning with search results</reasoning>
<answer>final answer</answer>
```

**Training Configuration:**
- Supports multiple base models: Llama-3.2-3B, Qwen2.5-3B/7B variants
- Configurable multi-turn conversations (default: 2 turns)
- Memory optimization with FSDP parameter/gradient/optimizer offloading
- Wandb experiment tracking integration

### Key Configuration Parameters

**Training Scripts:**
- `max_turns`: Number of search-reasoning rounds (default: 2)
- `retriever.url`: Retrieval server endpoint
- `retriever.topk`: Number of documents to retrieve (default: 3)
- `algorithm.adv_estimator`: 'gae' for PPO, 'grpo' for GRPO
- `total_training_steps`: 1005 steps for full training

**Model Settings:**
- `max_response_length`: 500 tokens per reasoning/search turn
- `max_obs_length`: 500 tokens for retrieved documents
- `temperature=1`: Sampling temperature during generation

### Environment Requirements

- **GPU Memory**: 8+ GPUs recommended for training (FSDP offloading available)
- **Retrieval Server**: Separate environment with faiss-gpu for efficient dense retrieval
- **Dependencies**: vLLM 0.6.3, transformers <4.48, flash-attn, Ray framework
- **Storage**: Wikipedia corpus (~18GB) and retrieval indexes required

## Development Notes

- The framework uses Ray for distributed training orchestration
- vLLM is used for efficient inference during rollout
- FSDP (Fully Sharded Data Parallel) is supported for memory efficiency
- Custom stopping criteria are implemented for search query detection
- The system supports both single-node and multi-node training setups
- 在实现时，用最简洁的代码实现功能，不要引入不必要的复杂性。
- 如果有些功能已经有现成的代码，请直接使用，不要重复造轮子。