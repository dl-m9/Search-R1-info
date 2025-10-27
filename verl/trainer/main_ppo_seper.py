# Enhanced PPO trainer with SePer integration
# Based on main_ppo.py with SePer reward integration

import os
import random
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.fs import create_local_fs
from verl.utils.ray_utils import DataProtoRayDistributor
from verl import DataProto
from verl.utils.reward_score import qa_em_seper


def _select_rm_score_fn(data_source):
    """Select reward function based on data source."""
    if data_source in ['nq_search', 'hotpotqa_search', 'triviaqa_search']:
        return qa_em_seper.compute_score_em_seper
    else:
        # Fallback to original EM scorer
        from verl.utils.reward_score.qa_em import compute_score_em
        return lambda solution_str, ground_truth, **kwargs: compute_score_em(
            solution_str, ground_truth, **kwargs)


class EnhancedRewardManager:
    """Enhanced reward manager with SePer integration."""

    def __init__(
        self,
        tokenizer,
        num_examine=0,
        format_score=0.,
        seper_weight=0.5,
        seper_config=None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.seper_weight = seper_weight
        self.seper_config = seper_config or {}

        # Initialize SePer calculator if enabled
        if self.seper_weight > 0:
            from verl.utils.reward_score.seper_reward import init_seper_calculator
            self.seper_calculator = init_seper_calculator(self.seper_config)
            print(f"SePer calculator initialized with weight: {self.seper_weight}")
        else:
            self.seper_calculator = None
            print("SePer disabled")

    def __call__(self, data: DataProto):
        """Enhanced reward computation with SePer integration."""

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

            # Select appropriate reward function
            compute_score_fn = _select_rm_score_fn(data_source)

            # Compute enhanced score with SePer
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
                seper_weight=self.seper_weight,
                seper_config=self.seper_config
            )

            # Debug printing
            if data_source not in already_print_data_sources and self.num_examine > 0:
                if len(already_print_data_sources) < self.num_examine:
                    already_print_data_sources[data_source] = True
                    print(f"Data source: {data_source}")
                    print(f"Solution: {sequences_str}")
                    print(f"Ground truth: {ground_truth}")
                    print(f"Score: {score}")
                    print("-" * 50)

            # Assign reward to last token of response
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config: DictConfig):
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(OmegaConf.to_yaml(config))

    # Create filesystem
    fs = create_local_fs(config.trainer.default_hdfs_dir)

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)

    # Set environment variable for SePer model auto-detection
    os.environ['ACTOR_MODEL_PATH'] = config.actor_rollout_ref.model.path

    # Enhanced reward manager with SePer
    reward_fn = EnhancedRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        format_score=config.reward_model.get('format_score', 0.),
        seper_weight=config.reward_model.get('seper_weight', 0.5),
        seper_config=config.reward_model.get('seper_config', {})
    )

    # Validation reward manager (can disable SePer for efficiency)
    val_seper_weight = config.reward_model.get('val_seper_weight', 0.0)
    val_reward_fn = EnhancedRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        format_score=config.reward_model.get('format_score', 0.),
        seper_weight=val_seper_weight,
        seper_config=config.reward_model.get('seper_config', {}) if val_seper_weight > 0 else None
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn
    )

    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()