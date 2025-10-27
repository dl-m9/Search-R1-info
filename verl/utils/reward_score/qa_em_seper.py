# Enhanced QA EM reward with SePer integration
# Copyright 2024 Bytedance Ltd. and/or its affiliates

import re
import random
from typing import Dict, Any

from .qa_em import extract_solution, normalize_answer, em_check, subem_check
from .seper_reward import compute_seper_reward, init_seper_calculator


def compute_score_em_seper(
    solution_str: str,
    ground_truth: Dict[str, Any],
    method: str = 'strict',
    format_score: float = 0.,
    score: float = 1.,
    seper_weight: float = 0.5,
    seper_config: Dict = None
) -> float:
    """
    Enhanced EM score with SePer integration.

    Args:
        solution_str: solution text from Search-R1 model
        ground_truth: ground truth answers
        method: method to extract solution
        format_score: score for correct format
        score: score for correct answer (EM)
        seper_weight: weight for SePer reward (0-1)
        seper_config: SePer configuration

    Returns:
        Combined reward score
    """
    # Initialize SePer calculator if needed
    if seper_weight > 0 and seper_config is not None:
        init_seper_calculator(seper_config)

    # Extract answer from solution string
    answer = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")

    # Initialize with format score (for invalid format)
    em_score = 0.0
    if answer is not None:
        # Compute EM score
        if em_check(answer, ground_truth['target']):
            em_score = score
        else:
            em_score = format_score

    # Compute SePer reward if enabled
    seper_score = 0.0
    if seper_weight > 0:
        try:
            seper_score = compute_seper_reward(
                solution_str=solution_str,
                ground_truth=ground_truth,
                seper_delta_weight=seper_weight,
                seper_baseline_weight=0.0
            )

            if do_print:
                print(f"SePer score: {seper_score:.4f}, weight: {seper_weight}")

        except Exception as e:
            print(f"Warning: SePer computation failed: {e}")
            seper_score = 0.0

    # Combine scores
    final_score = (1 - seper_weight) * em_score + seper_score

    if do_print:
        print(f"EM score: {em_score}, SePer score: {seper_score:.4f}")
        print(f"Combined score: {final_score:.4f}")
        print(f"--------------------------------")

    return final_score


def compute_score_subem_seper(
    solution_str: str,
    ground_truth: Dict[str, Any],
    method: str = 'strict',
    format_score: float = 0.,
    score: float = 1.,
    seper_weight: float = 0.5,
    seper_config: Dict = None
) -> float:
    """
    Enhanced Sub-EM score with SePer integration.

    Args:
        solution_str: solution text from Search-R1 model
        ground_truth: ground truth answers
        method: method to extract solution
        format_score: score for correct format
        score: score for correct answer (Sub-EM)
        seper_weight: weight for SePer reward (0-1)
        seper_config: SePer configuration

    Returns:
        Combined reward score
    """
    # Initialize SePer calculator if needed
    if seper_weight > 0 and seper_config is not None:
        init_seper_calculator(seper_config)

    # Extract answer from solution string
    answer = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")

    # Initialize with format score (for invalid format)
    subem_score = 0.0
    if answer is not None:
        # Compute Sub-EM score
        normalized_prediction = normalize_answer(answer)
        for golden_answer in ground_truth['target']:
            golden_answer = normalize_answer(golden_answer)
            if golden_answer in normalized_prediction:
                subem_score = score
                break
        else:
            subem_score = format_score

    # Compute SePer reward if enabled
    seper_score = 0.0
    if seper_weight > 0:
        try:
            seper_score = compute_seper_reward(
                solution_str=solution_str,
                ground_truth=ground_truth,
                seper_delta_weight=seper_weight,
                seper_baseline_weight=0.0
            )

            if do_print:
                print(f"SePer score: {seper_score:.4f}, weight: {seper_weight}")

        except Exception as e:
            print(f"Warning: SePer computation failed: {e}")
            seper_score = 0.0

    # Combine scores
    final_score = (1 - seper_weight) * subem_score + seper_score

    if do_print:
        print(f"Sub-EM score: {subem_score}, SePer score: {seper_score:.4f}")
        print(f"Combined score: {final_score:.4f}")
        print(f"--------------------------------")

    return final_score