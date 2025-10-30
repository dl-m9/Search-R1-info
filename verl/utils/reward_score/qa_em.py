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

import re
import string
import random
import sys
import os
import torch
from verl import DataProto
import numpy as np
from typing import Union, Optional
import threading

# Thread-local storage for reward metrics
_reward_metrics = threading.local()

# Thread-local storage for validation flag
_validation_flag = threading.local()


# Directly add the project root and the `seper` folder to sys.path
sys.path.insert(0, '/forest/forest/Search-R1-info')
sys.path.insert(0, '/forest/forest/Search-R1-info/seper')

from seper.calculate import gen_answers_batch, calculate_uncertainty_soft_batch, create_collate_fn, process_item_for_seper
from seper.models.huggingface_models import HuggingfaceModel
from seper.uncertainty_measures.semantic_entropy import EntailmentDeberta

# Try to import SEPER client (optional)
try:
    from verl.utils.reward_score.seper_client import get_seper_client, is_seper_service_available
    SEPER_CLIENT_AVAILABLE = True
except ImportError:
    SEPER_CLIENT_AVAILABLE = False
    get_seper_client = None
    is_seper_service_available = None


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer (for consistency with original logic)
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    result = extract_question_information(solution_str=solution_str)
    question = result['question']
    information_blocks = result['information_blocks']
    answer = extract_solution(solution_str=solution_str)
    
    
    do_print = random.randint(1, 64) == 1
    # do_print = True  # Uncomment to always print
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers 黄金答案: {ground_truth['target']}")
        print(f"Extracted answer 提取的答案: {answer}")
        print(f"Solution string 解决方案字符串: {solution_str}")
        print(f"--------------------------------")
    
    # Check if we're in validation mode (skip info gain computation during validation)
    is_validation = getattr(_validation_flag, 'skip_info_gain', False)
    
    # optional: compute information gain if handles are provided and we have retrieval
    info_gain_score_weight = 1
    if len(information_blocks) == 0:
        info_gain_score_weight = 0.0
    if is_validation:
        info_gain_score_weight = 0.0  # Skip info gain during validation
    info_gain_score = 0.0
    debug_print = False
    if debug_print:
        print(f"[DEBUG] SEPER_CLIENT_AVAILABLE: {SEPER_CLIENT_AVAILABLE}")
        print(f"[DEBUG] len(information_blocks): {len(information_blocks)}")
        print(f"[DEBUG] question: {question}")
        print(f"[DEBUG] information_blocks: {information_blocks}")
    # Try SEPER service first if available
    # Skip info gain computation during validation for efficiency
    if not is_validation and SEPER_CLIENT_AVAILABLE and len(information_blocks) > 0:
        if debug_print:
            print(f"[DEBUG] Attempting to get SEPER client...")
        # Use environment variable if set, otherwise use default
        service_url = os.getenv('SEPER_SERVICE_URL', 'http://0.0.0.0:0310')
        seper_client = get_seper_client(service_url=service_url)
        if debug_print:
            print(f"[DEBUG] seper_client: {seper_client}")
            print(f"[DEBUG] service_url: {service_url}")
        if seper_client is not None:
            if debug_print:
                print(f"[DEBUG] Checking service availability...")
            # Use the client's health_check method directly instead of is_seper_service_available()
            service_available = seper_client.health_check()
            if debug_print:
                print(f"[DEBUG] seper_client.health_check(): {service_available}")
            if service_available:
                try:
                    context = "\n".join(information_blocks)
                    if debug_print:
                        print(f"[DEBUG] Calling compute_info_gain with question: {question[:50]}..., context length: {len(context)}")
                        print(f"[DEBUG] ground_truth['target'] type: {type(ground_truth['target'])}, value: {ground_truth['target']}")
                    info_gain_score = seper_client.compute_info_gain(
                        question=question,
                        context=context,
                        answers=ground_truth['target']  # _normalize_answers will handle all cases
                    )
                    if debug_print:
                        print(f"[DEBUG] compute_info_gain returned: {info_gain_score}")
                except Exception as e:
                    print(f"[WARNING] SEPER service call failed, falling back to local computation: {e}")
                    import traceback
                    traceback.print_exc()
                    info_gain_score = 0.0
            else:
                print(f"[DEBUG] Service health check failed, skipping SEPER computation")
                print(f"[DEBUG] Service URL: {seper_client.service_url}")
        else:
            print(f"[DEBUG] get_seper_client() returned None")
    
    

    
    if answer is None:
        output_score = 0
        all_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            
            output_score = score  # Only EM check score
            all_score = score + info_gain_score_weight * info_gain_score  # Weighted sum
        else:
            output_score = format_score  # Only EM check score
            all_score = info_gain_score_weight * info_gain_score + format_score  # Weighted sum
    print("--------------------------------")
    print("info_gain_score: ", info_gain_score)
    print("output_score: ", output_score)
    print("all_score: ", all_score)
    # Store metrics in thread-local storage
    _reward_metrics.info_gain_score = info_gain_score
    _reward_metrics.output_score = output_score  # Only EM check score
    _reward_metrics.all_score = all_score  # Weighted sum
    
    return all_score  # Return weighted sum for training


def get_last_reward_metrics():
    """Get reward metrics from the last compute_score_em call.
    
    Returns:
        dict with 'info_gain_score', 'output_score' (EM check only), and 'all_score' (weighted sum)
    """
    try:
        return {
            'info_gain_score': getattr(_reward_metrics, 'info_gain_score', 0.0),
            'output_score': getattr(_reward_metrics, 'output_score', 0.0),  # EM check only
            'all_score': getattr(_reward_metrics, 'all_score', 0.0),  # Weighted sum
        }
    except:
        return {'info_gain_score': 0.0, 'output_score': 0.0, 'all_score': 0.0}



def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1., *, generator=None, entailment_model=None, device=None):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    result = extract_question_information(solution_str=solution_str)
    question = result['question']
    information_blocks = result['information_blocks']
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def extract_question_information(solution_str):
    """Extract question and retrieved information from a solution string.

    Returns a dict containing:
    - question: the question text parsed from the user turn (after "Question:")
    - retrieved_information: concatenated string of all <information>...</information> blocks
    - information_blocks: list of individual information blocks
    """
    # 1) Try to locate the user block and pull the text after "Question:"
    question = None
    user_block_match = re.search(r"<\|im_start\|>user(.*?)<\|im_end\|>", solution_str, re.DOTALL | re.IGNORECASE)
    if user_block_match:
        user_block = user_block_match.group(1)
        q_match = re.search(r"Question:\s*(.*?)(?:\n|$)", user_block, re.IGNORECASE | re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
    else:
        # Fallback: search globally for a "Question:" field
        q_match = re.search(r"Question:\s*(.*?)(?:\n|<\|im_end\|>|$)", solution_str, re.IGNORECASE | re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()

    # 2) Extract retrieved information blocks
    info_pattern = r"<information>(.*?)</information>"
    information_blocks = [s.strip() for s in re.findall(info_pattern, solution_str, re.DOTALL)]
    retrieved_information = "\n".join(information_blocks) if information_blocks else ""

    return {
        'question': question,
        'retrieved_information': retrieved_information,
        'information_blocks': information_blocks,
    }


if __name__ == "__main__":
    solution_str = "Question: What is the capital of France? <information>France is a country in Europe.</information> <information>Paris is the capital of France.</information> <answer>Paris</answer>"
    result = extract_question_information(solution_str)
    question = result['question']
    information_blocks = result['information_blocks']
    print(f"[DEBUG] question: {question}")
    print(f"[DEBUG] information_blocks: {information_blocks}")
    print(f"[DEBUG] information_blocks type: {type(information_blocks)}")
    print(f"[DEBUG] information_blocks length: {len(information_blocks)}")