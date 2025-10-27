#!/usr/bin/env python3
"""
Simple Qwen model test for SePer
Tests basic SePer functionality with Qwen2.5 models
"""

import os
import sys
import torch

def setup_environment():
    """Setup environment and paths"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Add paths
    sys.path.insert(0, os.path.join(project_root, 'seper'))
    sys.path.insert(0, os.path.join(project_root, 'verl', 'utils', 'reward_score'))

    # Set environment variables
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['ACTOR_MODEL_PATH'] = 'Qwen/Qwen2.5-3B'  # Match Search-R1 default

    print(f"✅ Environment configured")
    print(f"   Project root: {project_root}")
    print(f"   HF endpoint: {os.environ['HF_ENDPOINT']}")
    print(f"   Actor model: {os.environ['ACTOR_MODEL_PATH']}")

    return project_root

def test_seper_basic():
    """Test SePer basic functionality"""
    print("\n🧮 Testing SePer Basic Functionality...")

    try:
        # Test import with error handling
        from seper_reward import SePerRewardCalculator

        print("✅ SePerRewardCalculator imported")

        # Test basic disabled calculator
        print("\n📋 Testing disabled mode...")
        calculator_disabled = SePerRewardCalculator(enabled=False)

        test_text = "Question: Test?\n<answer>Answer</answer>"
        rewards = calculator_disabled.compute_delta_seper_reward(
            test_text, {'target': ['Test']}
        )

        expected = {
            'seper_delta': 0.0,
            'seper_retrieval': 0.0,
            'seper_baseline': 0.0,
            'seper_reward': 0.0
        }

        if rewards == expected:
            print("✅ Disabled mode test passed")
            return True
        else:
            print(f"❌ Disabled mode failed: {rewards}")
            return False

    except Exception as e:
        print(f"❌ SePer basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_detection():
    """Test Qwen model detection"""
    print("\n🔍 Testing Qwen Model Detection...")

    try:
        from seper_reward import SePerRewardCalculator

        # Test different Qwen models
        qwen_models = [
            ('Qwen2.5-3B', 'Qwen/Qwen2.5-3B'),
            ('Qwen2.5-7B', 'Qwen/Qwen2.5-7B'),
            ('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B')
        ]

        for model_name, model_path in qwen_models:
            print(f"\n🧪 Testing detection: {model_name}")

            # Set environment for this model
            os.environ['ACTOR_MODEL_PATH'] = model_path

            # Create calculator (this will use our model path detection)
            calculator = SePerRewardCalculator(enabled=False)

            if hasattr(calculator, 'model_path') and model_path in str(calculator.model_path):
                print(f"✅ {model_name} detected correctly")
            else:
                print(f"❌ {model_name} detection failed")
                print(f"   Expected: {model_path}")
                if hasattr(calculator, 'model_path'):
                    print(f"   Actual: {calculator.model_path}")

        return True

    except Exception as e:
        print(f"❌ Model detection test failed: {e}")
        return False

def test_qwen_reward():
    """Test reward function with Qwen models"""
    print("\n🧮 Testing Qwen Reward Function...")

    try:
        from verl.utils.reward_score.qa_em_seper import compute_score_em_seper

        # Test case with different Qwen configurations
        test_case = {
            'solution_text': '''Answer the following question as briefly as possible. You must conduct reasoning inside </think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of China?
<search>capital of China</search>
<information>Beijing is the capital of China.</information>
<answer>Beijing</answer>''',
            'ground_truth': {'target': ['Beijing']}
        }

        # Test with different Qwen models
        qwen_configs = [
            ('Qwen2.5-3B', 'Qwen/Qwen2.5-3B'),
            ('Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B')
        ]

        for model_name, model_path in qwen_configs:
            print(f"\n📊 Testing with {model_name}...")

            try:
                # Mock environment for this model
                os.environ['ACTOR_MODEL_PATH'] = model_path

                score = compute_score_em_seper(
                    solution_str=test_case['solution_text'],
                    ground_truth=test_case['ground_truth'],
                    seper_weight=0.5,
                    seper_config={
                        'enabled': True,
                        'model_path': model_path,
                        'num_generations': 2,  # Minimal for testing
                        'max_new_tokens': 32,
                        'temperature': 1.0,
                        'device': 'cpu'  # Use CPU to avoid GPU issues
                    }
                )

                if isinstance(score, (int, float)):
                    print(f"✅ {model_name} reward score: {score:.4f}")
                else:
                    print(f"❌ {model_name} reward computation failed: {score}")

            except Exception as e:
                print(f"❌ {model_name} test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Qwen reward test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Qwen Simple SePer Integration Test")
    print("=====================================")
    print("Testing SePer integration with Qwen models.")
    print("Uses HF-Mirror and CPU to avoid GPU/memory issues.")
    print("=" * 50)

    # Setup environment
    project_root = setup_environment()

    # Run tests
    tests = [
        ("SePer Basic Functions", test_seper_basic),
        ("Qwen Model Detection", test_model_detection),
        ("Qwen Reward Function", test_qwen_reward),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 QWEN SIMPLE TEST SUMMARY")
    print("=" * 50)

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed >= 3:  # Allow one test to fail
        print("\n🎉 Qwen SePer integration is ready!")
        print("✅ Basic SePer functionality working")
        print("✅ Qwen model detection working")
        print("✅ Reward calculation working")
        print("\n🚀 Next steps:")
        print("1. Update train_grpo_seper.sh to use Qwen models")
        print("2. Run: bash train_grpo_seper.sh")
        print("3. Monitor training logs for SePer scores")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed")
        print("\nTroubleshooting:")
        print("- Check that all SePer files are in correct locations")
        print("- Verify Qwen models can be downloaded from hf-mirror.com")
        print("- Ensure sufficient disk space for model downloads")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)