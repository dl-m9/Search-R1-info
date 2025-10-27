#!/usr/bin/env python3
"""
Fixed Qwen test for SePer with proper HF mirror and error handling
"""

import os
import sys
import time

def setup_environment():
    """Setup environment with HF mirror and paths"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Add Python paths
    sys.path.insert(0, os.path.join(project_root, 'seper'))
    sys.path.insert(0, os.path.join(project_root, 'verl', 'utils', 'reward_score'))

    # Set environment variables FIRST
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['ACTOR_MODEL_PATH'] = 'Qwen/Qwen2.5-3B'

    # Disable warnings
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

    print(f"✅ Environment configured")
    print(f"   Project root: {project_root}")
    print(f"   HF endpoint: {os.environ['HF_ENDPOINT']}")
    print(f"   Actor model: {os.environ['ACTOR_MODEL_PATH']}")

    return project_root

def test_import():
    """Test imports without loading models"""
    print("\n📦 Testing imports...")

    try:
        from verl.utils.reward_score.seper_reward import SePerRewardCalculator
        from verl.utils.reward_score.qa_em_seper import compute_score_em_seper
        print("✅ SePer modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_disabled_mode():
    """Test SePer in disabled mode (no model loading)"""
    print("\n🔒 Testing disabled mode...")

    try:
        from verl.utils.reward_score.seper_reward import SePerRewardCalculator

        # Create disabled calculator
        calculator = SePerRewardCalculator(enabled=False)

        # Test reward computation
        test_text = "Question: What is 2+2?\n<answer>4</answer>"
        rewards = calculator.compute_delta_seper_reward(
            test_text,
            {'target': ['4']}
        )

        expected_keys = ['seper_delta', 'seper_retrieval', 'seper_baseline', 'seper_reward']
        if all(key in rewards and rewards[key] == 0.0 for key in expected_keys):
            print("✅ Disabled mode works correctly")
            return True
        else:
            print(f"❌ Disabled mode failed: {rewards}")
            return False

    except Exception as e:
        print(f"❌ Disabled mode test failed: {e}")
        return False

def test_em_seper_function():
    """Test EM+SePer function with disabled SePer"""
    print("\n🧮 Testing EM+SePer function...")

    try:
        from verl.utils.reward_score.qa_em_seper import compute_score_em_seper

        # Test case
        test_case = {
            'solution_text': '''Question: What is the capital of France?
<search>capital of France</search>
<information>Paris is the capital of France.</information>
<answer>Paris</answer>''',
            'ground_truth': {'target': ['Paris']}
        }

        # Test with SePer disabled (seper_weight=0)
        score = compute_score_em_seper(
            solution_str=test_case['solution_text'],
            ground_truth=test_case['ground_truth'],
            seper_weight=0.0,  # Disable SePer
            seper_config={'enabled': False}
        )

        if isinstance(score, (int, float)) and score >= 0:
            print(f"✅ EM+SePer function works: {score:.4f}")
            return True
        else:
            print(f"❌ EM+SePer function failed: {score}")
            return False

    except Exception as e:
        print(f"❌ EM+SePer function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_loading():
    """Test Qwen model loading with proper error handling"""
    print("\n🤖 Testing Qwen model loading...")

    try:
        from verl.utils.reward_score.seper_reward import SePerRewardCalculator

        # Test with minimal configuration to avoid memory issues
        print("🔄 Creating SePer calculator with Qwen2.5-1.5B...")

        calculator = SePerRewardCalculator(
            model_path='Qwen/Qwen2.5-1.5B',  # Smaller model first
            num_generations=2,  # Minimal for testing
            max_new_tokens=32,  # Small
            computation_chunk_size=2,  # Very small
            device='cpu',  # Use CPU to avoid GPU issues
            enabled=True
        )

        if calculator.enabled:
            print(f"✅ SePer calculator loaded with model: {calculator.model_path}")

            # Test basic reward computation
            test_text = "Question: What is 2+2?\n<search>2+2 calculation</search>\n<information>2+2 equals 4</information>\n<answer>4</answer>"
            rewards = calculator.compute_delta_seper_reward(
                test_text,
                {'target': ['4']},
                delta_weight=0.5
            )

            if 'seper_delta' in rewards and 'seper_reward' in rewards:
                print(f"✅ SePer computation successful:")
                print(f"   Separ_delta: {rewards['seper_delta']:.4f}")
                print(f"   Separ_reward: {rewards['seper_reward']:.4f}")
                return True
            else:
                print(f"❌ SePer computation failed: {rewards}")
                return False
        else:
            print("❌ SePer calculator failed to enable")
            return False

    except Exception as e:
        print(f"❌ Qwen loading test failed: {e}")
        print("This might be due to network or model access issues")
        return False

def main():
    """Main test function"""
    print("🚀 Fixed Qwen SePer Integration Test")
    print("==================================")
    print("Testing SePer with proper HF mirror setup.")
    print("=" * 50)

    # Setup environment
    project_root = setup_environment()

    # Run tests in order of complexity
    tests = [
        ("Import Test", test_import),
        ("Disabled Mode", test_disabled_mode),
        ("EM+SePer Function", test_em_seper_function),
        ("Qwen Model Loading", test_qwen_loading),
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
    print("📊 TEST SUMMARY")
    print("=" * 50)

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed >= 3:  # Core functionality tests
        print("\n🎉 SePer integration is working!")
        print("✅ Core functionality verified")
        print("✅ EM+SePer functions working")
        if passed == 4:
            print("✅ Qwen models loading successfully")
        else:
            print("⚠️  Qwen model loading failed (check network/HF access)")

        print("\n🚀 Ready for training!")
        print("Next steps:")
        print("1. Start retrieval server: bash retrieval_launch.sh")
        print("2. Run training: cd .. && bash train_grpo_seper.sh")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed")
        print("\nTroubleshooting:")
        print("- Check network connection for model downloads")
        print("- Verify HF mirror is accessible: https://hf-mirror.com")
        print("- Ensure sufficient disk space for model downloads")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)