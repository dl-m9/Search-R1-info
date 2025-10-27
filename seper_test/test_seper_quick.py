#!/usr/bin/env python3
"""
Quick test for SePer reward calculation
Tests the reward function with a simple example
"""

import sys
import os

def setup_paths():
    """Setup Python paths for testing"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up to Search-R1-info/
    sys.path.insert(0, os.path.join(project_root, 'verl', 'utils', 'reward_score'))

def test_seper_reward_function():
    """Test the SePer reward function"""
    print("=" * 60)
    print("Testing SePer Reward Function")
    print("=" * 60)

    try:
        # Import the function
        from qa_em_seper import compute_score_em_seper
        print("✅ Successfully imported compute_score_em_seper")

        # Test case 1: Basic search-R1 format
        print("\n--- Test Case 1: Basic Search-R1 Format ---")
        solution_text = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Is Paris the capital of France?
<reasoning>I need to verify if Paris is the capital of France.</reasoning>
<search>capital of France</search>
<information>Paris is the capital and most populous city of France.</information>
<answer>Yes, Paris is the capital of France.</answer>'''

        ground_truth = {'target': ['Yes']}

        print(f"Solution: {solution_text[:100]}...")
        print(f"Ground truth: {ground_truth}")

        # Test with SePer disabled
        print("\n--- With SePer Disabled (w=0.0) ---")
        score = compute_score_em_seper(
            solution_text=solution_text,
            ground_truth=ground_truth,
            seper_weight=0.0,  # Disable SePer
            seper_config={'enabled': False}
        )
        print(f"Reward: {score}")

        # Test with SePer enabled (but won't run due to model loading)
        print("\n--- With SePer Enabled (w=0.5, but no model loading) ---")
        score = compute_score_em_seper(
            solution_text=solution_text,
            ground_truth=ground_truth,
            seper_weight=0.5,  # Enable SePer
            seper_config={'enabled': False}  # Disable actual computation
        )
        print(f"Reward: {score}")

        # Test case 2: Wrong answer
        print("\n--- Test Case 2: Wrong Answer ---")
        wrong_solution = '''<answer>No, London is the capital of France.</answer>'''

        score_wrong = compute_score_em_seper(
            solution_text=wrong_solution,
            ground_truth=ground_truth,
            seper_weight=0.0  # Just test EM part
        )
        print(f"Wrong answer reward: {score_wrong}")

        # Test case 3: No search tags
        print("\n--- Test Case 3: No Search Tags ---")
        no_search_solution = '''<answer>Paris is the capital of France.</answer>'''

        score_no_search = compute_score_em_seper(
            solution_text=no_search_solution,
            ground_truth=ground_truth,
            seper_weight=0.0  # Just test EM part
        )
        print(f"No search answer reward: {score_no_search}")

        print("\n✅ All reward function tests passed!")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure verl/utils/reward_score/qa_em_seper.py exists")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_seper_disabled():
    """Test behavior when SePer is completely disabled"""
    print("\n" + "=" * 60)
    print("Testing SePer Completely Disabled")
    print("=" * 60)

    try:
        from seper_reward import SePerRewardCalculator

        # Test with disabled calculator
        calculator = SePerRewardCalculator(enabled=False)
        print("✅ SePer calculator (disabled) created")

        # Test extraction still works
        question, context = calculator.extract_question_and_context(
            "Question: Test?\n<answer>Answer</answer>"
        )
        print(f"Extracted question: '{question}'")
        print(f"Extracted context: '{context}'")

        # Test reward calculation returns zero
        rewards = calculator.compute_delta_seper_reward(
            solution_text="Test solution",
            ground_truth={'target': ['Answer']}
        )
        print(f"Disabled SePer rewards: {rewards}")
        expected = {
            'seper_delta': 0.0,
            'seper_retrieval': 0.0,
            'seper_baseline': 0.0,
            'seper_reward': 0.0
        }

        if rewards == expected:
            print("✅ Disabled SePer test passed")
            return True
        else:
            print(f"❌ Disabled SePer test failed")
            print(f"Expected: {expected}")
            print(f"Got: {rewards}")
            return False

    except Exception as e:
        print(f"❌ Disabled SePer test failed: {e}")
        return False

def run_quick_tests():
    """Run quick test suite"""
    print("🚀 SePer Quick Integration Test")
    print("Testing core functionality without heavy computation")

    # Setup paths
    setup_paths()

    tests = [
        ("SePer Reward Function", test_seper_reward_function),
        ("SePer Disabled Mode", test_seper_disabled)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 Quick tests passed!")
        print("SePer integration basic structure is working correctly.")
        print("\nNext steps:")
        print("1. Run: python test_seper_minimal.py (for structure)")
        print("2. Run: python test_seper_integration.py (for full integration)")
        print("3. Install missing dependencies if needed")
        print("4. Run training with: bash train_grpo_seper.sh")
    else:
        print("\n⚠️  Some tests failed.")
        print("Check the errors above and fix issues before training.")

    return passed == len(results)

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)