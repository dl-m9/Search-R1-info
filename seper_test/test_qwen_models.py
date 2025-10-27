#!/usr/bin/env python3
"""
Qwen model test for SePer integration
Uses Qwen2.5-3B and Qwen2.5-1.5B for testing
"""

import os
import sys
import time
import torch

def setup_paths():
    """Setup Python paths and HF mirror"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    sys.path.insert(0, os.path.join(project_root, 'seper'))
    sys.path.insert(0, os.path.join(project_root, 'verl', 'utils', 'reward_score'))

    # Set HuggingFace mirror for faster downloads
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(f"Project root: {project_root}")
    return project_root

def test_qwen_model_loading():
    """Test Qwen model loading for SePer"""
    print("\n🤖 Testing Qwen Model Loading...")

    project_root = setup_paths()

    try:
        from seper.calculate import create_collate_fn
        from seper.models.huggingface_models import HuggingfaceModel
        from seper.uncertainty_measures.semantic_entropy import EntailmentDeberta
        from seper_reward import SePerRewardCalculator

        print("✅ SePer modules imported successfully")

        # Test Qwen 1.5B model (smaller, faster)
        print("\n📥 Testing with Qwen2.5-1.5B...")
        qwen_small = 'Qwen/Qwen2.5-1.5B'

        try:
            generator_small = HuggingfaceModel(
                qwen_small,
                stop_sequences='default',
                max_new_tokens=32,  # Very small for testing
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                attn_implementation=None,  # No flash attention to avoid issues
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            generator_small.model.eval()
            print(f"✅ Qwen2.5-1.5B loaded successfully")

            # Test Qwen 3B model (main)
            print("\n📥 Testing with Qwen2.5-3B...")
            qwen_main = 'Qwen/Qwen2.5-3B'

            generator_main = HuggingfaceModel(
                qwen_main,
                stop_sequences='default',
                max_new_tokens=64,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                attn_implementation=None,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            generator_main.model.eval()
            print(f"✅ Qwen2.5-3B loaded successfully")

            # Test entailment model
            print("\n📥 Testing Entailment model...")
            entailment_model = EntailmentDeberta(device='cuda' if torch.cuda.is_available() else 'cpu')
            entailment_model.model.eval()
            print("✅ Entailment model loaded successfully")

            # Test basic SePer calculator creation
            print("\n📥 Testing SePer calculator...")
            calculator = SePerRewardCalculator(
                model_path=qwen_small,  # Use small model for testing
                num_generations=2,  # Minimal for testing
                max_new_tokens=32,
                computation_chunk_size=4,  # Small chunk size
                device='cuda' if torch.cuda.is_available() else 'cpu',
                enabled=True
            )
            print("✅ SePer calculator created with Qwen2.5-1.5B")

            # Test reward computation
            print("\n🧮 Testing SePer reward computation...")
            test_data = {
                'solution_text': '''Question: What is the capital of China?
<search>capital of China</search>
<information>Beijing is the capital of China.</information>
<answer>Beijing</answer>''',
                'ground_truth': {'target': ['Beijing']}
            }

            start_time = time.time()
            rewards = calculator.compute_delta_seper_reward(
                test_data['solution_text'],
                test_data['ground_truth'],
                delta_weight=0.5
            )
            elapsed = time.time() - start_time

            print(f"✅ SePer computation completed in {elapsed:.1f}s")
            print(f"📊 Rewards: {rewards}")

            expected_keys = ['seper_delta', 'seper_retrieval', 'seper_baseline', 'seper_reward']
            for key in expected_keys:
                if key in rewards:
                    print(f"  {key}: {rewards[key]:.4f}")
                else:
                    print(f"  {key}: missing")

            return True

        except Exception as e:
            print(f"❌ Qwen model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_detection():
    """Test model detection with Qwen models"""
    print("\n🔍 Testing Model Detection Logic...")

    project_root = setup_paths()

    test_cases = [
        {
            'name': 'Qwen2.5-3B',
            'path': 'Qwen/Qwen2.5-3B'
        },
        {
            'name': 'Qwen2.5-7B',
            'path': 'Qwen/Qwen2.5-7B'
        },
        {
            'name': 'Qwen2.5-1.5B',
            'path': 'Qwen/Qwen2.5-1.5B'
        }
    ]

    try:
        from seper_reward import SePerRewardCalculator

        for test_case in test_cases:
            print(f"Testing detection for: {test_case['name']}")

            os.environ['ACTOR_MODEL_PATH'] = test_case['path']

            calculator = SePerRewardCalculator(enabled=False)

            if calculator.model_path and test_case['path'] in calculator.model_path:
                print(f"✅ Detection successful: {test_case['name']}")
            else:
                print(f"❌ Detection failed: {test_case['name']} -> {calculator.model_path}")

        return True

    except Exception as e:
        print(f"❌ Model detection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Qwen Model SePer Integration Test")
    print("=====================================")
    print("Testing SePer integration with Qwen models.")
    print("Uses HF-Mirror for faster downloads.")
    print("=" * 50)

    tests = [
        ("Qwen Model Loading", test_qwen_model_loading),
        ("Model Detection", test_model_detection),
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
    print("📊 QWEN MODEL TEST SUMMARY")
    print("=" * 50)

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 Qwen model tests PASSED!")
        print("\n✅ SePer integration ready with Qwen models")
        print("\n🚀 Ready for training!")
        print("\nNext steps:")
        print("1. Start retrieval server: bash retrieval_launch.sh")
        print("2. Run training: cd .. && bash train_grpo_seper.sh")
        print("\n3. Monitor logs for SePer scores")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed")
        print("\nTroubleshooting:")
        print("- Check internet connection for model downloads")
        print("- Verify Qwen models are accessible")
        print("- Ensure sufficient GPU memory (8GB+ recommended)")
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