#!/usr/bin/env python3
"""
Simple test to verify basic SePer integration structure
"""

import os
import sys

def main():
    print("🧪 Simple SePer Integration Test")
    print("=" * 50)

    # Get project root (should be Search-R1-info/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")

    # Test 1: Check key files exist
    print("\n📁 Checking key files...")
    key_files = [
        'verl/utils/reward_score/seper_reward.py',
        'verl/utils/reward_score/qa_em_seper.py',
        'verl/trainer/main_ppo_seper.py',
        'verl/trainer/config/ppo_trainer_seper.yaml',
        'train_grpo_seper.sh',
        'seper/seper/calculate.py',
        'seper/seper/models/huggingface_models.py'
    ]

    missing_files = []
    for file_path in key_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)

    # Test 2: Check Python import structure
    print("\n🐍 Checking Python imports...")

    try:
        # Add paths for testing
        sys.path.insert(0, os.path.join(project_root, 'verl', 'utils', 'reward_score'))
        sys.path.insert(0, os.path.join(project_root, 'seper'))

        # Test import of SePer reward calculator
        from seper_reward import SePerRewardCalculator
        print("✅ SePerRewardCalculator importable")

        # Test basic functionality
        calculator = SePerRewardCalculator(enabled=False)
        print("✅ SePerRewardCalculator instantiable (disabled)")

        # Test question/context extraction
        test_text = "Question: Test?\n<answer>Answer</answer>"
        question, context = calculator.extract_question_and_context(test_text)
        print(f"✅ Question extraction works: '{question}'")
        print(f"✅ Context extraction works: '{context}'")

        import_success = True
        import_error = None

    except Exception as e:
        print(f"❌ Import/Function test failed: {e}")
        import_success = False
        import_error = str(e)

    # Test 3: Check configuration files
    print("\n⚙️ Checking configuration...")
    config_files = [
        'train_grpo_seper.sh'
    ]

    for config_file in config_files:
        full_path = os.path.join(project_root, config_file)
        if os.path.exists(full_path):
            print(f"✅ {config_file}")
            # Check for SePer settings
            with open(full_path, 'r') as f:
                content = f.read()
                if 'seper_weight' in content:
                    print(f"   - Contains SePer configuration")
                if 'model_path=null' in content:
                    print(f"   - Auto model detection enabled")
        else:
            print(f"❌ {config_file}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    success_count = 0
    total_count = 3

    if len(missing_files) == 0:
        print("✅ Files structure: PASSED")
        success_count += 1
    else:
        print(f"❌ Files structure: FAILED ({len(missing_files)} missing)")

    if import_success:
        print("✅ Python imports: PASSED")
        success_count += 1
    else:
        print(f"❌ Python imports: FAILED ({import_error})")

    # Check if SePer settings exist
    train_script = os.path.join(project_root, 'train_grpo_seper.sh')
    if os.path.exists(train_script):
        with open(train_script, 'r') as f:
            content = f.read()
            if 'seper_weight=0.7' in content:
                print("✅ Configuration: PASSED")
                success_count += 1
            else:
                print("❌ Configuration: FAILED")
    else:
        print("❌ Configuration: FAILED")

    print(f"\nOverall: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Start retrieval server: bash retrieval_launch.sh")
        print("2. Run SePer-enabled training: bash train_grpo_seper.sh")
        return True
    else:
        print("\n⚠️ Some tests failed.")
        print("\nTroubleshooting:")
        print("- Check file paths above")
        print("- Ensure dependencies are installed:")
        print("  pip install transformers torch numpy hydra")
        print("- Check Python paths are correct")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)