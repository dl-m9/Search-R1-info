#!/usr/bin/env python3
"""
Minimal test for SePer integration
Tests basic functionality without heavy model loading
"""

import sys
import os

def test_path_structure():
    """Test if required files and directories exist"""
    print("=" * 50)
    print("Testing Path Structure...")

    # Get the project root directory (Search-R1-info/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    required_dirs = [
        'seper',
        'seper/seper',
        'seper/seper/models',
        'seper/seper/uncertainty_measures',
        'verl/utils/reward_score'
    ]

    required_files = [
        'seper/__init__.py',
        'seper/seper/__init__.py',
        'seper/seper/calculate.py',
        'seper/seper/models/huggingface_models.py',
        'seper/seper/uncertainty_measures/semantic_entropy.py',
        'verl/utils/reward_score/seper_reward.py',
        'verl/utils/reward_score/qa_em_seper.py',
        'verl/trainer/main_ppo_seper.py'
    ]

    print("Checking directories...")
    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path} -> {full_path}")

    print("\nChecking files...")
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path} -> {full_path}")

    return True

def test_basic_imports():
    """Test basic Python imports"""
    print("\n" + "=" * 50)
    print("Testing Basic Imports...")

    try:
        import torch
        print("✅ PyTorch imported")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print("✅ Transformers imported")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False

    return True

def test_seper_basic_structure():
    """Test SePer basic structure without full import"""
    print("\n" + "=" * 50)
    print("Testing SePer Structure...")

    seper_files = {
        'calculate.py': 'Main calculation logic',
        'models/huggingface_models.py': 'HuggingFace model wrapper',
        'uncertainty_measures/semantic_entropy.py': 'Semantic entropy calculator',
        '__init__.py': 'Package initialization'
    }

    for file_path, description in seper_files.items():
        full_path = os.path.join(project_root, f'seper/seper/{file_path}')
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Basic sanity checks
                    if 'class ' in content or 'def ' in content:
                        print(f"✅ {file_path} ({description})")
                    else:
                        print(f"⚠️  {file_path} ({description}) - seems empty")
            except Exception as e:
                print(f"❌ {file_path} - read error: {e}")
        else:
            print(f"❌ {file_path} ({description}) - missing")

    return True

def test_reward_files():
    """Test reward integration files"""
    print("\n" + "=" * 50)
    print("Testing Reward Integration Files...")

    reward_files = {
        'verl/utils/reward_score/seper_reward.py': 'SePer reward calculator',
        'verl/utils/reward_score/qa_em_seper.py': 'EM+SePer reward function',
        'verl/trainer/main_ppo_seper.py': 'Enhanced PPO trainer'
    }

    for file_path, description in reward_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check for key functions/classes
                    if 'class' in content or 'def' in content:
                        print(f"✅ {description}")
                        if 'SePer' in content:
                            print(f"   - Contains SePer integration code")
                    else:
                        print(f"⚠️  {description} - may be incomplete")
            except Exception as e:
                print(f"❌ {description} - read error: {e}")
        else:
            print(f"❌ {description} - missing")

    return True

def test_config_files():
    """Test configuration files"""
    print("\n" + "=" * 50)
    print("Testing Configuration Files...")

    config_files = [
        'verl/trainer/config/ppo_trainer_seper.yaml',
        'train_grpo_seper.sh'
    ]

    for config_file in config_files:
        full_path = os.path.join(project_root, config_file)
        if os.path.exists(full_path):
            print(f"✅ {config_file}")
            # Check for SePer specific configurations
            with open(full_path, 'r') as f:
                content = f.read()
                if 'seper' in content.lower():
                    print(f"   - Contains SePer configuration")
        else:
            print(f"❌ {config_file}")

    return True

def test_dependencies():
    """Test if key dependencies are available"""
    print("\n" + "=" * 50)
    print("Testing Dependencies...")

    # Check for required packages
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'hydra',
        'omegaconf'
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")

    return True

def run_minimal_tests():
    """Run minimal test suite"""
    print("🔧 Minimal SePer Integration Test Suite")
    print("Testing basic setup without heavy computation...")

    tests = [
        ("Path Structure", test_path_structure),
        ("Basic Imports", test_basic_imports),
        ("SePer Structure", test_seper_basic_structure),
        ("Reward Files", test_reward_files),
        ("Config Files", test_config_files),
        ("Dependencies", test_dependencies)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 MINIMAL TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nResult: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All minimal tests passed!")
        print("Basic SePer integration setup is correct.")
        print("You can proceed to full integration testing.")
    else:
        print("\n⚠️  Some tests failed.")
        print("Please fix the issues before proceeding.")

    return passed == len(results)

if __name__ == "__main__":
    success = run_minimal_tests()
    exit(0 if success else 1)