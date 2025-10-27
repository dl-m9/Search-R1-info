#!/bin/bash

# Test runner for SePer integration
# Runs all test suites in sequence

echo "🚀 SePer Integration Test Suite"
echo "This will run all test suites to verify SePer integration"
echo ""

# Function to run a test and check result
run_test() {
    local test_name="$1"
    local test_command="$2"

    echo "=========================================================="
    echo "Running: $test_name"
    echo "Command: $test_command"
    echo "=========================================================="
    echo ""

    if eval "$test_command"; then
        echo "✅ $test_name PASSED"
        return 0
    else
        echo "❌ $test_name FAILED"
        return 1
    failed_count=$((failed_count + 1))
}

# Initialize counters
total_count=0
failed_count=0

# Test 1: Quick functionality test
run_test "Quick Functionality Test" "cd .. && python seper_test/test_seper_quick.py"
total_count=$((total_count + 1))

echo ""
echo "⏸ Quick test completed. Proceeding with structure tests..."
echo ""

# Test 2: Minimal structure test
run_test "Minimal Structure Test" "cd .. && python seper_test/test_seper_minimal.py"
total_count=$((total_count + 1))

echo ""
echo "⏸ Structure test completed."
echo ""

# Test 3: Full integration test (optional - heavy)
read -p "Run full integration test? (requires models - may take time) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_test "Full Integration Test" "cd .. && python seper_test/test_seper_integration.py"
    total_count=$((total_count + 1))

    echo ""
    echo "⏸ Full integration test completed."
else
    echo "⏭ Skipping full integration test (heavy computation)"
fi

echo ""
echo "=========================================================="
echo "📊 FINAL TEST SUMMARY"
echo "=========================================================="
echo ""

if [ $failed_count -eq 0 ]; then
    echo "🎉 ALL $total_count TESTS PASSED!"
    echo ""
    echo "✅ SePer integration is ready for training"
    echo ""
    echo "Next steps:"
    echo "1. Start retrieval server: conda activate retriever && bash retrieval_launch.sh"
    echo "2. Run training: bash train_grpo_seper.sh"
else
    echo "⚠️  $failed_count/$total_count TESTS FAILED"
    echo ""
    echo "Please check the error messages above and fix issues before training."
    echo ""
    echo "Common issues and solutions:"
    echo "- Import errors: Check Python paths and install missing packages"
    echo "- File missing: Ensure all SePer files are in correct locations"
    echo "- Memory errors: Use smaller models or disable SePer for testing"
fi

echo ""
exit $failed_count