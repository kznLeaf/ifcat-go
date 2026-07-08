#!/bin/bash

# 1. Check if the Go test function name is provided as an argument
if [ -z "$1" ]; then
    echo "Error: Please provide the Go test function name."
    echo "Usage: $0 <test_function_name> (e.g., $0 TestKdd99_10)"
    exit 1
fi

TEST_FUNC=$1
ITERATIONS=10
TOTAL_AUC=0

echo "=================================================="
echo " Starting evaluation for: $TEST_FUNC"
echo " Total planned iterations: $ITERATIONS"
echo "=================================================="

for i in $(seq 1 $ITERATIONS); do
    # 2. Run the Go test.
    # Added regex anchors ^ and $ to ensure exact matching of the function name.
    # Standard output is redirected to /dev/null to keep the console clean.
    go test -run "^${TEST_FUNC}\$" -v > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Iteration $i failed: 'go test' exited with an error. Please check your Go code or directory."
        exit 1
    fi

    # 3. Run the Python script to fetch the AUC result
    AUC=$(/opt/homebrew/bin/uv run /Users/kznleaf/Projects/Python/PythonProject/scripts/.venv/bin/python /Users/kznleaf/Projects/Python/PythonProject/scripts/draw_roc.py)
    
    echo "✅ Iteration $i complete | AUC: $AUC"

    # Use awk to handle floating-point accumulation
    TOTAL_AUC=$(awk "BEGIN {print $TOTAL_AUC + $AUC}")
done

# 4. Calculate and print the average AUC
AVG_AUC=$(awk "BEGIN {print $TOTAL_AUC / $ITERATIONS}")

echo "=================================================="
echo " 🎉 Evaluation Complete!"
echo " 🎯 Final Average AUC over $ITERATIONS runs: $AVG_AUC"
echo "=================================================="
