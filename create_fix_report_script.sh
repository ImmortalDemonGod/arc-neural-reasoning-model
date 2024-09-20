#!/bin/bash

# Set the path to your project
PROJECT_PATH="/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc"

# Function to count failures in the most recent report section
count_recent_failures() {
    if [ ! -f fix_report.txt ]; then
        echo 0
        return
    fi
    tac fix_report.txt | sed '/--- Fix Report:/q' | tac | grep -c -- "- "
}

# Check if fix_report.txt exists; if not, run the script to create it
if [ ! -f fix_report.txt ]; then
    echo "fix_report.txt not found. Generating initial fix report..."
    python pytest_error_fixer.py --verbose --debug --max-retries 1 "$PROJECT_PATH" >> test.txt 2>&1
fi

# Get the initial failure count
initial_failures=$(count_recent_failures)
initial_failures=${initial_failures:-0}

echo "Initial failures: $initial_failures"

# Calculate the target failure count
target_failures=$((initial_failures / 2))
target_failures=${target_failures:-0}

echo "Initial failures: $initial_failures"
echo "Target failures: $target_failures"

current_failures=$initial_failures
iteration=1

while [ $current_failures -gt $target_failures ]; do
    echo "Iteration $iteration: Current failures: $current_failures"
    
    # Run the pytest_error_fixer script
    python pytest_error_fixer.py --verbose --debug --max-retries 6 "$PROJECT_PATH" >> test.txt 2>&1
    
    # Update the current failure count
    current_failures=$(count_recent_failures)
    
    # Increment the iteration counter
    ((iteration++))
    
    # Optional: Add a delay between runs if needed
    # sleep 10
done

echo "Process completed. Final failure count: $current_failures"
