#!/bin/bash

# Define the tmp directory within the working directory
TMP_DIR="/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/tmp"

# Ensure the tmp directory exists
mkdir -p $TMP_DIR

# Remove old output files in the tmp directory
rm -rf $TMP_DIR/compressed_output.txt $TMP_DIR/uncompressed_output.txt $TMP_DIR/test_output.log
rm -rf /Volumes/Totallynotaharddrive/arc-neural-reasoning-model/tmp/*.txt

# Correct path to the Python script
PYTHON_SCRIPT="/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/tmp/1filellm/onefilellm.py"

# Run the Python script if it exists
if [ -f "$PYTHON_SCRIPT" ]; then
    python "$PYTHON_SCRIPT" /Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc > $TMP_DIR/uncompressed_output.txt 2>&1
else
    echo "Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Run pytest with specified options and redirect logs to the tmp directory
cd /Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc
pytest -s -v --log-cli-level=DEBUG --capture=no --maxfail=1 > $TMP_DIR/test_output.log 2>&1

# Specify the path for compressed output
COMPRESSED_OUTPUT="$TMP_DIR/compressed_output.txt"
TEST_OUTPUT_LOG="$TMP_DIR/test_output.log"

# Create a temporary file to hold the combined contents
COMBINED_OUTPUT="$TMP_DIR/combined_output.txt"

# Combine the contents of both files into the combined output file
if [ -f "$COMPRESSED_OUTPUT" ] && [ -s "$COMPRESSED_OUTPUT" ]; then
    cat "$COMPRESSED_OUTPUT" > "$COMBINED_OUTPUT"
    echo "Contents of $COMPRESSED_OUTPUT added to combined output."
else
    echo "$COMPRESSED_OUTPUT not found or is empty."
fi

if [ -f "$TEST_OUTPUT_LOG" ] && [ -s "$TEST_OUTPUT_LOG" ]; then
    cat "$TEST_OUTPUT_LOG" >> "$COMBINED_OUTPUT"
    echo "Contents of $TEST_OUTPUT_LOG added to combined output."
else
    echo "$TEST_OUTPUT_LOG not found or is empty."
fi

# Copy the combined content to the clipboard
if [ -f "$COMBINED_OUTPUT" ] && [ -s "$COMBINED_OUTPUT" ]; then
    cat "$COMBINED_OUTPUT" | pbcopy
    echo "Combined contents copied to clipboard."
else
    echo "No content available to copy to clipboard."
fi