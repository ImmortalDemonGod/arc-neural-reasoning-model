#!/bin/bash

# Define possible base directories
BASE_DIR_1="/Volumes/Totallynotaharddrive/arc-neural-reasoning-model"
BASE_DIR_2="/workspaces/arc-neural-reasoning-model"

# Function to check if a directory exists
dir_exists() {
    [ -d "$1" ]
}

# Determine which base directory to use
if dir_exists "$BASE_DIR_1"; then
    BASE_DIR="$BASE_DIR_1"
elif dir_exists "$BASE_DIR_2"; then
    BASE_DIR="$BASE_DIR_2"
else
    echo "Error: Neither $BASE_DIR_1 nor $BASE_DIR_2 exists."
    exit 1
fi

# Define the tmp directory within the working directory
TMP_DIR="$BASE_DIR/tmp"

# Ensure the tmp directory exists
mkdir -p "$TMP_DIR"

# Remove old output files in the tmp directory
rm -rf "$TMP_DIR"/compressed_output.txt "$TMP_DIR"/uncompressed_output.txt "$TMP_DIR"/test_output.log
rm -rf "$BASE_DIR"/tmp/*.txt

# Check if GITHUB_TOKEN is set, if not, generate a random token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN not set. Generating a random token."
    export GITHUB_TOKEN=$(openssl rand -hex 16)
fi

PYTHON_SCRIPT="$BASE_DIR/tmp/1filellm/onefilellm.py"

# Run the Python script if it exists
if [ -f "$PYTHON_SCRIPT" ]; then
    python "$PYTHON_SCRIPT" "$BASE_DIR/gpt2_arc" > "$TMP_DIR/uncompressed_output.txt" 2>&1
else
    echo "Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Run pytest with specified options and redirect logs to the tmp directory
cd "$BASE_DIR/gpt2_arc"
pytest -s -v --log-cli-level=ERROR --capture=no --maxfail=2 > "$TMP_DIR/test_output.log" 2>&1

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
    if command -v pbcopy &> /dev/null; then
        cat "$COMBINED_OUTPUT" | pbcopy
        echo "Combined contents copied to clipboard using pbcopy."
    elif command -v xclip &> /dev/null; then
        cat "$COMBINED_OUTPUT" | xclip -selection clipboard
        echo "Combined contents copied to clipboard using xclip."
    else
        echo "Unable to copy to clipboard. Neither pbcopy nor xclip is available."
    fi
else
    echo "No content available to copy to clipboard."
fi
