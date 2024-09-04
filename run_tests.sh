#!/bin/bash

# Define the tmp directory within the working directory
TMP_DIR="./tmp/"

# Ensure the tmp directory exists
mkdir -p $TMP_DIR

# Remove old output files in the tmp directory
rm -rf $TMP_DIR/compressed_output.txt $TMP_DIR/uncompressed_output.txt $TMP_DIR/test_output.log

# Correct path to the Python script
PYTHON_SCRIPT="./tmp/1filellm/onefilellm.py"

# Run the Python script if it exists
if [ -f "$PYTHON_SCRIPT" ]; then
    python "$PYTHON_SCRIPT" /workspaces/arc-neural-reasoning-model/gpt2_arc > $TMP_DIR/uncompressed_output.txt 2>&1
else
    echo "Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# Run pytest with specified options and redirect logs to the tmp directory
cd ./gpt2_arc/
pytest -s -v --log-cli-level=ERROR > $TMP_DIR/test_output.log 2>&1

# Check if xclip is installed
if ! command -v xclip &> /dev/null
then
    echo "xclip could not be found. Please install it using your package manager (e.g., sudo apt install xclip) to enable clipboard functionality."
    exit 1
fi

# Create an empty compressed_output.txt if it doesn't exist
touch $TMP_DIR/compressed_output.txt

# Copy the new compressed_output.txt and pytest results to clipboard
cat $TMP_DIR/compressed_output.txt | xclip -selection clipboard
cat $TMP_DIR/test_output.log | xclip -selection clipboard

echo "New output files copied to clipboard successfully."
