#!/bin/bash

# Directory containing the files
directory="/workspaces/arc-neural-reasoning-model/gpt2_arc/src"

# Find all Python files in the directory and subdirectories, then iterate over each
find "$directory" -type f -name "*.py" | while read -r file; do
    # Run the py2mojo command on each file with all flags
    if py2mojo --float-precision 32 --extension mojo --convert-def-to-fn --convert-class-to-struct "$file" 2>&1 | tee /tmp/py2mojo_output.log | grep -q "Error: For converting a \"def\" function to \"fn\""; then
        # If "def to fn" error is detected, rerun without the --convert-def-to-fn flag
        echo "Def to fn error detected in $file. Retrying without --convert-def-to-fn."
        py2mojo --float-precision 32 --extension mojo --convert-class-to-struct "$file"
    else
        echo "Processed $file successfully."
    fi
done
