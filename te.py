import subprocess
import re
import os
import sys

# Configuration
COMMAND = [
    "mojo",
    "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/training/train.🔥",
    "--max_epochs",
    "1",
    "--fast_dev_run"
]
ERROR_FILE = "previous_errors.txt"

# Regular expression to match error lines
# Example error line:
# /path/to/file.mojo:401:76: error: unexpected character
error_regex = re.compile(r"^(.*?):(\d+):(\d+): error: (.*)$")

def run_command(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        return result.stdout, result.stderr
    except Exception as e:
        print(f"Failed to run command: {e}")
        sys.exit(1)

def parse_errors(stderr):
    errors = set()
    for line in stderr.splitlines():
        match = error_regex.match(line)
        if match:
            file_path, line_num, col_num, message = match.groups()
            # Create a unique identifier for each error
            error_id = f"{file_path}:{line_num}:{col_num}: {message.strip()}"
            errors.add(error_id)
    return errors

def load_previous_errors(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_current_errors(file_path, errors):
    with open(file_path, "w") as f:
        for error in sorted(errors):
            f.write(f"{error}\n")

def compare_errors(current, previous):
    new_errors = current - previous
    resolved_errors = previous - current
    return new_errors, resolved_errors

def main():
    print("Running Mojo command...")
    stdout, stderr = run_command(COMMAND)
    
    print("\nCommand Output:")
    print(stdout)
    
    print("\nParsing errors...")
    current_errors = parse_errors(stderr)
    print(f"Total Errors: {len(current_errors)}")
    
    previous_errors = load_previous_errors(ERROR_FILE)
    print(f"Previous Errors: {len(previous_errors)}")
    
    new_errors, resolved_errors = compare_errors(current_errors, previous_errors)
    
    if new_errors:
        print("\nNew Errors:")
        for error in sorted(new_errors):
            print(f"  - {error}")
    else:
        print("\nNo new errors found.")
    
    if resolved_errors:
        print("\nResolved Errors:")
        for error in sorted(resolved_errors):
            print(f"  - {error}")
    else:
        print("\nNo errors have been resolved.")
    
    # Save the current errors for the next run
    save_current_errors(ERROR_FILE, current_errors)

if __name__ == "__main__":
    main()
