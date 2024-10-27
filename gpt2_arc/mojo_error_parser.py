import subprocess
import re
import os
import sys
import json

# Configuration
COMMAND = [
    "mojo",
    "/workspaces/arc-neural-reasoning-model/gpt2_arc/src/training/train.🔥",
    "--max_epochs",
    "1",
    "--fast_dev_run"
]
ERROR_FILE = "previous_errors.json"
TOP_ERROR_FILE = "top_error.json"

# Regular expression to match error lines
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
    errors = []
    lines = stderr.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = error_regex.match(line)
        if match:
            file_path, line_num, col_num, message = match.groups()
            # Initialize error details
            error_details = {
                "file_path": file_path,
                "line_num": line_num,
                "col_num": col_num,
                "message": message.strip(),
                "code_line": "",
                "caret": "",
                "notes": []
            }
            # Attempt to capture code line and caret
            if i + 2 < len(lines):
                error_details["code_line"] = lines[i + 1].strip()
                error_details["caret"] = lines[i + 2].strip()
                i += 3
            # Capture any following notes
            while i < len(lines) and lines[i].strip().startswith("note:"):
                error_details["notes"].append(lines[i].strip())
                i += 1
            errors.append(error_details)
        else:
            i += 1
    return errors

def load_previous_errors(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def save_current_errors(file_path, errors):
    with open(file_path, "w") as f:
        json.dump(sorted(errors, key=lambda x: (x['file_path'], x['line_num'], x['col_num'], x['message'])), f, indent=2)

def compare_errors(current, previous):
    current_ids = set(f"{e['file_path']}:{e['line_num']}:{e['col_num']}: {e['message']}" for e in current)
    previous_ids = set(f"{e['file_path']}:{e['line_num']}:{e['col_num']}: {e['message']}" for e in previous)
    
    new_errors = current_ids - previous_ids
    resolved_errors = previous_ids - current_ids
    return new_errors, resolved_errors

def load_previous_top_error(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)

def save_top_error(file_path, error):
    with open(file_path, "w") as f:
        json.dump(error, f, indent=2)

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
        for error_id in sorted(new_errors):
            # Find the error details
            error = next((e for e in current_errors if f"{e['file_path']}:{e['line_num']}:{e['col_num']}: {e['message']}" == error_id), None)
            if error:
                print(f"  - {error_id}")
                print(f"    {error['code_line']}")
                print(f"    {error['caret']}")
                for note in error['notes']:
                    print(f"    {note}")
    else:
        print("\nNo new errors found.")
    
    if resolved_errors:
        print("\nResolved Errors:")
        for error_id in sorted(resolved_errors):
            print(f"  - {error_id}")
    else:
        print("\nNo errors have been resolved.")
    
    # Save the current errors for the next run
    save_current_errors(ERROR_FILE, current_errors)
    
    # Handle the topmost error
    if current_errors:
        top_error = current_errors[0]  # Assuming the first parsed error is the topmost
        print(f"\nTopmost Error:")
        print(f"  - {top_error['file_path']}:{top_error['line_num']}:{top_error['col_num']}: {top_error['message']}")
        print(f"    {top_error['code_line']}")
        print(f"    {top_error['caret']}")
        for note in top_error['notes']:
            print(f"    {note}")
        
        previous_top_error = load_previous_top_error(TOP_ERROR_FILE)
        
        if top_error != previous_top_error:
            print("\nThe topmost error has changed since the last run.")
            if previous_top_error:
                print(f"Previous Top Error:\n  - {previous_top_error['file_path']}:{previous_top_error['line_num']}:{previous_top_error['col_num']}: {previous_top_error['message']}")
            else:
                print("No previous top error recorded.")
        else:
            print("\nThe topmost error remains the same as the last run.")
        
        # Save the current top error for the next run
        save_top_error(TOP_ERROR_FILE, top_error)
    else:
        print("\nNo errors to display as the topmost error.")

if __name__ == "__main__":
    main()
