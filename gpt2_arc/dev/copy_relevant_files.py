import os
import json
import subprocess
from pytest_error_fixer import PytestErrorFixer

def copy_to_clipboard(text):
    """Copy the given text to the clipboard."""
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(text.encode('utf-8'))

def main():
    fixer = PytestErrorFixer(project_dir=".")
    test_files = list(fixer.relevant_files_mapping.keys())

    print("Select a test file to copy relevant files to clipboard:")
    for idx, test_file in enumerate(test_files, start=1):
        print(f"{idx}. {test_file}")

    choice = int(input("Enter the number of the test file: ")) - 1
    if choice < 0 or choice >= len(test_files):
        print("Invalid choice. Exiting.")
        return

    selected_test_file = test_files[choice]
    relevant_files = fixer.relevant_files_mapping[selected_test_file]

    combined_content = ""
    for file_path in relevant_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                combined_content += f"\n\n# {file_path}\n\n"
                combined_content += f.read()
        else:
            print(f"File not found: {file_path}")

    copy_to_clipboard(combined_content)
    print("Relevant files' contents copied to clipboard.")

if __name__ == "__main__":
    main()
