import subprocess
import re
from collections import defaultdict
import json
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import os

class PytestErrorFixer:
    def __init__(self, project_dir, max_retries=3, progress_log="progress_log.json"):
        print(f"Initializing PytestErrorFixer with project directory: {project_dir}")
        self.project_dir = project_dir
        self.max_retries = max_retries
        self.model = Model("gpt-4-turbo")
        self.io = InputOutput(yes=True)
        self.coder = Coder.create(main_model=self.model, io=self.io)
        self.progress_log = progress_log
        self.error_log = "error_log.json"
        self.init_progress_log()

    def init_progress_log(self):
        # Initialize the progress log file if it doesn't exist
        print("Initializing progress log...")
        if not os.path.exists(self.progress_log):
            with open(self.progress_log, 'w') as f:
                json.dump([], f)

    def log_progress(self, status, error, test_file):
        # Log the progress of fixing an error
        print(f"Logging progress: {status} for error in {test_file}")
        with open(self.progress_log, 'r+') as f:
            log = json.load(f)
            log.append({"error": error, "file": test_file, "status": status})
            f.seek(0)
            json.dump(log, f, indent=4)

    def run_full_test(self):
        cmd = [
            "pytest",
            "--cov=gpt2_arc",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            "gpt2_arc/"
        ]
        # for now will will just us this test_cmd to test the code : pytest gpt2_arc/tests/test_end_to_end.py -v --tb=short
        test_cmd = [
            "pytest",
            "gpt2_arc/tests/test_end_to_end.py",
            "-v",
            "--tb=short"
        ]
        print("Running full test suite...")
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        print("Test suite completed.")
        print("Parsing errors from test output...")
        print("Test stdout:", result.stdout)
        print("Test stderr:", result.stderr)
        return result.stdout, result.stderr

    def parse_errors(self, output):
        error_pattern = r"(gpt2_arc/.*\.py)::\w+\s+(.*)"
        errors = re.findall(error_pattern, output)
        print("Parsed errors:", errors)
        return defaultdict(list, errors)

    def save_errors(self, errors):
        # Save parsed errors in a JSON log for scalability
        print("Saving errors to log...")
        with open(self.error_log, 'w') as f:
            json.dump(errors, f, indent=4)

    def load_errors(self):
        # Load errors from JSON log
        print("Loading errors from log...")
        with open(self.error_log, 'r') as f:
            errors = json.load(f)
            print("Loaded errors:", errors)
            return errors

    def predict_relevant_files(self, error):
        print(f"Predicting relevant files for error: {error}")
        prompt = f"Which files are most likely involved in fixing this pytest error: {error}"
        response = self.coder.run(prompt)
        # Extract relevant files from aider's prediction
        files = re.findall(r"gpt2_arc/.*\.py", response)
        print("Predicted relevant files:", files)
        return files

    def fix_error(self, test_file, error):
        cmd = [
            "pytest",
            "-v",
            "--tb=short",
            "--log-cli-level=DEBUG",
            f"{test_file}::{error.split()[0]}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Prompt aider to suggest a fix based on test output
        prompt = f"Fix this pytest error:\n\n{result.stdout}\n\n{result.stderr}"
        self.coder.run(prompt)

        # Run the test again to check if it's fixed
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "PASSED" in result.stdout:
            print(f"Error fixed in {test_file}: {error}")
        else:
            print(f"Error not fixed in {test_file}: {error}")
        print("Fix result stdout:", result.stdout)
        print("Fix result stderr:", result.stderr)
        return "PASSED" in result.stdout

    def main(self):
        # Run full test suite and parse errors
        print("Starting main process...")
        stdout, stderr = self.run_full_test()
        print("Initial test run completed.")
        errors = self.parse_errors(stdout + stderr)
        print("Errors parsed. Saving to log...")
        self.save_errors(errors)
        print("Errors saved.")
"""
        # Process each error
        for test_file, error_list in errors.items():
            for error in error_list:
                print(f"Processing error: {error} in {test_file}")
                relevant_files = self.predict_relevant_files(error)
                print(f"Relevant files predicted: {relevant_files}")
                self.coder = Coder.create(main_model=self.model, io=self.io, fnames=relevant_files)

                for attempt in range(self.max_retries):
                    if self.fix_error(test_file, error):
                        print(f"Fixed: {test_file} - {error}")
                        self.log_progress("fixed", error, test_file)
                        print(f"Successfully fixed: {test_file} - {error}")
                        break
                    else:
                        print(f"Retry {attempt + 1} failed for: {test_file} - {error}")
                else:
                    print(f"Failed to fix after {self.max_retries} attempts: {test_file} - {error}")
                    self.log_progress("failed", error, test_file)

        # Run the full test suite again to verify all fixes
        print("Re-running full test suite to verify fixes...")
        final_stdout, final_stderr = self.run_full_test()
        print("Final test suite run completed.")
        print("Final test results:")
        print(final_stdout)
        print(final_stderr)
"""



if __name__ == "__main__":
    fixer = PytestErrorFixer("/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc")
    fixer.main()
