import subprocess
import re
from collections import defaultdict
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

class PytestErrorFixer:
    def __init__(self, project_dir, max_retries=3):
        self.project_dir = project_dir
        self.max_retries = max_retries
        self.model = Model("gpt-4-turbo")
        self.io = InputOutput(yes=True)
        self.coder = Coder.create(main_model=self.model, io=self.io)

    def run_full_test(self):
        cmd = [
            "pytest",
            "--cov=gpt2_arc",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
            "gpt2_arc/"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout, result.stderr

    def parse_errors(self, output):
        error_pattern = r"(gpt2_arc/.*\.py)::\w+\s+(.*)"
        errors = re.findall(error_pattern, output)
        return defaultdict(list, errors)

    def predict_relevant_files(self, error):
        prompt = f"Which files are most likely involved in fixing this pytest error: {error}"
        response = self.coder.run(prompt)
        # Parse the response to extract file names
        # This is a simplistic approach and might need refinement
        files = re.findall(r"gpt2_arc/.*\.py", response)
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
        
        prompt = f"Fix this pytest error:\n\n{result.stdout}\n\n{result.stderr}"
        self.coder.run(prompt)
        
        # Run the test again to check if it's fixed
        result = subprocess.run(cmd, capture_output=True, text=True)
        return "PASSED" in result.stdout

    def main(self):
        stdout, stderr = self.run_full_test()
        errors = self.parse_errors(stdout + stderr)
        
        for test_file, error_list in errors.items():
            for error in error_list:
                relevant_files = self.predict_relevant_files(error)
                self.coder = Coder.create(main_model=self.model, io=self.io, fnames=relevant_files)
                
                for attempt in range(self.max_retries):
                    if self.fix_error(test_file, error):
                        print(f"Fixed: {test_file} - {error}")
                        break
                else:
                    print(f"Failed to fix after {self.max_retries} attempts: {test_file} - {error}")

        # Run full test suite again to verify all fixes
        final_stdout, final_stderr = self.run_full_test()
        print("Final test results:")
        print(final_stdout)
        print(final_stderr)

if __name__ == "__main__":
    fixer = PytestErrorFixer("path/to/your/project")
    fixer.main()
