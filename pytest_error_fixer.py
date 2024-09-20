import asyncio
import subprocess
import logging
import re
from collections import defaultdict
import json
import os
import uuid
from typing import List, Dict, Any, Tuple, Optional
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from dotenv import load_dotenv
from raptor.raptor_rag import Raptor_RAG_Wrapper

load_dotenv()  # This loads the variables from .env

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='pytest_error_fixer.log', filemode='a')

print("DEBUG: Imported all necessary modules")

class PytestErrorFixer:
    def __init__(self, project_dir, max_retries=3, progress_log="progress_log.json", initial_temperature=0.4, temperature_increment=0.1, args=None):
        """
        Initialize the PytestErrorFixer with configuration settings.

        Parameters:
        - project_dir (str): The path to the project directory.
        - max_retries (int): Maximum number of retries for fixing an error.
        - progress_log (str): Path to the progress log file.
        - initial_temperature (float): Initial temperature setting for the AI model.
        - temperature_increment (float): Increment for temperature on each retry.
        """
        self.args = args
        self.project_dir = os.path.abspath(project_dir)
        self.max_retries = max_retries
        self.progress_log = progress_log
        self.initial_temperature = initial_temperature
        self.temperature_increment = temperature_increment
        if args and args.verbose:
            print(f"DEBUG: Initialized PytestErrorFixer with initial_temperature={initial_temperature}, temperature_increment={temperature_increment}, max_retries={max_retries}")

        self.initial_temperature = initial_temperature
        self.temperature_increment = temperature_increment
        if args and args.verbose:
            print(f"DEBUG: Initialized PytestErrorFixer with initial_temperature={initial_temperature}, temperature_increment={temperature_increment}, max_retries={max_retries}")
        self.project_dir = os.path.abspath(project_dir)
        self.max_retries = max_retries
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.model = Model("gpt-4o-mini")
        self.io = InputOutput(yes=True)
        self.coder = Coder.create(main_model=self.model, io=self.io)
        self.progress_log = progress_log
        self.error_log = "error_log.json"
        self.branch_name = "pytest-aider-automation"
        self.ensure_branch()
        self.raptor_wrapper = Raptor_RAG_Wrapper()
        if args and args.verbose:
            print(f"DEBUG: Initialized PytestErrorFixer with Raptor_RAG_Wrapper")
        self.init_progress_log()
        
        if args and args.verbose:
            print("DEBUG: PytestErrorFixer initialization completed")

        # Define the relevant_files_mapping attribute
        self.relevant_files_mapping = {
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_benchmark.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/helpers.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/benchmark.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_gpt2.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_end_to_end.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/evaluate.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/benchmark.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/train.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_arc_dataset.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_differential_pixel_accuracy.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/helpers.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_train.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/train.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/benchmark.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/evaluate.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_trainer.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_results_collector.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/experiment_tracker.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/tests/test_model_evaluation.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/training/trainer.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/utils/helpers.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/data/arc_dataset.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/utils/experiment_tracker.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/src/utils/results_collector.py"
            ],
            "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/test_integration_experiment.py": [
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/data/arc_dataset.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/models/gpt2.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/training/trainer.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/config.py",
                "/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc/src/utils/results_collector.py"
            ]
        }

        self.initialize_raptor_wrapper()
    
    
    def check_test_results(self, stdout: str) -> Dict[str, Any]:
        print("DEBUG: Entering check_test_results method")
        print(f"DEBUG: stdout length: {len(stdout)}")
        
        # Look for the summary line
        summary_match = re.search(r'=+ (\d+) passed, (\d+) failed, (\d+) warning.* in .*', stdout)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2))
            warnings = int(summary_match.group(3))
            
            print(f"DEBUG: Found summary line. Passed: {passed}, Failed: {failed}, Warnings: {warnings}")
            return {
                "all_passed": failed == 0,
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            }
        else:
            print("DEBUG: Couldn't find summary line, falling back to simple check")
            # If we can't find the summary line, fall back to the previous method
            passed = "PASSED" in stdout or "passed" in stdout.lower()
            failed = "FAILED" in stdout or "failed" in stdout.lower()
            return {
                "all_passed": passed and not failed,
                "passed": None,  # We don't know the exact number
                "failed": None,
                "warnings": None
            }
    
    
    def verify_fix(self, branch_name: str, test_file: str, function: str) -> bool:
        """
        Verify if the fix for a specific test function in a test file was successful.

        Parameters:
        - branch_name (str): The name of the git branch to check.
        - test_file (str): The path to the test file.
        - function (str): The name of the test function.

        Returns:
        - bool: True if the fix was successful, False otherwise.
        """
        if args and args.verbose:
            print(f"DEBUG: Verifying fix on branch {branch_name} for {function} in {test_file}")
        try:
            subprocess.run(["git", "checkout", branch_name], cwd=self.project_dir, check=True)
            if args.verbose:
                print(f"DEBUG: Switched to branch {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to switch to branch {branch_name}: {str(e)}")
            return False

        stdout, stderr = self.run_test(test_file, function)
        if args and args.verbose:
            print(f"DEBUG: Test output:\n{stdout[:500]}...")  # Print first 500 characters
            print(f"DEBUG: Test error output:\n{stderr[:500]}...")  # Print first 500 characters

        test_results = self.check_test_results(stdout)
        print(f"DEBUG: Verification test results: {test_results}")

        if test_results["all_passed"]:
            if args.verbose:
                print(f"DEBUG: Fix verified successfully for {function} in {test_file}")
            return True
        else:
            if args.verbose:
                print(f"DEBUG: Fix verification failed for {function} in {test_file}")
            return False

    def check_git_status(self):
        """
        Check and return the current git status of the project directory.

        Returns:
        - str: The git status output.
        """
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.project_dir).decode('utf-8')
            print(f"DEBUG: Git status:\n{status}")
            return status
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to get git status: {str(e)}")
            return ""

    def get_relevant_files(self, test_file_path: str) -> List[str]:
        """
        Retrieve a list of relevant files associated with a given test file.

        Parameters:
        - test_file_path (str): The path to the test file.

        Returns:
        - List[str]: A list of relevant file paths.
        """
        # Strip the base path to match the dictionary keys
        relative_path = os.path.relpath(test_file_path, self.project_dir)
        
        # Replace backslashes with forward slashes for consistency
        relative_path = relative_path.replace('\\', '/')
        
        relevant_files = self.relevant_files_mapping.get(relative_path, [])
        
        # Convert relative paths to absolute paths
        return [os.path.join(self.project_dir, file) for file in relevant_files]

    def init_progress_log(self):
        """Initialize the progress log file if it doesn't already exist."""
        # Initialize the progress log file if it doesn't exist
        logging.info("Initializing progress log...")
        if not os.path.exists(self.progress_log):
            with open(self.progress_log, 'w') as f:
                json.dump([], f)

    def initialize_raptor_wrapper(self):
        """Initialize the Raptor_RAG_Wrapper with content from relevant files."""
        # Initialize the Raptor_RAG_Wrapper with relevant files
        for test_file, relevant_files in self.relevant_files_mapping.items():
            for file_path in relevant_files:
                full_path = os.path.join(self.project_dir, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as file:
                        content = file.read()
                        self.raptor_wrapper.add_documents(content)
                    print(f"DEBUG: Added content from {file_path} to Raptor_RAG_Wrapper")
                else:
                    print(f"DEBUG: File not found: {full_path}")
        print("DEBUG: Initialized Raptor_RAG_Wrapper with relevant files")

    def ensure_branch(self):
        """Ensure that the specified git branch exists and switch to it."""
        """Ensure that the branch exists and switch to it."""
        try:
            # Check if the branch exists
            branches = subprocess.check_output(["git", "branch"], cwd=self.project_dir).decode('utf-8')
            if self.branch_name not in branches:
                # Create the branch if it doesn't exist
                subprocess.run(["git", "checkout", "-b", self.branch_name], cwd=self.project_dir, check=True)
            else:
                # Switch to the branch
                subprocess.run(["git", "checkout", self.branch_name], cwd=self.project_dir, check=True)
            logging.info(f"Switched to branch: {self.branch_name}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to switch to branch {self.branch_name}: {str(e)}")
            raise

    def log_progress(self, status: str, error: Dict[str, Any], test_file: str, files_used: List[str], changes: str, temperature: float):
        logging.info(f"Logging progress: {status} for error in {test_file}")
        print(f"DEBUG: Logging progress - status: {status}, test_file: {test_file}")
        
        commit_sha = self.get_commit_sha()
        timestamp = self.get_current_timestamp()

        log_entry = {
            "error": error,
            "file": test_file,
            "status": status,
            "commit_sha": commit_sha,
            "timestamp": timestamp,
            "files_used": files_used,
            "changes": changes,
            "temperature": temperature
        }

        try:
            with open(self.progress_log, 'r+') as f:
                try:
                    log = json.load(f)
                except json.JSONDecodeError:
                    print("DEBUG: Progress log is empty or invalid. Initializing new log.")
                    log = []
                
                log.append(log_entry)
                f.seek(0)
                json.dump(log, f, indent=4)
                f.truncate()
            print(f"DEBUG: Successfully logged progress to {self.progress_log}")
        except Exception as e:
            print(f"ERROR: Failed to log progress: {str(e)}")

        print(f"DEBUG: Logged progress - status: {status}, test_file: {test_file}, temperature: {temperature}")


    def get_git_status(self) -> str:
        """
        Retrieve the current git status.

        Returns:
        - str: The git status output.
        """
        """Retrieve the current git status."""
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.project_dir).decode('utf-8')
            print(f"DEBUG: Git status retrieved: {status[:100]}...")  # Print first 100 characters
            return status
        except subprocess.CalledProcessError:
            logging.warning("Failed to get git status. Is this a git repository?")
            return ""

    def get_commit_sha(self) -> str:
        """
        Get the current commit SHA of the project directory.

        Returns:
        - str: The commit SHA.
        """
        try:
            for attempt in range(self.max_retries):
                temperature = self.initial_temperature + (attempt * self.temperature_increment)
                logging.info(f"Attempt {attempt + 1}/{self.max_retries} with temperature {temperature:.2f}")
                return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.project_dir).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            logging.warning("Failed to get commit SHA. Is this a git repository?")
            return "N/A"

    def get_current_timestamp(self) -> str:
        """
        Get the current timestamp in a specific format.

        Returns:
        - str: The current timestamp.
        """
        return subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).strip().decode('utf-8')

    def discover_test_files(self) -> List[str]:
        """
        Discover and return a list of test files in the project directory.

        Returns:
        - List[str]: A list of test file paths.
        """
        test_files = []
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
        return test_files

    def run_test(self, test_file: str, function: str = None) -> Tuple[str, str]:
        """
        Run a specific test file or function and return the output and error.

        Parameters:
        - test_file (str): The path to the test file.
        - function (str, optional): The specific test function to run.

        Returns:
        - Tuple[str, str]: The standard output and error output from the test run.
        """
        cmd = ["pytest", "-v", "--tb=short", "--log-cli-level=DEBUG"]
        if function:
            cmd.append(f"{test_file}::{function}")
        else:
            cmd.append(test_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)
        return result.stdout, result.stderr

    def parse_errors(self, output: str, test_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse errors from the test output and return them in a structured format.

        Parameters:
        - output (str): The combined standard output and error output from the test run.
        - test_file (str): The path to the test file.

        Returns:
        - Dict[str, List[Dict[str, Any]]]: A dictionary mapping file paths to lists of error details.
        """
        print(f"DEBUG: Entering parse_errors method")
        print(f"DEBUG: Output length: {len(output)}")

        # Extract the 'FAILURES' section from the log
        failures_match = re.search(r"={10,} FAILURES ={10,}\n(.*?)(?=\n=|$)", output, re.DOTALL)
        if failures_match:
            failures_section = failures_match.group(1)
            print(f"DEBUG: Found FAILURES section. Length: {len(failures_section)}")
        else:
            print("DEBUG: No FAILURES section found in output")
            return {}

        # Use regex to find all failure blocks along with their test names
        failure_matches = re.findall(
            r"_{10,}\s+(.*?)\s+_{10,}\n(.*?)(?=\n_{10,}|\Z)",
            failures_section,
            re.DOTALL
        )

        parsed_errors = defaultdict(list)

        for test_name, block in failure_matches:
            failure_info = self.parse_failure_block(block)
            if failure_info:
                file_path = failure_info['file_path']
                parsed_errors[file_path].append({
                    'function': failure_info['function'],
                    'error_type': failure_info['error_type'],
                    'error_details': failure_info['error_details'],
                    'line_number': failure_info['line_number'],
                    'code_snippet': failure_info['code_snippet'],
                    'captured_output': failure_info.get('captured_output', ''),
                    'captured_log': failure_info.get('captured_log', ''),
                    'test_file': test_file
                })

        print(f"DEBUG: Parsed {len(parsed_errors)} errors")
        for file, errors in parsed_errors.items():
            print(f"DEBUG: File {file} has {len(errors)} errors")

        return parsed_errors

    def parse_failure_block(self, block: str) -> Optional[Dict[str, Any]]:
        """
        Parse a block of failure information from the test output.

        Parameters:
        - block (str): A block of text containing failure information.

        Returns:
        - Optional[Dict[str, Any]]: A dictionary with parsed failure details, or None if parsing fails.
        """
        print(f"DEBUG: Entering parse_failure_block method")
        print(f"DEBUG: Block length: {len(block)}")

        # Split the block into sections based on captured outputs/logs
        sections = re.split(r"(-{10,}.*?-{10,})\n", block)
        content_sections = []
        i = 0
        while i < len(sections):
            if not sections[i].startswith('-----'):
                content_sections.append(sections[i])
            i += 1

        main_content = content_sections[0].strip().split('\n')
        # Skip empty lines at the start
        while main_content and not main_content[0].strip():
            main_content = main_content[1:]
        if not main_content:
            return None

        # First line should be file path with line number and function
        file_line_func_match = re.match(r'(.*?):(\d+): in (.*)', main_content[0])
        if not file_line_func_match:
            return None
        file_path = file_line_func_match.group(1)
        line_number = file_line_func_match.group(2)
        function = file_line_func_match.group(3)

        print(f"DEBUG: Parsed file path: {file_path}, line number: {line_number}, function: {function}")

        # The rest of the lines are code snippet and error messages
        code_snippet = []
        error_messages = []
        i = 1
        while i < len(main_content):
            line = main_content[i]
            if line.startswith('E   '):
                # Start collecting error messages
                error_messages.append(line[1:].strip())  # Remove one leading space
            else:
                code_snippet.append(line)
            i += 1
        error_message = '\n'.join(error_messages)
        error_type = error_message.split(':')[0] if error_message else ''

        print(f"DEBUG: Parsed error type: {error_type}")
        print(f"DEBUG: Error message length: {len(error_message)}")

        # Collect captured outputs/logs if available
        captured_output = ''
        captured_log = ''
        for idx, section in enumerate(sections):
            if 'Captured stdout call' in section:
                captured_output = sections[idx + 1].strip()
            if 'Captured log call' in section:
                captured_log = sections[idx + 1].strip()

        return {
            'file_path': file_path,
            'line_number': line_number,
            'function': function,
            'code_snippet': '\n'.join(code_snippet),
            'error_type': error_type.strip(),
            'error_details': error_message.strip(),
            'captured_output': captured_output,
            'captured_log': captured_log
        }

    def save_errors(self, new_errors: Dict[str, List[Dict[str, Any]]]):
        """
        Save new errors to the error log, merging with existing errors.

        Parameters:
        - new_errors (Dict[str, List[Dict[str, Any]]]): New errors to save.
        """
        if os.path.exists(self.error_log):
            with open(self.error_log, 'r') as f:
                existing_errors = json.load(f)
        else:
            existing_errors = {}

        # Merge new errors with existing errors, avoiding duplicates
        for file_path, errors in new_errors.items():
            if file_path not in existing_errors:
                existing_errors[file_path] = errors
            else:
                for error in errors:
                    if error not in existing_errors[file_path]:
                        existing_errors[file_path].append(error)

        with open(self.error_log, 'w') as f:
            json.dump(existing_errors, f, indent=4)

    def load_errors(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load errors from the error log file.

        Returns:
        - Dict[str, List[Dict[str, Any]]]: A dictionary of errors loaded from the log.
        """
        # Load errors from JSON log
        logging.info("Loading errors from log...")
        if os.path.exists(self.error_log):
            with open(self.error_log, 'r') as f:
                errors = json.load(f)
                logging.debug("Loaded errors: %s", errors)
                return errors
        else:
            logging.info("Error log not found. Creating a new one.")
            with open(self.error_log, 'w') as f:
                json.dump({}, f)
            return {}

    def extract_file_paths_from_errors(self, errors: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """
        Extract file paths from error details and return them.

        Parameters:
        - errors (Dict[str, List[Dict[str, Any]]]): A dictionary of errors.

        Returns:
        - Dict[str, List[str]]: A dictionary mapping error keys to lists of relevant file paths.
        """
        error_file_paths = {}
        for file_path, error_list in errors.items():
            for error in error_list:
                relevant_paths = set()
                absolute_file_path = os.path.abspath(os.path.join(self.project_dir, file_path))
                relevant_paths.add(absolute_file_path)
                
                # Extract paths from error details and code snippet
                paths = re.findall(r'gpt2_arc/[^\s:]+\.py', error['error_details'] + error['code_snippet'])
                absolute_paths = [os.path.abspath(os.path.join(self.project_dir, p)) for p in paths]
                relevant_paths.update(absolute_paths)
                
                if 'test_file' in error:
                    relevant_paths.add(os.path.abspath(os.path.join(self.project_dir, error['test_file'])))
                
                # Add predefined relevant files for the specific test file
                test_file_name = os.path.basename(error['test_file'])
                if test_file_name in self.relevant_files_mapping:
                    for rel_path in self.relevant_files_mapping[test_file_name]:
                        abs_path = os.path.abspath(os.path.join(self.project_dir, rel_path))
                        relevant_paths.add(abs_path)
                
                error_key = f"{error['function']} - {error['error_type']}"
                error_file_paths[error_key] = list(relevant_paths)
            
            logging.debug(f"Extracted file paths for {file_path}: {error_file_paths}")
        return error_file_paths

    async def fix_error(self, error: Dict[str, Any], file_path: str) -> Tuple[str, str, str, str]:
        """
        Attempt to fix a specific error in a file and return the results.

        Parameters:
        - error (Dict[str, Any]): The error details.
        - file_path (str): The path to the file containing the error.

        Returns:
        - Tuple[str, str, str, str]: The branch name, AI responses, standard output, and error output.
        """
        logging.debug(f"Starting fix_error for {error['function']} in {file_path}")
        all_ai_responses = []
        print(f"DEBUG: Starting fix_error for {error['function']} in {file_path}")
        
        subprocess.run(["git", "stash"], cwd=self.project_dir, check=True)
        branch_name = f"fix-{error['function']}-{uuid.uuid4().hex[:8]}"
        logging.debug(f"Creating branch: {branch_name}")
        print(f"DEBUG: Creating branch: {branch_name}")
        
        try:
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=self.project_dir, check=True)
            print(f"DEBUG: Successfully created and switched to branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to create or switch to branch {branch_name}: {str(e)}")
            return branch_name, "", "", ""

        logging.info(f"Attempting to fix error in {file_path}: {error['function']}")
        print(f"DEBUG: Attempting to fix error in {file_path}: {error['function']}")

        all_relevant_files = self.get_relevant_files(error['test_file'])
        print(f"DEBUG: Relevant files: {all_relevant_files}")

        self.coder = Coder.create(main_model=self.model, io=self.io, fnames=all_relevant_files)
        print(f"DEBUG: Created new Coder instance with model: {self.model}")

        for attempt in range(self.max_retries):
            temperature = self.initial_temperature + (attempt * self.temperature_increment)
            logging.info(f"Attempt {attempt + 1}/{self.max_retries} with temperature {temperature:.2f}")
            print(f"DEBUG: Starting attempt {attempt + 1}/{self.max_retries} with temperature {temperature:.2f}")

            # Log the start of the fix attempt
            self.log_progress("attempt_start", error, file_path, all_relevant_files, f"Starting attempt {attempt + 1}", temperature)

            initial_git_status = self.get_git_status()
            print(f"DEBUG: Initial git status:\n{initial_git_status}")

            stdout, stderr = self.run_test(error['test_file'], error['function'])
            print(f"DEBUG: Test output before fix attempt:\n{stdout[:500]}...")
            print(f"DEBUG: Test error output before fix attempt:\n{stderr[:500]}...")

            prompt = await self.construct_prompt(error, stdout, stderr, attempt)
            print(f"DEBUG: Constructed prompt (first 500 chars):\n{prompt[:500]}...")

            try:
                response = self.coder.run(prompt)
                print(f"DEBUG: AI model response received (first 500 chars):\n{response[:500]}...")
                
                changes = self.parse_aider_response(response)
                print(f"DEBUG: Parsed changes:\n{changes}")

                # Log the AI response
                self.log_progress("ai_response", error, file_path, all_relevant_files, changes, temperature)

                if not changes:
                    print("DEBUG: No changes detected. Prompting AI to execute its plan.")
                    execute_plan_prompt = (
                        f"{prompt}\n\n"
                        "The previous attempt did not result in any code changes. "
                        "Please execute your plan and provide specific search and replace statements to fix the error."
                    )
                    response = self.coder.run(execute_plan_prompt)
                    changes = self.parse_aider_response(response)
                    print(f"DEBUG: Changes after explicit prompt:\n{changes}")

                    # Log the additional AI response
                    self.log_progress("ai_response_additional", error, file_path, all_relevant_files, changes, temperature)
                
                # Apply the changes

                stdout, stderr = self.run_test(error['test_file'], error['function'])
                print(f"DEBUG: Test output after fix attempt:\n{stdout[:500]}...")
                print(f"DEBUG: Test error output after fix attempt:\n{stderr[:500]}...")

                # Commit changes regardless of test result
                try:
                    subprocess.run(["git", "add", "."], cwd=self.project_dir, check=True)
                    commit_message = f"Fix attempt {attempt + 1} for {error['function']} in {file_path}"
                    subprocess.run(["git", "commit", "-m", commit_message], cwd=self.project_dir, check=True)
                    print(f"DEBUG: Committed changes for attempt {attempt + 1}")
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Failed to commit changes: {str(e)}")

                test_results = self.check_test_results(stdout)
                print(f"DEBUG: Test results: {test_results}")

                if test_results["all_passed"]:
                    print(f"DEBUG: All tests passed for {error['function']} in {file_path}")
                    print(f"DEBUG: Passed: {test_results['passed']}, Failed: {test_results['failed']}, Warnings: {test_results['warnings']}")
                    print(f"DEBUG: Full test output:\n{stdout}")
                    self.log_progress("fixed", error, file_path, all_relevant_files, changes, temperature)
                    return branch_name, "\n".join(all_ai_responses), stdout, stderr
                else:
                    print(f"DEBUG: Tests still failing for {error['function']} in {file_path}")
                    print(f"DEBUG: Passed: {test_results['passed']}, Failed: {test_results['failed']}, Warnings: {test_results['warnings']}")
                    self.log_progress("failed", error, file_path, all_relevant_files, changes, temperature)

            except Exception as e:
                logging.error(f"Error while applying changes: {str(e)}")
                print(f"DEBUG: Exception occurred: {str(e)}")
                # Commit even if an exception occurred
                try:
                    subprocess.run(["git", "add", "."], cwd=self.project_dir, check=True)
                    commit_message = f"Failed fix attempt {attempt + 1} for {error['function']} in {file_path} (Exception occurred)"
                    subprocess.run(["git", "commit", "-m", commit_message], cwd=self.project_dir, check=True)
                    print(f"DEBUG: Committed changes for failed attempt {attempt + 1}")
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Failed to commit changes after exception: {str(e)}")

                # Log the exception
                self.log_progress("exception", error, file_path, all_relevant_files, str(e), temperature)

        print(f"DEBUG: All fix attempts completed for {error['function']} in {file_path}")
        return branch_name, "\n".join(all_ai_responses), stdout, stderr

    def parse_aider_response(self, response: str) -> str:
        """
        Parse the AI model's response to extract search/replace statements.

        Parameters:
        - response (str): The response from the AI model.

        Returns:
        - str: A JSON string representing the parsed changes.
        """
        """Parse the Aider response to extract search/replace statements."""
        changes = []
        lines = response.split('\n')
        in_search = False
        in_replace = False
        current_change = {"file": "", "search": "", "replace": ""}

        for line in lines:
            if line.strip().endswith(".py"):
                current_change["file"] = line.strip()
            elif line.strip() == "<<<<<<< SEARCH":
                in_search = True
                in_replace = False
                current_change["search"] = ""
            elif line.strip() == "=======":
                in_search = False
                in_replace = True
                current_change["replace"] = ""
            elif line.strip() == ">>>>>>> REPLACE":
                in_replace = False
                changes.append(current_change.copy())
                current_change = {"file": "", "search": "", "replace": ""}
            elif in_search:
                current_change["search"] += line + "\n"
            elif in_replace:
                current_change["replace"] += line + "\n"

        return json.dumps(changes, indent=2)


    async def summarize_relevant_files(self, test_file: str) -> str:
        """
        Summarize the contents and purpose of relevant files for a test file.

        Parameters:
        - test_file (str): The path to the test file.

        Returns:
        - str: A summary of the relevant files.
        """
        print(f"DEBUG: Entering summarize_relevant_files for {test_file}")
        print(f"DEBUG: Test file path: {test_file}")
        print(f"DEBUG: Relevant files mapping keys: {self.relevant_files_mapping.keys()}")
        relevant_files = self.relevant_files_mapping.get(test_file, [])
        print(f"DEBUG: Relevant files: {relevant_files}")
        summaries = []

        for file in relevant_files:
            try:
                full_path = os.path.join(self.project_dir, file)
                print(f"DEBUG: Attempting to summarize file: {full_path}")
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"DEBUG: File content length: {len(content)}")
                    summary = await asyncio.to_thread(
                        self.raptor_wrapper.answer_question,
                        f"Provide a brief summary of this file's contents and purpose. File: {file}\n\nContent:\n{content[:1000]}..."  # Limit to first 1000 characters
                    )
                    summaries.append(f"{file}:\n{summary}")
                else:
                    print(f"DEBUG: File not found: {full_path}")
                    summaries.append(f"{file}: File not found")
            except Exception as e:
                print(f"DEBUG: Error summarizing {file}: {str(e)}")
                summaries.append(f"{file}: Error summarizing - {str(e)}")

        full_summary = "\n\n".join(summaries)
        print(f"DEBUG: Full summary length: {len(full_summary)}")
        return full_summary

    async def construct_prompt(self, error: Dict[str, Any], stdout: str, stderr: str, attempt: int) -> str:
        """
        Construct a prompt for the AI model to fix an error.

        Parameters:
        - error (Dict[str, Any]): The error details.
        - stdout (str): The standard output from the test run.
        - stderr (str): The error output from the test run.
        - ai_responses (str): Previous AI responses.
        - attempt (int): The current attempt number.

        Returns:
        - str: The constructed prompt.
        """
        print(f"DEBUG: Entering construct_prompt for attempt {attempt}")
        
        error_prompt = (
            f"Fix pytest error in {error['test_file']} - {error['function']}:\n"
            f"Type: {error['error_type']}\n"
            f"Details: {error['error_details']}...\n"
            f"Code: {error['code_snippet']}...\n"
        )
        print(f"DEBUG: Error prompt length: {len(error_prompt)}")

        analysis_prompt = (
            "Analyze and fix using:\n"
            "1. Analysis: Understand error context.\n"
            "2. Framework: Develop fix approach.\n"
            "3. Plan: Outline specific fix.\n"
            "4. Execution: Implement and verify.\n"
        )
        print(f"DEBUG: Analysis prompt length: {len(analysis_prompt)}")

        test_summary = await self.summarize_test_output(stdout, stderr)
        print(f"DEBUG: Test summary length: {len(test_summary)}")

        relevant_files_summary = await self.summarize_relevant_files(error['test_file'])
        print(f"DEBUG: Relevant files summary length: {len(relevant_files_summary)}")
        print(f"DEBUG: Relevant files summary content:\n{relevant_files_summary}")

        git_history_note = (
            f"Note: This is attempt {attempt + 1}. Previous fix attempts (if any) "
            "have been committed to the git history. Please analyze the current "
            "state of the code and propose a new approach if necessary."
        )
        print(f"DEBUG: Git history note length: {len(git_history_note)}")

        full_prompt = (
            f"{error_prompt}\n{analysis_prompt}\n"
            f"Relevant Files Summary:\n{relevant_files_summary}\n"
            f"Test Summary:\n{test_summary}\n"
            f"{git_history_note}"
        )
        print(f"DEBUG: Full prompt length before truncation: {len(full_prompt)}")

        truncated_prompt = self.trunc_to_token_limit(full_prompt)
        print(f"DEBUG: Truncated prompt length: {len(truncated_prompt)}")
        return truncated_prompt

    def debug_fix_single_error(self, error: Dict[str, Any], file_path: str) -> bool:
        """
        Debug and attempt to fix a single error in a file.

        Parameters:
        - error (Dict[str, Any]): The error details.
        - file_path (str): The path to the file containing the error.

        Returns:
        - bool: True if the error was fixed, False otherwise.
        """
        print(f"DEBUG: Starting debug_fix_single_error for {error['function']} in {file_path}")

        # Extract relevant files for this specific error
        error_dict = {file_path: [error]}
        relevant_files = self.extract_file_paths_from_errors(error_dict)

        # Flatten the list of relevant files
        all_relevant_files = list(set(path for paths in relevant_files.values() for path in paths))

        print("DEBUG: Relevant files for this error:")
        for path in all_relevant_files:
            print(f"  - {path}")

        # Create a new Coder instance with the relevant files for this error
        print("DEBUG: Creating new Coder instance")
        debug_coder = Coder.create(main_model=self.model, io=self.io, fnames=all_relevant_files)

        # Run the test to get the error output
        cmd = ["pytest", "-v", "--tb=short", "--log-cli-level=DEBUG", f"{error['test_file']}::{error['function']}"]
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Debug: Print test output
        print("\nDEBUG: Initial test output:")
        print(result.stdout)
        print("\nDEBUG: Initial test error output:")
        print(result.stderr)

        # Construct a structured prompt using parsed error details
        error_prompt_template = (
            "Fix the following pytest error:\n\n"
            "Test File: {test_file}\n"
            "Function: {function}\n"
            "Error Type: {error_type}\n"
            "Error Details: {error_details}\n"
            "Code Snippet:\n{code_snippet}\n"
            "Captured Output:\n{captured_output}\n"
            "Captured Log:\n{captured_log}\n"
        )
        
        error_prompt = error_prompt_template.format(
            test_file=file_path,
            function=error['function'],
            error_type=error['error_type'],
            error_details=error['error_details'],
            code_snippet=error['code_snippet'],
            captured_output=error.get('captured_output', ''),
            captured_log=error.get('captured_log', '')
        )
        
        # Analysis framework prompt
        analysis_prompt = (
            "Please analyze the error using the Analysis, Framework, Plan, Execution System:\n"
            "1. Analysis: Provide a detailed understanding of the error context.\n"
            "2. Framework: Develop a framework for addressing the error.\n"
            "3. Plan: Outline a specific plan to fix the error.\n"
            "4. Execution: Implement the plan and verify the fix.\n"
        )
        
        # Combine the error details with the analysis framework
        combined_prompt = f"{error_prompt}\n\n{analysis_prompt}"
        
        # Send the structured prompt to aider
        print("DEBUG: Sending structured prompt to AI model")
        try:
            debug_coder.run(combined_prompt)
            print("DEBUG: AI model suggested changes. Applying changes...")
        except Exception as e:
            print(f"DEBUG: Error while applying changes: {str(e)}")
            # Optionally log the failed attempt
            self.log_progress("failed", error, file_path, all_relevant_files, "No changes", 0.0)
            return False

        # Run the test again to check if it's fixed
        print("\nDEBUG: Re-running test to check if error is fixed...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Debug: Print re-run test output
        print("\nDEBUG: Re-run test output:")
        print(result.stdout)
        print("\nDEBUG: Re-run test error output:")
        print(result.stderr)

        if "PASSED" in result.stdout:
            print(f"DEBUG: Fixed: {file_path} - {error['function']}")
            # Log the successful fix
            self.log_progress("fixed", error, file_path, all_relevant_files, "Changes applied", 0.0)
            return True
        else:
            print(f"DEBUG: Failed to fix: {file_path} - {error['function']}")
            return False
    
    
    async def summarize_test_output(self, stdout: str, stderr: str) -> str:
        """
        Summarize the test output, focusing on errors and context.

        Parameters:
        - stdout (str): The standard output from the test run.
        - stderr (str): The error output from the test run.

        Returns:
        - str: A summary of the test output.
        """
        print(f"DEBUG: Entering summarize_test_output")
        print(f"DEBUG: stdout length: {len(stdout)}, stderr length: {len(stderr)}")
        
        full_output = stdout + "\n" + stderr
        print(f"DEBUG: Combined output length: {len(full_output)}")
        
        try:
            print("DEBUG: Attempting to summarize with RAPTOR")
            summary = await asyncio.to_thread(
                self.raptor_wrapper.answer_question,
                f"Summarize the following pytest output, focusing on error messages, line numbers, and brief context. Limit the summary to about 200 words:\n\n{full_output}"
            )
            print(f"DEBUG: RAPTOR summary length: {len(summary)}")
            return f"Test Output Summary:\n{summary}"
        except Exception as e:
            print(f"DEBUG: Error in summarize_test_output: {str(e)}")
            logging.error(f"Error summarizing test output: {str(e)}")
            return f"Test Output Summary: Error occurred while summarizing - {str(e)}"

    def trunc_to_token_limit(self, text: str, max_tokens: int = 9500) -> str:
        """
        Truncate text to fit within a specified token limit.

        Parameters:
        - text (str): The text to truncate.
        - max_tokens (int): The maximum number of tokens allowed.

        Returns:
        - str: The truncated text.
        """
        print(f"DEBUG: Entering trunc_to_token_limit")
        print(f"DEBUG: Input text length: {len(text)}")
        # Simple approximation: assume 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            truncated = text[:max_chars] + "..."
            print(f"DEBUG: Text truncated. New length: {len(truncated)}")
            return truncated
        print("DEBUG: Text not truncated")
        return text
    
    
    def display_progress_log(self):
        print("DEBUG: Displaying contents of progress log")
        try:
            with open(self.progress_log, 'r') as f:
                log = json.load(f)
                print(json.dumps(log, indent=2))
        except Exception as e:
            print(f"ERROR: Failed to read progress log: {str(e)}")

    def generate_fix_report(self, successful_fixes, failed_fixes):
        timestamp = self.get_current_timestamp()
        report_filename = "fix_report.txt"
        
        print(f"DEBUG: Generating fix report with timestamp {timestamp}")
        
        with open(report_filename, 'a') as report:
            report.write(f"\n\n--- Fix Report: {timestamp} ---\n")
            report.write("Successful Fixes:\n")
            for fix in successful_fixes:
                report.write(f"- {fix[1]} in {fix[0]} (Branch: {fix[2]})\n")
            
            report.write("\nFailed Fixes:\n")
            for fix in failed_fixes:
                report.write(f"- {fix[1]} in {fix[0]} (Branch: {fix[2]})\n")
        
        print(f"DEBUG: Fix report appended to {report_filename}")

    def print_git_diff(self):
        try:
            diff = subprocess.check_output(["git", "diff"], cwd=self.project_dir).decode('utf-8')
            print(f"DEBUG: Git diff:\n{diff[:500]}...")  # Print first 500 characters
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to get git diff: {str(e)}")

    def print_file_contents(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            print(f"DEBUG: Contents of {file_path} (first 500 chars):\n{content[:500]}...")
        except Exception as e:
            print(f"ERROR: Failed to read file {file_path}: {str(e)}")


async def run_error_fixing_process(self):
    print("DEBUG: Starting error fixing process")
    # Delete the existing error log if it exists
    if os.path.exists(self.error_log):
        os.remove(self.error_log)
        print(f"DEBUG: Deleted existing error log: {self.error_log}")
    
    # Load existing errors from the log (which will now be empty)
    all_errors = self.load_errors()
    if all_errors:
        logging.info("Existing errors found in error log. Skipping test execution.")
    else:
        test_files = self.discover_test_files()
        logging.info(f"Discovered {len(test_files)} test files.")

        for test_file in test_files:
            logging.info(f"Running test file: {test_file}")
            stdout, stderr = self.run_test(test_file)
            errors = self.parse_errors(stdout + stderr, test_file)
            if errors:
                logging.info(f"Errors found in {test_file}. Saving to log...")
                self.save_errors(errors)
            else:
                logging.info(f"No errors found in {test_file}")

        # Reload errors after running tests
        all_errors = self.load_errors()
    logging.info(f"Loaded errors: {json.dumps(all_errors, indent=2)}")

    successful_fixes = []
    failed_fixes = []

    print(f"DEBUG: Starting to process {len(all_errors)} files with errors")

    for file_path, error_list in all_errors.items():
        print(f"DEBUG: Processing file: {file_path} with {len(error_list)} errors")
        for error in error_list:
            print(f"DEBUG: Processing error: {error['function']} in {file_path}")
            
            if args and args.debug_single_error:
                branch, ai_responses, stdout, stderr = await self.debug_fix_single_error(error, file_path)
            else:
                branch, ai_responses, stdout, stderr = await self.fix_error(error, file_path)
            
            # Verify the fix
            if self.verify_fix(branch, error['test_file'], error['function']):
                successful_fixes.append((file_path, error['function'], branch))
                print(f"DEBUG: Fix successful for {error['function']} in {file_path}")
            else:
                failed_fixes.append((file_path, error['function'], branch))
                print(f"DEBUG: Fix failed for {error['function']} in {file_path}")
            
            print(f"DEBUG: Finished processing error: {error['function']} in {file_path}")
            print(f"DEBUG: Branch: {branch}, stdout length: {len(stdout)}, stderr length: {len(stderr)}")

    print("DEBUG: Finished processing all errors")
    print("DEBUG: Displaying final progress log")
    self.display_progress_log()

    print("DEBUG: Generating fix report")
    with open("fix_report.txt", "w") as report:
        report.write("Successful Fixes:\n")
        for fix in successful_fixes:
            report.write(f"- {fix[1]} in {fix[0]} (Branch: {fix[2]})\n")
        
        report.write("\nFailed Fixes:\n")
        for fix in failed_fixes:
            report.write(f"- {fix[1]} in {fix[0]} (Branch: {fix[2]})\n")
    
    # Replace the existing fix report generation with the new method
    self.generate_fix_report(successful_fixes, failed_fixes)

    logging.info("Error fixing and verification completed.")
    
    # Switch back to master branch after completion
    try:
        subprocess.run(["git", "checkout", "master"], cwd=self.project_dir, check=True)
        logging.info("Switched back to master branch.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch back to master branch: {str(e)}")
    
    print("DEBUG: Main process completed. Check the logs for details.")


async def main():
    """Initialize and run the PytestErrorFixer with command-line arguments."""
    fixer = PytestErrorFixer(
        args.project_dir,
        initial_temperature=args.initial_temperature,
        temperature_increment=args.temperature_increment,
        max_retries=args.max_retries,
        args=args
    )
    await fixer.run_error_fixing_process()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PytestErrorFixer")
    parser.add_argument("project_dir", help="Path to the project directory")
    parser.add_argument("--initial-temperature", type=float, default=0.4, help="Initial temperature setting for the AI model (default: 0.4)")
    parser.add_argument("--temperature-increment", type=float, default=0.1, help="Temperature increment for each retry (default: 0.1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of fix attempts per error (default: 3)")
    parser.add_argument("--debug-single-error", action="store_true", help="Use debug_fix_single_error instead of fix_error")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"DEBUG: Starting PytestErrorFixer with arguments: {args}")

        asyncio.run(main())


