import subprocess
import logging
import re
from collections import defaultdict
import json
import os
import os
import logging
import re
from typing import List, Dict, Any
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PytestErrorFixer:
    def __init__(self, project_dir, max_retries=3, progress_log="progress_log.json", initial_temperature=0.4, temperature_increment=0.1):
        self.initial_temperature = initial_temperature
        self.temperature_increment = temperature_increment
        print(f"DEBUG: Initialized PytestErrorFixer with initial_temperature={initial_temperature}, temperature_increment={temperature_increment}, max_retries={max_retries}")
        self.project_dir = project_dir
        self.max_retries = max_retries
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.model = Model("gpt-4o-mini")
        self.io = InputOutput(yes=True)
        self.coder = Coder.create(main_model=self.model, io=self.io)
        self.progress_log = progress_log
        self.error_log = "error_log.json"
        self.init_progress_log()

        # Define a mapping of test files to their relevant files
        self.relevant_files_mapping = {
            "gpt2_arc/tests/test_benchmark.py": [
                "gpt2_arc/benchmark.py",
                "gpt2_arc/src/models/gpt2.py",
                "gpt2_arc/src/config.py",
                "gpt2_arc/src/data/arc_dataset.py",
                "gpt2_arc/src/utils/experiment_tracker.py",
                "gpt2_arc/src/utils/results_collector.py",
                "gpt2_arc/src/training/train.py",
                "gpt2_arc/src/training/trainer.py"
            ],
            "gpt2_arc/tests/test_end_to_end.py": [
                "gpt2_arc/src/data/arc_dataset.py",
                "gpt2_arc/src/models/gpt2.py",
                "gpt2_arc/src/training/trainer.py",
                "gpt2_arc/src/config.py",
                "gpt2_arc/src/utils/experiment_tracker.py",
                "gpt2_arc/src/utils/results_collector.py",
                "gpt2_arc/src/evaluate.py",
                "gpt2_arc/benchmark.py",
                "gpt2_arc/train.py",
                # Add more relevant files as needed
            ],
            "gpt2_arc/tests/test_arc_dataset.py": [
                "gpt2_arc/src/data/arc_dataset.py",
                "gpt2_arc/src/utils/experiment_tracker.py",
                "gpt2_arc/src/models/gpt2.py",
                "gpt2_arc/src/config.py",
            ],
            "gpt2_arc/tests/test_differential_pixel_accuracy.py": [
                "gpt2_arc/src/utils/helpers.py",
                "gpt2_arc/src/models/gpt2.py",
                "gpt2_arc/src/config.py",
                "gpt2_arc/src/data/arc_dataset.py",
                # Add more relevant files as needed
            ],
            "gpt2_arc/tests/test_train.py": [
                "gpt2_arc/src/training/train.py",
                "gpt2_arc/src/training/trainer.py",
                "gpt2_arc/src/models/gpt2.py",
                "gpt2_arc/src/data/arc_dataset.py",
                "gpt2_arc/src/config.py",
                "gpt2_arc/src/utils/results_collector.py",
                "gpt2_arc/src/utils/experiment_tracker.py",
                "gpt2_arc/benchmark.py",
                "gpt2_arc/src/evaluate.py"
            ],
            "gpt2_arc/tests/test_trainer.py": [
                "gpt2_arc/src/config.py",
                "gpt2_arc/src/data/arc_dataset.py",
                "gpt2_arc/src/models/gpt2.py",
                "gpt2_arc/src/training/trainer.py",
                "gpt2_arc/src/utils/experiment_tracker.py",
                "gpt2_arc/src/utils/results_collector.py"
        }

    def init_progress_log(self):
        # Initialize the progress log file if it doesn't exist
        logging.info("Initializing progress log...")
        if not os.path.exists(self.progress_log):
            with open(self.progress_log, 'w') as f:
                json.dump([], f)

    def log_progress(self, status: str, error: Dict[str, Any], test_file: str, files_used: List[str], changes: str, temperature: float):
        logging.info(f"Logging progress: {status} for error in {test_file}")
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

        with open(self.progress_log, 'r+') as f:
            log = json.load(f)
            log.append(log_entry)
            f.seek(0)
            json.dump(log, f, indent=4)

        print(f"DEBUG: Logged progress - status: {status}, test_file: {test_file}, temperature: {temperature}")

    def get_git_status(self) -> str:
        """Retrieve the current git status."""
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.project_dir).decode('utf-8')
            return status
        except subprocess.CalledProcessError:
            logging.warning("Failed to get git status. Is this a git repository?")
            return ""

    def get_commit_sha(self) -> str:
        try:
            for attempt in range(self.max_retries):
                temperature = self.initial_temperature + (attempt * self.temperature_increment)
                logging.info(f"Attempt {attempt + 1}/{self.max_retries} with temperature {temperature:.2f}")
                return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.project_dir).strip().decode('utf-8')
        except subprocess.CalledProcessError:
            logging.warning("Failed to get commit SHA. Is this a git repository?")
            return "N/A"

    def get_current_timestamp(self) -> str:
        return subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).strip().decode('utf-8')

    def discover_test_files(self):
        test_files = []
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
        return test_files

    def run_test(self, test_file: str, function: str = None) -> tuple:
        cmd = ["pytest", "-v", "--tb=short", "--log-cli-level=DEBUG"]
        if function:
            cmd.append(f"{test_file}::{function}")
        else:
            cmd.append(test_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_dir)
        return result.stdout, result.stderr

    def parse_errors(self, output, test_file):
        # Extract the 'FAILURES' section from the log
        failures_match = re.search(r"={10,} FAILURES ={10,}\n(.*?)(?=\n=|$)", output, re.DOTALL)
        if failures_match:
            failures_section = failures_match.group(1)
        else:
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

        print("Parsed errors:", dict(parsed_errors))
        return parsed_errors

    def parse_failure_block(self, block):
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

    def save_errors(self, new_errors):
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

    def load_errors(self):
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

    def extract_file_paths_from_errors(self, errors):
        error_file_paths = {}
        for file_path, error_list in errors.items():
            for error in error_list:
                relevant_paths = set()
                absolute_file_path = os.path.abspath(file_path)
                relevant_paths.add(absolute_file_path)
                
                # Extract paths from error details and code snippet
                paths = re.findall(r'gpt2_arc/[^\s:]+\.py', error['error_details'] + error['code_snippet'])
                absolute_paths = [os.path.abspath(os.path.join(self.project_dir, p)) for p in paths]
                relevant_paths.update(absolute_paths)
                
                if 'test_file' in error:
                    relevant_paths.add(os.path.abspath(error['test_file']))
                
                error_key = f"{error['function']} - {error['error_type']}"
                error_file_paths[error_key] = list(relevant_paths)
            
            print(f"DEBUG: Extracted file paths for {file_path}: {error_file_paths}")
        return error_file_paths

    def fix_error(self, error: Dict[str, Any], file_path: str) -> bool:
        logging.info(f"Attempting to fix error in {file_path}: {error['function']}")

        error_dict = {file_path: [error]}
        relevant_files = self.extract_file_paths_from_errors(error_dict)
        all_relevant_files = list(set(path for paths in relevant_files.values() for path in paths))

        # Add predefined relevant files for the specific test file
        test_file_relevant_files = self.relevant_files_mapping.get(error['test_file'], [])
        all_relevant_files.extend(test_file_relevant_files)
        test_file_name = os.path.basename(error['test_file'])
        debug_tips_file = os.path.join(self.project_dir, "debug tips", f"{test_file_name.replace('.py', '.md')}")
        if os.path.exists(debug_tips_file):
            all_relevant_files.append(debug_tips_file)
            print(f"DEBUG: Added debug tips file: {debug_tips_file}")
        else:
            print(f"DEBUG: Debug tips file not found: {debug_tips_file}")

        print(f"DEBUG: All relevant files: {all_relevant_files}")

        self.coder = Coder.create(main_model=self.model, io=self.io, fnames=all_relevant_files)

        all_changes = []
        for attempt in range(self.max_retries):
            temperature = self.initial_temperature + (attempt * self.temperature_increment)
            logging.info(f"Attempt {attempt + 1}/{self.max_retries} with temperature {temperature:.2f}")

            # Get the initial git status
            initial_git_status = self.get_git_status()

            stdout, stderr = self.run_test(error['test_file'], error['function'])
            accumulated_changes = "\n".join(all_changes)
            prompt = self.construct_prompt(error, stdout, stderr, accumulated_changes, attempt)
        
            print(f"DEBUG: Attempt {attempt + 1} - Temperature: {temperature}")
            print(f"DEBUG: Prompt:\n{prompt[:500]}...")  # Print first 500 characters of prompt

            try:
                response = self.coder.run(prompt)
                logging.info("AI model suggested changes. Applying changes...")
            
                # Parse the Aider response for search/replace statements
                changes = self.parse_aider_response(response)
                all_changes.append(changes)
                print(f"DEBUG: Changes made by Aider:\n{changes}")
            
                # Re-run the test
                stdout, stderr = self.run_test(error['test_file'], error['function'])

                print(f"DEBUG: Test output after fix attempt:\n{stdout[:500]}...")  # Print first 500 characters of stdout

                if "PASSED" in stdout:
                    logging.info(f"Fixed: {file_path} - {error['function']} on attempt {attempt + 1}")
                    self.log_progress("fixed", error, file_path, all_relevant_files, changes, temperature)
                    return True
                else:
                    logging.info(f"Fix attempt {attempt + 1} failed for: {file_path} - {error['function']}")
                    self.log_progress("failed", error, file_path, all_relevant_files, changes, temperature)
                
                    # Revert changes if the fix failed
                    self.revert_changes()
            except Exception as e:
                logging.error(f"Error while applying changes: {str(e)}")
                print(f"DEBUG: Exception occurred: {str(e)}")
                self.revert_changes()

        logging.warning(f"Failed to fix after {self.max_retries} attempts: {file_path} - {error['function']}")
        return False

    def parse_aider_response(self, response: str) -> str:
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


    def revert_changes(self):
        try:
            subprocess.run(["git", "reset", "--hard"], cwd=self.project_dir, check=True)
            subprocess.run(["git", "clean", "-fd"], cwd=self.project_dir, check=True)
            logging.info("Changes reverted successfully.")
        except subprocess.CalledProcessError:
            logging.error("Failed to revert changes. Manual intervention may be required.")


    def construct_prompt(self, error: Dict[str, Any], stdout: str, stderr: str, changes: str, attempt: int) -> str:
        error_prompt = (
            f"Fix the following pytest error:\n\n"
            f"Test File: {error['test_file']}\n"
            f"Function: {error['function']}\n"
            f"Error Type: {error['error_type']}\n"
            f"Error Details: {error['error_details']}\n"
            f"Code Snippet:\n{error['code_snippet']}\n"
            f"Captured Output:\n{error.get('captured_output', '')}\n"
            f"Captured Log:\n{error.get('captured_log', '')}\n"
        )

        analysis_prompt = (
            "Please analyze the error using the Analysis, Framework, Plan, Execution System:\n"
            "1. Analysis: Provide a detailed understanding of the error context.\n"
            "2. Framework: Develop a framework for addressing the error.\n"
            "3. Plan: Outline a specific plan to fix the error.\n"
            "4. Execution: Implement the plan and verify the fix.\n"
        )

        previous_attempt_prompt = ""
        if attempt > 0:
            previous_attempt_prompt = (
                f"\n\nPrevious fix attempt failed. This is attempt {attempt + 1}.\n"
                f"Changes made in the previous attempt:\n{changes}\n"
                "Please analyze why the previous fix didn't work and suggest a different approach. Collect more data if needed with debug statements or add additional relevant files."
            )

        full_prompt = f"{error_prompt}\n\n{analysis_prompt}\n\nTest Output:\n{stdout}\n\nTest Error Output:\n{stderr}{previous_attempt_prompt}"
    
        print(f"DEBUG: Constructed prompt for attempt {attempt + 1}:\n{full_prompt[:500]}...")  # Print first 500 characters of prompt
    
        return full_prompt

    def debug_fix_single_error(self, error, file_path):
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
            return True
        else:
            print(f"DEBUG: Failed to fix: {file_path} - {error['function']}")
            return False

    def main(self):
        logging.info("Starting main process...")
        # Load existing errors from the log
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

        fixed_errors = []
        failed_errors = []

        for file_path, error_list in all_errors.items():
            for error in error_list:
                logging.info(f"Processing error: {error} in {file_path}")
                if self.fix_error(error, file_path):
                    fixed_errors.append(f"{file_path} - {error['function']}")
                else:
                    failed_errors.append(f"{file_path} - {error['function']}")

        logging.info("Error fixing completed.")

        if fixed_errors:
            logging.info("Fixed errors:")
            for error in fixed_errors:
                logging.info(f"- {error}")

        if failed_errors:
            logging.warning("Failed to fix errors:")
            for error in failed_errors:
                logging.warning(f"- {error}")

        logging.info("Re-running full test suite to verify fixes...")
        for test_file in self.discover_test_files():
            stdout, stderr = self.run_test(test_file)
            logging.info(f"Final test results for {test_file}:")
            logging.info(stdout)
            logging.info(stderr)

        logging.info("Error fixing and verification completed.")
        
        print("DEBUG: Main process completed. Check the logs for detailed information.")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PytestErrorFixer")
    parser.add_argument("project_dir", help="Path to the project directory")
    parser.add_argument("--initial-temperature", type=float, default=0.4, help="Initial temperature setting for the AI model (default: 0.4)")
    parser.add_argument("--temperature-increment", type=float, default=0.1, help="Temperature increment for each retry (default: 0.1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of fix attempts per error (default: 3)")
    args = parser.parse_args()

    print(f"DEBUG: Starting PytestErrorFixer with arguments: {args}")

    fixer = PytestErrorFixer(
        args.project_dir,
        initial_temperature=args.initial_temperature,
        temperature_increment=args.temperature_increment,
        max_retries=args.max_retries
    )
    fixer.main()
