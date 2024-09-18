import subprocess
import logging
import re
from collections import defaultdict
import json
from aider.coders import Coder
from aider.coders import AskCoder
from aider.models import Model
from aider.io import InputOutput
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PytestErrorFixer:
    def __init__(self, project_dir, max_retries=3, progress_log="progress_log.json"):
        logging.info(f"Initializing PytestErrorFixer with project directory: {project_dir}")
        self.project_dir = project_dir
        self.max_retries = max_retries
        # Get the API key from the environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.model = Model("gpt-4o-2024-08-06")
        self.io = InputOutput(yes=True)
        self.coder = Coder.create(main_model=self.model, io=self.io)
        self.progress_log = progress_log
        self.error_log = "error_log.json"
        self.init_progress_log()

    def init_progress_log(self):
        # Initialize the progress log file if it doesn't exist
        logging.info("Initializing progress log...")
        if not os.path.exists(self.progress_log):
            with open(self.progress_log, 'w') as f:
                json.dump([], f)

    def log_progress(self, status, error, test_file):
        # Log the progress of fixing an error
        logging.info(f"Logging progress: {status} for error in {test_file}")
        # Get the latest commit SHA
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')

        # Get the current timestamp
        timestamp = subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).strip().decode('utf-8')

        with open(self.progress_log, 'r+') as f:
            log = json.load(f)
            log.append({
                "error": error,
                "file": test_file,
                "status": status,
                "commit_sha": commit_sha,  # Include the commit SHA
                "timestamp": timestamp  # Include the timestamp
            })
            f.seek(0)
            json.dump(log, f, indent=4)

    def discover_test_files(self):
        test_files = []
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
        return test_files

    def run_individual_test(self, test_file):
        cmd = [
            "pytest",
            "-v",
            "--tb=short",
            "--log-cli-level=DEBUG",
            test_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
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
                relevant_paths.add(file_path)  # The test file itself
                
                # Extract paths from error details and code snippet
                paths = re.findall(r'gpt2_arc/[^\s:]+\.py', error['error_details'] + error['code_snippet'])
                relevant_paths.update(paths)
                
                # Add the test file if it's not already included
                if 'test_file' in error:
                    relevant_paths.add(error['test_file'])
                
                # Create a unique key for this error
                error_key = f"{error['function']} - {error['error_type']}"
                error_file_paths[error_key] = list(relevant_paths)
        
        return error_file_paths

    def fix_error(self, test_file, error):
        # Extract relevant files for this specific error
        error_dict = {test_file: [error]}
        relevant_files = self.extract_file_paths_from_errors(error_dict)
        
        # Flatten the list of relevant files
        all_relevant_files = list(set(path for paths in relevant_files.values() for path in paths))
        
        # Create a new Coder instance with the relevant files
        self.coder = Coder.create(main_model=self.model, io=self.io, fnames=all_relevant_files)
        
        # Run the test to get the error output
        cmd = ["pytest", "-v", "--tb=short", "--log-cli-level=DEBUG", f"{test_file}::{error['function']}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Prompt aider to suggest a fix based on test output
        prompt = f"Fix this pytest error:\n\n{result.stdout}\n\n{result.stderr}"
        self.coder.run(prompt)  # This will apply the changes to the files
        
        # Run the test again to check if it's fixed
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "PASSED" in result.stdout:
            self.log_progress("fixed", error, test_file)
            return True
        else:
            # If the test still fails, revert the commit
            try:
                commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
                logging.info(f"Reverting commit {commit_sha} due to failed test")
                subprocess.run(["git", "revert", "--no-edit", commit_sha], check=True)
                self.log_progress("reverted", error, test_file)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error during revert: {e}")
                self.log_progress("revert_failed", error, test_file)
            return False

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

        # Prompt aider to suggest a fix based on test output
        prompt = f"Fix this pytest error:\n\n{result.stdout}\n\n{result.stderr}"
        print("DEBUG: Sending prompt to AI model")
        try:
            debug_coder.run(prompt)  # This will apply the changes to the files
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

    def debug_main(self):
        print("DEBUG: Starting debug_main process...")

        # Load all errors from the error log
        all_errors = self.load_errors()
        print("DEBUG: Loaded errors:", json.dumps(all_errors, indent=2))

        # Process only the first error for debugging
        for file_path, error_list in all_errors.items():
            if error_list:  # Check if there are any errors
                error = error_list[0]  # Use only the first error for debugging
                print(f"DEBUG: Processing error: {error} in {file_path}")
                fixed = self.debug_fix_single_error(error, file_path)
                if fixed:
                    print(f"DEBUG: Successfully fixed error in {file_path}")
                else:
                    print(f"DEBUG: Failed to fix error in {file_path}")
                break  # Only process the first error, then stop

        print("DEBUG: Error fixing debug completed.")

        print("Starting main process...")
        test_files = self.discover_test_files()
        print(f"Discovered {len(test_files)} test files.")

        for test_file in test_files:
            print(f"Running test file: {test_file}")
            stdout, stderr = self.run_individual_test(test_file)
            errors = self.parse_errors(stdout + stderr, test_file)
            if errors:
                print(f"Errors found in {test_file}. Saving to log...")
                self.save_errors(errors)
            else:
                print(f"No errors found in {test_file}")

        print("All test files processed.")

        # Load all errors from the error log
        all_errors = self.load_errors()
        print("Loaded errors:", all_errors)

        # Process each error
        for file_path, error_list in all_errors.items():
            for error in error_list:
                print(f"Processing error: {error} in {file_path}")
                
                # Extract relevant files for this specific error
                error_dict = {file_path: [error]}
                relevant_files = self.extract_file_paths_from_errors(error_dict)
                
                # Flatten the list of relevant files
                all_relevant_files = list(set(path for paths in relevant_files.values() for path in paths))
                
                print("Relevant files for this error:")
                for path in all_relevant_files:
                    print(f"  - {path}")
                
                # Create a new Coder instance with the relevant files for this error
                self.coder = Coder.create(main_model=self.model, io=self.io, fnames=all_relevant_files)
                
                # Run the test to get the error output
                cmd = ["pytest", "-v", "--tb=short", "--log-cli-level=DEBUG", f"{error['test_file']}::{error['function']}"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Debug: Print test output
                print("\nTest output:")
                print(result.stdout)
                print("\nTest error output:")
                print(result.stderr)
                
                # Prompt aider to suggest a fix based on test output
                fix_prompt = f"Analyze the errors please follow the Analysis, Framework, Plan, Execution System as described make sure to have critques. Let's now fix this pytest error:\n\n{result.stdout}\n\n{result.stderr}"
                analysis_prompt = """just to re-iterate before the next one. Learn This. 
your primary job is to utilize the Analysis, Framework, Plan, Execution System. It consist of performing: 
Analysis: 
^IYour job is to build a detailed, descriptive, rational/probabilistic, verbose understanding and comprehension of both the content of the supplied reques
t and the actual needs of the plan/execution through a written pleonastic, long-winded, and semi-scholarly Document. Each Section within this analysis doc
ument shall also have a independent critique addendum which is inherently prolix and palaverous. After completing the Document, you shall do an additional
 cruitique addendum to critique the whole. after this you shall as if you should critique your critique, or if you should proceed with updating the docume
nt with the knowledge of the critique. after you have updated the document, you shall ask if you should proceed with the creation of the framework for the
 plan. 
Framework: 
^IYour job is to build a detailed vivid and detailed Framework to be able to create the Plan. Each Section within this analysis document shall also have a
n independent critique addendum which is inherently prolix and palaverous. After completing the Document, you shall do an additional cruitique addendum to
 critique the whole. after this you shall as if you should critique your critique, or if you should proceed with updating the document with the knowledge 
of the critique. after you have updated the document, you shall ask if you should proceed with the creation of the plan 
Plan: 
^IYour Job is to build an illustrative, explicit, specific, detailed Plan. word choice should be very nuanced to convey the most information possible. Eac
h Section within this analysis document shall also have an independent critique addendum which is inherently prolix and palaverous. After completing the D
ocument, you shall do an additional cruitique addendum to critique the whole. after this you shall as if you should critique your critique, or if you shou
ld proceed with updating the document with the knowledge of the critique. after you have updated the document, you shall ask if you should proceed with th
e Execution 
 
Please cre4ate an understanding of this to a intensive detailed specific clear degree, and update your memory. I know you are here to assist     
with code analysis and improvements this is the framework that will maximize your ability for that task"""
                combined_prompt = f"{fix_prompt}\n\n{analysis_prompt}"
                print("Sending prompt to AI model...")
                print(f"Word count: {len(combined_prompt.split())}")
                try:
                    self.coder.run(combined_prompt)  # This will apply the changes to the files
                    print("AI model suggested changes. Applying changes...")
                except Exception as e:
                    print(f"Error while applying changes: {str(e)}")
                
                # Run the test again to check if it's fixed
                print("\nRe-running test to check if error is fixed...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Debug: Print re-run test output
                print("\nRe-run test output:")
                print(result.stdout)
                print("\nRe-run test error output:")
                print(result.stderr)
                
                if "PASSED" in result.stdout:
                    print(f"Fixed: {file_path} - {error}")
                    self.log_progress("fixed", error, file_path)
                else:
                    print(f"Failed to fix: {file_path} - {error}")
                    self.log_progress("failed", error, file_path)

        print("Error fixing completed.")

        # Run the full test suite again to verify all fixes
        print("Re-running full test suite to verify fixes...")
        test_files = self.discover_test_files()
        for test_file in test_files:
            stdout, stderr = self.run_individual_test(test_file)
            print(f"Final test results for {test_file}:")
            print(stdout)
            print(stderr)

        print("Error fixing and verification completed.")





if __name__ == "__main__":
    fixer = PytestErrorFixer("/Volumes/Totallynotaharddrive/arc-neural-reasoning-model/gpt2_arc")
    fixer.debug_main()  # Use the new debug_main method instead of main
