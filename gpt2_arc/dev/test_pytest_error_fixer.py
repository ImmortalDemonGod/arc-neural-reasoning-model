import os
import json
import pytest
from unittest.mock import patch, MagicMock
from pytest_error_fixer import PytestErrorFixer

# Reusable fixtures for test setup
@pytest.fixture
def error_fixer(tmp_path):
    # Initialize PytestErrorFixer with a temporary directory for the progress log
    fixer = PytestErrorFixer("test_project_dir")
    fixer.progress_log = tmp_path / "test_progress_log.json"
    fixer.error_log = tmp_path / "test_error_log.json"
    return fixer

@pytest.fixture
def sample_errors():
    # Sample errors for testing
    return {
        "gpt2_arc/test_file.py": [
            "test_function AssertionError: assert 1 == 2",
            "test_another_function TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        ]
    }

# 1. Test for progress log initialization
def test_init_progress_log(error_fixer):
    error_fixer.init_progress_log()
    assert os.path.exists(error_fixer.progress_log)
    with open(error_fixer.progress_log, 'r') as f:
        assert json.load(f) == []  # Ensure the log is empty upon initialization

# 2. Test for logging progress in the progress log
def test_log_progress(error_fixer):
    error_fixer.init_progress_log()
    error_fixer.log_progress("fixed", "test error", "test_file.py")
    with open(error_fixer.progress_log, 'r') as f:
        log = json.load(f)
        assert len(log) == 1
        assert log[0] == {"error": "test error", "file": "test_file.py", "status": "fixed"}

# 3. Test for running full test suite and capturing output
@patch('subprocess.run')
def test_run_full_test(mock_run, error_fixer):
    # Mock the output of subprocess.run to simulate pytest execution
    mock_run.return_value = MagicMock(stdout="Test output", stderr="Test error")
    stdout, stderr = error_fixer.run_full_test()
    
    # Assert that stdout and stderr are captured correctly
    assert stdout == "Test output"
    assert stderr == "Test error"
    mock_run.assert_called_once()

# 4. Test for parsing errors from pytest output
def test_parse_errors(error_fixer):
    # Simulate pytest output with multiple errors
    sample_output = """
    gpt2_arc/test_file.py::test_function FAILED
    gpt2_arc/another_file.py::test_another_function FAILED
    """
    errors = error_fixer.parse_errors(sample_output)
    
    # Verify that errors are correctly parsed and associated with the right test files
    assert "gpt2_arc/test_file.py" in errors
    assert "gpt2_arc/another_file.py" in errors
    assert "test_function FAILED" in errors["gpt2_arc/test_file.py"]
    assert "test_another_function FAILED" in errors["gpt2_arc/another_file.py"]

# 5. Test for saving and loading errors to/from a JSON file
def test_save_and_load_errors(error_fixer, sample_errors):
    # Save errors to a file
    error_fixer.save_errors(sample_errors)
    
    # Load errors back and verify they match the original data
    loaded_errors = error_fixer.load_errors()
    assert loaded_errors == sample_errors

# 6. Test for predicting relevant files using aider's output
@patch.object(PytestErrorFixer, 'coder')
def test_predict_relevant_files(mock_coder, error_fixer):
    # Mock aider's file prediction output
    mock_coder.run.return_value = "The files likely involved are gpt2_arc/file1.py and gpt2_arc/file2.py"
    
    # Predict files for a test error
    files = error_fixer.predict_relevant_files("test error")
    
    # Assert that the correct files are predicted
    assert files == ["gpt2_arc/file1.py", "gpt2_arc/file2.py"]
    mock_coder.run.assert_called_once()

# 7. Test for fixing errors and retrying if needed
@patch('subprocess.run')
@patch.object(PytestErrorFixer, 'coder')
def test_fix_error(mock_coder, mock_run, error_fixer):
    # Simulate failed and successful pytest runs
    mock_run.side_effect = [
        MagicMock(stdout="Test failed", stderr="Error occurred"),
        MagicMock(stdout="Test PASSED", stderr="")
    ]
    
    # Simulate aider suggesting fixes
    mock_coder.run.return_value = "Suggested fix"
    
    # Run the fix_error method and verify it retries and eventually succeeds
    result = error_fixer.fix_error("gpt2_arc/test_file.py", "test_function")
    
    # Assert that the error is eventually fixed
    assert result == True
    assert mock_run.call_count == 2
    mock_coder.run.assert_called_once()

# 8. Edge case: Test for handling invalid error output (additional coverage)
def test_parse_errors_invalid_format(error_fixer):
    invalid_output = "This is not a valid pytest output"
    errors = error_fixer.parse_errors(invalid_output)
    assert errors == {}

# 9. Edge case: Test for retry exhaustion when errors remain unfixed
@patch('subprocess.run')
@patch.object(PytestErrorFixer, 'coder')
def test_retry_exhaustion(mock_coder, mock_run, error_fixer):
    # Simulate constant failure in pytest runs
    mock_run.side_effect = [
        MagicMock(stdout="Test failed", stderr="Error occurred")
    ] * 3  # Retry the maximum number of times
    
    mock_coder.run.return_value = "Suggested fix"
    
    # Run the fix_error method and ensure it retries up to the max limit
    result = error_fixer.fix_error("gpt2_arc/test_file.py", "test_function")
    
    # Assert that the retries are exhausted
    assert result == False
    assert mock_run.call_count == 3  # Ensure the retry mechanism works
    mock_coder.run.assert_called_once()
