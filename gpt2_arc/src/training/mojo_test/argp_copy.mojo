from python import Python, PythonObject
from sys.param_env import env_get_int, env_get_string

fn get_float_param[name: StringLiteral, default: StringLiteral]() raises -> PythonObject:
    """Convert string parameter to Python float with error handling."""
    var raw_value = env_get_string[name, default]()
    try:
        var float_func = Python.evaluate("float")
        return float_func(raw_value)
    except:
        print("Warning: Unable to parse float for parameter '", name, "'. Using default value: ", default)
        var float_func = Python.evaluate("float")
        return float_func(default)

fn get_parameters() raises -> PythonObject:
    """Retrieve and validate all compile-time parameters."""
    var params = Python.import_module("builtins").dict()
    
    # Integer parameters
    params["max_epochs"] = env_get_int["max_epochs", 100]()
    params["batch_size"] = env_get_int["batch_size", 16]()
    params["num_workers"] = env_get_int["num_workers", 4]()
    params["grokfast_window_size"] = env_get_int["grokfast_window_size", 100]()
    params["prefetch_factor"] = env_get_int["prefetch_factor", 2]()
    
    # Boolean parameters (as integers)
    params["use_gpu"] = env_get_int["use_gpu", 0]() != 0
    params["fast_dev_run"] = env_get_int["fast_dev_run", 0]() != 0
    params["no_persistent_workers"] = env_get_int["no_persistent_workers", 0]() != 0
    params["no_logging"] = env_get_int["no_logging", 0]() != 0
    params["no_checkpointing"] = env_get_int["no_checkpointing", 0]() != 0
    params["no_progress_bar"] = env_get_int["no_progress_bar", 0]() != 0
    params["enable_symbol_freq"] = env_get_int["enable_symbol_freq", 0]() != 0
    params["use_synthetic_data"] = env_get_int["use_synthetic_data", 0]() != 0
    params["use_optuna"] = env_get_int["use_optuna", 0]() != 0
    params["include_pad_in_accuracy"] = env_get_int["include_pad_in_accuracy", 1]() != 0
    
    # String parameters
    params["grokfast_type"] = env_get_string["grokfast_type", "ema"]()
    params["optuna_study_name"] = env_get_string["optuna_study_name", ""]()
    params["optuna_storage"] = env_get_string["optuna_storage", "sqlite:///optuna_results.db"]()
    params["model_checkpoint"] = env_get_string["model_checkpoint", ""]()
    params["project"] = env_get_string["project", "gpt2-arc"]()
    params["results_dir"] = env_get_string["results_dir", "./results"]()
    params["run_name"] = env_get_string["run_name", "default_run"]()
    params["matmul_precision"] = env_get_string["matmul_precision", "medium"]()
    params["synthetic_data_path"] = env_get_string["synthetic_data_path", ""]()
    params["log_level"] = env_get_string["log_level", "INFO"]()
    params["accelerator"] = env_get_string["accelerator", "gpu"]()
    params["profiler_dirpath"] = env_get_string["profiler_dirpath", "./profiler_logs"]()
    params["profiler_filename"] = env_get_string["profiler_filename", "profile"]()
    
    # Float parameters with validation
    params["learning_rate"] = get_float_param["learning_rate", "1e-4"]()
    params["grokfast_alpha"] = get_float_param["grokfast_alpha", "0.98"]()
    params["grokfast_lamb"] = get_float_param["grokfast_lamb", "2.0"]()
    params["val_check_interval"] = get_float_param["val_check_interval", "0.01"]()
    params["train_split"] = get_float_param["train_split", "0.8"]()
    params["val_split"] = get_float_param["val_split", "0.1"]()
    params["test_split"] = get_float_param["test_split", "0.1"]()

    # Validate split parameters sum to 1.0
    var math = Python.import_module("math")
    var total_split = (params["train_split"].__float__() + 
                      params["val_split"].__float__() + 
                      params["test_split"].__float__())
    if math.fabs(total_split - 1.0) > 1e-6:
        raise Error("Split parameters must sum to 1.0")

    return params

fn validate_mamba_ratio(value: Float64) raises:
    """Validate that mamba_ratio is between 0.0 and 1.0."""
    if value < 0.0 or value > 1.0:
        raise Error("mamba_ratio must be between 0.0 and 1.0")

fn main() raises:
    # Get all parameters
    var params = get_parameters()
    
    print("\nParameters loaded:")
    print("Training Configuration:")
    print("  max_epochs:", params["max_epochs"])
    print("  batch_size:", params["batch_size"])
    print("  learning_rate:", params["learning_rate"])
    
    print("\nData Split:")
    print("  train_split:", params["train_split"])
    print("  val_split:", params["val_split"])
    print("  test_split:", params["test_split"])
    
    print("\nGrokFast Configuration:")
    print("  type:", params["grokfast_type"])
    print("  alpha:", params["grokfast_alpha"])
    print("  lambda:", params["grokfast_lamb"])
    print("  window_size:", params["grokfast_window_size"])
    
    print("\nSystem Configuration:")
    print("  num_workers:", params["num_workers"])
    print("  use_gpu:", params["use_gpu"])
    print("  accelerator:", params["accelerator"])