from python import Python, PythonObject
from sys.param_env import env_get_int, env_get_string
from optuna_manager import configure_optuna_study

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

fn get_param_value[name: StringLiteral]() raises -> PythonObject:
    """Get the raw parameter value from environment."""
    return env_get_string[name, ""]()

fn get_int_param[name: StringLiteral, default: StringLiteral]() raises -> PythonObject:
    """Get integer parameter with string-based default value."""
    var int_func = Python.evaluate("int")
    try:
        var value = get_param_value[name]()
        return int_func(value)
    except:
        return int_func(default)

fn get_bool_param[name: StringLiteral, default: StringLiteral]() raises -> PythonObject:
    """Get boolean parameter from integer string."""
    var bool_func = Python.evaluate("bool")
    return bool_func(get_int_param[name, default]())

fn get_parameters() raises -> PythonObject:
    """Get parameters with proper default handling."""
    var params = Python.import_module("builtins").dict()
    var set_value = Python.evaluate("lambda d, k, v: d.update({k: v})")
    
    try:
        # Integer parameters - always provide defaults
        set_value(params, "max_epochs", env_get_int["max_epochs", 100]())
        set_value(params, "batch_size", env_get_int["batch_size", 16]())
        set_value(params, "num_workers", env_get_int["num_workers", 4]())
        set_value(params, "grokfast_window_size", env_get_int["grokfast_window_size", 100]())
        set_value(params, "prefetch_factor", env_get_int["prefetch_factor", 2]())
        
        # Boolean parameters (as integers)
        set_value(params, "use_gpu", env_get_int["use_gpu", 0]() != 0)
        set_value(params, "fast_dev_run", env_get_int["fast_dev_run", 0]() != 0)
        set_value(params, "no_persistent_workers", env_get_int["no_persistent_workers", 0]() != 0)
        set_value(params, "no_logging", env_get_int["no_logging", 0]() != 0)
        set_value(params, "no_checkpointing", env_get_int["no_checkpointing", 0]() != 0)
        set_value(params, "no_progress_bar", env_get_int["no_progress_bar", 0]() != 0)
        set_value(params, "enable_symbol_freq", env_get_int["enable_symbol_freq", 0]() != 0)
        set_value(params, "use_synthetic_data", env_get_int["use_synthetic_data", 0]() != 0)
        set_value(params, "use_optuna", env_get_int["use_optuna", 0]() != 0)
        set_value(params, "include_pad_in_accuracy", env_get_int["include_pad_in_accuracy", 1]() != 0)
        
        # String parameters
        set_value(params, "grokfast_type", env_get_string["grokfast_type", "ema"]())
        set_value(params, "optuna_study_name", env_get_string["optuna_study_name", ""]())
        set_value(params, "optuna_storage", env_get_string["optuna_storage", "sqlite:///optuna_results.db"]())
        set_value(params, "model_checkpoint", env_get_string["model_checkpoint", ""]())
        set_value(params, "project", env_get_string["project", "gpt2-arc"]())
        set_value(params, "results_dir", env_get_string["results_dir", "./results"]())
        set_value(params, "run_name", env_get_string["run_name", "default_run"]())
        set_value(params, "matmul_precision", env_get_string["matmul_precision", "medium"]())
        set_value(params, "synthetic_data_path", env_get_string["synthetic_data_path", ""]())
        set_value(params, "log_level", env_get_string["log_level", "INFO"]())
        set_value(params, "accelerator", env_get_string["accelerator", "gpu"]())
        set_value(params, "profiler_dirpath", env_get_string["profiler_dirpath", "./profiler_logs"]())
        set_value(params, "profiler_filename", env_get_string["profiler_filename", "profile"]())
        
        # Float parameters
        set_value(params, "learning_rate", get_float_param["learning_rate", "1e-4"]())
        set_value(params, "grokfast_alpha", get_float_param["grokfast_alpha", "0.98"]())
        set_value(params, "grokfast_lamb", get_float_param["grokfast_lamb", "2.0"]())
        set_value(params, "val_check_interval", get_float_param["val_check_interval", "0.01"]())
        set_value(params, "train_split", get_float_param["train_split", "0.8"]())
        set_value(params, "val_split", get_float_param["val_split", "0.1"]())
        set_value(params, "test_split", get_float_param["test_split", "0.1"]())

        # Validate split parameters sum to 1.0
        var math = Python.import_module("math")
        var to_float = Python.evaluate("float")
        var total_split = (
            to_float(params["train_split"]) + 
            to_float(params["val_split"]) + 
            to_float(params["test_split"])
        )
        
        if math.fabs(total_split - 1.0) > 1e-6:
            raise Error("Split parameters must sum to 1.0")

        return params
    except:
        raise Error("Failed to get parameters: check environment variables and defaults")
    
fn validate_mamba_ratio(value: Float64) raises:
    """Validate that mamba_ratio is between 0.0 and 1.0."""
    if value < 0.0 or value > 1.0:
        raise Error("mamba_ratio must be between 0.0 and 1.0")

struct ModelConfigSaver:
    """A callback to save model configuration during training checkpoints."""
    
    var config: PythonObject
    var callback_base: PythonObject
    
    fn __init__(inout self) raises:
        # Import required Python modules
        var pl_callbacks = Python.import_module("pytorch_lightning.callbacks")
        self.callback_base = pl_callbacks.Callback()
        # Use evaluate() to get Python's None
        self.config = Python.evaluate("None")
    
    fn set_config(inout self, config: PythonObject):
        """Set the configuration to be saved with checkpoints."""
        self.config = config
    
    fn on_save_checkpoint(self, trainer: PythonObject, pl_module: PythonObject, 
                         inout checkpoint: PythonObject) raises:
        """Save model configuration to checkpoint.
        
        Args:
            trainer: PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            checkpoint: Checkpoint dictionary to modify.
        """
        if not self.config.is_(Python.evaluate("None")):
            try:
                # Create a new dictionary for model config using Python's dict()
                var model_config = Python.evaluate("dict()")
                var config_dict = self.config.__dict__
                
                # Copy all items to the new dictionary using Python's dict update
                for key in config_dict:
                    var  update_expr = "lambda d, k, v: d.update({k: v})"
                    var update_func = Python.evaluate(update_expr)
                    update_func(model_config, key, config_dict[key])
                
                # Update the checkpoint using Python's dict update
                var  checkpoint_update_expr = "lambda d, k, v: d.update({k: v})"
                var checkpoint_update = Python.evaluate(checkpoint_update_expr)
                checkpoint_update(checkpoint, "model_config", model_config)
                
            except:
                print("Warning: Failed to save model configuration")
    
    fn convert_to_python(self) raises -> PythonObject:
        # Create a simple Python callback class
        var class_def = "type('ModelConfigSaver', (), {})"
        var py_saver = Python.evaluate(class_def)()
        
        # Copy our config to the Python object
        py_saver.config = self.config
        
        # Add the save checkpoint method
        py_saver.on_save_checkpoint = Python.evaluate("lambda self, trainer, pl_module, checkpoint: checkpoint.update({'model_config': self.config})")
        
        return py_saver

# Prepare validation or test data
fn prepare_val_or_test_data(eval_set: PythonObject, args: PythonObject, is_validation: Bool = True) raises -> PythonObject:
    """Prepare validation or test data from evaluation set."""
    var set_type = "validation" if is_validation else "test"
    var logger = Python.import_module("logging").getLogger("train")
    
    # Create message using Python string formatting
    var debug_msg = Python.evaluate("'Preparing {} data from arckit evaluation set'").format(set_type)
    logger.debug(debug_msg)
    
    # Create Python list to store results
    var py_list = Python.evaluate("[]")
    
    for task in eval_set.tasks:
        var examples = task.train if is_validation else task.test
        
        for example in examples:
            # Create Python dictionary directly
            var sample_dict = Python.dict()
            sample_dict["input"] = example[0]
            sample_dict["output"] = example[1]
            sample_dict["task_id"] = task.id
            
            py_list.append(sample_dict)
    
    # Use Python string formatting for the final message
    var final_msg = Python.evaluate("'Prepared {} samples for {} dataset'").format(
        Python.evaluate("len")(py_list),
        set_type
    )
    logger.debug(final_msg)
    return py_list


# Load dataset
fn get_none() raises -> PythonObject:
    """Helper function to get Python's None value."""
    return Python.evaluate("None")

fn load_dataset(
    args: PythonObject, 
    config: PythonObject, 
    dataset_type: String = "train", 
    all_synthetic_data: PythonObject = PythonObject()
) raises -> PythonObject:
    var logger = Python.import_module("logging").getLogger("train")
    
    # Convert Mojo string to Python string at the start
    var py_dataset_type = Python.evaluate("str")(dataset_type)
    
    # Use Python string formatting for debug message
    var debug_msg = Python.evaluate("'load_dataset called with dataset_type={}, args.use_synthetic_data={}'").format(
        py_dataset_type,
        args.use_synthetic_data
    )
    logger.debug(debug_msg)
    
    var dataset: PythonObject
    var none_value = get_none()
    
    # Use Python string methods
    var dataset_type_lower = py_dataset_type.lower()
    
    if dataset_type_lower == "train":
        if args.use_synthetic_data:
            if all_synthetic_data != none_value:
                dataset = all_synthetic_data["train_dataset"]
            else:
                var err_msg = Python.evaluate("'all_synthetic_data is required when use_synthetic_data is True'")
                logger.error(err_msg)
                raise Error(err_msg.__str__())
        else:
            var arckit = Python.import_module("arckit")
            dataset = Python.import_module("gpt2_arc.src.data.arc_dataset").ARCDataset(
                data_source=arckit.load_data()[0],
                is_test=False,
                max_samples=args.max_train_samples,
                num_symbols=config.training.num_symbols,
                pad_symbol_idx=config.training.pad_symbol_idx,
                symbol_freq=config.training.symbol_freq if args.enable_symbol_freq else none_value
            )
    else:
        var arckit = Python.import_module("arckit")
        var eval_set = arckit.load_data()[1]
        var is_validation = dataset_type_lower == "val"
        var data_source = prepare_val_or_test_data(eval_set, args, is_validation)
        dataset = Python.import_module("gpt2_arc.src.data.arc_dataset").ARCDataset(
            data_source=data_source,
            is_test=(dataset_type_lower == "test"),
            num_symbols=config.training.num_symbols,
            pad_symbol_idx=config.training.pad_symbol_idx,
            symbol_freq=config.training.symbol_freq if args.enable_symbol_freq else none_value
        )
    
    # Use Python's len() function
    var dataset_len = Python.evaluate("len")(dataset)
    
    if dataset_len == 0:
        # Keep everything as Python strings
        var error_msg = Python.evaluate("'No samples loaded for {} dataset.'").format(py_dataset_type)
        logger.error(error_msg)
        raise Error(error_msg.__str__())
    
    # Use Python string formatting for debug message
    var final_msg = Python.evaluate("'{} dataset loaded with {} samples'").format(
        py_dataset_type.capitalize(),
        dataset_len
    )
    logger.debug(final_msg)
    return dataset

# Load and split synthetic data
fn load_and_split_synthetic_data(args: PythonObject, config: PythonObject) raises -> PythonObject:
    var logger = Python.import_module("logging").getLogger("train")
    logger.debug("Entering load_and_split_synthetic_data function")
    
    var none_value = Python.evaluate("None")
    var synthetic_dataset = Python.import_module("gpt2_arc.src.data.arc_dataset").ARCDataset(
        data_source=args.synthetic_data_path,
        is_test=False,
        max_samples=args.max_train_samples,
        num_symbols=config.training.num_symbols,
        pad_symbol_idx=config.training.pad_symbol_idx,
        symbol_freq=config.training.symbol_freq if args.enable_symbol_freq else none_value
    )
    
    var total_samples = Python.evaluate("len")(synthetic_dataset)
    
    # Use Python string formatting
    var debug_msg = Python.evaluate("'Total synthetic samples loaded: {}'").format(total_samples)
    logger.debug(debug_msg)
    
    # Check total samples and raise error if needed
    if total_samples == 0:
        var error_msg = "No synthetic samples were loaded. Please check your synthetic data files."
        logger.error(error_msg)
        raise Error(error_msg)
    
    # Create and return dictionary using Python dict
    var result = Python.dict()
    result["train_dataset"] = synthetic_dataset
    return result

# 1. Add proper parameter validation function
fn validate_parameters(args: PythonObject) raises:
    """Validate all input parameters before processing."""
    var logger = Python.import_module("logging").getLogger("train")
    
    # Validate numeric ranges
    if args.batch_size <= 0:
        raise Error("batch_size must be positive")
    if args.max_epochs <= 0:
        raise Error("max_epochs must be positive")
    
    # Validate split parameters
    var total = Float64(args.train_split) + Float64(args.val_split) + Float64(args.test_split)
    if abs(total - 1.0) > 1e-6:
        raise Error("Split parameters must sum to 1.0")
    
    # Validate paths and existence
    if args.use_synthetic_data:
        var os = Python.import_module("os")
        if not os.path.exists(args.synthetic_data_path):
            raise Error("Synthetic data path does not exist: " + args.synthetic_data_path.__str__())


fn safe_str_to_python(value: String) raises -> PythonObject:
    """Safely convert Mojo string to Python string."""
    try:
        return Python.evaluate("str")(value)
    except:
        raise Error("Failed to convert string to Python string")


fn setup_imports() raises -> PythonObject:
    """Initialize and return required Python modules."""
    var modules = Python.dict()
    modules["logging"] = Python.import_module("logging")
    modules["torch"] = Python.import_module("torch")
    modules["pytorch_lightning"] = Python.import_module("pytorch_lightning")
    return modules

fn setup_configs(args: PythonObject) raises -> PythonObject:
    """Create and return model configurations."""
    var config_module = Python.import_module("gpt2_arc.src.config")
    var model_config = config_module.ModelConfig(
        n_embd=args["n_embd"],
        n_head=args["n_head"],
        n_layer=args["n_layer"],
        dropout=args["dropout"]
    )
    var training_config = config_module.TrainingConfig(
        batch_size=args["batch_size"],
        learning_rate=args["learning_rate"],
        max_epochs=args["max_epochs"],
        use_gpu=args["use_gpu"],
        log_level=args["log_level"]
    )
    return config_module.Config(model=model_config, training=training_config)

fn setup_model(config: PythonObject) raises -> PythonObject:
    """Initialize and return the model."""
    var GPT2ARC = Python.import_module("gpt2_arc.src.models.gpt2").GPT2ARC
    var symbol_freq_dict = Python.dict()
    return GPT2ARC(
        config=config,
        num_classes=config.training.num_classes,
        symbol_freq=symbol_freq_dict,
        pad_symbol_idx=config.training.pad_symbol_idx
    )

fn setup_trainer(
    model: PythonObject,
    train_data: PythonObject,
    val_data: PythonObject,
    config: PythonObject,
    args: PythonObject
) raises -> PythonObject:
    """Initialize and return the trainer."""
    var ARCTrainer = Python.import_module("gpt2_arc.src.training.trainer").ARCTrainer
    return ARCTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        config=config,
        args=args
    )

fn train_model(args: PythonObject) raises -> None:
    # Try to isolate which part fails during compilation
    var modules = setup_imports()
    var logger = modules["logging"].getLogger("train")
    logger.info("Initializing training")

    var config = setup_configs(args)
    
    var train_data = load_dataset(args, config, dataset_type="train")
    var val_data = load_dataset(args, config, dataset_type="val")

    var model = setup_model(config)
    var trainer = setup_trainer(model, train_data, val_data, config, args)
    var pytorch_lightning = modules["pytorch_lightning"]
    var torch = modules["torch"]
    
    var accelerator = "cpu"
    if args["use_gpu"].to_bool() and torch.cuda.is_available():
        accelerator = "gpu"
    
    var pl_trainer = pytorch_lightning.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True
    )
    
    try:
        logger.info("Starting model training")
        pl_trainer.fit(trainer)
        logger.info("Training completed successfully")
    except:
        logger.error("An error occurred during training")
        var error_info = Python.evaluate("str(sys.exc_info()[1])")
        logger.error("Error details: " + error_info)



# Main function
fn main() raises:
    var args = get_parameters()
    validate_parameters(args)  # Validate before using
    train_model(args)