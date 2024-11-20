from python import Python, PythonObject

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

fn main():
    pass