from python import Python, PythonObject
from sys.param_env import env_get_int, env_get_string

# Load dataset
fn get_none() raises -> PythonObject:
    """Helper function to get Python's None value."""
    return Python.evaluate("None")

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

fn main():
    pass