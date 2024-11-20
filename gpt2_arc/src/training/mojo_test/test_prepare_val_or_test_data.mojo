from python import Python, PythonObject

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

