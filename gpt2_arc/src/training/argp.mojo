from python import Python, PythonObject
# Setup Python modules
fn setup_python_modules() -> List[PythonObject] raises:
    var argparse = Python.import_module("argparse")
    var logging = Python.import_module("logging")
    var os = Python.import_module("os")
    var sys = Python.import_module("sys")
    var torch = Python.import_module("torch")
    var numpy = Python.import_module("numpy")
    var pytorch_lightning = Python.import_module("pytorch_lightning")
    var optuna = Python.import_module("optuna")
    var arckit = Python.import_module("arckit")
    var concurrent_futures = Python.import_module("concurrent.futures")
    var random = Python.import_module("random")
    var tqdm = Python.import_module("tqdm")
    var datetime_module = Python.import_module("datetime")
    var validators = Python.import_module("validators")
    var exceptions = Python.import_module("exceptions")

    return [
        argparse,
        logging,
        os,
        sys,
        torch,
        numpy,
        pytorch_lightning,
        optuna,
        arckit,
        concurrent_futures,
        random,
        tqdm,
        datetime_module,
        validators,
        exceptions
    ]

fn create_argument_parser() -> PythonObject:
    var modules: List[PythonObject] = setup_python_modules()
    var argparse = modules[0]
    var validators = modules[13]
    var exceptions = modules[14]

    var parser: PythonObject = argparse.ArgumentParser(description="Train the ARC Neural Reasoning Model")

    # Add a minimal argument
    try:
        parser.add_argument("--max_epochs", type=Python.int, required=True, help="Maximum number of epochs")
    except exceptions.Exception as e:
        print("Error adding argument:", e)

    return parser
