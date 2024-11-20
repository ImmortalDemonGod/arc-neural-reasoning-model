from sys.param_env import env_get_int
from python import Python

fn main() raises:
    # Retrieve compile-time parameters
    var max_epochs: Int = env_get_int["max_epochs", 100]()  # Default to 100 if not provided
    print("Training with max_epochs:", max_epochs)
    
    # Import Python modules directly where they are needed
    var torch = Python.import_module("torch") 
    print("PyTorch version:", torch.__version__)

    # Example of using another module
    var numpy = Python.import_module("numpy")
    print("NumPy version:", numpy.__version__)