from python import Python, PythonObject

struct ModelConfigSaver:
    var config: PythonObject
    
    fn __init__(inout self) raises:
        self.config = Python.evaluate("None")
    
    fn convert_to_python(self) raises -> PythonObject:
        try:
            # Create methods dict first - we know this works
            var methods = Python.evaluate("{'on_save_checkpoint': lambda self, trainer, pl_module, checkpoint: checkpoint.update({'model_config': self.config})}")
            
            # Create class with methods - we know type() works
            var py_class = Python.evaluate("type('ModelConfigSaver', (), {})")
            
            # Create instance - we know this works
            var instance = py_class()
            
            # Set config attribute - we know attribute setting works
            instance.config = self.config
            
            return instance
        except:
            print("Failed in convert_to_python with error")
            raise

fn test_callbacks() raises:
    print("=== Testing ModelConfigSaver Callback ===")
    
    # Create list - we know this works
    var callbacks = Python.evaluate("[]")
    print("1. Created callbacks list")
    
    # Create config saver
    var config_saver = ModelConfigSaver()
    print("2. Created ModelConfigSaver")
    
    try:
        # Convert to Python object
        var py_config_saver = config_saver.convert_to_python()
        print("3. Created Python object")
        
        # Append to list - we know this works
        callbacks.append(py_config_saver)
        print("4. Added to callbacks list")
        
        # Check length - we know this works
        var list_len = Python.evaluate("len")(callbacks)
        print("5. List length:", list_len)
    except:
        print("Failed in callback operations")

fn main() raises:
    test_callbacks()