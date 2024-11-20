from python import Python, PythonObject

struct ModelConfigSaver:
    var config: PythonObject
    
    fn __init__(inout self) raises:
        self.config = Python.evaluate("None")
    
    fn convert_to_python(self) raises -> PythonObject:
        # Create Python object in a single line to avoid syntax issues
        try:
            # First create the dictionary of methods
            var methods = Python.evaluate("{'config': None, 'on_save_checkpoint': lambda self, trainer, pl_module, checkpoint: checkpoint.update({'model_config': self.config})}")
            
            # Then create the class using type()
            var py_class = Python.evaluate("type('ModelConfigSaver', (), {})")
            
            # Create instance
            var instance = py_class()
            
            # Set our config
            instance.config = self.config
            
            print("Successfully created Python object")
            return instance
        except:
            print("Failed to create Python object")
            raise

fn test_callbacks() raises:
    print("=== Testing ModelConfigSaver Callback ===")
    
    # Create list of callbacks
    var callbacks = Python.evaluate("[]")
    print("1. Created callbacks list")
    
    # Create our Mojo ModelConfigSaver
    var config_saver = ModelConfigSaver()
    print("2. Created ModelConfigSaver")
    
    try:
        var py_config_saver = config_saver.convert_to_python()
        print("3. Created Python object")
        
        callbacks.append(py_config_saver)
        print("4. Added to callbacks list")
        
        var list_len = Python.evaluate("len")(callbacks)
        print("5. List length:", list_len)
    except:
        print("Failed in callback operations")

fn main() raises:
    test_callbacks()