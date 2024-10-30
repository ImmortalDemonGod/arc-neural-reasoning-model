from python import Python, PythonObject

struct ModelConfigSaver:
    var config: PythonObject
    
    fn __init__(inout self) raises:
        self.config = Python.evaluate("None")
    
    fn convert_to_python(self) raises -> PythonObject:
        # Create a Python callback class with required methods
        var callback_code = """
class _ModelConfigSaver:
    def __init__(self):
        self.config = None
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['model_config'] = self.config
callback = _ModelConfigSaver()
"""
        var py_saver = Python.evaluate(callback_code)
        py_saver.config = self.config
        return py_saver

fn test_callbacks() raises:
    print("=== Testing ModelConfigSaver Callback ===")
    # Create list of callbacks
    var callbacks = Python.evaluate("[]")
    print("1. Created callbacks list")
    
    # Create our Mojo ModelConfigSaver
    var config_saver = ModelConfigSaver()
    print("2. Created ModelConfigSaver")
    
    # Convert to Python object before appending
    var py_config_saver = config_saver.convert_to_python()
    callbacks.append(py_config_saver)
    print("3. Appended converted config saver")
    
    # Check the list
    var list_len = Python.evaluate("len")(callbacks)
    print("4. Callbacks length:", list_len)
    
    # Test None handling
    var none_value = Python.evaluate("None")
    var final_callbacks = callbacks if list_len > 0 else none_value
    print("5. Final callbacks type:", Python.type(final_callbacks))
    print("=== Test Complete ===")

fn main() raises:
    test_callbacks()