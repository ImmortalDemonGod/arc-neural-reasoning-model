# Mojo Development: Lessons Learned and Best Practices
*A practical guide based on real-world Python to Mojo conversion experience*

## Understanding Mojo Types

### Float Handling
1. **Float64 Reality**
   - Float64 in Mojo is actually a SIMD scalar: `SIMD[DType.float64, 1]`
   - Don't try to use Float64 directly with Python objects/dictionaries
   - When working with Python interop, keep values as Python floats

2. **String to Float Conversion**
   ```mojo
   # Wrong way:
   var x: Float64 = float(some_string)  # This doesn't work!
   
   # Correct way:
   var float_func = Python.evaluate("float")
   var py_float = float_func(some_string)
   var mojo_float = Float64(py_float)  # Only if you need Mojo Float64
   ```

### String Parameters
1. **StringLiteral vs String**
   - `StringLiteral` is compile-time only
   - Functions like `env_get_string` require `StringLiteral` parameters
   - Parameters must be compile-time constants

2. **Best Practice**
   ```mojo
   # Wrong:
   fn get_param(name: String) -> String  # Won't work with env_get_string
   
   # Correct:
   fn get_param[name: StringLiteral]() -> String  # Use compile-time parameter
   ```

## Parameter Handling

### Environment Parameters
1. **Available Types**
   - `env_get_string`: For string parameters
   - `env_get_int`: For integer parameters
   - No built-in float parameter function

2. **Command Line Usage**
   ```bash
   mojo build -D param_name="value" script.mojo
   ```

3. **Best Practice Pattern**
   ```mojo
   fn get_float_param[name: StringLiteral, default: StringLiteral]() raises -> PythonObject:
       var raw_value = env_get_string[name, default]()
       var float_func = Python.evaluate("float")
       return float_func(raw_value)
   ```

## Python Interoperability

### Dictionary Handling
1. **Creating Python Dictionaries**
   ```mojo
   var params = Python.import_module("builtins").dict()
   ```

2. **Type Compatibility**
   - Store Python types in Python dictionaries
   - Convert to Mojo types only when needed for Mojo operations
   - Keep data in Python format when primarily working with Python interop

### Type Conversion Patterns
1. **Python to Mojo**
   ```mojo
   # Python float to Mojo Float64
   var mojo_float = Float64(python_float)
   
   # Python int to Mojo Int
   var mojo_int = Int(python_int)
   ```

2. **Mojo to Python**
   ```mojo
   # Mojo Float64 to Python float
   var py_float = float_func(str(mojo_float))
   ```

## Error Handling

### Best Practices
1. **Parameter Validation**
   ```mojo
   # Example of float range validation
   fn validate_ratio(value: Float64) raises:
       if value < 0.0 or value > 1.0:
           raise Error("Value must be between 0.0 and 1.0")
   ```

2. **Type Conversion Safety**
   ```mojo
   try:
       var float_val = float_func(string_val)
   except:
       print("Warning: Using default value")
       var float_val = float_func(default_val)
   ```

## Debugging Tips

1. **Print Debugging**
   - Add type information to debug prints
   ```mojo
   print("DEBUG: value =", value)
   print("DEBUG: type =", Python.type(value))
   ```

2. **Incremental Testing**
   - Comment out parts of code to isolate issues
   - Test one parameter type at a time
   - Validate each conversion step separately

## Common Pitfalls

1. **Type Mismatch**
   ```mojo
   # Wrong:
   params["value"] = Float64(1.0)  # Can't store Mojo Float64 in Python dict
   
   # Correct:
   params["value"] = float_func("1.0")  # Store as Python float
   ```

2. **Dynamic vs Compile-time Parameters**
   ```mojo
   # Wrong:
   fn get_param(name: String)  # Runtime string
   
   # Correct:
   fn get_param[name: StringLiteral]()  # Compile-time parameter
   ```

3. **Absolute Value for Floats**
   ```mojo
   # Wrong:
   abs(float_value)  # Won't work with Python floats
   
   # Correct:
   math.fabs(float_value)  # Use Python's math.fabs for floats
   ```

## Best Practices Summary

1. **Type Consistency**
   - Keep values in Python types when working primarily with Python
   - Convert to Mojo types only when needed for Mojo-specific operations

2. **Parameter Handling**
   - Use compile-time parameters with StringLiteral
   - Handle float parameters through string conversion
   - Validate parameters immediately after parsing

3. **Error Handling**
   - Add detailed error messages
   - Use try/except blocks for type conversions
   - Provide sensible defaults

4. **Code Organization**
   - Group parameters by type (int, float, string, bool)
   - Add validation functions for specific parameter types
   - Include debug output for troubleshooting

## Useful Command Examples

```bash
# Basic parameter passing
mojo build -D learning_rate="0.001" script.mojo

# Multiple parameters
mojo build -D learning_rate="0.001" -D batch_size="32" -D train_split="0.8" script.mojo

# Running with debug output
mojo script.mojo -D debug="1"
```

Would you like me to expand on any of these sections or add additional topics?  # Python to Mojo Conversion Guide: Lessons & Best Practices

## 1. Working with Python Objects in Mojo

### Python Integration Basics
```mojo
from python import Python, PythonObject

# Import Python modules
var numpy = Python.import_module("numpy")
var torch = Python.import_module("torch")
```

### Key Differences with None
```mojo
# Wrong:
self.config = Python.None

# Correct:
self.config = Python.evaluate("None")
```

### None Comparison
```mojo
# Wrong:
if self.config is not Python.None

# Correct:
if not self.config.is_(Python.evaluate("None"))
```

## 2. Dictionary Operations

### Creating Python Dictionaries
```mojo
# Wrong:
var dict = {"key": "value"}  # Mojo doesn't support dict literals yet

# Correct:
var dict = Python.evaluate("{'key': 'value'}")
# or
var dict = Python.evaluate("dict()")
```

### Updating Dictionaries
```mojo
# Wrong:
checkpoint["model_config"] = config_dict

# Correct:
# Method 1: Using Python's update()
let update_expr = "lambda d, k, v: d.update({k: v})"
var update_func = Python.evaluate(update_expr)
update_func(dict, key, value)

# Method 2: Using evaluate
dict.update(Python.evaluate("{'key': value}"))
```

## 3. Function Arguments and Mutability

### Argument Conventions
```mojo
# Immutable reference (default)
fn read_only(config: PythonObject)

# Mutable reference
fn mutable(inout config: PythonObject)

# Transfer ownership
fn take_ownership(owned config: PythonObject)
```

### Working with Mutable Objects
```mojo
# When modifying a Python object, mark it as inout
fn update_dict(inout dict: PythonObject) raises:
    dict.update(Python.evaluate("{'new_key': 'new_value'}"))
```

## 4. Error Handling

### Using raises
```mojo
# Mark functions that might raise exceptions
fn might_fail() raises -> PythonObject:
    try:
        return Python.evaluate("some_operation()")
    except:
        print("Operation failed")
        raise
```

## 5. Creating Python Classes in Mojo

### Dynamic Class Creation
```mojo
fn create_python_class() raises -> PythonObject:
    return Python.evaluate("""
    class MyClass:
        def __init__(self, value):
            self.value = value
            
        def process(self):
            return self.value * 2
    MyClass
    """)
```

## 6. List Operations

### Working with Python Lists
```mojo
# Create empty Python list
var list = Python.evaluate("[]")

# Append to list
list.append(item)

# List comprehension alternative
var result = Python.evaluate("[]")
for item in items:
    if some_condition:
        result.append(item)
```

## 7. String Operations

### String Concatenation
```mojo
# Python string concatenation
var message = "Current value: " + String(value)

# For complex strings, use Python's format
var formatted = Python.evaluate(f"'Value: {value}, Type: {type}'")
```

## 8. Best Practices

### 1. Type Handling
- Use `PythonObject` for Python-originated data structures
- Use native Mojo types (Int, String, etc.) for simple values
- Always mark Python object modification with `inout`

### 2. Error Handling
- Use `raises` keyword for functions that interact with Python
- Implement proper try/except blocks
- Provide meaningful error messages

### 3. Memory Management
- Be explicit about ownership with `owned` and `inout`
- Clean up resources properly in error cases
- Use Python's context managers through `Python.evaluate()` when needed

### 4. Performance
- Minimize Python object creation and modification
- Batch operations when possible
- Use native Mojo types and operations where appropriate

## 9. Common Pitfalls to Avoid

1. **Dictionary Literals**
   - Don't use Mojo dictionary literals for Python dicts
   - Use `Python.evaluate()` instead

2. **None Handling**
   - Don't use `Python.None`
   - Use `Python.evaluate("None")` instead

3. **Mutability**
   - Don't forget `inout` when modifying Python objects
   - Don't modify borrowed references

4. **Type Conversion**
   - Don't assume automatic type conversion
   - Explicitly convert between Mojo and Python types

## 10. Debugging Tips

1. **Print Debugging**
```mojo
print("Debug:", String(python_object))  # Convert to String for printing
```

2. **Type Checking**
```mojo
print("Type:", Python.type(object).__name__)
```

3. **Exception Information**
```mojo
try:
    # code that might fail
except:
    print("Error occurred")
    var traceback = Python.import_module("traceback")
    traceback.print_exc()
```

## 11. Migration Strategy

1. **Incremental Conversion**
   - Convert small, self-contained functions first
   - Maintain Python interop for complex operations
   - Test thoroughly after each conversion

2. **Keep Python Fallback**
   - Maintain Python versions of critical code
   - Use Python implementations when Mojo features are missing
   - Plan for gradual optimization

3. **Documentation**
   - Document Python dependencies
   - Note any workarounds used
   - Include examples of both Python and Mojo usage

Remember that Mojo is evolving rapidly, and some current limitations might be resolved in future versions. Keep your code modular and well-documented to make future optimizations easier. # Python to Mojo Conversion Guide: Lessons & Best Practices

## 1. Python Object Handling

### None Values
```mojo
# Don't use
var x = Python.None  # Error: expected name in attribute reference
var x = None        # Error: None not defined

# Do use
var none_value = Python.evaluate("None")
var x = none_value
```

### String Operations
```mojo
# Don't use
logger.debug("Value: " + String(python_value))  # Error: String conversion issues

# Do use
var msg = Python.evaluate("'Value: {}'").format(python_value)
logger.debug(msg)
```

### List Comprehensions
```mojo
# Don't use
var items = [x.name for x in python_list]  # Error: list comprehension not supported

# Do use
var items = Python.evaluate("[x.name for x in python_list]")
```

## 2. Type Handling

### PythonObject Declaration
```mojo
# Declare PythonObject variables before use
var data: PythonObject  # Declare type
if condition:
    data = some_value
else:
    data = other_value
```

### Keyword Arguments
```mojo
# Don't use
var obj = Constructor(**kwargs)  # Error: keyword unpacking not supported

# Do use
var obj = Python.evaluate("Constructor(**kwargs)")
```

## 3. Python Integration

### Python Functions/Methods
```mojo
# Use Python's built-in functions instead of Mojo equivalents
# Don't use
var length = len(python_list)  # May fail
var absolute = abs(value)      # May fail

# Do use
var length = Python.evaluate("len")(python_list)
var math = Python.import_module("math")
var absolute = math.fabs(value)
```

### Module Imports
```mojo
# Import Python modules when needed
var module = Python.import_module("module_name")
var submodule = Python.import_module("package.submodule")
```

## 4. Error Handling

### Exception Handling
```mojo
# Don't use
try:
    # code
except SpecificError as e:  # Error: specific exceptions not supported
    # handle error

# Do use
try:
    # code
except:
    # handle all errors
```

## 5. Common Patterns

### Conditional Statements with Python Objects
```mojo
# For 'is None' checks
var is_none = Python.evaluate("value is None").to_bool()

# For 'in' operator
var contains = Python.evaluate("key in dict").to_bool()
```

### String Formatting
```mojo
# Pattern for complex string formatting
var formatted = Python.evaluate("'Complex {} with {:.2f}'").format(
    value1,
    value2
)
```

## 6. Best Practices

1. **Python Object Consistency**
   - Keep values as PythonObjects when working with Python libraries
   - Convert to Mojo types only when necessary for Mojo-specific operations

2. **String Operations**
   - Use Python's string formatting instead of Mojo string concatenation
   - Format strings in Python land when dealing with PythonObjects

3. **Error Handling**
   - Use simple try/except blocks without exception types
   - Convert Python exceptions to Mojo Error types when needed

4. **Variable Scope**
   - Declare PythonObject variables before conditional blocks
   - Ensure proper variable scope for all PythonObjects

5. **Type Conversions**
   - Minimize conversions between Mojo and Python types
   - Use Python's native operations when working with PythonObjects

## 7. Common Gotchas

1. **None Handling**
   ```mojo
   # Always use Python.evaluate("None") instead of Python.None
   var none_value = Python.evaluate("None")
   ```

2. **String Conversion**
   ```mojo
   # Don't convert PythonObjects to String directly
   # Use Python's string formatting instead
   ```

3. **List/Dict Operations**
   ```mojo
   # Use Python's native operations for Python containers
   var py_list = Python.evaluate("[]")
   py_list.append(item)  # Instead of push_back or other Mojo methods
   ```

4. **Boolean Operations**
   ```mojo
   # Convert Python booleans to Mojo booleans when needed
   var py_bool = Python.evaluate("condition")
   var mojo_bool = py_bool.to_bool()
   ```

## 8. Performance Tips

1. Minimize conversions between Mojo and Python types
2. Batch operations in Python when possible
3. Keep data in one type system (either Mojo or Python) as much as possible

## 9. Debugging Tips

1. Use Python's print/logging for debugging PythonObjects
2. Check variable types when getting conversion errors
3. Use Python's native operations when working with Python objects

## 10. General Recommendations

1. Plan your type system boundaries carefully
2. Keep Python operations in Python
3. Use Mojo's type system for performance-critical sections
4. Document type conversions and assumptions

Remember: When in doubt, keep operations in Python land when working with Python objects and libraries. Only convert to Mojo types when you need Mojo-specific functionality or performance optimizations.