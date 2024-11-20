from python import Python, PythonObject

fn test_python_steps() raises:
    print("=== Testing Individual Python Evaluations ===")
    
    print("\nStep 1: Create empty dict")
    try:
        var empty_dict = Python.evaluate("{}")
        print("Success: Created empty dict")
    except:
        print("Failed to create empty dict")
        
    print("\nStep 2: Create simple type")
    try:
        var simple_type = Python.evaluate("type('TestClass', (), {})")
        print("Success: Created simple type")
    except:
        print("Failed to create simple type")
        
    print("\nStep 3: Create instance")
    try:
        var instance = Python.evaluate("type('TestClass', (), {})()")
        print("Success: Created instance")
    except:
        print("Failed to create instance")
        
    print("\nStep 4: Create dict with lambda")
    try:
        var methods = Python.evaluate("{'test_method': lambda self: None}")
        print("Success: Created methods dict")
    except:
        print("Failed to create methods dict")
        
    print("\nStep 5: Create type with method")
    try:
        var type_with_method = Python.evaluate("type('TestClass', (), {'test_method': lambda self: None})")
        print("Success: Created type with method")
    except:
        print("Failed to create type with method")
        
    print("\nStep 6: Create list and append")
    try:
        var test_list = Python.evaluate("[]")
        var test_obj = Python.evaluate("object()")
        test_list.append(test_obj)
        print("Success: Created and appended to list")
    except:
        print("Failed list operations")

fn main() raises:
    test_python_steps()