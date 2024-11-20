from python import Python, PythonObject

fn test_class_creation() raises:
    print("=== Testing Class Creation Steps ===")
    
    # Step 1: Create empty class
    try:
        var simple_class = Python.evaluate("type('SimpleClass', (), {})")
        print("1. Created empty class definition")
        
        # Step 2: Try to instantiate
        var instance = Python.evaluate("type('SimpleClass', (), {})()")
        print("2. Created class instance")
        
        # Step 3: Try with a method
        var class_with_method = """
class_dict = {'test_method': lambda self: 'test'}
TestClass = type('TestClass', (), class_dict)
TestClass()"""
        var test_obj = Python.evaluate(class_with_method)
        print("3. Created class with method")
        
        # Step 4: Try to call method
        var result = test_obj.test_method()
        print("4. Called method:", result)
        
        # Step 5: Create a list and append instance
        var obj_list = Python.evaluate("[]")
        obj_list.append(test_obj)
        print("5. Added to list, length:", Python.evaluate("len")(obj_list))
        
    except:
        print("Failed at last tried step")

fn main() raises:
    test_class_creation()