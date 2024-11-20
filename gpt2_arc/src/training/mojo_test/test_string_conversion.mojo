from python import Python, PythonObject

fn main() raises -> None:
    # Test Case 1: Converting PythonObject to String using str()
    var py_float = Python.evaluate("float")(123.456)
    var message1 = Python.evaluate("'Memory allocated: {} bytes'").format(py_float)
    Python.evaluate("print")(message1)

    # Test Case 2: Using Python's format() method for string formatting
    var py_int = Python.evaluate("int")(789)
    var message2 = Python.evaluate("'Memory allocated: {} bytes'").format(py_int)
    Python.evaluate("print")(message2)

    # Test Case 3: Handling None values and converting them to strings
    var py_none = Python.evaluate("None")
    var message3 = Python.evaluate("'Value is: {}'").format(py_none)
    Python.evaluate("print")(message3)

    # Test Case 4: Working with Python list and its length for concatenation
    var py_list = Python.evaluate("[]")
    py_list.append(1)
    py_list.append(2)
    py_list.append(3)
    var list_length = Python.evaluate("len")(py_list)
    var message4 = Python.evaluate("'List contains {} elements.'").format(list_length)
    Python.evaluate("print")(message4)
