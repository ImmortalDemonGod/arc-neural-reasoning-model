import sys
import os

# Add the root directory of the project to the PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import torch
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

def test_identical_inputs_and_targets():
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    target_tensor = torch.tensor([[1, 2], [3, 4]])
    prediction_tensor = torch.tensor([[1, 2], [3, 4]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for identical input and target"

def test_completely_different_inputs_and_targets():
    input_tensor = torch.tensor([[1, 1], [1, 1]])
    target_tensor = torch.tensor([[0, 0], [0, 0]])
    prediction_tensor = torch.tensor([[0, 0], [0, 0]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for correct prediction of all differing pixels"

def test_partial_differences():
    input_tensor = torch.tensor([[1, 2], [3, 4]])
    target_tensor = torch.tensor([[1, 0], [3, 0]])
    prediction_tensor = torch.tensor([[1, 0], [3, 4]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 0.5, "Expected accuracy of 0.5 for partial correct predictions"

def test_empty_tensors():
    input_tensor = torch.tensor([])
    target_tensor = torch.tensor([])
    prediction_tensor = torch.tensor([])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for empty tensors"

def test_single_pixel_difference():
    input_tensor = torch.tensor([[1]])
    target_tensor = torch.tensor([[0]])
    prediction_tensor = torch.tensor([[0]])
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)
    assert accuracy == 1.0, "Expected accuracy of 1.0 for single pixel difference"

def test_differential_pixel_accuracy_with_arckit_data():                                                                                                 
    # Load a specific task using arckit                                                                                                                  
    task_id = "fe9372f3"  # Replace with the ID of the task you want to test                                                                             
    task_data = arckit.load_task(task_id)                                                                                                                
                                                                                                                                                        
    # Process the task data                                                                                                                              
    input_tensor = torch.tensor(task_data['input']).unsqueeze(0)  # Add batch dimension                                                                  
    target_tensor = torch.tensor(task_data['output']).unsqueeze(0)  # Add batch dimension                                                                
                                                                                                                                                        
    # Initialize the model                                                                                                                               
    model_config = ModelConfig(n_embd=64, n_head=2, n_layer=1)                                                                                           
    model = GPT2ARC(model_config)                                                                                                                        
    model.eval()  # Set the model to evaluation mode                                                                                                     
                                                                                                                                                        
    # Make predictions                                                                                                                                   
    with torch.no_grad():                                                                                                                                
        prediction_tensor = model(input_tensor)                                                                                                          
                                                                                                                                                        
    # Evaluate the differential_pixel_accuracy                                                                                                           
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor)                                                         
    print(f"Differential Pixel Accuracy for task {task_id}: {accuracy}")                                                                                 
                                                                                                                                                        
    # Add assertions as needed                                                                                                                           
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"  

# Run the tests
if __name__ == "__main__":
    test_identical_inputs_and_targets()
    test_completely_different_inputs_and_targets()
    test_partial_differences()
    test_empty_tensors()
    test_single_pixel_difference()
    print("All tests passed!")
