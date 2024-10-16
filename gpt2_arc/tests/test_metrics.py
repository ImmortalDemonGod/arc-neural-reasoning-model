# gpt2_arc/tests/test_metrics.py                                                                                                                                     
                                                                                                                                                                    
import pytest                                                                                                                                                        
import torch                                                                                                                                                         
from gpt2_arc.src.models.gpt2 import GPT2ARC                                                                                                                         
from gpt2_arc.src.config import Config, TrainingConfig, ModelConfig                                                                                                  
                                                                                                                                                                    
def generate_test_data(num_samples=100):                                                                                                                             
    """                                                                                                                                                              
    Generates synthetic predictions and targets for testing.                                                                                                         
    Returns:                                                                                                                                                         
        preds (torch.Tensor): Predicted class indices.                                                                                                               
        targets (torch.Tensor): Ground truth class indices.                                                                                                          
    """                                                                                                                                                              
    preds = torch.randint(0, 3, (num_samples,))                                                                                                                      
    targets = torch.randint(0, 3, (num_samples,))                                                                                                                    
    return preds, targets                                                                                                                                            
                                                                                                                                                                    
@pytest.fixture                                                                                                                                                      
def model():                                                                                                                                                         
    config = Config(                                                                                                                                                 
        model=ModelConfig(n_embd=4, n_head=2, n_layer=3, dropout=0.1, mamba_ratio=1.0, d_state=2, d_conv=4),                                                         
        training=TrainingConfig(balance_symbols=True, balancing_method="weighting"),                                                                                 
        evaluation=None                                                                                                                                              
    )                                                                                                                                                                
    symbol_freq_dict = {'0': 0.5, '1': 0.3, '2': 0.2}                                                                                                                
    return GPT2ARC(config=config, num_classes=3, symbol_freq=symbol_freq_dict)                                                                                       
                                                                                                                                                                    
def test_training_metrics(model):                                                                                                                                    
    preds, targets = generate_test_data()                                                                                                                            
    # Mock a batch                                                                                                                                                   
    batch = (torch.tensor([0]*len(preds)), targets, ['task']*len(preds))                                                                                             
                                                                                                                                                                    
    # Perform a training step                                                                                                                                        
    loss = model.training_step(batch, batch_idx=0)                                                                                                                   
                                                                                                                                                                    
    # Check that metrics are computed and logged                                                                                                                     
    assert 'train_loss' in model.logged_metrics                                                                                                                      
    assert 'train_precision' in model.logged_metrics                                                                                                                 
    assert 'train_recall' in model.logged_metrics                                                                                                                    
    assert 'train_f1' in model.logged_metrics                                                                                                                        
                                                                                                                                                                    
    # Verify that the loss is a scalar tensor                                                                                                                        
    assert isinstance(loss, torch.Tensor)                                                                                                                            
    assert loss.dim() == 0, "Loss should be a scalar tensor"                                                                                                         
                                                                                                                                                                    
def test_validation_metrics(model):                                                                                                                                  
    preds, targets = generate_test_data()                                                                                                                            
    # Mock a batch                                                                                                                                                   
    batch = (torch.tensor([0]*len(preds)), targets, ['task']*len(preds))                                                                                             
                                                                                                                                                                    
    # Perform a validation step                                                                                                                                      
    model.validation_step(batch, batch_idx=0)                                                                                                                        
                                                                                                                                                                    
    # Check that metrics are computed and logged                                                                                                                     
    assert 'val_loss' in model.logged_metrics                                                                                                                        
    assert 'val_precision' in model.logged_metrics                                                                                                                   
    assert 'val_recall' in model.logged_metrics                                                                                                                      
    assert 'val_f1' in model.logged_metrics
from gpt2_arc.src.utils.helpers import differential_pixel_accuracy

@pytest.fixture
def pad_symbol_idx():
    return 10

def test_pixel_accuracy_with_padding(pad_symbol_idx):
    # Sample tensors with padding tokens
    input_tensor = torch.tensor([[[1, 2, pad_symbol_idx],
                                   [pad_symbol_idx, 4, 5],
                                   [6, pad_symbol_idx, 8]]], dtype=torch.float)
    
    target_tensor = torch.tensor([[[1, 0, pad_symbol_idx],
                                    [pad_symbol_idx, 4, 5],
                                    [6, pad_symbol_idx, 8]]], dtype=torch.long)
    
    prediction_tensor = torch.tensor([[[1, 3, pad_symbol_idx],
                                        [pad_symbol_idx, 4, 0],
                                        [6, pad_symbol_idx, 8]]], dtype=torch.long)
    
    # Calculate differential pixel accuracy excluding padding tokens
    accuracy, _, _ = differential_pixel_accuracy(input_tensor, target_tensor, prediction_tensor, pad_symbol_idx=pad_symbol_idx)
    
    # Valid pixels: positions where target != pad_symbol_idx
    # Differences:
    # (2 != 0): input=2, target=0, prediction=3 => False
    # (5 != 0): input=5, target=5, prediction=0 => False (since prediction==target)
    # Hence, total_diff_pixels = 1 (only position (0,1,1) where target=0)
    # Correct_diff_predictions = 0 (since prediction=3 != target=0)
    # Expected accuracy = 0 / 1 = 0.0
    
    expected_accuracy = 0.0
    assert accuracy == expected_accuracy, f"Expected accuracy {expected_accuracy}, got {accuracy}"
