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