# gpt2_arc/tests/test_balancing.py                                                                                                                                   
                                                                                                                                                                    
import pytest                                                                                                                                                        
import torch                                                                                                                                                         
from torch.utils.data import DataLoader, WeightedRandomSampler                                                                                                       
from gpt2_arc.src.data.arc_dataset import ARCDataset                                                                                                                 
from gpt2_arc.src.models.gpt2 import GPT2ARC                                                                                                                         
from gpt2_arc.src.config import Config, TrainingConfig, ModelConfig                                                                                                  
                                                                                                                                                                    
from collections import Counter                                                                                                                                      
                                                                                                                                                                    
def generate_synthetic_data():                                                                                                                                       
    """                                                                                                                                                              
    Generates synthetic data with an imbalanced class distribution.                                                                                                  
    Returns:                                                                                                                                                         
        List[Dict]: A list of samples with 'input', 'output', and 'symbol'.                                                                                          
    """                                                                                                                                                              
    data = []                                                                                                                                                        
    # Class 0: 50 samples                                                                                                                                            
    for _ in range(50):                                                                                                                                              
        data.append({"input": torch.tensor([0]), "output": torch.tensor(0), "symbol": "0"})                                                                          
    # Class 1: 30 samples                                                                                                                                            
    for _ in range(30):                                                                                                                                              
        data.append({"input": torch.tensor([1]), "output": torch.tensor(1), "symbol": "1"})                                                                          
    # Class 2: 20 samples                                                                                                                                            
    for _ in range(20):                                                                                                                                              
        data.append({"input": torch.tensor([2]), "output": torch.tensor(2), "symbol": "2"})                                                                          
    return data                                                                                                                                                      
                                                                                                                                                                    
@pytest.fixture                                                                                                                                                      
def imbalanced_dataset():                                                                                                                                            
    data = generate_synthetic_data()                                                                                                                                 
    dataset = ARCDataset(data_source=data, balance_symbols=True)                                                                                                     
    symbol_freq = dataset.get_symbol_frequencies()                                                                                                                   
    symbol_freq_dict = {str(i): freq for i, freq in enumerate(symbol_freq)}                                                                                          
    return dataset, symbol_freq_dict                                                                                                                                 
                                                                                                                                                                    
def test_weighted_random_sampler(imbalanced_dataset):                                                                                                                
    dataset, symbol_freq_dict = imbalanced_dataset                                                                                                                   
    num_classes = 3                                                                                                                                                  
                                                                                                                                                                    
    # Initialize model (though not used in this test)                                                                                                                
    config = Config(                                                                                                                                                 
        model=ModelConfig(n_embd=4, n_head=2, n_layer=3, dropout=0.1, mamba_ratio=1.0, d_state=2, d_conv=4),                                                         
        training=TrainingConfig(balance_symbols=True, balancing_method="weighting"),                                                                                 
        evaluation=None  # Not needed for this test                                                                                                                  
    )                                                                                                                                                                
    model = GPT2ARC(config=config, num_classes=num_classes, symbol_freq=symbol_freq_dict)                                                                            
                                                                                                                                                                    
    # Compute class weights (inverse of frequencies)                                                                                                                 
    class_weights = 1.0 / torch.tensor(list(symbol_freq_dict.values()), dtype=torch.float)                                                                           
                                                                                                                                                                    
    # Assign weights to each sample based on its class                                                                                                               
    sample_weights = [class_weights[str(sample['symbol'])] for sample in dataset.data]                                                                               
    sample_weights = torch.tensor(sample_weights)                                                                                                                    
                                                                                                                                                                    
    # Initialize WeightedRandomSampler                                                                                                                               
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)                                                       
                                                                                                                                                                    
    # Initialize DataLoader with the sampler                                                                                                                         
    train_loader = DataLoader(                                                                                                                                       
        dataset,                                                                                                                                                     
        batch_size=10,                                                                                                                                               
        sampler=sampler,                                                                                                                                             
        shuffle=False,                                                                                                                                               
        pin_memory=False,                                                                                                                                            
        prefetch_factor=2,                                                                                                                                           
        persistent_workers=False                                                                                                                                     
    )                                                                                                                                                                
                                                                                                                                                                    
    # Collect sampled classes                                                                                                                                        
    sampled_classes = []                                                                                                                                             
    for batch_idx, (inputs, targets, task_ids) in enumerate(train_loader):                                                                                           
        sampled_classes.extend(targets.tolist())                                                                                                                     
        if batch_idx >= 9:  # Collect 100 samples                                                                                                                    
            break                                                                                                                                                    
                                                                                                                                                                    
    # Calculate sampled class frequencies                                                                                                                            
    sampled_count = Counter(sampled_classes)                                                                                                                         
    total_samples = sum(sampled_count.values())                                                                                                                      
    sampled_distribution = {cls: count / total_samples for cls, count in sampled_count.items()}                                                                      
                                                                                                                                                                    
    # Expected distribution is approximately uniform                                                                                                                 
    expected_distribution = {cls: 1/num_classes for cls in range(num_classes)}                                                                                       
                                                                                                                                                                    
    # Allow a margin of error (e.g., ±10%)                                                                                                                           
    margin = 0.1                                                                                                                                                     
    for cls in expected_distribution:                                                                                                                                
        assert abs(sampled_distribution.get(cls, 0) - expected_distribution[cls]) < margin, \                                                                        
        f"Class {cls} distribution mismatch. Expected ~{expected_distribution[cls]:.2f}, Got {sampled_distribution.get(cls, 0):.2f}"      