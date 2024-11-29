import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import logging
from typing import Dict, Tuple, Optional

from gpt2_arc.src.models.gpt2 import GPT2ARC, Attention, TransformerBlock, MambaLayer, BitLinearNew  # Added BitLinearNew import
from gpt2_arc.src.config import Config

# Configure logging for detailed test output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MockGPT2Config:
    """Comprehensive mock configuration with all required parameters"""
    class ModelConfig:
        def __init__(self):
            self.n_embd = 64
            self.n_head = 8
            self.dropout = 0.1
            self.n_layer = 4
            self.mamba_ratio = 0.5
            self.d_state = 16
            self.d_conv = 4
            self.mamba_depth = 1
            self.mamba_expand = 2

    class TrainingConfig:
        def __init__(self):
            self.num_classes = 10
            self.batch_size = 16
            self.num_workers = 2
            self.use_gpu = False
            self.pad_symbol_idx = 0
            self.include_pad_in_loss = True
            self.symbol_freq = {i: 1.0 for i in range(10)}

    def __init__(self):
        self.model = self.ModelConfig()
        self.training = self.TrainingConfig()
        self.debug = True
        self.test_data_path = "path/to/test/data"

class BaseTestFixtures:
    """Base fixtures shared across all test classes"""
    
    @pytest.fixture
    def config(self) -> MockGPT2Config:
        return MockGPT2Config()

    @pytest.fixture
    def base_model(self, config) -> GPT2ARC:
        with patch('gpt2_arc.src.models.gpt2.MambaBlock') as mock_mamba:
            return GPT2ARC(config=config, num_classes=config.training.num_classes)

    @pytest.fixture
    def attention_component(self, config) -> Attention:
        return Attention(
            n_embd=config.model.n_embd,
            n_head=config.model.n_head,
            dropout=config.model.dropout
        )

    @pytest.fixture
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width = 2, 6, 6
        inputs = torch.randn(batch_size, 1, height, width)
        targets = torch.randint(0, 10, (batch_size, height * width))
        return inputs, targets

    @pytest.fixture
    def model_with_custom_weights(self) -> GPT2ARC:
        config = MockGPT2Config()
        config.training.symbol_freq = {i: (i + 1) / 10 for i in range(10)}
        return GPT2ARC(config=config, num_classes=10)

    @pytest.fixture
    def gpu_model(self, config) -> GPT2ARC:
        """Additional fixture for GPU testing"""
        config.training.use_gpu = True
        with patch('gpt2_arc.src.models.gpt2.MambaBlock') as mock_mamba:
            model = GPT2ARC(config=config, num_classes=config.training.num_classes)
            if torch.cuda.is_available():
                model = model.cuda()
            return model

class TestModelInitialization(BaseTestFixtures):
    """Tests for model initialization and configuration"""

    def test_model_structure(self, base_model, config):
        """Verify model structure and component initialization"""
        assert isinstance(base_model, GPT2ARC)
        assert len(base_model.blocks) == config.model.n_layer
        assert base_model.fc_out.out_features == config.training.num_classes

    def test_custom_weight_initialization(self, base_model):
        """Test custom weight initialization across different layer types"""
        for name, module in base_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    @pytest.mark.parametrize("weight_type", ["conv", "linear", "norm"])
    def test_layer_specific_initialization(self, base_model, weight_type):
        """Test initialization for specific layer types"""
        if weight_type == "conv":
            layer = nn.Conv2d(1, 64, 3)
        elif weight_type == "linear":
            layer = nn.Linear(64, 64)
        else:
            layer = nn.LayerNorm(64)
            
        base_model._init_weights(layer)
        
        if weight_type in ["conv", "linear"]:
            assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
            if layer.bias is not None:
                assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

class TestForwardPropagation(BaseTestFixtures):
    """Tests for forward pass behavior and shape handling"""

    @pytest.mark.parametrize("input_shape", [
        (2, 1, 6, 6),
        (2, 1, 8, 8),
        (2, 1, 12, 12)
    ])
    def test_input_shape_handling(self, base_model, input_shape):
        """Test model's ability to handle different input shapes"""
        inputs = torch.randn(*input_shape)
        output = base_model(inputs)
        expected_shape = (input_shape[0], input_shape[2] * input_shape[3], 
                         base_model.config.training.num_classes)
        assert output.shape == expected_shape

    def test_attention_mechanism(self, attention_component, config):
        """Test attention component with masking"""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.model.n_embd)
        mask = torch.ones(batch_size, seq_len)
        mask[:, seq_len//2:] = 0
        output = attention_component(x, mask)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_mamba_integration(self, base_model, config):
        """Test Mamba block integration and output"""
        batch_size, seq_len = 2, 36
        x = torch.randn(batch_size, 1, 6, 6)
        output = base_model(x)
        assert output.shape == (batch_size, seq_len, config.training.num_classes)
        assert not torch.isnan(output).any()

class TestTrainingBehavior(BaseTestFixtures):
    """Tests for training-related functionality"""

    def test_training_step_execution(self, base_model, sample_batch):
        """Test complete training step"""
        inputs, targets = sample_batch
        loss = base_model.training_step((inputs, targets, None), 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_validation_step_with_padding(self, model_with_custom_weights):
        """Test validation step with padding tokens"""
        batch_size, seq_len = 2, 36
        inputs = torch.randn(batch_size, 1, 6, 6)
        targets = torch.randint(0, 10, (batch_size, seq_len))
        targets[:, 0] = model_with_custom_weights.pad_symbol_idx
        
        val_loss = model_with_custom_weights.validation_step((inputs, targets, None), 0)
        assert isinstance(val_loss, torch.Tensor)
        assert not torch.isnan(val_loss)

    def test_test_step_metrics(self, model_with_custom_weights):
        """Test metrics calculation during test step"""
        batch_size, seq_len = 2, 36
        inputs = torch.randn(batch_size, 1, 6, 6)
        targets = torch.randint(0, 10, (batch_size, seq_len))
        targets[:, :5] = model_with_custom_weights.pad_symbol_idx
        
        test_loss = model_with_custom_weights.test_step((inputs, targets, None), 0)
        assert isinstance(test_loss, torch.Tensor)
        assert not torch.isnan(test_loss)

    @pytest.mark.parametrize("dropout_value", [0.0, 0.5])
    def test_dropout_behavior(self, config, dropout_value):
        """Test dropout behavior in different modes"""
        config.model.dropout = dropout_value
        with patch('gpt2_arc.src.models.gpt2.MambaBlock'):
            model = GPT2ARC(config, config.training.num_classes)
        
        model.train()
        inputs = torch.randn(2, 1, 6, 6)
        out1 = model(inputs)
        out2 = model(inputs)
        
        if dropout_value == 0.0:
            assert torch.allclose(out1, out2, rtol=1e-4)
        else:
            assert not torch.allclose(out1, out2)

    def test_validation_step_detailed_metrics(self, model_with_custom_weights):
        """Test detailed validation step metrics calculation (lines 230-249)"""
        batch_size, seq_len = 2, 36
        inputs = torch.randn(batch_size, 1, 6, 6)
        targets = torch.randint(0, 10, (batch_size, seq_len))
        
        # Create various padding scenarios
        targets[:, 0:5] = model_with_custom_weights.pad_symbol_idx  # Leading padding
        targets[:, -5:] = model_with_custom_weights.pad_symbol_idx  # Trailing padding
        
        # Add logging callback to capture metrics
        metrics = {}
        def log_callback(name, value, **kwargs):
            metrics[name] = value
        model_with_custom_weights.log = log_callback
        
        val_loss = model_with_custom_weights.validation_step((inputs, targets, None), 0)
        
        # Verify all metrics are calculated
        assert 'val_loss' in metrics
        assert 'val_acc_with_pad' in metrics
        assert 'val_acc_without_pad' in metrics
        assert all(0 <= v <= 1.0 for v in metrics.values() if isinstance(v, float))

    def test_test_step_detailed_metrics(self, model_with_custom_weights):
        """Test detailed test step metrics calculation (lines 263-271)"""
        batch_size, seq_len = 2, 36
        inputs = torch.randn(batch_size, 1, 6, 6)
        targets = torch.randint(0, 10, (batch_size, seq_len))
        
        # Create mixed token scenarios
        targets[:, ::2] = model_with_custom_weights.pad_symbol_idx  # Alternating padding
        
        # Add logging callback to capture metrics
        metrics = {}
        def log_callback(name, value, **kwargs):
            metrics[name] = value
        model_with_custom_weights.log = log_callback
        
        test_loss = model_with_custom_weights.test_step((inputs, targets, None), 0)
        
        # Verify all metrics are calculated
        assert 'test_loss' in metrics
        assert 'test_acc_with_pad' in metrics
        assert 'test_acc_without_pad' in metrics
        assert all(0 <= v <= 1.0 for v in metrics.values() if isinstance(v, float))
   
    def test_accuracy_calculation_helper(self, base_model):
        """Test the accuracy calculation helper method"""
        preds = torch.tensor([[0, 1, 2], [3, 4, 0]])
        targets = torch.tensor([[0, 1, 2], [3, 0, 0]])  # Some padding tokens
        
        acc_with_pad, acc_without_pad = base_model._calculate_accuracies(preds, targets)
        
        expected_with_pad = 4/6  # 4 correct out of 6 total
        expected_without_pad = 4/4  # 4 correct out of 4 non-padding
        
        assert torch.allclose(acc_with_pad, torch.tensor(expected_with_pad))
        assert torch.allclose(acc_without_pad, torch.tensor(expected_without_pad))

    def test_accuracy_calculation_all_padding(self, base_model):
        """Test accuracy calculation with all padding tokens"""
        preds = torch.tensor([[0, 1, 2], [3, 4, 5]])
        targets = torch.full_like(preds, base_model.pad_symbol_idx)
        
        acc_with_pad, acc_without_pad = base_model._calculate_accuracies(preds, targets)
        
        assert torch.allclose(acc_with_pad, torch.tensor(0.0))
        assert torch.allclose(acc_without_pad, torch.tensor(0.0))


class TestDeviceHandling(BaseTestFixtures):
    """Tests for device handling and GPU support"""

    def test_bitlinear_device_handling(self, gpu_model):
        """Test BitLinearNew layer device handling (lines 201-203)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")
        
        # Verify all BitLinearNew layers are on GPU
        for module in gpu_model.modules():
            if isinstance(module, BitLinearNew):
                assert next(module.parameters()).is_cuda

    def test_model_device_transfer(self, base_model):
        """Test model transfer between devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")
            
        # Test CPU to GPU transfer
        inputs = torch.randn(2, 1, 6, 6)
        base_model = base_model.cuda()
        inputs = inputs.cuda()
        outputs = base_model(inputs)
        assert outputs.is_cuda

        # Test GPU to CPU transfer
        base_model = base_model.cpu()
        inputs = inputs.cpu()
        outputs = base_model(inputs)
        assert not outputs.is_cuda

class TestComponentIntegration(BaseTestFixtures):
    """Tests for component integration and handling"""

    def test_component_interactions(self, base_model):
        """Test component interactions (lines 328-331)"""
        batch_size, seq_len = 2, 36
        inputs = torch.randn(batch_size, 1, 6, 6)
        
        # Track intermediate outputs
        intermediate_outputs = []
        def hook_fn(module, input, output):
            intermediate_outputs.append(output)
        
        # Register hooks for each component
        for block in base_model.blocks:
            if isinstance(block, TransformerBlock):
                block.attention.register_forward_hook(hook_fn)
            elif isinstance(block, MambaLayer):
                block.mamba_block.register_forward_hook(hook_fn)
        
        outputs = base_model(inputs)
        
        # Verify component chain
        assert len(intermediate_outputs) == len(base_model.blocks)
        assert all(not torch.isnan(output).any() for output in intermediate_outputs)

class TestErrorHandling(BaseTestFixtures):
    """Tests for error conditions and edge cases"""

    def test_invalid_input_dimensions(self, base_model):
        """Test handling of invalid input dimensions"""
        with pytest.raises(RuntimeError):
            invalid_input = torch.randn(2, 1, 5, 7)  # Non-square input
            base_model(invalid_input)

    def test_zero_batch_size(self, base_model):
        """Test handling of empty batch"""
        with pytest.raises(RuntimeError):
            empty_input = torch.randn(0, 1, 6, 6)
            base_model(empty_input)

    def test_extreme_values(self, base_model):
        """Test handling of extreme input values"""
        extreme_input = torch.full((2, 1, 6, 6), float('inf'))
        output = base_model(extreme_input)
        assert torch.isfinite(output).all()

    def test_invalid_config(self):
        """Test handling of invalid configuration"""
        invalid_config = MagicMock()
        invalid_config.model = None
        
        with pytest.raises(AttributeError):
            GPT2ARC(config=invalid_config, num_classes=10)

class TestIntegrationScenarios(BaseTestFixtures):
    """Tests for end-to-end scenarios and component integration"""

    def test_transformer_mamba_composition(self, base_model, config):
        """Test correct composition of Transformer and Mamba blocks"""
        transformer_count = sum(1 for block in base_model.blocks 
                              if isinstance(block, TransformerBlock))
        mamba_count = sum(1 for block in base_model.blocks 
                         if isinstance(block, MambaLayer))
        
        expected_transformer = int(config.model.n_layer / (1 + config.model.mamba_ratio))
        expected_mamba = config.model.n_layer - expected_transformer
        
        assert transformer_count == expected_transformer
        assert mamba_count == expected_mamba

    def test_end_to_end_training(self, base_model):
        """Test complete forward and backward pass"""
        batch_size, seq_len = 2, 36
        inputs = torch.randn(batch_size, 1, 6, 6)
        targets = torch.randint(0, 10, (batch_size, seq_len))
        
        outputs = base_model(inputs)
        loss = base_model.training_step((inputs, targets, None), 0)
        loss.backward()
        
        for name, param in base_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_dataloader_configuration(self, model_with_custom_weights):
        """Test dataloader setup and configuration"""
        with patch('torch.utils.data.DataLoader') as mock_loader:
            dataloader = model_with_custom_weights.test_dataloader()
            mock_loader.assert_called_once()

class TestLossFunctionConfiguration(BaseTestFixtures):
    """Tests specifically for loss function configuration and behavior"""

    def test_loss_function_with_class_weights(self):
        """Test loss function initialization with class weights"""
        config = MockGPT2Config()
        # Set specific symbol frequencies for testing
        config.training.symbol_freq = {
            0: 0.5,  # More common class
            1: 0.3,
            2: 0.2   # Less common class
        }
        model = GPT2ARC(config=config, num_classes=3)
        
        # Verify loss function has weights
        assert hasattr(model.loss_fn, 'weight')
        assert torch.allclose(model.loss_fn.weight[0], torch.tensor(0.5))
        assert torch.allclose(model.loss_fn.weight[2], torch.tensor(0.2))
        assert model.loss_fn.ignore_index == config.training.pad_symbol_idx

    def test_loss_function_without_class_weights(self):
        """Test loss function initialization without class weights"""
        config = MockGPT2Config()
        config.training.symbol_freq = None
        model = GPT2ARC(config=config, num_classes=10)
        
        # Verify loss function has no weights but has ignore_index
        assert not hasattr(model.loss_fn, 'weight')
        assert model.loss_fn.ignore_index == config.training.pad_symbol_idx
    
    def test_loss_function_device_handling(self):
        """Test loss function weights device handling"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")
            
        config = MockGPT2Config()
        config.training.symbol_freq = {0: 0.5, 1: 0.3, 2: 0.2}
        model = GPT2ARC(config=config, num_classes=3)
        
        # Check initial device
        assert model.loss_fn.weight.device.type == "cpu"
        
        # Move to GPU
        model = model.cuda()
        assert model.loss_fn.weight.device.type == "cuda"
        assert torch.allclose(
            model.loss_fn.weight,
            torch.tensor([0.5, 0.3, 0.2], device="cuda")
        )

    def test_loss_function_missing_indices(self):
        """Test handling of missing indices in symbol frequencies"""
        config = MockGPT2Config()
        config.training.symbol_freq = {0: 0.5, 3: 0.25}  # Skip indices 1, 2
        model = GPT2ARC(config=config, num_classes=5)
        
        expected = torch.tensor([0.5, 1.0, 1.0, 0.25, 1.0])
        assert torch.allclose(model.loss_fn.weight, expected)

class TestTrainingStepMetrics(BaseTestFixtures):
    """Tests for training step metrics calculation"""

    def test_training_metrics_all_correct(self, base_model):
        """Test metrics calculation when all predictions are correct"""
        batch_size, height, width = 2, 6, 6
        inputs = torch.randn(batch_size, 1, height, width)
        targets = torch.ones(batch_size, height * width, dtype=torch.long)
        
        # Mock the forward pass to return perfect predictions
        def mock_forward(*args, **kwargs):
            logits = torch.zeros(batch_size, height * width, base_model.config.training.num_classes)
            logits[:, :, 1] = 100.0  # High confidence for class 1
            return logits
        
        with patch.object(base_model, 'forward', side_effect=mock_forward):
            metrics = {}
            def log_callback(name, value, **kwargs):
                metrics[name] = value
            base_model.log = log_callback
            
            loss = base_model.training_step((inputs, targets, None), 0)
            
            assert metrics['train_acc_with_pad'] == 1.0
            assert metrics['train_acc_without_pad'] == 1.0

    def test_training_metrics_with_padding(self, base_model):
        """Test metrics calculation with mixed padding and non-padding tokens"""
        batch_size, height, width = 2, 6, 6
        inputs = torch.randn(batch_size, 1, height, width)
        targets = torch.ones(batch_size, height * width, dtype=torch.long)
        # Add padding tokens
        targets[:, :height] = base_model.pad_symbol_idx
        
        metrics = {}
        def log_callback(name, value, **kwargs):
            metrics[name] = value
        base_model.log = log_callback
        
        loss = base_model.training_step((inputs, targets, None), 0)
        
        # Verify padding is handled correctly in metrics
        assert 'train_acc_with_pad' in metrics
        assert 'train_acc_without_pad' in metrics
        assert metrics['train_acc_with_pad'] != metrics['train_acc_without_pad']

class TestInputFormatting(BaseTestFixtures):
    """Tests for input formatting and reshaping"""

    def test_2d_to_4d_conversion(self, base_model):
        """Test conversion from 2D to 4D input format"""
        batch_size = 2
        seq_length = 36  # 6x6
        input_2d = torch.randn(batch_size, seq_length)
        
        output = base_model(input_2d)
        assert output.shape == (batch_size, seq_length, base_model.config.training.num_classes)

    def test_invalid_sequence_length(self, base_model):
        """Test handling of non-square sequence length"""
        batch_size = 2
        seq_length = 35  # Not a perfect square
        input_2d = torch.randn(batch_size, seq_length)
        
        with pytest.raises(ValueError, match="Sequence length must be a perfect square"):
            base_model(input_2d)


class TestLossFunctionWeights(BaseTestFixtures):
    """Tests for loss function weight initialization and device handling"""

    def test_class_weights_initialization_and_device(self):
        """Test class weights initialization and device placement"""
        config = MockGPT2Config()
        # Set up symbol frequencies with missing indices
        config.training.symbol_freq = {
            0: 0.5,
            2: 0.2,  # Skip index 1 to test .get() default
            4: 0.1   # Skip index 3 to test .get() default
        }
        
        model = GPT2ARC(config=config, num_classes=5)
        
        # Check weight tensor values
        expected_weights = torch.tensor([0.5, 1.0, 0.2, 1.0, 0.1], dtype=torch.float32)
        assert torch.allclose(model.loss_fn.weight, expected_weights)
        
        # Test device placement
        if torch.cuda.is_available():
            model = model.cuda()
            assert model.loss_fn.weight.device.type == "cuda"
            
            # Test if weights move to CPU
            model = model.cpu()
            assert model.loss_fn.weight.device.type == "cpu"

    def test_class_weights_with_device_transfer(self):
        """Test class weights device transfer during model initialization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for testing")
            
        config = MockGPT2Config()
        config.training.use_gpu = True
        config.training.symbol_freq = {i: 1.0/(i+1) for i in range(5)}
        
        with patch('torch.cuda.is_available', return_value=True):
            model = GPT2ARC(config=config, num_classes=5).cuda()
            # Verify weights are on GPU
            assert model.loss_fn.weight.is_cuda
            # Verify weight values maintained after device transfer
            expected_weights = torch.tensor([1.0, 0.5, 1/3, 0.25, 0.2], dtype=torch.float32).cuda()
            assert torch.allclose(model.loss_fn.weight, expected_weights)

class TestDataLoaderConfiguration(BaseTestFixtures):
    """Tests for DataLoader configuration"""

    @pytest.mark.parametrize("use_gpu", [True, False])
    def test_dataloader_gpu_settings(self, use_gpu):
        """Test DataLoader configuration with different GPU settings"""
        config = MockGPT2Config()
        config.training.use_gpu = use_gpu
        model = GPT2ARC(config=config, num_classes=10)
        
        with patch('torch.utils.data.DataLoader') as mock_loader:
            dataloader = model.test_dataloader()
            # Verify pin_memory setting matches GPU usage
            call_args = mock_loader.call_args[1]
            assert call_args['pin_memory'] == use_gpu
            assert call_args['num_workers'] == config.training.num_workers
            assert not call_args['shuffle']  # Test data shouldn't be shuffled

class TestDataLoaderConfiguration(BaseTestFixtures):
    """Tests for DataLoader configuration and creation"""

    def test_test_dataloader_creation(self, model_with_custom_weights):
        """Test complete DataLoader creation process"""
        with patch('gpt2_arc.src.data.arc_dataset.ARCDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_loader:
            
            # Configure mock dataset
            mock_dataset.return_value = "mock_dataset"
            
            # Get dataloader
            dataloader = model_with_custom_weights.test_dataloader()
            
            # Verify dataset creation
            mock_dataset.assert_called_once_with(
                data_source=model_with_custom_weights.config.test_data_path,
                is_test=True,
                num_symbols=model_with_custom_weights.config.training.num_symbols,
                pad_symbol_idx=model_with_custom_weights.config.training.pad_symbol_idx,
                symbol_freq=model_with_custom_weights.config.training.symbol_freq,
                debug=model_with_custom_weights.config.debug
            )
            
            # Verify dataloader creation with correct parameters
            mock_loader.assert_called_once_with(
                "mock_dataset",
                batch_size=model_with_custom_weights.config.training.batch_size,
                num_workers=model_with_custom_weights.config.training.num_workers,
                shuffle=False,
                pin_memory=model_with_custom_weights.config.training.use_gpu
            )

    @pytest.mark.parametrize("num_workers,use_gpu", [
        (0, False),
        (2, False),
        (4, True),
        (8, True)
    ])
    def test_dataloader_worker_and_memory_settings(self, num_workers, use_gpu):
        """Test DataLoader worker and memory pin settings"""
        config = MockGPT2Config()
        config.training.num_workers = num_workers
        config.training.use_gpu = use_gpu
        
        model = GPT2ARC(config=config, num_classes=10)
        
        with patch('gpt2_arc.src.data.arc_dataset.ARCDataset') as mock_dataset, \
             patch('torch.utils.data.DataLoader') as mock_loader:
            
            mock_dataset.return_value = "mock_dataset"
            dataloader = model.test_dataloader()
            
            # Verify dataloader settings
            _, kwargs = mock_loader.call_args
            assert kwargs['num_workers'] == num_workers
            assert kwargs['pin_memory'] == use_gpu
            assert not kwargs['shuffle']
            assert kwargs['batch_size'] == config.training.batch_size

    def test_dataloader_error_handling(self):
        """Test DataLoader error handling for invalid configurations"""
        config = MockGPT2Config()
        config.training.num_workers = -1  # Invalid worker count
        model = GPT2ARC(config=config, num_classes=10)
        
        with patch('gpt2_arc.src.data.arc_dataset.ARCDataset') as mock_dataset:
            mock_dataset.return_value = "mock_dataset"
            with pytest.raises(ValueError):
                dataloader = model.test_dataloader()


if __name__ == "__main__":
    pytest.main([__file__])