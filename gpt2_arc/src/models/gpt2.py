# gpt2_arc/src/models/gpt2.py

import logging

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from typing import Dict, Optional
import torch.nn.init as init
from bitnet import BitLinearNew

#from zeta.nn import MambaBlock
from gpt2_arc.src.models.mamba_block_internal import MambaBlock

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.key = BitLinearNew(n_embd, n_embd)
        self.query = BitLinearNew(n_embd, n_embd)
        self.value = BitLinearNew(n_embd, n_embd)
        self.proj = BitLinearNew(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)  # Add this line
        logger.debug(f"Initialized Attention with n_embd={n_embd}, n_head={n_head}")

    def forward(self, x, mask=None):
        B, T, C = x.size()
        if not torch._dynamo.is_compiling():
            logger.debug(f"Attention input shape: {x.shape}")
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)))
        if mask is not None:
            att = att.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Apply dropout to attention probabilities
        att = self.dropout(att)  # Add this line
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.proj(y)
        if not torch._dynamo.is_compiling():
            logger.debug(f"Attention output shape: {output.shape}")
        return output


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            BitLinearNew(n_embd, 4 * n_embd), nn.ReLU(), nn.Dropout(dropout), BitLinearNew(4 * n_embd, n_embd)
        )
        logger.debug(f"Initialized FeedForward with n_embd={n_embd}")

    def forward(self, x):
        if not torch._dynamo.is_compiling():
            logger.debug(f"FeedForward input shape: {x.shape}")
        output = self.net(x)
        if not torch._dynamo.is_compiling():
            logger.debug(f"FeedForward output shape: {output.shape}")
        return output


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.attention = Attention(n_embd, n_head, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        logger.debug(
            f"Initialized TransformerBlock with n_embd={n_embd}, n_head={n_head}"
        )

    def forward(self, x, mask=None):
        if not torch._dynamo.is_compiling():
            logger.debug(f"TransformerBlock input shape: {x.shape}")
        # Attention sublayer with residual dropout
        attn_output = self.attention(self.ln1(x), mask)
        attn_output = self.dropout(attn_output)  # Apply dropout
        x = x + attn_output

        # Feed-forward sublayer with residual dropout
        ff_output = self.feed_forward(self.ln2(x))
        ff_output = self.dropout(ff_output)  # Apply dropout
        x = x + ff_output
        if not torch._dynamo.is_compiling():
            logger.debug(f"TransformerBlock output shape: {x.shape}")
        return x


class MambaLayer(nn.Module):
    def __init__(self, n_embd, d_state, d_conv, dropout, depth, expand):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mamba_block = MambaBlock(
            dim=n_embd,
            depth=depth,           # Use depth from config
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,         # Use expand from config
            conv_bias=True,        # Default value
            bias=False             # Default value
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        logger.debug(
            f"Initialized MambaLayer with n_embd={n_embd}, d_state={d_state}, d_conv={d_conv}, dropout={dropout}"
        )

    def forward(self, x):
        if not torch._dynamo.is_compiling():
            logger.debug(f"MambaLayer input shape: {x.shape}")
        x_norm = self.layer_norm(x)
        x_mamba = self.mamba_block(x_norm)
        x_mamba = self.dropout(x_mamba)
        output = x + x_mamba
        if not torch._dynamo.is_compiling():
            logger.debug(f"MambaLayer output shape: {output.shape}")
        return output

from torch.utils.data import DataLoader
from src.data.arc_dataset import ARCDataset
from gpt2_arc.src.config import Config

class GPT2ARC(pl.LightningModule):
    def __init__(self, config: Config, num_classes: int, symbol_freq: Optional[Dict[int, float]] = None):
        # Define an example input array for model summary
        self.example_input_array = torch.zeros(1, 1, 6, 6)  # Adjust dimensions as needed
        super().__init__()
        self.config = config
        self.symbol_freq = symbol_freq if symbol_freq is not None else {}
        logger.debug(f"Symbol frequencies loaded: {self.symbol_freq}")
        self.pad_symbol_idx = self.config.training.pad_symbol_idx
        self.include_pad_in_loss = self.config.training.include_pad_in_loss
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.config.model.n_embd,  # Accessing the 'model' attribute within Config
            kernel_size=3,
            padding=1
        ).to(torch.float32)
        # Initialize blocks with interleaved TransformerBlocks and MambaLayer(s)
        self.blocks = nn.ModuleList()
        num_transformer_blocks = self.config.model.n_layer
        total_mamba_layers = int(num_transformer_blocks * self.config.model.mamba_ratio)
        
        logger.debug(f"Total TransformerBlocks: {num_transformer_blocks}")
        logger.debug(f"Total MambaLayers to add: {total_mamba_layers}")

        # Distribute MambaLayers across TransformerBlocks
        mamba_layer_positions = []
        if total_mamba_layers > 0:
            step = num_transformer_blocks / total_mamba_layers
            mamba_layer_positions = [int(i * step) for i in range(total_mamba_layers)]

        current_mamba_index = 0
        for layer_idx in range(num_transformer_blocks):
            # Add a TransformerBlock
            self.blocks.append(TransformerBlock(self.config.model.n_embd, self.config.model.n_head, self.config.model.dropout))
            logger.debug(f"Layer {len(self.blocks)}: Added TransformerBlock")

            # Check if we should add a MambaLayer after this TransformerBlock
            if current_mamba_index < len(mamba_layer_positions) and layer_idx == mamba_layer_positions[current_mamba_index]:
                # Add a MambaLayer
                self.blocks.append(
                    MambaLayer(
                        n_embd=self.config.model.n_embd,
                        d_state=self.config.model.d_state,
                        d_conv=self.config.model.d_conv,
                        dropout=self.config.model.dropout,
                        depth=self.config.model.mamba_depth,
                        expand=self.config.model.mamba_expand
                    )
                )
                logger.debug(f"Layer {len(self.blocks)}: Added MambaLayer after TransformerBlock {layer_idx + 1}")
                current_mamba_index += 1
        self.ln_f = nn.LayerNorm(self.config.model.n_embd)
        assert isinstance(self.config.model.n_embd, int), "model.n_embd must be an integer"
        assert isinstance(num_classes, int), "num_classes must be an integer"
        self.fc_out = BitLinearNew(int(self.config.model.n_embd), num_classes)  # Use num_classes directly

        # Initialize loss function with class weights if needed
        if self.config.training.balance_symbols and self.config.training.balancing_method == "weighting":
            if not self.symbol_freq:
                raise ValueError("symbol_freq must be provided when balance_symbols is True and balancing_method is 'weighting'")
            # Prevent division by zero by adding a small epsilon to each frequency
            epsilon = 1e-6
            symbol_freq_values = torch.tensor([
                freq if freq > 0 else epsilon for freq in self.symbol_freq.values()
            ], dtype=torch.float)
            
            # Compute class weights as inverse of symbol frequencies
            class_weights = 1.0 / symbol_freq_values
            
            
            # Ensure that the length of class_weights matches num_classes
            assert class_weights.size(0) == self.config.training.num_classes, (
                f"class_weights length ({class_weights.size(0)}) does not match num_classes ({self.config.training.num_classes})"
            )
            
            if self.config.training.include_pad_in_loss:
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                logger.debug("Including padding class in loss calculation with class weights.")
            else:
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.config.training.pad_symbol_idx)
                logger.debug("Excluding padding class from loss calculation with class weights.")
            logger.debug(f"Class Weights: {class_weights}")  # Added line
        else:
            if self.config.training.include_pad_in_loss:
                # Include padding in loss without class weights
                self.loss_fn = nn.CrossEntropyLoss()
                logger.debug("Including padding class in loss calculation without class weights.")
            else:
                # Exclude padding class from loss without class weights
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.training.pad_symbol_idx)
                logger.debug("Excluding padding class from loss calculation without class weights.")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # Calculate fan_in for Conv2d
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            std = 1.0 / fan_in**0.5
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, BitLinearNew):
            fan_in = module.in_features
            std = 1.0 / fan_in**0.5
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        # No initialization for nn.LayerNorm, using default

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.config.training.num_classes), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        
        # Accuracy including padding
        correct_with_pad = (preds == targets).float()
        total_with_pad = torch.numel(targets)
        acc_with_pad = correct_with_pad.sum() / total_with_pad if total_with_pad > 0 else torch.tensor(0.0)
        
        # Accuracy excluding padding
        mask = targets != self.pad_symbol_idx
        correct_without_pad = (preds == targets).float() * mask
        total_without_pad = mask.sum()
        acc_without_pad = correct_without_pad.sum() / total_without_pad if total_without_pad > 0 else torch.tensor(0.0)
        
        # Log both accuracies: with padding and without padding
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_with_pad', acc_with_pad, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_without_pad', acc_without_pad, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_dataloader(self):
        # Initialize the test dataset
        test_dataset = ARCDataset(
            data_source=self.config.test_data_path,  # Ensure this path is correctly set in your configuration
            is_test=True,
            num_symbols=self.config.training.num_symbols,
            pad_symbol_idx=self.config.training.pad_symbol_idx,
            symbol_freq=self.config.training.symbol_freq,
            debug=self.config.debug
        )
        
        # Create and return the DataLoader
        return DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=False,  # Typically, shuffling is not needed for test data
            pin_memory=self.config.training.use_gpu  # Optimize memory usage based on GPU availability
        )

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.config.training.num_classes), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        
        # Accuracy including padding
        correct_with_pad = (preds == targets).float()
        total_with_pad = torch.numel(targets)
        acc_with_pad = correct_with_pad.sum() / total_with_pad if total_with_pad > 0 else torch.tensor(0.0)
        
        # Accuracy excluding padding
        mask = targets != self.pad_symbol_idx
        correct_without_pad = (preds == targets).float() * mask
        total_without_pad = mask.sum()
        acc_without_pad = correct_without_pad.sum() / total_without_pad if total_without_pad > 0 else torch.tensor(0.0)
        
        # Log both accuracies
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_with_pad', acc_with_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_without_pad', acc_without_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.config.training.num_classes), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        
        # Accuracy including padding
        correct_with_pad = (preds == targets).float()
        total_with_pad = torch.numel(targets)
        acc_with_pad = correct_with_pad.sum() / total_with_pad if total_with_pad > 0 else torch.tensor(0.0)
        
        # Accuracy excluding padding
        mask = targets != self.pad_symbol_idx
        correct_without_pad = (preds == targets).float() * mask
        total_without_pad = mask.sum()
        acc_without_pad = correct_without_pad.sum() / total_without_pad if total_without_pad > 0 else torch.tensor(0.0)
        
        # Log both accuracies
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_with_pad', acc_with_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_without_pad', acc_without_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def forward(self, input_ids, attention_mask=None):
        if not torch._dynamo.is_compiling():
            logger.debug(f"GPT2ARC input shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        
        # Check if input_ids is already in the correct shape
        if input_ids.dim() == 4:
            x = input_ids.float()
        else:
            # Reshape input_ids to [batch_size, 1, height, width]
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            height = width = int(seq_length ** 0.5)
            x = input_ids.float().view(batch_size, 1, height, width)
        
        x = self.conv1(x)
        logger.debug(f"After conv1 shape: {x.shape}")
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # Flatten spatial dimensions
        x = x.permute(0, 2, 1)  # Rearrange to (batch_size, sequence_length, channels)
        logger.debug(f"Reshaped for transformer blocks: {x.shape}")

        for i, block in enumerate(self.blocks):
            # Check if the block is a TransformerBlock or MambaLayer
            if isinstance(block, TransformerBlock):
                x = block(x, attention_mask)
                logger.debug(f"After TransformerBlock {i + 1}: shape {x.shape}")
            else:
                x = block(x)
                logger.debug(f"After MambaLayer {i + 1}: shape {x.shape}")
        
        x = self.ln_f(x)
        x = self.fc_out(x)  # Apply final linear layer
        return x
