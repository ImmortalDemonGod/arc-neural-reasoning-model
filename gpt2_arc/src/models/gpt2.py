# gpt2_arc/src/models/gpt2.py

import logging

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from typing import Dict, Optional, Tuple
import torch.nn.init as init
from bitnet import BitLinearNew
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
from gpt2_arc.src.data.arc_dataset import ARCDataset
from gpt2_arc.src.config import Config
from zeta.nn import MambaBlock
#from gpt2_arc.src.models.mamba_block_internal import MambaBlock


class Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.key = BitLinearNew(n_embd, n_embd)
        self.query = BitLinearNew(n_embd, n_embd)
        self.value = BitLinearNew(n_embd, n_embd)
        self.proj = BitLinearNew(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)  # Add this line
        logger.debug(f"Initialized Attention with n_embd={n_embd}, n_head={n_head}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        logger.debug(f"Model input shape: {x.shape}")
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
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            BitLinearNew(n_embd, 4 * n_embd), nn.ReLU(), nn.Dropout(dropout), BitLinearNew(4 * n_embd, n_embd)
        )
        logger.debug(f"Initialized FeedForward with n_embd={n_embd}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not torch._dynamo.is_compiling():
            logger.debug(f"FeedForward input shape: {x.shape}")
        output = self.net(x)
        if not torch._dynamo.is_compiling():
            logger.debug(f"FeedForward output shape: {output.shape}")
        return output


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.attention = Attention(n_embd, n_head, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        logger.debug(
            f"Initialized TransformerBlock with n_embd={n_embd}, n_head={n_head}"
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        logger.debug(f"Final output shape: {x.shape}")
        return x


class MambaLayer(nn.Module):
    def __init__(self, n_embd: int, d_state: int, d_conv: int, dropout: float, depth: int, expand: int):
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not torch._dynamo.is_compiling():
            logger.debug(f"MambaLayer input shape: {x.shape}")
        x_norm = self.layer_norm(x)
        x_mamba = self.mamba_block(x_norm)
        x_mamba = self.dropout(x_mamba)
        output = x + x_mamba
        if not torch._dynamo.is_compiling():
            logger.debug(f"MambaLayer output shape: {output.shape}")
        return output



class GPT2ARC(pl.LightningModule):
    def __init__(self, config: Config, num_classes: int, symbol_freq: Optional[Dict[int, float]] = None, pad_symbol_idx: int = 10):
        super().__init__()
        self.example_input_array = torch.zeros(1, 1, 6, 6)  # Adjust dimensions as needed
        self.config = config
        self.symbol_freq = symbol_freq if symbol_freq is not None else {}
        self.pad_symbol_idx = pad_symbol_idx  # Add this line
        self.include_pad_in_loss = self.config.training.include_pad_in_loss  # Reintroduced
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.config.model.n_embd,  # Accessing the 'model' attribute within Config
            kernel_size=3,
            padding=1
        )
        # Initialize blocks with interleaved TransformerBlocks and MambaLayer(s)
        self.blocks = nn.ModuleList()
        mamba_ratio = self.config.model.mamba_ratio
        total_layers = self.config.model.n_layer  # Total number of layers
        num_transformer_layers = int(total_layers / (1 + mamba_ratio)) if mamba_ratio > 0 else total_layers
        num_mamba_layers = total_layers - num_transformer_layers

        logger.debug(f"Total layers: {total_layers}")
        logger.debug(f"mamba_ratio: {mamba_ratio}")
        logger.debug(f"Number of TransformerLayers: {num_transformer_layers}")
        logger.debug(f"Number of MambaLayers: {num_mamba_layers}")

        # Calculate positions to insert MambaLayers
        mamba_layer_positions = []
        if num_mamba_layers > 0:
            step = num_transformer_layers / num_mamba_layers
            mamba_layer_positions = [int((i + 1) * step) for i in range(num_mamba_layers)]

        current_mamba_index = 0
        for layer_idx in range(1, total_layers + 1):
            # Add a TransformerBlock
            self.blocks.append(
                TransformerBlock(
                    self.config.model.n_embd,
                    self.config.model.n_head,
                    self.config.model.dropout
                )
            )
            logger.debug(f"Layer {layer_idx}: Added TransformerBlock")

            # Check if a MambaLayer should be added after this TransformerBlock
            if current_mamba_index < len(mamba_layer_positions) and layer_idx == mamba_layer_positions[current_mamba_index]:
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
                logger.debug(f"Layer {layer_idx + 1}: Added MambaLayer after TransformerBlock {layer_idx}")
                current_mamba_index += 1
        self.ln_f = nn.LayerNorm(self.config.model.n_embd)
        assert isinstance(self.config.model.n_embd, int), "model.n_embd must be an integer"
        assert isinstance(num_classes, int), "num_classes must be an integer"
        self.fc_out = BitLinearNew(int(self.config.model.n_embd), num_classes)  # Use num_classes directly

        # Initialize loss function with class weights if needed
        if self.symbol_freq:
            # Create a tensor of class weights based on symbol frequencies
            class_weights = torch.tensor([self.symbol_freq.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32)
            class_weights = class_weights.to(self.device)  # Ensure weights are on the correct device
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.pad_symbol_idx)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_symbol_idx)
        

        # Initialize weights
        self.apply(self._init_weights)

    def _calculate_accuracies(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method to calculate accuracies with and without padding
        
        Args:
            preds: Model predictions
            targets: Target labels
            
        Returns:
            Tuple containing (accuracy with padding, accuracy without padding)
        """
        # Accuracy including padding
        correct_with_pad = (preds == targets).float()
        total_with_pad = torch.numel(targets)
        acc_with_pad = correct_with_pad.sum() / total_with_pad if total_with_pad > 0 else torch.tensor(0.0)
        
        # Accuracy excluding padding
        mask = targets != self.pad_symbol_idx
        correct_without_pad = (preds == targets).float() * mask
        total_without_pad = mask.sum()
        acc_without_pad = correct_without_pad.sum() / total_without_pad if total_without_pad > 0 else torch.tensor(0.0)
        
        return acc_with_pad, acc_without_pad
    
    def _init_weights(self, module: nn.Module) -> None:
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


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.config.training.num_classes), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        acc_with_pad, acc_without_pad = self._calculate_accuracies(preds, targets)
        
        # Log all metrics
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
        logger.debug("Entering GPT2ARC.test_dataloader")
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=False,  # Typically, shuffling is not needed for test data
            pin_memory=self.config.training.use_gpu  # Optimize memory usage based on GPU availability
        )
        return dataloader


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.config.training.num_classes), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        acc_with_pad, acc_without_pad = self._calculate_accuracies(preds, targets)
        
        # Log all metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_with_pad', acc_with_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_without_pad', acc_without_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.view(-1, self.config.training.num_classes), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        acc_with_pad, acc_without_pad = self._calculate_accuracies(preds, targets)
        
        # Log all metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_with_pad', acc_with_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_without_pad', acc_without_pad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logger.debug(f"GPT2ARC forward - Input shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        
        if input_ids.dim() == 4:
            x = input_ids
        else:
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            height = width = int(seq_length ** 0.5)
            x = input_ids.view(batch_size, 1, height, width)
        
        x = self.conv1(x)
        logger.debug(f"After conv1 shape: {x.shape}")
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        logger.debug(f"Reshaped for transformer blocks: {x.shape}")

        for i, block in enumerate(self.blocks):
            if isinstance(block, TransformerBlock):
                x = block(x, attention_mask)  # Pass the mask to MambaLayer
                logger.debug(f"After TransformerBlock {i + 1}: shape {x.shape}")
            else:
                x = block(x, attention_mask)  # Pass the mask to MambaLayer
                logger.debug(f"After MambaLayer {i + 1}: shape {x.shape}")
        
        x = self.ln_f(x)
        x = self.fc_out(x)
        logger.debug(f"GPT2ARC forward - Final output shape: {x.shape}, dtype: {x.dtype}")
        return x
