# gpt2_arc/src/models/gpt2.py

import logging

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
import torch.nn.init as init
from zeta.nn import MambaBlock

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
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
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.proj(y)
        if not torch._dynamo.is_compiling():
            logger.debug(f"Attention output shape: {output.shape}")
        return output


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = Attention(n_embd, n_head)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        logger.debug(
            f"Initialized TransformerBlock with n_embd={n_embd}, n_head={n_head}"
        )

    def forward(self, x, mask=None):
        if not torch._dynamo.is_compiling():
            logger.debug(f"TransformerBlock input shape: {x.shape}")
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        if not torch._dynamo.is_compiling():
            logger.debug(f"TransformerBlock output shape: {x.shape}")
        return x


class MambaLayer(nn.Module):
    def __init__(self, n_embd, d_state, d_conv, dropout):
        super().__init__()
        self.mamba_block = MambaBlock(
            dim=n_embd,
            depth=1,               # You can adjust the depth as needed
            d_state=d_state,
            d_conv=d_conv,
            expand=2,              # Default value
            dt_rank="auto",        # Default value
            conv_bias=True,        # Default value
            bias=False,            # Default value
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        logger.debug(
            f"Initialized MambaLayer with n_embd={n_embd}, d_state={d_state}, d_conv={d_conv}, dropout={dropout}"
        )

    def forward(self, x):
        if not torch._dynamo.is_compiling():
            logger.debug(f"MambaLayer input shape: {x.shape}")
        x_norm = self.layer_norm(x)
        output = x + self.mamba_block(x_norm)
        if not torch._dynamo.is_compiling():
            logger.debug(f"MambaLayer output shape: {output.shape}")
        return output


from gpt2_arc.src.config import ModelConfig


class GPT2ARC(pl.LightningModule):
    def __init__(self, config: ModelConfig, num_classes: int):
        # Define an example input array for model summary
        self.example_input_array = torch.zeros(1, 1, 6, 6)  # Adjust dimensions as needed
        super().__init__()
        self.config = config
        # Replace token embedding with a convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.config.n_embd, kernel_size=3, padding=1).to(torch.float32)
        # Initialize blocks with interleaved TransformerBlocks and MambaLayer(s)
        self.blocks = nn.ModuleList()
        for layer_idx in range(self.config.n_layer):
            # Add a TransformerBlock
            self.blocks.append(TransformerBlock(self.config.n_embd, self.config.n_head))
            
            # Add MambaLayer(s) according to mamba_ratio
            for _ in range(self.config.mamba_ratio):
                self.blocks.append(
                    MambaLayer(
                        n_embd=self.config.n_embd,
                        d_state=self.config.d_state,
                        d_conv=self.config.d_conv,
                        dropout=self.config.dropout
                    )
                )
            logger.debug(f"Layer {layer_idx + 1}: Added 1 TransformerBlock and {self.config.mamba_ratio} MambaLayer(s)")
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.fc_out = nn.Linear(self.config.n_embd, num_classes)  # Add final linear layer
        
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
        elif isinstance(module, nn.Linear):
            fan_in = module.in_features
            std = 1.0 / fan_in**0.5
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        # No initialization for nn.LayerNorm, using default

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
