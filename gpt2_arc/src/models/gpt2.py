# gpt2_arc/src/models/gpt2.py

import torch
from torch import nn
import torch.nn.functional as F
import logging

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
        logger.debug(f"Attention input shape: {x.shape}")
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        if mask is not None:
            att = att.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.proj(y)
        logger.debug(f"Attention output shape: {output.shape}")
        return output

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        logger.debug(f"Initialized FeedForward with n_embd={n_embd}")

    def forward(self, x):
        logger.debug(f"FeedForward input shape: {x.shape}")
        output = self.net(x)
        logger.debug(f"FeedForward output shape: {output.shape}")
        return output

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = Attention(n_embd, n_head)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        logger.debug(f"Initialized TransformerBlock with n_embd={n_embd}, n_head={n_head}")

    def forward(self, x, mask=None):
        logger.debug(f"TransformerBlock input shape: {x.shape}")
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        logger.debug(f"TransformerBlock output shape: {x.shape}")
        return x

from src.config import ModelConfig

class GPT2ARC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(50257, config.n_embd)
        self.position_embedding = nn.Embedding(1024, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.config = {
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout
        }
        logger.debug(f"Initialized GPT2ARC with vocab_size={vocab_size}, n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}")

    def forward(self, input_ids, attention_mask=None):
        logger.debug(f"GPT2ARC input shape: {input_ids.shape}")
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        for i, block in enumerate(self.blocks):
            logger.debug(f"Processing block {i}")
            x = block(x, attention_mask)
        
        x = self.ln_f(x)
        logger.debug(f"GPT2ARC output shape: {x.shape}")
        return x
