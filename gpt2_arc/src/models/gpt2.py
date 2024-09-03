# gpt2_arc/src/models/gpt2.py

from torch import nn
from transformers import GPT2Config, GPT2Model


class GPT2ARC(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(GPT2ARC, self).__init__()
        self.config = GPT2Config.from_pretrained(model_name)
        self.gpt2 = GPT2Model.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
