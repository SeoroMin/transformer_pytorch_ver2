import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, device=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.device = device
        
    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        maximum_value = torch.max(output)#torch.FloatTensor([math.sqrt(torch.max(value))]).to(args.device)
        if maximum_value > 1.0:
            maximum_value = torch.FloatTensor([math.sqrt(torch.max(output))]).to(self.device)
            output.divide_(maximum_value)
        
        return output, attn
