"""
@Author: danyzhou
@Updated: 10/23/25 2:58 PM

Advantest Confidential - All Rights Reserved
"""
import torch
from torch import nn
import math
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super().__init__(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={dim})")

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))

        pe[:, 0::2] = torch.sin(position/div_term)
        pe[:, 1::2] = torch.cos(position/div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, emb):
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:emb.size(0)]
        emb = self.drop_out(emb)
        return emb

def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = (query @ value.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return self_attn @ value, self_attn


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def forward(self, head, d_model, query, key, value, dropout=0.1, mask=None):
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model

        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.atten = None

        n_batch = query.size(0)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        key = self.linear_query(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.linear_query(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)

        x, self_attn = self_attention(query, key, value, dropout, mask)

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)
