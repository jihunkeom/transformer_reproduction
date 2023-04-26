import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from partials import *

class SelfMHA(nn.Module):
    def __init__(self, model_dim, h, p_dropout):
        super().__init__()
        self.d_k = model_dim // h
        self.h = h
        self.projections = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(3)])
        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, embedded, mask):
        batch_size = embedded.size(1)
        q, k, v = [
            lin(x).view(-1, batch_size, self.h, self.d_k).permute(1, 2, 0, 3) for lin, x in zip(self.projections, (embedded, embedded, embedded))
            ]
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)
        mask = mask.transpose(0, 1).unsqueeze(1).repeat_interleave(self.h, 1).unsqueeze(2).repeat_interleave(embedded.size(0), 2)
        # scores = scores.masked_fill_(mask, -1e10)
        scores.masked_fill_(mask, -1e10)
        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)
        
        attn = torch.matmul(weights, v).permute(2, 0, 1, 3).contiguous().view(-1, batch_size, self.h*self.d_k)
        del q
        del k 
        del v
        return self.linear(attn)

class EncoderBlock(nn.Module):
    def __init__(self, model_dim, h, ffn_dim, p_dropout):
        super().__init__()

        self.MHA = SelfMHA(model_dim=model_dim, h=h, p_dropout=p_dropout)
        self.FFN = FeedForwardNetwork(model_dim, ffn_dim, p_dropout)
        self.sublayer_connections = nn.ModuleList([SublayerConnection(model_dim, p_dropout) for _ in range(2)])

    def forward(self, x, mask):
        attn = self.sublayer_connections[0](x, lambda x: self.MHA(x, mask))
        output = self.sublayer_connections[1](attn, self.FFN)
        return output
        

class Encoder(nn.Module):
    def __init__(self, model_dim, h, ffn_dim, p_dropout, n):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(model_dim, h, ffn_dim, p_dropout) for _ in range(n)])

    def forward(self, x, mask):
        for block in self.encoder_blocks:
            x = block(x, mask)

        return x
