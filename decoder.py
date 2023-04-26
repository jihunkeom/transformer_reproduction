import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from partials import *

class MaskedMHA(nn.Module):
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
        subsequent_mask = torch.triu(torch.ones((embedded.size(0), embedded.size(0))), diagonal=1).unsqueeze(0).unsqueeze(0)
        subsequent_mask = subsequent_mask.repeat_interleave(batch_size, 0)
        subsequent_mask = subsequent_mask.repeat_interleave(self.h, 1).type(torch.bool).to(embedded.device)
        # scores = scores.masked_fill_(subsequent_mask, -1e10)
        mask = mask.transpose(0, 1).unsqueeze(1).repeat_interleave(self.h, 1).unsqueeze(2).repeat_interleave(embedded.size(0), 2)
        
        scores.masked_fill_(subsequent_mask, -1e10)
        scores.masked_fill_(mask, -1e10)
        
        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)
        
        attn = torch.matmul(weights, v).permute(2, 0, 1, 3).contiguous().view(-1, batch_size, self.h*self.d_k)
        del q
        del k 
        del v
        return self.linear(attn)

class EncDecMHA(nn.Module):
    def __init__(self, model_dim, h, p_dropout):
        super().__init__()
        self.d_k = model_dim // h
        self.h = h
        self.projections = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(3)])
        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, enc_output, embedded, src_mask, tgt_mask):
        batch_size = embedded.size(1)
        q, k, v = [
            lin(x).view(-1, batch_size, self.h, self.d_k).permute(1, 2, 0, 3) for lin, x in zip(self.projections, (embedded, enc_output, enc_output))
            ]
        
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)
        src_mask = src_mask.transpose(0, 1).unsqueeze(1).repeat_interleave(self.h, 1).unsqueeze(2).repeat_interleave(embedded.size(0), 2)
        scores.masked_fill_(src_mask, -1e10)
        
        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)

        attn = torch.matmul(weights, v).permute(2, 0, 1, 3).contiguous().view(-1, batch_size, self.h*self.d_k)
        
        del q
        del k 
        del v
        return self.linear(attn)


class DecoderBlock(nn.Module):
    def __init__(self, model_dim, h, ffn_dim, p_dropout):
        super().__init__()

        self.MMHA = MaskedMHA(model_dim, h, p_dropout)
        self.EDMHA = EncDecMHA(model_dim, h, p_dropout)
        self.FFN = FeedForwardNetwork(model_dim, ffn_dim, p_dropout)
        self.sublayer_connections = nn.ModuleList([SublayerConnection(model_dim, p_dropout) for _ in range(3)])

    def forward(self, enc_output, embedded, src_mask, tgt_mask):
        self_attn = self.sublayer_connections[0](embedded, lambda embedded: self.MMHA(embedded, tgt_mask))
        enc_dec_attn = self.sublayer_connections[1](embedded, lambda embedded: self.EDMHA(enc_output, embedded, src_mask, tgt_mask))
        output = self.sublayer_connections[2](enc_dec_attn, self.FFN)

        return output

class Decoder(nn.Module):
    def __init__(self, model_dim, h, ffn_dim, p_dropout, n):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(model_dim=model_dim, h=h, ffn_dim=ffn_dim, p_dropout=p_dropout) for _ in range(n)])

    def forward(self, enc_output, tgt, src_mask, tgt_mask):
        for block in self.decoder_blocks:
            x = block(enc_output, tgt, src_mask, tgt_mask)

        return x
