import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device

        pe_table = torch.zeros(300, 1, hidden_size).to(self.device)
        pos = torch.arange(0, 300, dtype=torch.float).unsqueeze(1)

        denom = torch.pow(10000, torch.arange(0, hidden_size, 2)/hidden_size)

        pe_table[:, 0, 0::2] = torch.sin(pos/denom)
        pe_table[:, 0, 1::2] = torch.cos(pos/denom)
        self.register_buffer("pe", pe_table)
        
    def forward(self, embedded):
        output = embedded + self.pe[:embedded.size(0)].requires_grad_(False)
        return output


class SublayerConnection(nn.Module):
    def __init__(self, dim, p_dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x, layer):
        # return self.layer_norm(torch.add(x, self.dropout(layer(x))))
        return x + self.dropout(layer(self.layer_norm(x)))

class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim, hidden_dim, p_dropout):
        super().__init__()
        self.dense1 = nn.Linear(model_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        output = self.dense2(self.dropout(F.relu(self.dense1(x))))
        return output