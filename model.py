import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from preprocess import *
from encoder import Encoder
from decoder import Decoder
from partials import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, device, vocab_size, model_dim=512, PAD_IDX=0, p_dropout=0.1, n=6, h=8):
        super().__init__()

        self.model_dim = model_dim
        self.device = device
        self.PAD_IDX = PAD_IDX

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_IDX)
        self.pos_enc = nn.ModuleList([PositionalEncoding(model_dim, self.device) for _ in range(2)])
        self.encoder = Encoder(model_dim=model_dim, h=h, ffn_dim=model_dim*4, p_dropout=p_dropout, n=n)
        self.decoder = Decoder(model_dim=model_dim, h=h, ffn_dim=model_dim*4, p_dropout=p_dropout, n=n)
        self.dense = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.ModuleList([nn.Dropout(p=p_dropout) for _ in range(2)])
        self.dense.weight = self.embedding.weight
        

    def forward(self, src, src_mask, tgt, tgt_mask):
        src_embedded = torch.mul(self.embedding(src), math.sqrt(self.model_dim))
        src_embedded = self.dropout[0](self.pos_enc[0](src_embedded))
        tgt_embedded = torch.mul(self.embedding(tgt), math.sqrt(self.model_dim))
        tgt_embedded = self.dropout[1](self.pos_enc[1](tgt_embedded))

        enc_output = self.encoder(src_embedded, src_mask)
        output = self.decoder(enc_output, tgt_embedded, src_mask, tgt_mask)
        pred = self.dense(output)
        
        return pred

    def translate(self, src, src_mask, tgt_input, EOS_IDX=3, max_len=20):
        src_embedded = torch.mul(self.embedding(src), math.sqrt(self.model_dim))
        src_embedded = self.dropout[0](self.pos_enc[0](src_embedded))
        enc_output = self.encoder(src_embedded, src_mask)

        for t in range(max_len):
            tgt_mask = self._inference_mask(tgt_input)
            tgt_embedded = torch.mul(self.embedding(tgt_input), math.sqrt(self.model_dim))
            tgt_embedded = self.dropout[1](self.pos_enc[1](tgt_embedded))
            out = self.decoder(enc_output, tgt_embedded, src_mask, tgt_mask)
            prob = self.dense(out[-1, :])
            
            next_word = prob.argmax(-1)
            # print(next_word)
            next_word = next_word.data[0]
            tgt_input = torch.cat([tgt_input, torch.empty(1, 1, device=self.device).fill_(next_word)], dim=0).to(device=self.device, dtype=torch.long)
            if next_word.item() == EOS_IDX:
                break
        
        return [i.item() for i in tgt_input]

    def _inference_mask(self, tgt):
        return (tgt == self.PAD_IDX).to(self.device)

class TransformerEmb(nn.Module):
    def __init__(self, device, vocab_size, model_dim=512, PAD_IDX=0, p_dropout=0.1, n=6, h=8):
        super().__init__()

        self.model_dim = model_dim
        self.device = device
        self.PAD_IDX = PAD_IDX

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(300, model_dim)
        self.encoder = Encoder(model_dim=model_dim, h=h, ffn_dim=model_dim*4, p_dropout=p_dropout, n=n)
        self.decoder = Decoder(model_dim=model_dim, h=h, ffn_dim=model_dim*4, p_dropout=p_dropout, n=n)
        self.dense = nn.Linear(model_dim, vocab_size)
        self.dropout = nn.ModuleList([nn.Dropout(p=p_dropout) for _ in range(2)])
        self.dense.weight = self.embedding.weight
        

    def forward(self, src, src_mask, tgt, tgt_mask):        
        src_embedded = torch.mul(self.embedding(src), math.sqrt(self.model_dim))
        src_pos = torch.arange(0, src.size(0)).unsqueeze(1).repeat(1, src.size(1)).to(self.device)
        src_embedded = self.dropout[0](torch.add(src_embedded, self.pos_emb(src_pos)))

        tgt_embedded = torch.mul(self.embedding(tgt), math.sqrt(self.model_dim))
        tgt_pos = torch.arange(0, tgt.size(0)).unsqueeze(1).repeat(1, tgt.size(1)).to(self.device)
        tgt_embedded = self.dropout[1](torch.add(tgt_embedded, self.pos_emb(tgt_pos)))

        enc_output = self.encoder(src_embedded, src_mask)
        output = self.decoder(enc_output, tgt_embedded, src_mask, tgt_mask)
        pred = self.dense(output)
        
        return pred

    def translate(self, src, src_mask, tgt_input, EOS_IDX=3, max_len=20):
        src_embedded = torch.mul(self.embedding(src), math.sqrt(self.model_dim))
        src_pos = torch.arange(0, src.size(0)).unsqueeze(1).repeat(1, src.size(1)).to(self.device)
        src_embedded = self.dropout[0](torch.add(src_embedded, self.pos_emb(src_pos)))
        enc_output = self.encoder(src_embedded, src_mask)

        for t in range(max_len):
            tgt_mask = self._inference_mask(tgt_input)
            tgt_embedded = torch.mul(self.embedding(tgt_input), math.sqrt(self.model_dim))
            tgt_pos = torch.arange(0, tgt_input.size(0)).unsqueeze(1).repeat(1, tgt_input.size(1)).to(self.device)
            tgt_embedded = self.dropout[1](torch.add(tgt_embedded, self.pos_emb(tgt_pos)))
            out = self.decoder(enc_output, tgt_embedded, src_mask, tgt_mask)
            prob = self.dense(out[-1, :])
            
            next_word = prob.argmax(-1)
            # print(next_word)
            next_word = next_word.data[0]
            tgt_input = torch.cat([tgt_input, torch.empty(1, 1, device=self.device).fill_(next_word)], dim=0).to(device=self.device, dtype=torch.long)
            if next_word.item() == EOS_IDX:
                break
        
        return [i.item() for i in tgt_input]

    def _inference_mask(self, tgt):
        return (tgt == self.PAD_IDX).to(self.device)