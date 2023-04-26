import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def init_weights(model):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model

def create_mask(sentence, pad_token, device):
    mask = (sentence == pad_token).to(device)
    return mask

def create_tgt_mask(sentence, pad_token, device):
    pad_mask = (sentence == pad_token).to(device)
    subsequent_mask = torch.triu(sentence, diagonal=1) == score
    mask = pad_mask or subsequent_mask
    return mask

def make_variables(source, target, device, pipeline, PAD_IDX=0, SOS_IDX=2, EOS_IDX=3):
    batch_size = len(source)
    src_text_list, tgt_text_list = [], []

    for i in range(batch_size):
        src_text = torch.tensor(pipeline(source[i]), dtype=torch.int64)
        tgt_text = torch.tensor([SOS_IDX] + pipeline(target[i]) + [EOS_IDX], dtype=torch.int64)

        src_text_list.append(src_text)
        tgt_text_list.append(tgt_text)

    src = pad_sequence(src_text_list, batch_first=False, padding_value=PAD_IDX).to(device)
    tgt = pad_sequence(tgt_text_list, batch_first=False, padding_value=PAD_IDX).to(device)

    return src, tgt

def rate(step, model_size, warmup):
    if step == 0:
        step = 1
    return (model_size**(-0.5)) * min(step**(-0.5), step*warmup**(-1.5))

def convert_back(sent):
    tmp = "".join([w for w in sent])
    output = tmp.split("▁")
    if output[0] == "":
        return output[1:]
    return output