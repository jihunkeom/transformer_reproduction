import time, math, copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer

from preprocess import *
from utils import *
from encoder import *
from decoder import *
from model import *
from evaluation import translate

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:3")

    HIDDEN_SIZE = 512
    EPOCHS = 40
    h = 8
    dropout = 0.1
    PATH = "ende_base_dropout.pt"
    # PATH = "ende_base_posemb.pt"

    
    special_symbols = ["<pad>"]
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

    all_vocab = torch.load("en_de_bpe_vocab.pth")
    # all_vocab = torch.load("en_fr_bpe_vocab.pth")
    vocab_size = len(all_vocab)
    en_de_bpe = load_sp_model("en_de_bpe.model")
    tokenizer = sentencepiece_tokenizer(en_de_bpe)
    # en_fr_bpe = load_sp_model("en_fr_bpe.model")
    # tokenizer = sentencepiece_tokenizer(en_fr_bpe)

    pipeline = lambda x: all_vocab(list(tokenizer([normalizeString(x)]))[0])

    dataloader = torch.load("trainloader_ende_32.pkl")
    # dataloader = torch.load("trainloader_enfr_32.pkl")

    model = Transformer(device=device, vocab_size=vocab_size, model_dim=HIDDEN_SIZE, PAD_IDX=0, p_dropout=dropout, n=6, h=h).to(device)
    # model = TransformerEmb(device=device, vocab_size=vocab_size, model_dim=HIDDEN_SIZE, PAD_IDX=0, p_dropout=dropout, n=6, h=h).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer = optimizer,
        lr_lambda = lambda step: rate(step, model_size=HIDDEN_SIZE, warmup=4000),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    try:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]+1
        print_loss = checkpoint["loss"]
    except:
        model = init_weights(model)
        epoch = 0
        print_loss = 0
        
    print(f'Resume training at {epoch}!')

    start_time = time.time()
    
    #총 100k x 12 step만큼 학습하자
    for e in range(epoch, EPOCHS+1):
        for i, pairs in enumerate(dataloader):
            src, tgt = make_variables(pairs[0], pairs[1], device, pipeline, PAD_IDX, SOS_IDX, EOS_IDX)
            src_mask = create_mask(src, PAD_IDX, device)
            tgt_mask = create_mask(tgt[:-1], PAD_IDX, device)
            pred = model(src, src_mask, tgt[:-1], tgt_mask)
            loss = criterion(pred.reshape(-1, vocab_size), tgt[1:].reshape(-1))
            print_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if i==0:
                print("Current Epoch: " + str(e) + " / " + str(EPOCHS), end=", ")
                print(str(i)+ " / " + str(len(dataloader)), end=", ")
                print("Time Elapsed: " + str(int(time.time() - start_time)) + " sec")
                print_loss = 0

            elif (i % 5000) == 0:
                print_loss /= 5000
                print_loss = round(print_loss, 5)
                print("Current Epoch: " + str(e) + " / " + str(EPOCHS), end=", ")
                print(str(i)+ " / " + str(len(dataloader)), end=", ")
                print("Time Elapsed: " + str(int(time.time() - start_time)) + " sec", end=", ")
                print("Loss: " + str(print_loss))
                print("Perplexity: " + str(math.exp(print_loss)))
                print_loss = 0

        torch.save({
            "epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "loss": print_loss
        }, PATH)