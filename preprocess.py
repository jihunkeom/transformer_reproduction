import pickle
import string
import re
import unicodedata
import sys

from collections import OrderedDict
import numpy
from tqdm.auto import tqdm

import torch
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"

    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z ]+", r" ", s)
    
    return s.strip()

def read_data(src, tgt):
    src_corpus, tgt_corpus = [], []

    with open("/home/user22/RNN/data/en_de/train." + str(src) + ".txt", "r") as f:
        for line in f:
            text = normalizeString(line)
            src_corpus.append(text)

    with open("/home/user22/RNN/data/en_de/train." + str(tgt) + ".txt", "r") as f:
        for line in f:
            text = normalizeString(line)
            tgt_corpus.append(text)

    pairs = []
    for i in range(len(src_corpus)):
            if (len(src_corpus[i].split()) > 0) and (len(tgt_corpus[i].split()) > 0):
                if (len(src_corpus[i].split()) < 100) and (len(tgt_corpus[i].split()) < 100):
                    pairs.append([src_corpus[i], tgt_corpus[i]])

    return pairs

def read_french(src, tgt):
    src, tgt = [], []

    with open(f"./en-fr/europarl-v7.fr-en.{src}", "r") as f:
        for line in f:
            text = normalizeString(line)
            src.append(text)
    with open(f"./en-fr/europarl-v7.fr-en.{tgt}", "r") as f:
        for line in f:
            text = normalizeString(line)
            tgt.append(text)

    print(len(src), len(tgt))

    with open(f"./en-fr/giga-fren.release2.fixed.{src}", "r") as f:
        for line in f:
            text = normalizeString(line)
            src.append(text)

    with open(f"./en-fr/giga-fren.release2.fixed.{tgt}", "r") as f:
        for line in f:
            text = normalizeString(line)
            tgt.append(text)

    print(len(src), len(tgt))

    with open(f"./en-fr/undoc.2000.fr-en.{src}", "r") as f:
        for line in f:
            text = normalizeString(line)
            src.append(text)
    with open(f"./en-fr/undoc.2000.fr-en.{tgt}", "r") as f:
        for line in f:
            text = normalizeString(line)
            tgt.append(text)

    print(len(src), len(tgt))

    with open(f"./en-fr/giga-fren.release2.fixed.{src}", "r") as f:
        for line in f:
            text = normalizeString(line)
            src.append(text)

    with open(f"./en-fr/giga-fren.release2.fixed.{tgt}", "r") as f:
        for line in f:
            text = normalizeString(line)
            tgt.append(text)

    print(len(src), len(tgt))

    pairs = []
    for i in range(len(src)):
        if (len(src[i].split()) > 0) and (len(tgt[i].split()) > 0):
            if (len(src[i].split()) < 100) and (len(tgt[i].split()) < 100):
                pairs.append([src[i], tgt[i]])
    
    return pairs

def read_test_data(src, tgt):
    src_corpus, tgt_corpus = [], []

    with open(f"./test_sets/newstest2014-fren-src.{src}.sgm", "r") as f:
        for line in f:
            if line.startswith('<seg id="'):
                start = line.index(">")
                end = line[start:].index("<")
                line = line.replace("-", " ")
                text = normalizeString(line[start+1 : start+end])
                src_corpus.append(text)    

    with open(f"./test_sets/newstest2014-fren-ref.{tgt}.sgm", "r") as f:
        for line in f:
            if line.startswith('<seg id="'):
                start = line.index(">")
                end = line[start:].index("<")
                line = line.replace("-", " ")
                text = normalizeString(line[start+1 : start+end])
                tgt_corpus.append(text)

    pairs = []
    for i in range(len(src_corpus)):
            if (len(src_corpus[i].split()) > 0) and (len(tgt_corpus[i].split()) > 0):
                pairs.append([src_corpus[i], tgt_corpus[i]])

    return pairs

def build_vocab():
    vocab = OrderedDict()

    with open("en_de_bpe.vocab", "r") as f:
        for line in f:
            word = line.split()[0]
            vocab[word] = 1

    return vocab

class TrainData(Dataset):
    def __init__(self, src, tgt):
        self.pairs = read_data(src, tgt)

    def __getitem__(self, index):
        return self.pairs[index][0], self.pairs[index][1]

    def __len__(self):
        return len(self.pairs)

class FrenchData(Dataset):
    def __init__(self, src, tgt):
        self.pairs = read_french(src, tgt)

    def __getitem__(self, index):
        return self.pairs[index][0], self.pairs[index][1]

    def __len__(self):
        return len(self.pairs)

class TestData(Dataset):
    def __init__(self, src, tgt):
        self.pairs = read_test_data(src, tgt)

    def __getitem__(self, index):
        return self.pairs[index][0], self.pairs[index][1]

    def __len__(self):
        return len(self.pairs)

if __name__ == "__main__":
    if sys.argv[1] == "ende":
        train_data = TrainData("en", "de")
        trainloader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True, num_workers=8)
        torch.save(trainloader, "trainloader_ende_16.pkl")
        test_data = TestData("en", "de")
        testloader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True, num_workers=8)
        torch.save(testloader, "testloader_ende.pkl")

    elif sys.argv[1] == "enfr":
        train_data = FrenchData("en", "fr")
        trainloader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, num_workers=8)
        torch.save(trainloader, "trainloader_enfr_32.pkl")
        test_data = TestData("en", "fr")
        testloader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True, num_workers=8)
        torch.save(testloader, "testloader_enfr.pkl")