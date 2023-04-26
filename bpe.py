import pickle
import string
import re
import unicodedata
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer, sentencepiece_tokenizer
from torchtext.vocab import vocab
from tqdm.auto import tqdm
from collections import OrderedDict
from preprocess import *

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"

    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z ]+", r" ", s)
    
    return s

def merge_data(src, tgt):
    src_corpus, tgt_corpus = [], []

    with open("/home/user22/RNN/data/en_de/train." + str(src) + ".txt", "r") as f:
        for line in tqdm(f):
            text = normalizeString(line)
            src_corpus.append(text)

    with open("/home/user22/RNN/data/en_de/train." + str(tgt) + ".txt", "r") as f:
        for line in tqdm(f):
            text = normalizeString(line)
            tgt_corpus.append(text)

    with open("train.en.de.txt", "w") as f:
        for i in tqdm(range(len(src_corpus))):
            if (len(src_corpus[i].split()) > 0) and (len(tgt_corpus[i].split()) > 0):
                f.write(src_corpus[i] + "\n" + tgt_corpus[i])
                    
if __name__ == "__main__":
    merge_data("en", "de")
    generate_sp_model('train.en.de.txt', vocab_size=37000, model_type="bpe", model_prefix="en_de_bpe")

    en_de_bpe = load_sp_model("en_de_bpe.model")

    token_generator = sentencepiece_tokenizer(en_de_bpe)
    
    PATH = "en_de_bpe_vocab.pth"
    
    special_symbols = ["<pad>"]
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

    all_vocab_ = build_vocab()
    all_vocab = vocab(all_vocab_, specials=special_symbols, special_first=True)
    all_vocab.set_default_index(UNK_IDX)

    torch.save(all_vocab, PATH)