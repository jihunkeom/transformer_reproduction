import torch
from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer
from torchtext.data.metrics import bleu_score

from preprocess import *
from utils import *
from model import Transformer, TransformerEmb

def translate(model, vocab, src, device, pipeline, PAD_IDX=0, SOS_IDX=2, EOS_IDX=3, max_len=20):
    src_text = torch.tensor(pipeline(src), dtype=torch.int64).unsqueeze(1).to(device)
    tgt_input = torch.tensor([SOS_IDX], dtype=torch.int64).unsqueeze(1).to(device)
    mask = create_mask(src_text, PAD_IDX, device)

    model.eval()
    with torch.no_grad():
        translated = model.translate(src_text, mask, tgt_input, EOS_IDX, max_len)

    return vocab.lookup_tokens(translated)[1:-1]

def evalutate(model, dataloader, device, vocab, pipeline, max_len=20, PAD_IDX=0):
    # model.eval()
    outputs = []
    gold_label = []
    
    # with torch.no_grad():
    for i, pairs in enumerate(dataloader):
        src, tgt = pairs[0][0], pairs[1][0]
        output = translate(model, vocab, src, device, pipeline)
        output = convert_back(output)
        outputs.append(output)
        # answer = list(tokenizer([tgt]))[0]
        answer = tgt.lower().split()
        gold_label.append([answer])
        print(answer)
        print(output)
        print(f"BLEU 1 : {bleu_score(outputs, gold_label, 1, [1])} || BLEU 2 : {bleu_score(outputs, gold_label, 4, [0.45, 0.3, 0.2, 0.05])}")
        print("-"*30)

    return bleu_score(outputs, gold_label, 1, [1]), bleu_score(outputs, gold_label, 4, [0.45, 0.3, 0.2, 0.05])

if __name__ == "__main__":
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    HIDDEN_SIZE = 512
    PATH = "ende_base_dropout.pt"
    # PATH = "ende_base_posemb.pt"

    special_symbols = ["<pad>"]
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

    all_vocab_ = build_vocab()
    all_vocab = vocab(all_vocab_, specials=special_symbols, special_first=True)
    all_vocab.set_default_index(UNK_IDX)
    vocab_size = len(all_vocab)

    en_de_bpe = load_sp_model("en_de_bpe.model")
    tokenizer = sentencepiece_tokenizer(en_de_bpe)
    pipeline = lambda x: all_vocab(list(tokenizer([normalizeString(x)]))[0])

    dataloader = torch.load("testloader_ende.pkl")

    model = Transformer(device=device, vocab_size=vocab_size, model_dim=512, PAD_IDX=0, p_dropout=0.1, n=6).to(device)
    # model = TransformerEmb(device=device, vocab_size=vocab_size, model_dim=512, PAD_IDX=0, p_dropout=0.1, n=6, h=8).to(device)

    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    bleu = evalutate(model, dataloader, device, all_vocab, pipeline)
    print(bleu)

    # sentence = "Orlando Bloom and Miranda Kerr still love each other"
    # print(translate(model, all_vocab, sentence, device, pipeline))
