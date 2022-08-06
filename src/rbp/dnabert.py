import torch
from torch import nn
from transformers import BertForSequenceClassification
import numpy as np
import collections


def get_defaults():
    import os
    # modeldir = "/h/mkarimza/software/DNABERT/trainedModels/6-new-12w-0"
    modeldir = os.path.join(os.getcwd(), "trainedModels/6-new-12w-0")
    if not os.path.exists(modeldir):
        print("See https://github.com/jerryji1993/DNABERT")
        print("Download pre-trained DNABERT/6-new-12w-0")
        print("Save it to {}".format(modeldir))
        raise ValueError("Pre-trained model doesn't exist")
    modelpath = os.path.join(modeldir, "pytorch_model.bin")
    modelconfigpath = os.path.join(modeldir, "config.json")
    tokenmappath = os.path.join(modeldir, "special_tokens_map.json")
    tokenconfigpath = os.path.join(modeldir, "tokenizer_config.json")
    vocabpath = os.path.join(modeldir, "vocab.txt")
    return modelpath, modelconfigpath, vocabpath, tokenmappath, tokenconfigpath


def load_config(modelconfigpath):
    import json
    with open(modelconfigpath, "r") as jsonfile:
        config = json.load(jsonfile)
    return config


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class DnaBert(nn.Module):
    def __init__(self, modelpath, modelconfigpath, vocab_file):
        super(DnaBert, self).__init__()
        self.transformer = BertForSequenceClassification.from_pretrained(
            modelpath, from_tf=False, config=modelconfigpath)
        self.vocab_dict = load_vocab(vocab_file)

    def forward(self, x):
        tokenized = tokenize_x(x, self.vocab_dict)
        out_t = self.transformer(tokenized)[0]
        return out_t


def tokenize_x(x, vocab_dict, max_len=512):
    x_ar = x.detach().cpu().numpy()
    dict_nuc = {0: "A", 1: "C",
                2: "G", 3: "T"}
    idx_n = np.where(
        np.apply_along_axis(np.sum, arr=x_ar, axis=1) == 0)
    seqs = torch.argmax(x, dim=1).detach().cpu().numpy()
    newvals = torch.zeros((seqs.shape[0], seqs.shape[1] + 10))
    for i in range(seqs.shape[0]):
        curseq = [dict_nuc[seqs[i][k]] for
                  k in range(len(seqs[i]))]
        idx_n = np.where(
            np.apply_along_axis(
                np.sum, arr=x_ar[i], axis=0) == 0)[0]
        if len(idx_n) > 0:
            curseq[idx_n[0]] = "N"
        curseq.extend(["N"] * 10)
        for j in range(x_ar.shape[2]):
            idx_st = j
            idx_end = idx_st + 6
            hexnuc = "".join([each for each in curseq[idx_st:idx_end]])
            newvals[i, j] = vocab_dict.get(hexnuc, 1)
    newvals = newvals.to(x.device)
    newvals = newvals.long()
    return newvals


def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def get_x():
    np.random.seed(42)
    x = np.zeros((32, 4, 110))
    for i in range(32):
        temp_seq = np.random.choice(
            np.array([0, 1, 2, 3]),
            110, replace=True)
        ad_hot = to_one_hot(temp_seq, 4)
        x[i] = ad_hot.T
    return x


def get_xy():
    np.random.seed(42)
    x = np.zeros((32, 4, 110))
    y = np.zeros((32, 1))
    for i in range(32):
        temp_seq = np.random.choice(
            np.array([0, 1, 2, 3]),
            110, replace=True)
        ad_hot = to_one_hot(temp_seq, 4)
        y[i] = np.sum(ad_hot[:, 2]) * np.sum(ad_hot[:, 3]) / 80
        x[i] = ad_hot.T
    return x, y


def train_model():
    list_defs = get_defaults()
    from torch.nn import MSELoss
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using gpu")
    else:
        device = torch.device("cpu")
    x, y = get_xy()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)
    net = DnaBert(modelpath=list_defs[0],
                  modelconfigpath=list_defs[1],
                  vocab_file=list_defs[2])
    net = net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=0.0001,
        weight_decay=0.01, eps=1e-4)
    crit_mse = MSELoss().to(device)
    MINIBATCH = 32
    TOT_IDX = int(x.shape[0] / MINIBATCH) + 1
    for epoch in range(200):
        loss_mse = 0
        for idx_batch in range(TOT_IDX):
            idx_st = idx_batch * MINIBATCH
            idx_end = min([idx_st + MINIBATCH, x.shape[0]])
            if idx_st >= x.shape[0]:
                break
            train1 = x[idx_st:idx_end].to(device)
            resp = y[idx_st:idx_end].to(device).float()
            pred = net(train1)
            admse = crit_mse(pred, resp)
            optimizer.zero_grad()
            admse.backward()
            optimizer.step()
            loss_mse += admse.item()
            del train1, resp, pred
        avg_loss = loss_mse / idx_batch
        print("Loss at epoch {} is {}".format(
                epoch, avg_loss))


if __name__ == "__main__":
    train_model()
