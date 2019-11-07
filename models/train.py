# comet.ml
import torch
import torch.nn as nn
from models.utils import encode_label
from models.vocab import Vocab
from models.dataset import MSDialog
from models.models import BaseCNN
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm
import gensim
import pandas as pd
import numpy as np
from quality_metrics.quality_metrics import get_accuracy, get_f1

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from data_load.loading import load_from_json

MAX_SEQ_LEN = 800
N_EPOCHS = 100


def main(emb_path='PreTrainedWord2Vec', data_path='data/msdialogue/'):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'DEVICE : {device}')
    params = {'batch_size': 128,
              'shuffle': True}

    # 1) Data loading
    # # Пока так для дебага
    # X, y = load_from_json(data_path)
    # # 1. One-Hot Encode
    # labels = {'O': 0, 'FQ': 1, 'IR': 2,
    #           'OQ': 3, 'GG': 4, 'FD': 5,
    #           'JK': 6, 'NF': 7, 'PF': 8,
    #           'RQ': 9, 'CQ': 10, 'PA': 11}
    # y_train = []
    # for l in y:
    #     l = l.split('_')
    #     cur_y = [0] * len(labels)
    #     for un_l in l:
    #         cur_y[labels[un_l]] = 1
    #     y_train.append(cur_y)
    # y_train = torch.tensor(y_train)
    # # 2. Нужный вид
    # X_train = []
    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         X_train.append(X[i][j])

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(emb_path)
    tokenizer = Vocab()
    tokenizer.build(word2vec)

    X_train = pd.read_csv(data_path+"train.tsv", sep="\t", header=None, index_col=None)
    y_train = encode_label(X_train[0].to_numpy())
    X_train = tokenizer.tokenize(X_train[1].to_numpy(), max_len=MAX_SEQ_LEN)

    X_val = pd.read_csv(data_path + "valid.tsv", sep="\t", header=None, index_col=None)
    y_val = encode_label(X_val[0].to_numpy())
    X_val = tokenizer.tokenize(X_val[1].to_numpy(), max_len=MAX_SEQ_LEN)

    X_test = pd.read_csv(data_path + "test.tsv", sep="\t", header=None, index_col=None)
    y_test = encode_label(X_test[0].to_numpy())
    X_test = tokenizer.tokenize(X_test[1].to_numpy(), max_len=MAX_SEQ_LEN)


    # 2. padding
    pad_val = tokenizer.get_pad()
    X_train = pad_sequence(X_train, batch_first=True,
                           padding_value=pad_val).to(torch.long)[1:, :MAX_SEQ_LEN]  # size: tensor(batch, max_seq_len)
    X_val = pad_sequence(X_val, batch_first=True,
                         padding_value=pad_val).to(torch.long)[1:, :MAX_SEQ_LEN]
    X_test = pad_sequence(X_test, batch_first=True,
                          padding_value=pad_val).to(torch.long)[1:, :MAX_SEQ_LEN]

    # 3) Batch iterator
    training = data.DataLoader(MSDialog(X_train, y_train), **params)
    validation = data.DataLoader(MSDialog(X_val, y_val), **params)
    testing = data.DataLoader(MSDialog(X_test, y_test), **params)

    # 4) Model, criterion and optimizer
    model = BaseCNN(word2vec, tokenizer.get_pad()).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MultiLabelSoftMarginLoss()
    # 5) training process

    for ep in tqdm(range(N_EPOCHS)):
        print(f'epoch: {ep}')
        i = 0
        for X, y in training:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y.to(torch.float32))
            loss.backward()
            optimizer.step()

            print(f'iter: {i}, loss: {loss}')
            i += 1

        with torch.set_grad_enabled(False):
            print('EVALUATION________')
            losses = []
            f1_scores = []
            accuracies = []
            for X, y in tqdm(validation):
                X, y = X.to(device), y.to(device)

                output = model(X)
                loss = criterion(output, y.to(torch.float32))
                losses.append(float(loss.cpu()))
                output = (output > 0.5).cpu().numpy()
                f1_scores.append(get_f1(y, output))
                f1_scores.append(get_accuracy(y, output))

            print(f'VAL: loss={np.mean(losses)}, f1-score={np.mean(f1_scores)}, accuracy={np.mean(accuracies)}')
            print('__________________')

    # 5) test evaluation process


if __name__ == "__main__":
    main(data_path, model)
