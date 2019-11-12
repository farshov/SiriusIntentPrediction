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
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os, sys, inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)


from data_load.loading import load_from_json

MAX_SEQ_LEN = 800
N_EPOCHS = 15
SAVE_PATH = 'BaseCNNbase'


# def main(emb_path='GoogleNews-vectors-negative300.bin.gz', data_path='data/msdialogue/'):
def main(emb_path='glove.6B.100d.txt', data_path='data/msdialogue/'):
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
    print('Building Embedding')
    if emb_path == 'glove.6B.100d.txt':
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(emb_path, tmp_file)
        word2vec = KeyedVectors.load_word2vec_format(tmp_file)
    else:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=True)
    EMB_DIM = word2vec.vectors.shape[1]
    word2vec.add('<UNK>', np.mean(word2vec.vectors.astype('float32'), axis=0))
    word2vec.add('<PAD>', np.array(np.zeros(EMB_DIM)))
    tokenizer = Vocab()
    tokenizer.build(word2vec)
    
    print('Loading Data')
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
    model = BaseCNN(word2vec, tokenizer.get_pad(), emb_dim=EMB_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.BCELoss()
    # 5) training process
    treshold = 0.5
    print('Train')
#     for X, y in training:
#         X, y = X.to(device), y.to(device)
#         break
    for ep in range(N_EPOCHS):
        if ep == 10:
            optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        print(f'epoch: {ep}')
#         j = 0
#         # model.train() 
#         losses = []
#         for i in range(50):
#             optimizer.zero_grad()
            
#             output = model(X)
#             loss = torch.tensor(0.0).to(output)
#             for i in range(output.shape[1]):
#                 criterion = nn.BCELoss()
#                 loss += criterion(output[:, i].unsqueeze(1), y[:, i].unsqueeze(1).to(torch.float32))
#             losses.append(float(loss.cpu())/output.shape[1])
#             loss.backward()
#             optimizer.step()

#             # print(f'iter: {j}, loss: {loss}')
#             j += 1
#         print(f'train loss={np.mean(losses)}')
        
        j = 0
        model.train() 
        losses = []
        for X, y in training:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = torch.tensor(0.0).to(output)
            for i in range(output.shape[1]):
                criterion = nn.BCELoss()
                loss += criterion(output[:, i].unsqueeze(1), y[:, i].unsqueeze(1).to(torch.float32))
            loss.backward()
            losses.append(float(loss.cpu()))
            optimizer.step()

            # print(f'iter: {j}, loss: {loss}')
            j += 1
        print(f'train loss={np.mean(losses)}')
        with torch.no_grad():
            model.eval()
            # print('EVALUATION________')
            losses = []
            f1_scores = []
            precisions = []
            recalls = []
            accuracies = []
            for X, y in validation:
                criterion = nn.MultiLabelSoftMarginLoss()
                
                X, y = X.to(device), y.to(device)

                output = model(X)
                loss = torch.tensor(0.0).to(output)
                for i in range(output.shape[1]):
                    criterion = nn.BCELoss()
                    loss += criterion(output[:, i].unsqueeze(1), y[:, i].unsqueeze(1).to(torch.float32))
                losses.append(float(loss.cpu()))
                output = output.cpu().numpy()
                for i in range(len(output)):
                    pred = output[i] > treshold
                    if sum(pred) == 0:
                        pred = output[i].max(axis=0, keepdims=1) == output[i]
                    output[i] = pred
                precisions.append(get_f1(y, output)[0])
                recalls.append(get_f1(y, output)[1])
                f1_scores.append(get_f1(y, output)[2])
                accuracies.append(get_accuracy(y, output))
            
            print('VAL:') 
            print(f'val_loss={np.mean(losses)}')
            print(f'accuracy={np.mean(accuracies)}')
            print(f'precision={np.mean(precisions)}')
            print(f'recall={np.mean(recalls)}')
            print(f'f1-score={np.mean(f1_scores)}')
            
            print('__________________')
    torch.save(model.state_dict(), SAVE_PATH)
    # 5) test evaluation process


if __name__ == "__main__":
    main(data_path, model)
