# comet.ml
import torch
from models.vocab import Vocab
from models.dataset import MSDialog
from models.models import BaseCNN
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm

MAX_SEQ_LEN = 800
N_EPOCHS = 100


def main(data_path, model):
    device = 'cpu'
    if torch.cuda.is_avaliable():
        device = 'cuda'
    params = {'batch_size': 128,
              'shuffle': True}
    # 1) Data loading
    # Data is just a list of strs contains utterance: X_train, y_train, X_test, y_test
    #

    # 2) Build vocab:
    # 1. tokenize
    tokenizer = Vocab()
    tokenizer.build(X_train)

    X_train = tokenizer.tokenize(X_train)  # size: (batch, seq_lens)
    X_test = tokenizer.tokenize(X_test)
    X_val = tokenizer.tokenize(X_val)

    # 2. padding
    pad_val = tokenizer.get_pad()
    X_train = pad_sequence(X_train, batch_first=True,
                           padding_value=pad_val)[1:, :MAX_SEQ_LEN]  # size: tensor(batch, max_seq_len)
    X_val = pad_sequence(X_val, batch_first=True,
                         padding_value=pad_val)[1:, :MAX_SEQ_LEN]
    X_test = pad_sequence(X_test, batch_first=True,
                          padding_value=pad_val)[1:, :MAX_SEQ_LEN]

    # 3) Batch iterator
    training = data.DataLoader(MSDialog(X_train, y_train), **params)
    validation = data.DataLoader(MSDialog(X_val, y_val), **params)
    testing = data.DataLoader(MSDialog(X_test, y_test), **params)

    # 4) Model, criterion and optimizer
    model = BaseCNN(tokenizer).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # 5) training process

    for ep in range(N_EPOCHS):
        print(f'epoch {ep}:')
        for X, y in tqdm(training):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            for X, y in validation:
                # Transfer to GPU
                X, y = X.to(device), y.to(device)

                output = model(X)

                # evaluation

    # 5) test evaluation process


if __name__ == "__main__":
    main(data_path, model)
