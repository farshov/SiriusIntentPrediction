# comet.ml
import torch
from vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

MAX_SEQ_LEN = 800

def main(data_path, model):
    # Data loading
    # Data is just a list of strs contains utterance: X_train, y_train, X_test, y_test

    # Build vocab:
    # 1. tokenize
    tokenizer = Vocab()
    tokenizer.build(X_train)

    X_train = tokenizer.tokenize(X_train)  # size: (batch, seq_lens)
    X_test = tokenizer.tokenize(X_test)
    X_val = tokenizer.tokenize(X_val)

    # 2. padding
    pad_val = tokenizer.get_pad()
    X_train = pad_sequence(X_train, batch_first=True, padding_value=pad_val)[1:, :MAX_SEQ_LEN]
    X_val = pad_sequence(X_val, batch_first=True, padding_value=pad_val)[1:, :MAX_SEQ_LEN]
    X_test = pad_sequence(X_test, batch_first=True, padding_value=pad_val)[1:, :MAX_SEQ_LEN]
    # Batch iterator



    # training process

    # evaluation process



if __name__ == "__main__":
    main(data_path, model)
