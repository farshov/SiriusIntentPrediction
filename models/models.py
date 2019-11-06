import torch
import torch.nn as nn


class BaseCNN(nn.Module):

    def __init__(self, vocab, n_conv_units=1024, p_dropout=0.6, filter_size=3,
                 pool_size=3, max_len=800, dim=300, n_classes=12):
        super(self, BaseCNN).__init__()
        self.n_conv_units = n_conv_units
        self.p_dropout = p_dropout
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.max_len = max_len
        self.embedder = nn.Embedding(len(vocab), dim)
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=dim, out_channels=n_conv_units, kernel_size=filter_size),
            nn.MaxPool1d(kernel_size=pool_size, padding=1),
            nn.Dropout(p=p_dropout)
            nn.Conv1d(in_channels=n_conv_units, out_channels=n_conv_units, kernel_size=filter_size),
            nn.MaxPool1d(kernel_size=pool_size, padding=1),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(in_channels=n_conv_units, out_channels=n_conv_units, kernel_size=filter_size),
            nn.MaxPool1d(kernel_size=pool_size, padding=1),
            nn.Dropout(p=p_dropout),
            nn.Linear(???),
            nn.ReLU(),
            nn.Linear(out_features=n_classes),
            nn.Sigmoid()
        )

    def forward(self, seq):

