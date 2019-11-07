import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block of base-CNN from paper arXiv:1901.03489v1
    """
    def __init__(self, input_size=100, output_size=1024, p_dropout=0.6, filter_size=3, pool_size=3):
        super(self, ConvBlock).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=filter_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=pool_size),
            nn.Dropout(p=p_dropout)
        )

    def forward(self, x):  # shape (batch, seq_len, dim)
        return self.block(x)


class TransposeChannels(nn.Module):
    """
    transpose to channels to have dimention 1 for Conv1D layer
    """
    def __init__(self):
        super(self, TransposeChannels).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)


class GlobalMaxPool(nn.Module):
    def __init__(self, dim):
        super(self, GlobalMaxPool).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)
