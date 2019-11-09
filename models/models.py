import torch.nn as nn
import torch
from models.layers import ConvBlock, TransposeChannels, GlobalMaxPool


class BaseCNN(nn.Module):
    def __init__(self, pretr_emb, pad_idx, n_conv_units=1024, p_dropout=0.6, filter_size=3,
                 pool_size=3, emb_dim=100, n_classes=12, dense_layer_units=256):
        super(BaseCNN, self).__init__()

        # input_size: (batch, seq)
        weights = torch.FloatTensor(pretr_emb.vectors)
        self.embedder = nn.Embedding.from_pretrained(weights)
        self.embedder.padding_idx = pad_idx
#         for param in self.embedder.parameters():
#             param.requires_grad = False
        # embedded_input_size: (batch, seq, 100)
        self.model = nn.Sequential(
            # (batch, 800, 100)
            TransposeChannels(),
            # (batch, 100, 800)
            ConvBlock(emb_dim, n_conv_units, p_dropout, filter_size, pool_size),
            # (batch, 1024, 268)
            ConvBlock(n_conv_units, n_conv_units, p_dropout, filter_size, pool_size),
            # (batch, 1024, ?????)
            nn.Conv1d(n_conv_units, n_conv_units, filter_size),
            # (batch, 1024, ?????)
            nn.ReLU(),
            GlobalMaxPool(dim=2),
            # (batch, 1024)
            nn.Linear(in_features=n_conv_units, out_features=dense_layer_units),
            nn.ReLU(),
            # (batch, 256)
            nn.Linear(in_features=dense_layer_units, out_features=n_classes),
            nn.Sigmoid()
            # (batch, 12)
        )

    def forward(self, seq):
        """
        :param seq: tensor of tokens idxs (batch_size, seq_len)
        :return: probs: tensor of probabilities (batch_size, n_classes(originally 12))
        """
        return self.model(self.embedder(seq))
