import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Implements the absolute Positional Encoding mechanism.

    Args:
        max_seq_len (int): hypothetical maximum sequence length (usually 1000)
        hidden_size (int): embeddings size
    """

    def __init__(self, max_seq_len, hidden_size):
        super().__init__()

        position = torch.arange(0.0, max_seq_len).unsqueeze(1)
        neg_log_term = - math.log(10000.0) / hidden_size
        div_term = torch.exp(torch.arange(0.0, hidden_size, 2) * neg_log_term)

        pe = torch.zeros(max_seq_len, hidden_size, requires_grad=False)
        pe[:, 0::2] = torch.sin(position * div_term)

        # handle cases when hidden size is odd (cos will have 1 less than sin)
        pe_cos = torch.cos(position * div_term)
        if hidden_size % 2 == 1:
            pe_cos = pe_cos[:, :-1]
        pe[:, 1::2] = pe_cos

        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)
        self.hidden_size = hidden_size

    def forward(self, emb):
        # self.pe = self.pe.to(emb.device)
        assert (emb.shape[1] <= self.pe.shape[1])
        return emb + self.pe[:, :emb.shape[1]]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    batch_size = 8
    vocab_size = 1000
    emb_size = 20
    seq_len = 100
    max_seq_len = 5000
    d_i, d_j = 4, 10

    x_emb = torch.randint(vocab_size, size=(batch_size, seq_len)).long()
    x_rand = torch.randn(batch_size, seq_len, emb_size)
    x_zero = torch.zeros(batch_size, seq_len, emb_size)

    embed = nn.Embedding(vocab_size, emb_size)
    torch.nn.init.xavier_normal_(embed.weight)
    pe = PositionalEncoding(max_seq_len, emb_size)

    x_rand = pe(x_rand)
    x_emb = pe(embed(x_emb)).data
    x_zero = pe(x_zero)

    plt.figure(figsize=(15, 5))
    plt.title('Random input')
    plt.plot(np.arange(seq_len), x_rand[0, :, d_i:d_j].numpy())
    plt.legend(['dim %d' % d for d in range(d_i, d_j)])
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.title('Embedding input')
    plt.plot(np.arange(seq_len), x_emb[0, :, d_i:d_j].numpy())
    plt.legend(['dim %d' % d for d in range(d_i, d_j)])
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.title('Zero input')
    plt.plot(np.arange(seq_len), x_zero[0, :, d_i:d_j].numpy())
    plt.legend(['dim %d' % d for d in range(d_i, d_j)])
    plt.show()
