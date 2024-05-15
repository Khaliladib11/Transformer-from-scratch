from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim=512, heads=8):
        """
        Multi-Head Attention class
        :param embed_dim: the embedding dimension
        :param heads: the number of heads, default equals 8
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # 512 by default
        self.heads = heads  # 8 heads by default
        self.head = int(self.embed_dim / self.heads)  # 512 / 8 = 64 by default
        # note: The embedding dimension must be divided by the number of heads

        # query, value, key: (64x64)
        self.query = nn.Linear(self.head, self.head, bias=False)  # the Query metrix
        self.value = nn.Linear(self.head, self.head, bias=False)  # the Value metrix
        self.key = nn.Linear(self.head, self.head, bias=False)  # the Key metrix

        # fully connected layer: 8*64x512 or 512x512
        self.fc_out = nn.Linear(self.head * self.heads, embed_dim)

    def forward(self, key, query, value, mask=None):
        # Input of size: batch_size x sequence length x embedding dims
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        # reshape from (batch_size x seq_len x embed_size) -> (batch_size x seq_len x heads x head)
        # example: from (32x10x512) -> (32x10x8x64)
        key = key.reshape(batch_size, k_len, self.heads, self.head)
        query = query.reshape(batch_size, q_len, self.heads, self.head)
        value = value.reshape(batch_size, v_len, self.heads, self.head)

        key = self.key(key)  # (32x10x8x64)
        query = self.query(query)  # (32x10x8x64)
        value = self.value(value)  # (32x10x8x64)

        ############### query x key ###############

        # query shape: batch_size x q_len, heads, head, e.g: (32x10x8x64)
        # key shape: batch_size x v_len, heads, head, e.g: (32x10x8x64)
        # product shape should be: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        # if mask (in decoder)
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / sqrt(self.head)

        scores = F.softmax(product, dim=-1)

        ############### scores x value ###############

        # scores shape: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        # value shape: batch_size, v_len, heads, head, e.g: (32x10x8x64)
        # output: batch_size, heads, v_len, head, e.g: (32x10x512)

        output = torch.einsum("nhql,nlhd->nqhd", [scores, value]).reshape(
            batch_size, q_len, self.heads * self.head
        )

        output = self.fc_out(output)  # (32x10x512) -> (32x10x512)

        return output
