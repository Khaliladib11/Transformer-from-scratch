import math
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
        self.d_k = int(self.embed_dim / self.heads)  # 512 / 8 = 64 by default
        # note: The embedding dimension must be divided by the number of heads, otherwise the model will not work
        assert embed_dim % heads == 0, f"embed_dim ({embed_dim}) must be divisible by heads ({heads})"

        # query, value, key: (embed_dim x embed_dim)
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)  # the Query metrix
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)  # the Value metrix
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)  # the Key metrix

        # fully connected layer: 8*64x512 or 512x512
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)
        
    def split_heads(self, x):
        """
        Split the last dimension of the input tensor into heads
        :param x: the input tensor: batch_size x seq_len x embed_dim (e.g. 64x1024x512)
        :return: the tensor with the last dimension split into heads (e.g. 64x1024x8x64)
        """
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len, self.heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, seq_len, heads, d_k) to (batch_size, heads, seq_len, d_k), not (batch_size, heads, d_k, seq_len)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)
        
        key = self.w_k(key)       # (batch_size, k_len, embed_dim)
        query = self.w_q(query)   # (batch_size, q_len, embed_dim)
        value = self.w_v(value)   # (batch_size, v_len, embed_dim)

        # Split heads: (batch_size, seq_len, embed_dim) -> (batch_size, heads, seq_len, d_k)
        key = self.split_heads(key)
        query = self.split_heads(query)
        value = self.split_heads(value)

        # Scaled dot-product attention scores
        product = torch.einsum("bhqd,bhkd->bhqk", [query, key])  # (batch_size, heads, q_len, k_len)
        
        product = product / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask: True = attend, False = don't attend
            # We mask positions where mask == False with large negative value
            product = product.masked_fill(~mask, float("-1e20"))

        scores = F.softmax(product, dim=-1)  # (batch_size, heads, q_len, k_len)

        # Attention output
        output = torch.einsum("bhqk,bhkd->bhqd", [scores, value])  # (batch_size, heads, q_len, d_k)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.heads * self.d_k)

        output = self.fc_out(output)  # (batch_size, q_len, embed_dim)

        return output