import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention head class
    """

    def __init__(self, embed_size, heads):
        """
        Self Attention constractor
        :param embed_size: the size of the embedding
        :param heads: the number of head to be fed into the multi head
        """

        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # Divide the embed size by the number of heads

        assert (self.head_dim * self.heads == self.embed_size), "head_dim*heads must be equal to embed_size"

        self.softmax = nn.Softmax(dim=-1)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_output = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, q, k, v, mask=None, e=1e-20):
        # dot product with weight matrix
        q = self.queries(q)
        k = self.keys(k)
        v = self.values(v)
        # Split the embedding into heads pieces
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        output, attention = self.scale_dot_product_attention(q, k, v, mask, e)

        output = self.concat(output)
        output = self.fc_output(output)

        return output

    def scale_dot_product_attention(self, q, k, v, mask, e):
        batch_size, heads, length, head_dim = k.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

    def concat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Concatenate function
        :param tensor: torch with shape [batch_size, heads, length, head_dim]
        :return: tensor with shape [batch_size, length, embed_size]
        """
        batch_size, heads, length, head_dim = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, length, self.embed_size)

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        split the tensor to number of head
        :param tensor: tensor with shape of [batch_size, length, embed_size]
        :return: tensor with shape [batch_size, heads, length, head_dim]
        """
        batch_size, length, embed_size = tensor.size()
        return tensor.reshape(batch_size, length, self.heads, self.head_dim).transpose(1, 2)
