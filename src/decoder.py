import torch.nn as nn
import torch.nn.functional as F
from utils import replicate
from attention import MultiHeadAttention
from embed import PositionalEncoding
from encoder import EncoderBlock


class DecoderBlock(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 heads=8,
                 expansion_factor=4,
                 dropout=0.2
                 ):
        """
        The DecoderBlock which will consist of the EncoderBlock used in the encoder, plus a decoder multi-head attention
        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(DecoderBlock, self).__init__()

        # First define the Decoder Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, heads)
        # normalization
        self.norm = nn.LayerNorm(embed_dim)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)
        # finally the encoderBlock (used as cross-attention)
        self.encoderBlock = EncoderBlock(embed_dim, heads, expansion_factor, dropout)
    
    def forward(self, encoder_output, decoder_input, src_mask=None, trg_mask=None):            
        # pass the inputs to the decoder multi-head attention (self-attention)
        self_attn_output = self.attention(decoder_input, decoder_input, decoder_input, trg_mask)
        # residual connection + normalization
        self_attn_out = self.dropout(self.norm(self_attn_output + decoder_input))
        # finally the encoderBlock for cross-attention (query from decoder, key and value from encoder)
        out = self.encoderBlock(key=encoder_output, query=self_attn_out, value=encoder_output, mask=src_mask)
        # return the output
        return out


class Decoder(nn.Module):

    def __init__(self,
                 target_vocab_size,
                 seq_len,
                 embed_dim=512,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2
                 ):
        """
        The Decoder part of the Transformer architecture

        it is a set of stacked decoders on top of each others, in the paper they used stack of 6 decoder
        :param target_vocab_size: the size of the target
        :param seq_len: the length of the sequence, in other words, the length of the words
        :param embed_dim: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each decoder
        :param heads: he number of heads in each decoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super(Decoder, self).__init__()

        # define the embedding
        self.embedding = nn.Embedding(target_vocab_size, embed_dim)
        # the positional embedding
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        # define the set of decoders
        self.blocks = replicate(DecoderBlock(embed_dim, heads, expansion_factor, dropout), num_blocks)
        # dropout for overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, trg_mask=None):
        x = self.dropout(self.positional_encoder(self.embedding(x)))  # 32x10x512

        for block in self.blocks:
            x = block(encoder_output, x, src_mask, trg_mask)

        return x
