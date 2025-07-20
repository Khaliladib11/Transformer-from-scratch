import torch.nn as nn
import torch.nn.functional as F
from .utils import replicate
from .attention import MultiHeadAttention
from .embed import PositionalEncoding


class DecoderBlock(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 heads=8,
                 expansion_factor=4,
                 dropout=0.2
                 ):
        """
        The DecoderBlock which consists of:
        1. Masked self-attention
        2. Cross-attention with encoder outputs
        3. Feed-forward network
        Each with residual connections and layer normalization
        
        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(DecoderBlock, self).__init__()

        # Self-attention for decoder
        self.self_attention = MultiHeadAttention(embed_dim, heads)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-attention with encoder
        self.cross_attention = MultiHeadAttention(embed_dim, heads)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_output, decoder_input, src_mask=None, trg_mask=None):
        """
        Forward pass through decoder block
        
        Args:
            encoder_output: output from encoder (batch_size, src_len, embed_dim)
            decoder_input: input to decoder (batch_size, trg_len, embed_dim)  
            src_mask: mask for encoder output (batch_size, 1, 1, src_len)
            trg_mask: causal mask for decoder (batch_size, 1, trg_len, trg_len)
        """
        
        # 1. Masked Self-Attention
        self_attn_output = self.self_attention(
            key=decoder_input, 
            query=decoder_input, 
            value=decoder_input, 
            mask=trg_mask
        )
        # Residual connection + normalization
        decoder_input = self.dropout(self.self_attn_norm(self_attn_output + decoder_input))
        
        # 2. Cross-Attention with encoder
        cross_attn_output = self.cross_attention(
            key=encoder_output,      # From encoder (src_len)
            query=decoder_input,     # From decoder (trg_len)  
            value=encoder_output,    # From encoder (src_len)
            mask=src_mask           # Mask encoder padding
        )
        # Residual connection + normalization (query shape = decoder shape)
        decoder_input = self.dropout(self.cross_attn_norm(cross_attn_output + decoder_input))
        
        # 3. Feed-Forward Network
        ff_output = self.feed_forward(decoder_input)
        # Residual connection + normalization
        output = self.dropout(self.ff_norm(ff_output + decoder_input))
        
        return output


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
