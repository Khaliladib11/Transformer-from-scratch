import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):

    def __init__(self,
                 embed_dim,
                 src_vocab_size,
                 target_vocab_size,
                 seq_len,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2,
                 pad_token_id=0):
        super(Transformer, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.pad_token_id = pad_token_id

        self.encoder = Encoder(seq_len=seq_len,
                               vocab_size=src_vocab_size,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        self.decoder = Decoder(target_vocab_size=target_vocab_size,
                               seq_len=seq_len,
                               embed_dim=embed_dim,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               heads=heads,
                               dropout=dropout)

        self.fc_out = nn.Linear(embed_dim, target_vocab_size)

    def make_src_mask(self, src):
        """
        Create padding mask for source sequence.
        Args:
            src: source tensor of shape (batch_size, src_len)
        Returns:
            src_mask: mask tensor of shape (batch_size, 1, 1, src_len)
        """
        batch_size, src_len = src.shape
        # Create mask where padding tokens are False, non-padding tokens are True
        src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return src_mask  # Shape: (batch_size, 1, 1, src_len)

    def make_trg_mask(self, trg):
        """
        Create combined causal and padding mask for target sequence.
        Args:
            trg: target tensor of shape (batch_size, trg_len)
        Returns:
            trg_mask: combined mask tensor of shape (batch_size, 1, trg_len, trg_len)
        """
        batch_size, trg_len = trg.shape
        device = trg.device
        
        # Create causal mask (lower triangular) directly on the correct device
        causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=device, dtype=torch.bool)).expand(
            batch_size, 1, trg_len, trg_len
        )
        
        # Create padding mask
        padding_mask = (trg != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(batch_size, 1, trg_len, trg_len)
        
        # Combine both masks (both must be True for attention to be allowed)
        trg_mask = causal_mask & padding_mask
        
        return trg_mask

    def forward(self, source, target):
        # Create masks (automatically on correct device)
        src_mask = self.make_src_mask(source)
        trg_mask = self.make_trg_mask(target)
        
        # Forward pass
        enc_out = self.encoder(source, src_mask)
        outputs = self.decoder(target, enc_out, src_mask, trg_mask)
        output = self.fc_out(outputs)
        return output
