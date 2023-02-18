import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):

    def __init__(self,
                 embed_dim,
                 src_vocab_size,
                 target_vocab_size,
                 seq_len,
                 num_blocks=6,
                 expansion_factor=4,
                 heads=8,
                 dropout=0.2):
        super(Transformer, self).__init__()
        self.target_vocab_size = target_vocab_size

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

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, source, target):
        trg_mask = self.make_trg_mask(target)
        enc_out = self.encoder(source)
        outputs = self.decoder(target, enc_out, trg_mask)
        output = F.softmax(self.fc_out(outputs), dim=-1)
        return output
