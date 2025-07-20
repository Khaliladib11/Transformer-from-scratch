from torch.utils.data import Dataset
import tiktoken
import torch

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=100, pad_token_id=0):
        """
        pairs: list of (english_text, french_text) tuples
        tokenizer: a tiktoken tokenizer object
        max_length: max sequence length
        pad_token_id: ID used for padding
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_text, trg_text = self.pairs[idx]

        # Tokenize
        src_tokens = self.tokenizer.encode(src_text)[:self.max_length]
        trg_tokens = self.tokenizer.encode(trg_text)[:self.max_length]

        # Pad
        src_padded = src_tokens + [self.pad_token_id] * (self.max_length - len(src_tokens))
        trg_padded = trg_tokens + [self.pad_token_id] * (self.max_length - len(trg_tokens))

        # Convert to tensors
        src_tensor = torch.tensor(src_padded, dtype=torch.long)
        trg_tensor = torch.tensor(trg_padded, dtype=torch.long)

        return src_tensor, trg_tensor