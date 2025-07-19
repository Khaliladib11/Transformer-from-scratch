"""
Pytest configuration and fixtures for Transformer tests
"""
import pytest
import torch
import sys
import os

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer import Transformer


@pytest.fixture
def device():
    """Fixture to get the available device (CUDA, MPS if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture
def small_model():
    """Fixture for a small transformer model for fast testing"""
    return Transformer(
        embed_dim=128,
        src_vocab_size=100,
        target_vocab_size=100,
        seq_len=16,
        num_blocks=2,
        expansion_factor=2,
        heads=4,
        dropout=0.1,
        pad_token_id=0
    )


@pytest.fixture
def medium_model():
    """Fixture for a medium transformer model"""
    return Transformer(
        embed_dim=256,
        src_vocab_size=1000,
        target_vocab_size=1000,
        seq_len=32,
        num_blocks=4,
        expansion_factor=4,
        heads=8,
        dropout=0.1,
        pad_token_id=0
    )


@pytest.fixture
def sample_sequences():
    """Fixture for sample input sequences with padding"""
    return {
        'source': torch.tensor([
            [1, 2, 3, 4, 5, 0, 0, 0],    # Length 5 + 3 padding
            [6, 7, 8, 0, 0, 0, 0, 0],    # Length 3 + 5 padding
            [9, 10, 11, 12, 13, 14, 0, 0] # Length 6 + 2 padding
        ]),
        'target': torch.tensor([
            [1, 2, 3, 4, 0, 0, 0, 0],    # Length 4 + 4 padding
            [5, 6, 7, 8, 9, 0, 0, 0],    # Length 5 + 3 padding
            [10, 11, 12, 0, 0, 0, 0, 0]  # Length 3 + 5 padding
        ])
    }


@pytest.fixture
def uniform_sequences():
    """Fixture for sequences without padding (all same length)"""
    return {
        'source': torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24]
        ]),
        'target': torch.tensor([
            [25, 26, 27, 28, 29, 30, 31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47, 48]
        ])
    }


@pytest.fixture(params=[
    {'embed_dim': 64, 'heads': 2, 'num_blocks': 1},
    {'embed_dim': 128, 'heads': 4, 'num_blocks': 2},
    {'embed_dim': 256, 'heads': 8, 'num_blocks': 3}
])
def model_configs(request):
    """Parametrized fixture for different model configurations"""
    config = request.param
    return Transformer(
        embed_dim=config['embed_dim'],
        src_vocab_size=50,
        target_vocab_size=50,
        seq_len=12,
        num_blocks=config['num_blocks'],
        heads=config['heads'],
        pad_token_id=0
    ) 