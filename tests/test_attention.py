"""
Test suite for Multi-Head Attention mechanism
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test Multi-Head Attention functionality"""
    
    @pytest.fixture
    def attention_layer(self):
        """Fixture for attention layer"""
        return MultiHeadAttention(embed_dim=128, heads=8)
    
    def test_attention_creation(self, attention_layer):
        """Test attention layer creation"""
        assert attention_layer.embed_dim == 128
        assert attention_layer.heads == 8
        assert attention_layer.d_k == 16  # 128 / 8
    
    def test_attention_forward_shape(self, attention_layer):
        """Test attention forward pass shape"""
        batch_size, seq_len, embed_dim = 2, 10, 128
        
        key = torch.randn(batch_size, seq_len, embed_dim)
        query = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        output = attention_layer(key, query, value)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_attention_with_mask(self, attention_layer):
        """Test attention with mask"""
        batch_size, seq_len, embed_dim = 1, 4, 128
        
        key = torch.randn(batch_size, seq_len, embed_dim)
        query = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        # Create causal mask
        mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len)).bool()
        
        output = attention_layer(key, query, value, mask)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()
    
    def test_different_sequence_lengths(self, attention_layer):
        """Test attention with different key/query lengths (cross-attention)"""
        batch_size, embed_dim = 2, 128
        key_len, query_len = 8, 5
        
        key = torch.randn(batch_size, key_len, embed_dim)
        query = torch.randn(batch_size, query_len, embed_dim)
        value = torch.randn(batch_size, key_len, embed_dim)
        
        output = attention_layer(key, query, value)
        
        # Output should match query sequence length
        assert output.shape == (batch_size, query_len, embed_dim)
    
    def test_attention_mask_effectiveness(self, attention_layer):
        """Test that mask actually prevents attention to masked positions"""
        batch_size, seq_len, embed_dim = 1, 4, 128
        
        # Create distinct patterns for each position
        key = torch.zeros(batch_size, seq_len, embed_dim)
        key[0, 0] = 1.0    # First position has value 1
        key[0, 1] = 2.0    # Second position has value 2
        key[0, 2] = 3.0    # Third position has value 3 
        key[0, 3] = 4.0    # Fourth position has value 4
        
        query = torch.ones(batch_size, seq_len, embed_dim) * 0.5
        value = key.clone()
        
        # Test without mask
        output_no_mask = attention_layer(key, query, value)
        
        # Test with mask that blocks last two positions
        mask = torch.ones(batch_size, 1, seq_len, seq_len).bool()
        mask[0, 0, :, 2:] = False  # Block positions 2 and 3
        
        output_with_mask = attention_layer(key, query, value, mask)
        
        # Outputs should be different
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-5)
    
    @pytest.mark.parametrize("embed_dim,heads", [
        (64, 2),
        (128, 4),
        (256, 8),
        (512, 16)
    ])
    def test_different_configurations(self, embed_dim, heads):
        """Test attention with different embed_dim and head configurations"""
        attention = MultiHeadAttention(embed_dim=embed_dim, heads=heads)
        
        batch_size, seq_len = 2, 6
        key = torch.randn(batch_size, seq_len, embed_dim)
        query = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        output = attention(key, query, value)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attention.d_k == embed_dim // heads
    
    def test_attention_gradients(self, attention_layer):
        """Test that gradients flow through attention layer"""
        batch_size, seq_len, embed_dim = 1, 3, 128
        
        key = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        query = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        value = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        
        output = attention_layer(key, query, value)
        loss = output.sum()
        loss.backward()
        
        assert key.grad is not None
        assert query.grad is not None
        assert value.grad is not None
        assert not torch.allclose(key.grad, torch.zeros_like(key.grad))
    
    def test_invalid_embed_dim_heads_combination(self):
        """Test error handling for invalid embed_dim/heads combination"""
        with pytest.raises(AssertionError):
            # embed_dim must be divisible by heads
            MultiHeadAttention(embed_dim=127, heads=8)
    
    def test_attention_deterministic(self, attention_layer):
        """Test that attention is deterministic with same inputs"""
        torch.manual_seed(42)
        batch_size, seq_len, embed_dim = 1, 4, 128
        
        key = torch.randn(batch_size, seq_len, embed_dim)
        query = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        # First forward pass
        output1 = attention_layer(key, query, value)
        
        # Second forward pass with same inputs
        output2 = attention_layer(key, query, value)
        
        assert torch.allclose(output1, output2) 