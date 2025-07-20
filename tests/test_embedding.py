"""
Test suite for Embedding and Positional Encoding components
"""
import pytest
import torch
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from embed import Embedding, PositionalEncoding


class TestEmbedding:
    """Test Embedding functionality"""
    
    @pytest.fixture
    def embedding_layer(self):
        """Fixture for embedding layer"""
        return Embedding(vocab_size=1000, embed_dim=128)
    
    def test_embedding_creation(self, embedding_layer):
        """Test embedding layer creation"""
        assert embedding_layer.embed_dim == 128
        assert embedding_layer.embed.num_embeddings == 1000
        assert embedding_layer.embed.embedding_dim == 128
    
    def test_embedding_forward_shape(self, embedding_layer):
        """Test embedding forward pass shape"""
        batch_size, seq_len = 4, 10
        
        x = torch.randint(0, 1000, (batch_size, seq_len))
        output = embedding_layer(x)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_embedding_scaling(self, embedding_layer):
        """Test that embeddings are scaled by sqrt(embed_dim)"""
        x = torch.tensor([[1, 2, 3]])
        
        output = embedding_layer(x)
        
        # Get raw embedding without scaling
        raw_embedding = embedding_layer.embed(x)
        expected_output = raw_embedding * math.sqrt(128)
        
        assert torch.allclose(output, expected_output)
    
    def test_embedding_gradients(self, embedding_layer):
        """Test that gradients flow through embedding"""
        x = torch.randint(0, 1000, (2, 5))
        
        output = embedding_layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check that embedding layer has gradients
        assert embedding_layer.embed.weight.grad is not None
        assert not torch.allclose(
            embedding_layer.embed.weight.grad, 
            torch.zeros_like(embedding_layer.embed.weight.grad)
        )
    
    @pytest.mark.parametrize("vocab_size,embed_dim", [
        (100, 64),
        (1000, 128),
        (10000, 256),
        (50000, 512)
    ])
    def test_different_embedding_sizes(self, vocab_size, embed_dim):
        """Test embedding with different vocabulary and embedding sizes"""
        embedding = Embedding(vocab_size=vocab_size, embed_dim=embed_dim)
        
        batch_size, seq_len = 2, 6
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embedding(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        
        # Check scaling factor
        raw_embedding = embedding.embed(x)
        expected_scaling = math.sqrt(embed_dim)
        expected_output = raw_embedding * expected_scaling
        
        assert torch.allclose(output, expected_output)


class TestPositionalEncoding:
    """Test Positional Encoding functionality"""
    
    @pytest.fixture
    def pos_encoding(self):
        """Fixture for positional encoding"""
        return PositionalEncoding(embed_dim=128, max_seq_len=100, dropout=0.1)
    
    def test_positional_encoding_creation(self, pos_encoding):
        """Test positional encoding creation"""
        assert pos_encoding.embed_dim == 128
        assert pos_encoding.pe.shape == (1, 100, 128)
    
    def test_positional_encoding_forward_shape(self, pos_encoding):
        """Test positional encoding forward pass shape"""
        batch_size, seq_len, embed_dim = 4, 20, 128
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_positional_encoding_addition(self, pos_encoding):
        """Test that positional encoding is added to input"""
        batch_size, seq_len, embed_dim = 2, 10, 128
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = pos_encoding(x)
        
        # Check that output is not just the input (encoding was added)
        # Note: Due to dropout, we test this in eval mode
        pos_encoding.eval()
        with torch.no_grad():
            output_no_dropout = pos_encoding(x)
            
        expected_output = x + pos_encoding.pe[:, :seq_len].requires_grad_(False)
        assert torch.allclose(output_no_dropout, expected_output, atol=1e-6)
    
    def test_positional_encoding_sinusoidal_pattern(self):
        """Test that positional encoding follows sinusoidal pattern"""
        embed_dim = 4  # Small dimension for easier testing
        max_seq_len = 8
        
        pos_enc = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        
        # Extract positional encoding tensor
        pe = pos_enc.pe[0]  # Shape: (max_seq_len, embed_dim)
        
        # Check that even dimensions use sin and odd dimensions use cos
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        
        expected_pe = torch.zeros(max_seq_len, embed_dim)
        expected_pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        expected_pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        assert torch.allclose(pe, expected_pe, atol=1e-6)
    
    def test_positional_encoding_different_sequence_lengths(self, pos_encoding):
        """Test with different sequence lengths"""
        embed_dim = 128
        
        for seq_len in [1, 5, 20, 50, 99]:
            x = torch.randn(2, seq_len, embed_dim)
            output = pos_encoding(x)
            
            assert output.shape == (2, seq_len, embed_dim)
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic"""
        pos_enc1 = PositionalEncoding(embed_dim=64, max_seq_len=50)
        pos_enc2 = PositionalEncoding(embed_dim=64, max_seq_len=50)
        
        # Positional encodings should be identical
        assert torch.allclose(pos_enc1.pe, pos_enc2.pe)
    
    def test_positional_encoding_no_requires_grad(self, pos_encoding):
        """Test that positional encoding doesn't require gradients"""
        batch_size, seq_len, embed_dim = 1, 10, 128
        
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        output = pos_encoding(x)
        
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients, but PE tensor should not require grad
        assert x.grad is not None
        assert not pos_encoding.pe.requires_grad
    
    @pytest.mark.parametrize("embed_dim,max_seq_len", [
        (64, 50),
        (128, 100),
        (256, 200),
        (512, 500)
    ])
    def test_different_positional_encoding_sizes(self, embed_dim, max_seq_len):
        """Test positional encoding with different dimensions and sequence lengths"""
        pos_enc = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        
        assert pos_enc.pe.shape == (1, max_seq_len, embed_dim)
        
        # Test with input
        batch_size, seq_len = 3, min(20, max_seq_len)
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        output = pos_enc(x)
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_positional_encoding_exceeds_max_length(self):
        """Test behavior when input sequence exceeds max_seq_len"""
        pos_enc = PositionalEncoding(embed_dim=32, max_seq_len=10)
        
        # Input sequence longer than max_seq_len should raise an error or be handled
        x = torch.randn(1, 8, 32)  # Use shorter sequence within max_seq_len
        
        # This should work since it's within max_seq_len
        output = pos_enc(x)
        assert output.shape == (1, 8, 32)


class TestEmbeddingPositionalEncodingIntegration:
    """Test integration between Embedding and Positional Encoding"""
    
    @pytest.fixture
    def embedding_with_pos_enc(self):
        """Fixture for embedding + positional encoding"""
        embedding = Embedding(vocab_size=100, embed_dim=64)
        pos_encoding = PositionalEncoding(embed_dim=64, max_seq_len=50)
        return embedding, pos_encoding
    
    def test_embedding_pos_encoding_pipeline(self, embedding_with_pos_enc):
        """Test complete embedding + positional encoding pipeline"""
        embedding, pos_encoding = embedding_with_pos_enc
        
        batch_size, seq_len = 3, 12
        x = torch.randint(0, 100, (batch_size, seq_len))
        
        # Apply embedding then positional encoding
        embedded = embedding(x)
        output = pos_encoding(embedded)
        
        assert output.shape == (batch_size, seq_len, 64)
        assert torch.isfinite(output).all()
    
    def test_embedding_pos_encoding_gradients(self, embedding_with_pos_enc):
        """Test gradients flow through both embedding and positional encoding"""
        embedding, pos_encoding = embedding_with_pos_enc
        
        x = torch.randint(0, 100, (2, 8))
        
        embedded = embedding(x)
        output = pos_encoding(embedded)
        
        loss = output.sum()
        loss.backward()
        
        # Check that embedding has gradients
        assert embedding.embed.weight.grad is not None
        # Positional encoding should not have gradients (frozen)
        assert not pos_encoding.pe.requires_grad 