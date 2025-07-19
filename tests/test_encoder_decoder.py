"""
Test suite for Encoder and Decoder components
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock


class TestEncoderBlock:
    """Test EncoderBlock functionality"""
    
    @pytest.fixture
    def encoder_block(self):
        """Fixture for encoder block"""
        return EncoderBlock(embed_dim=128, heads=4, expansion_factor=2, dropout=0.1)
    
    def test_encoder_block_creation(self, encoder_block):
        """Test encoder block creation"""
        assert encoder_block.attention.embed_dim == 128
        assert encoder_block.attention.heads == 4
    
    def test_encoder_block_forward(self, encoder_block):
        """Test encoder block forward pass"""
        batch_size, seq_len, embed_dim = 2, 6, 128
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = encoder_block(key=x, query=x, value=x)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()
    
    def test_encoder_block_with_mask(self, encoder_block):
        """Test encoder block with attention mask"""
        batch_size, seq_len, embed_dim = 1, 4, 128
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        mask = torch.ones(batch_size, 1, 1, seq_len).bool()
        mask[0, 0, 0, 2:] = False  # Mask last two positions
        
        output = encoder_block(key=x, query=x, value=x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, embed_dim)


class TestEncoder:
    """Test Encoder functionality"""
    
    @pytest.fixture
    def encoder(self):
        """Fixture for encoder"""
        return Encoder(
            seq_len=10,
            vocab_size=100,
            embed_dim=128,
            num_blocks=2,
            expansion_factor=2,
            heads=4,
            dropout=0.1
        )
    
    def test_encoder_creation(self, encoder):
        """Test encoder creation"""
        assert len(encoder.blocks) == 2
        assert encoder.embedding.embed_dim == 128
    
    def test_encoder_forward(self, encoder):
        """Test encoder forward pass"""
        batch_size, seq_len = 2, 8
        
        x = torch.randint(1, 100, (batch_size, seq_len))
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, 128)
        assert torch.isfinite(output).all()
    
    def test_encoder_with_mask(self, encoder):
        """Test encoder with padding mask"""
        batch_size, seq_len = 2, 6
        
        x = torch.randint(1, 100, (batch_size, seq_len))
        # Create padding mask
        mask = torch.ones(batch_size, 1, 1, seq_len).bool()
        mask[0, 0, 0, 4:] = False  # Mask last two positions for first sequence
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, 128)


class TestDecoderBlock:
    """Test DecoderBlock functionality"""
    
    @pytest.fixture
    def decoder_block(self):
        """Fixture for decoder block"""
        return DecoderBlock(embed_dim=128, heads=4, expansion_factor=2, dropout=0.1)
    
    def test_decoder_block_creation(self, decoder_block):
        """Test decoder block creation"""
        assert decoder_block.self_attention.embed_dim == 128
        assert decoder_block.cross_attention.embed_dim == 128
        assert decoder_block.self_attention.heads == 4
    
    def test_decoder_block_forward(self, decoder_block):
        """Test decoder block forward pass"""
        batch_size, src_len, tgt_len, embed_dim = 2, 8, 6, 128
        
        encoder_output = torch.randn(batch_size, src_len, embed_dim)
        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        
        output = decoder_block(encoder_output, decoder_input)
        
        assert output.shape == (batch_size, tgt_len, embed_dim)
        assert torch.isfinite(output).all()
    
    def test_decoder_block_with_masks(self, decoder_block):
        """Test decoder block with both masks"""
        batch_size, src_len, tgt_len, embed_dim = 1, 5, 4, 128
        
        encoder_output = torch.randn(batch_size, src_len, embed_dim)
        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        
        # Source mask (padding mask)
        src_mask = torch.ones(batch_size, 1, 1, src_len).bool()
        src_mask[0, 0, 0, 3:] = False  # Mask last two positions
        
        # Target mask (causal mask)
        trg_mask = torch.tril(torch.ones(batch_size, 1, tgt_len, tgt_len)).bool()
        
        output = decoder_block(encoder_output, decoder_input, src_mask, trg_mask)
        
        assert output.shape == (batch_size, tgt_len, embed_dim)
    
    def test_different_encoder_decoder_lengths(self, decoder_block):
        """Test decoder block with different encoder/decoder sequence lengths"""
        batch_size, embed_dim = 2, 128
        src_len, tgt_len = 10, 7  # Different lengths
        
        encoder_output = torch.randn(batch_size, src_len, embed_dim)
        decoder_input = torch.randn(batch_size, tgt_len, embed_dim)
        
        output = decoder_block(encoder_output, decoder_input)
        
        # Output should match decoder (query) sequence length
        assert output.shape == (batch_size, tgt_len, embed_dim)


class TestDecoder:
    """Test Decoder functionality"""
    
    @pytest.fixture
    def decoder(self):
        """Fixture for decoder"""
        return Decoder(
            target_vocab_size=100,
            seq_len=10,
            embed_dim=128,
            num_blocks=2,
            expansion_factor=2,
            heads=4,
            dropout=0.1
        )
    
    def test_decoder_creation(self, decoder):
        """Test decoder creation"""
        assert len(decoder.blocks) == 2
        assert decoder.embedding.num_embeddings == 100
        assert decoder.embedding.embedding_dim == 128
    
    def test_decoder_forward(self, decoder):
        """Test decoder forward pass"""
        batch_size, src_len, tgt_len, embed_dim = 2, 8, 6, 128
        
        target = torch.randint(1, 100, (batch_size, tgt_len))
        encoder_output = torch.randn(batch_size, src_len, embed_dim)
        
        output = decoder(target, encoder_output)
        
        assert output.shape == (batch_size, tgt_len, embed_dim)
        assert torch.isfinite(output).all()
    
    def test_decoder_with_masks(self, decoder):
        """Test decoder with both source and target masks"""
        batch_size, src_len, tgt_len, embed_dim = 1, 5, 4, 128
        
        target = torch.randint(1, 100, (batch_size, tgt_len))
        encoder_output = torch.randn(batch_size, src_len, embed_dim)
        
        # Create masks
        src_mask = torch.ones(batch_size, 1, 1, src_len).bool()
        trg_mask = torch.tril(torch.ones(batch_size, 1, tgt_len, tgt_len)).bool()
        
        output = decoder(target, encoder_output, src_mask, trg_mask)
        
        assert output.shape == (batch_size, tgt_len, embed_dim)


class TestEncoderDecoderIntegration:
    """Test encoder-decoder integration"""
    
    @pytest.fixture
    def encoder_decoder_pair(self):
        """Fixture for encoder-decoder pair"""
        encoder = Encoder(
            seq_len=12, vocab_size=50, embed_dim=64,
            num_blocks=1, heads=2, dropout=0.1
        )
        decoder = Decoder(
            target_vocab_size=50, seq_len=12, embed_dim=64,
            num_blocks=1, heads=2, dropout=0.1
        )
        return encoder, decoder
    
    def test_encoder_decoder_flow(self, encoder_decoder_pair):
        """Test complete encoder-decoder flow"""
        encoder, decoder = encoder_decoder_pair
        
        batch_size, src_len, tgt_len = 2, 6, 4
        
        source = torch.randint(1, 50, (batch_size, src_len))
        target = torch.randint(1, 50, (batch_size, tgt_len))
        
        # Encoder forward
        encoder_output = encoder(source)
        
        # Decoder forward
        decoder_output = decoder(target, encoder_output)
        
        assert encoder_output.shape == (batch_size, src_len, 64)
        assert decoder_output.shape == (batch_size, tgt_len, 64)
    
    def test_encoder_decoder_with_masks(self, encoder_decoder_pair):
        """Test encoder-decoder with proper masking"""
        encoder, decoder = encoder_decoder_pair
        
        batch_size, src_len, tgt_len = 1, 5, 4
        
        # Create sequences with padding
        source = torch.tensor([[1, 2, 3, 0, 0]])  # Padding at end
        target = torch.tensor([[4, 5, 6, 0]])
        
        # Create masks
        src_mask = torch.tensor([[[[True, True, True, False, False]]]])
        trg_mask = torch.tril(torch.ones(1, 1, 4, 4)).bool()
        # Add padding mask to target mask
        tgt_padding = torch.tensor([[[[True, True, True, False]]]])
        tgt_padding = tgt_padding.expand(1, 1, 4, 4)
        trg_mask = trg_mask & tgt_padding
        
        # Forward pass
        encoder_output = encoder(source, mask=src_mask)
        decoder_output = decoder(target, encoder_output, src_mask, trg_mask)
        
        assert encoder_output.shape == (1, 5, 64)
        assert decoder_output.shape == (1, 4, 64)
        assert torch.isfinite(encoder_output).all()
        assert torch.isfinite(decoder_output).all() 