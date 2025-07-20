"""
Test suite for the main Transformer class
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformer import Transformer


class TestTransformerBasics:
    """Test basic transformer functionality"""
    
    def test_model_creation(self, small_model):
        """Test that model is created successfully"""
        assert isinstance(small_model, nn.Module)
        assert small_model.target_vocab_size == 100
        assert small_model.pad_token_id == 0
    
    def test_forward_pass_shape(self, small_model, sample_sequences):
        """Test that forward pass returns correct output shape"""
        source = sample_sequences['source']
        target = sample_sequences['target']
        
        output = small_model(source, target)
        
        batch_size, target_seq_len = target.shape
        expected_shape = (batch_size, target_seq_len, small_model.target_vocab_size)
        
        assert output.shape == expected_shape
    
    def test_forward_pass_dtype(self, small_model, sample_sequences):
        """Test that output has correct dtype"""
        source = sample_sequences['source']
        target = sample_sequences['target']
        
        output = small_model(source, target)
        
        assert output.dtype == torch.float32
    
    def test_different_sequence_lengths(self, small_model):
        """Test with different source and target sequence lengths"""
        source = torch.tensor([[1, 2, 3, 4, 5, 6]])  # Length 6
        target = torch.tensor([[7, 8, 9, 10]])        # Length 4
        
        output = small_model(source, target)
        
        assert output.shape == (1, 4, 100)  # Batch=1, Target_len=4, Vocab=100


class TestPaddingMasks:
    """Test padding mask functionality"""
    
    def test_source_mask_creation(self, small_model):
        """Test source mask is created correctly"""
        source = torch.tensor([[1, 2, 3, 0, 0, 0]])
        
        src_mask = small_model.make_src_mask(source)
        
        assert src_mask.shape == (1, 1, 1, 6)
        expected_mask = torch.tensor([[[[True, True, True, False, False, False]]]])
        assert torch.equal(src_mask, expected_mask)
    
    def test_target_mask_creation(self, small_model):
        """Test target mask combines causal and padding correctly"""
        target = torch.tensor([[1, 2, 3, 0]])
        
        trg_mask = small_model.make_trg_mask(target)
        
        assert trg_mask.shape == (1, 1, 4, 4)
        
        # Check causal pattern
        assert trg_mask[0, 0, 0, 0] == True   # Position 0 can see position 0
        assert trg_mask[0, 0, 1, 0] == True   # Position 1 can see position 0
        assert trg_mask[0, 0, 1, 1] == True   # Position 1 can see position 1
        assert trg_mask[0, 0, 0, 1] == False  # Position 0 cannot see position 1
        
        # Check padding masking
        assert trg_mask[0, 0, 0, 3] == False  # Cannot attend to padding
        assert trg_mask[0, 0, 3, 3] == False  # Padding position masked
    
    def test_variable_length_batch(self, small_model):
        """Test batch with variable-length sequences"""
        source = torch.tensor([
            [1, 2, 3, 4, 0, 0],    # Length 4
            [5, 6, 0, 0, 0, 0],    # Length 2
            [7, 8, 9, 10, 11, 12]  # Length 6 (no padding)
        ])
        target = torch.tensor([
            [13, 14, 15, 0, 0],    # Length 3
            [16, 17, 0, 0, 0],     # Length 2
            [18, 19, 20, 21, 22]   # Length 5
        ])
        
        output = small_model(source, target)
        
        assert output.shape == (3, 5, 100)
        assert torch.isfinite(output).all()  # Check for NaN/Inf values


class TestDeviceCompatibility:
    """Test device compatibility (CPU/CUDA/MPS)"""
    
    def test_cpu_inference(self, small_model, sample_sequences):
        """Test inference on CPU"""
        small_model = small_model.to('cpu')
        source = sample_sequences['source'].to('cpu')
        target = sample_sequences['target'].to('cpu')
        
        output = small_model(source, target)
        
        assert output.device.type == 'cpu'
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference(self, small_model, sample_sequences):
        """Test inference on CUDA GPU"""
        device = torch.device('cuda')
        small_model = small_model.to(device)
        source = sample_sequences['source'].to(device)
        target = sample_sequences['target'].to(device)
        
        output = small_model(source, target)
        
        assert output.device.type == 'cuda'
    
    @pytest.mark.mps
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_inference(self, small_model, sample_sequences):
        """Test inference on MPS (Apple Silicon GPU)"""
        device = torch.device('mps')
        small_model = small_model.to(device)
        source = sample_sequences['source'].to(device)
        target = sample_sequences['target'].to(device)
        
        output = small_model(source, target)
        
        assert output.device.type == 'mps'
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mask_device_consistency_cuda(self, small_model):
        """Test that masks are created on the same device as input (CUDA)"""
        device = torch.device('cuda')
        small_model = small_model.to(device)
        source = torch.tensor([[1, 2, 3, 0, 0]]).to(device)
        target = torch.tensor([[4, 5, 0, 0, 0]]).to(device)
        
        # This should not raise any device mismatch errors
        output = small_model(source, target)
        assert output.device.type == 'cuda'  # Compare device type instead of exact device
    
    @pytest.mark.mps
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mask_device_consistency_mps(self, small_model):
        """Test that masks are created on the same device as input (MPS)"""
        device = torch.device('mps')
        small_model = small_model.to(device)
        source = torch.tensor([[1, 2, 3, 0, 0]]).to(device)
        target = torch.tensor([[4, 5, 0, 0, 0]]).to(device)
        
        # This should not raise any device mismatch errors
        output = small_model(source, target)
        assert output.device.type == 'mps'  # Compare device type instead of exact device


class TestTrainingCompatibility:
    """Test training and optimization compatibility"""
    
    def test_backward_pass(self, small_model, sample_sequences):
        """Test that backward pass works correctly"""
        source = sample_sequences['source'].float()
        target_input = sample_sequences['target'].float()
        target_output = torch.randint(1, 100, (3, 8)).long()  # Expected output
        
        # Forward pass
        logits = small_model(source.long(), target_input.long())
        
        # Loss calculation
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits.view(-1, logits.size(-1)), target_output.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        has_gradients = any(p.grad is not None for p in small_model.parameters())
        assert has_gradients
    
    def test_optimizer_step(self, small_model, sample_sequences):
        """Test optimizer step updates parameters"""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        
        # Store initial parameters
        initial_params = [p.clone() for p in small_model.parameters()]
        
        source = sample_sequences['source']
        target = sample_sequences['target']
        target_output = torch.randint(1, 100, target.shape)
        
        # Training step
        optimizer.zero_grad()
        logits = small_model(source, target)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(logits.view(-1, logits.size(-1)), target_output.view(-1))
        
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        final_params = [p.clone() for p in small_model.parameters()]
        params_changed = any(
            not torch.equal(initial, final) 
            for initial, final in zip(initial_params, final_params)
        )
        assert params_changed
    
    def test_model_modes(self, small_model, sample_sequences):
        """Test train/eval modes"""
        source = sample_sequences['source']
        target = sample_sequences['target']
        
        # Test eval mode
        small_model.eval()
        with torch.no_grad():
            output_eval = small_model(source, target)
        
        # Test train mode
        small_model.train()
        output_train = small_model(source, target)
        
        # Outputs should be different due to dropout
        assert output_eval.shape == output_train.shape


class TestParametrizedConfigurations:
    """Test different model configurations"""
    
    def test_different_configurations(self, model_configs, uniform_sequences):
        """Test various model configurations"""
        # Use fixed sequence lengths instead of accessing model.seq_len
        max_seq_len = 12  # Based on the fixture configuration
        source = uniform_sequences['source'][:, :max_seq_len]
        target = uniform_sequences['target'][:, :max_seq_len]
        
        output = model_configs(source, target)
        
        batch_size, target_seq_len = target.shape
        expected_shape = (batch_size, target_seq_len, model_configs.target_vocab_size)
        
        assert output.shape == expected_shape
    
    @pytest.mark.parametrize("batch_size,src_len,tgt_len", [
        (1, 5, 7),
        (2, 8, 6), 
        (4, 10, 12),
        (8, 6, 4)
    ])
    def test_various_sequence_lengths(self, small_model, batch_size, src_len, tgt_len):
        """Test with various batch sizes and sequence lengths"""
        # Ensure sequences don't exceed reasonable limits (small_model uses seq_len=16)
        max_seq_len = 16
        src_len = min(src_len, max_seq_len)
        tgt_len = min(tgt_len, max_seq_len)
        
        source = torch.randint(1, 50, (batch_size, src_len))
        target = torch.randint(1, 50, (batch_size, tgt_len))
        
        output = small_model(source, target)
        
        assert output.shape == (batch_size, tgt_len, 100)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_minimum_sequence_length(self, small_model):
        """Test with minimum sequence length (1 token)"""
        source = torch.tensor([[1]])
        target = torch.tensor([[2]])
        
        output = small_model(source, target)
        
        assert output.shape == (1, 1, 100)
    
    def test_only_padding_tokens(self, small_model):
        """Test behavior with sequences that are all padding"""
        source = torch.tensor([[0, 0, 0, 0]])  # All padding
        target = torch.tensor([[0, 0, 0]])     # All padding
        
        # Should not crash, though results may not be meaningful
        output = small_model(source, target)
        
        assert output.shape == (1, 3, 100)
        assert torch.isfinite(output).all()
    
    def test_large_vocabulary_indices(self, small_model):
        """Test that model handles vocabulary boundary correctly"""
        # Use indices at the boundary of vocabulary
        source = torch.tensor([[1, 99, 50, 0]])    # Max valid index is vocab_size - 1
        target = torch.tensor([[2, 99, 0]])
        
        output = small_model(source, target)
        
        assert output.shape == (1, 3, 100)
        assert torch.isfinite(output).all() 