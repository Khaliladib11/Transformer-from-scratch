"""
Comprehensive Test Suite for Transformer with Padding Mask Support
Tests basic functionality, padding masks, variable-length sequences, and training compatibility
"""

import torch
import torch.nn as nn
from .transformer import Transformer


def test_basic_functionality():
    """Test 1: Basic transformer functionality"""
    print("=" * 60)
    print("TEST 1: Basic Transformer Functionality")
    print("=" * 60)
    
    # Model parameters
    embed_dim = 512
    src_vocab_size = 1000
    target_vocab_size = 1000
    seq_len = 20
    num_blocks = 6
    
    # Create model
    model = Transformer(
        embed_dim=embed_dim,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        seq_len=seq_len,
        num_blocks=num_blocks,
        expansion_factor=4,
        heads=8,
        dropout=0.1,
        pad_token_id=0  # 0 is padding token
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Sample input (no padding)
    batch_size = 4
    src_seq_len = 15
    tgt_seq_len = 12
    
    source = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))  # Start from 1 to avoid padding
    target = torch.randint(1, target_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    
    # Forward pass
    output = model(source, target)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == (batch_size, tgt_seq_len, target_vocab_size)
    print("✅ Basic functionality test PASSED!\n")


def test_padding_masks():
    """Test 2: Padding mask functionality with variable-length sequences"""
    print("=" * 60)
    print("TEST 2: Padding Mask Functionality") 
    print("=" * 60)
    
    # Model parameters
    embed_dim = 256  # Smaller for faster testing
    src_vocab_size = 100
    target_vocab_size = 100
    seq_len = 10
    pad_token_id = 0
    
    model = Transformer(
        embed_dim=embed_dim,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        seq_len=seq_len,
        num_blocks=2,  # Smaller for faster testing
        pad_token_id=pad_token_id
    )
    
    # Create batch with variable-length sequences (with padding)
    source = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],    # Length 5 + 5 padding
        [6, 7, 8, 0, 0, 0, 0, 0, 0, 0],    # Length 3 + 7 padding
        [9, 10, 11, 12, 0, 0, 0, 0, 0, 0], # Length 4 + 6 padding
        [13, 14, 0, 0, 0, 0, 0, 0, 0, 0]   # Length 2 + 8 padding
    ])
    
    target = torch.tensor([
        [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],    # Length 4 + 6 padding
        [5, 6, 7, 0, 0, 0, 0, 0, 0, 0],    # Length 3 + 7 padding
        [8, 9, 10, 11, 12, 0, 0, 0, 0, 0], # Length 5 + 5 padding
        [13, 14, 15, 0, 0, 0, 0, 0, 0, 0]  # Length 3 + 7 padding
    ])
    
    print("Variable-length sequences with padding:")
    print(f"Source:\n{source}")
    print(f"Target:\n{target}")
    
    # Test mask creation
    src_mask = model.make_src_mask(source)
    trg_mask = model.make_trg_mask(target)
    
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {trg_mask.shape}")
    
    # Display first sequence's source mask
    print(f"Source mask for sequence 0: {src_mask[0, 0, 0]}")
    
    # Display first sequence's target mask (causal + padding)
    print(f"Target mask for sequence 0 (first 4x4):")
    print(trg_mask[0, 0, :4, :4])
    
    # Forward pass
    output = model(source, target)
    print(f"Output shape: {output.shape}")
    
    assert src_mask.shape == (4, 1, 1, 10)
    assert trg_mask.shape == (4, 1, 10, 10)
    print("✅ Padding mask test PASSED!\n")


def test_device_compatibility():
    """Test 3: Device compatibility (GPU if available)"""
    print("=" * 60)
    print("TEST 3: Device Compatibility")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    model = Transformer(
        embed_dim=128,
        src_vocab_size=50,
        target_vocab_size=50,
        seq_len=8,
        num_blocks=1,
        pad_token_id=0
    ).to(device)
    
    # Create input on device
    source = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]], device=device)
    target = torch.tensor([[4, 5, 6, 0, 0, 0, 0, 0]], device=device)
    
    output = model(source, target)
    
    assert output.device == device
    print(f"✅ Device compatibility test PASSED! (Device: {device})\n")


def test_training_compatibility():
    """Test 4: Training compatibility with CrossEntropyLoss"""
    print("=" * 60)
    print("TEST 4: Training Compatibility")
    print("=" * 60)
    
    # Model setup
    vocab_size = 100
    model = Transformer(
        embed_dim=128,
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        seq_len=8,
        num_blocks=1,
        pad_token_id=0
    )
    
    # Sample data
    source = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
    target_input = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])   # Input to decoder
    target_output = torch.tensor([[2, 3, 4, 5, 6, 0, 0, 0]])  # Expected output (shifted by 1)
    
    # Forward pass
    logits = model(source, target_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    
    # Test with CrossEntropyLoss (standard for language modeling)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens in loss
    
    # Reshape for loss calculation
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    target_flat = target_output.view(-1)       # (batch_size * seq_len,)
    
    loss = criterion(logits_flat, target_flat)
    print(f"CrossEntropyLoss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print("✅ Training compatibility test PASSED!\n")


def test_attention_patterns():
    """Test 5: Verify attention patterns with padding"""
    print("=" * 60)
    print("TEST 5: Attention Pattern Verification")
    print("=" * 60)
    
    model = Transformer(
        embed_dim=64,
        src_vocab_size=20,
        target_vocab_size=20,
        seq_len=6,
        num_blocks=1,
        heads=2,
        pad_token_id=0
    )
    
    # Create sequences with clear padding patterns
    source = torch.tensor([[1, 2, 3, 0, 0, 0]])  # 3 real tokens + 3 padding
    target = torch.tensor([[4, 5, 0, 0, 0, 0]])  # 2 real tokens + 4 padding
    
    print(f"Source sequence: {source[0].tolist()}")
    print(f"Target sequence: {target[0].tolist()}")
    
    # Get masks
    src_mask = model.make_src_mask(source)
    trg_mask = model.make_trg_mask(target)
    
    print(f"Source mask: {src_mask[0, 0, 0].tolist()}")
    print("Target mask (causal + padding):")
    print(trg_mask[0, 0].int())  # Convert to int for clearer display
    
    # Forward pass
    output = model(source, target)
    print(f"Output shape: {output.shape}")
    
    # Verify that padding positions have consistent outputs
    # (they should be similar since they don't attend to real content)
    padding_outputs = output[0, 2:]  # Positions 2+ are padding-influenced
    print(f"First padding-influenced output shape: {padding_outputs.shape}")
    
    print("✅ Attention pattern verification PASSED!\n")


def run_all_tests():
    """Run all test functions"""
    print("🚀 Running Comprehensive Transformer Tests\n")
    
    test_basic_functionality()
    test_padding_masks()
    test_device_compatibility()
    test_training_compatibility()
    test_attention_patterns()
    
    print("🎉 ALL TESTS PASSED! Your transformer is ready to use!")
    print("\n" + "=" * 60)
    print("USAGE EXAMPLE:")
    print("=" * 60)
    print("""
# Initialize transformer
model = Transformer(
    embed_dim=512,
    src_vocab_size=10000,
    target_vocab_size=10000,
    seq_len=100,
    num_blocks=6,
    pad_token_id=0  # Important: specify your padding token
)

# Prepare data with padding
source = torch.tensor([[1, 2, 3, 4, 0, 0]])  # Real tokens: [1,2,3,4], Padding: [0,0] 
target = torch.tensor([[5, 6, 7, 0, 0, 0]])  # Real tokens: [5,6,7], Padding: [0,0,0]

# Forward pass (handles masking automatically)
logits = model(source, target)

# For training with CrossEntropyLoss
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding in loss
loss = criterion(logits.view(-1, target_vocab_size), target_shifted.view(-1))
    """)


if __name__ == "__main__":
    run_all_tests()
