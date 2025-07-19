# Transformer from Scratch

![Transformer Architecture](./docs/images/transformer.png)

A complete PyTorch implementation of the Transformer architecture from the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). This implementation includes advanced features like padding masks for variable-length sequences and comprehensive testing.

## 🚀 Features

- ✅ **Complete Transformer Architecture**: Encoder-decoder model with multi-head self-attention
- ✅ **Padding Mask Support**: Handles variable-length sequences efficiently  
- ✅ **Proper Cross-Attention**: Correctly implemented decoder with separate self and cross-attention layers
- ✅ **Training Ready**: Compatible with PyTorch's `CrossEntropyLoss` and standard training loops
- ✅ **Device Agnostic**: Supports both CPU and GPU training
- ✅ **Comprehensive Testing**: Full test suite covering all functionality
- ✅ **Production Ready**: Clean, documented code following best practices

## 📦 Installation

This project uses modern Python dependency management with `uv` and `pyproject.toml`.

### Option 1: Using uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/Transformer-from-scratch.git
cd Transformer-from-scratch

# Install dependencies with uv
uv sync
```

### Option 2: Using pip
```bash
# Clone the repository  
git clone https://github.com/yourusername/Transformer-from-scratch.git
cd Transformer-from-scratch

# Install dependencies
pip install torch torchvision torchaudio
```

## 🎯 Quick Start

### Basic Usage

```python
import torch
from src.transformer import Transformer

# Initialize transformer
model = Transformer(
    embed_dim=512,
    src_vocab_size=10000,
    target_vocab_size=10000, 
    seq_len=100,
    num_blocks=6,
    expansion_factor=4,
    heads=8,
    dropout=0.1,
    pad_token_id=0  # Important: specify your padding token
)

# Prepare sequences (0 = padding token)
source = torch.tensor([[1, 2, 3, 4, 0, 0]])  # "hello world" + padding
target = torch.tensor([[5, 6, 7, 0, 0, 0]])  # "bonjour monde" + padding

# Forward pass - handles masking automatically
logits = model(source, target)
print(f"Output shape: {logits.shape}")  # (1, 6, 10000)
```

### Training Example

```python
import torch
import torch.nn as nn
from src.transformer import Transformer

# Model setup
model = Transformer(
    embed_dim=512,
    src_vocab_size=5000,
    target_vocab_size=5000,
    seq_len=50,
    pad_token_id=0
)

# Loss function - ignores padding tokens
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training step
def train_step(source, target_input, target_output):
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(source, target_input)
    
    # Compute loss
    loss = criterion(
        logits.view(-1, logits.size(-1)), 
        target_output.view(-1)
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example batch
source = torch.randint(1, 5000, (32, 20))      # Batch of source sequences
target_in = torch.randint(1, 5000, (32, 25))   # Decoder input
target_out = torch.randint(1, 5000, (32, 25))  # Expected output

loss = train_step(source, target_in, target_out)
print(f"Training loss: {loss:.4f}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python src/test.py

# Expected output:
# ============================================================
# TEST 1: Basic Transformer Functionality  
# ✅ Basic functionality test PASSED!
# 
# TEST 2: Padding Mask Functionality
# ✅ Padding mask test PASSED!
# 
# ... (more tests)
# 
# 🎉 ALL TESTS PASSED! Your transformer is ready to use!
```

The test suite covers:
- Basic transformer functionality
- Padding mask behavior with variable-length sequences
- Device compatibility (CPU/GPU)
- Training compatibility with CrossEntropyLoss
- Attention pattern verification

## 🏗️ Architecture Overview

### Core Components

```
Transformer
├── Encoder (Stack of 6 EncoderBlocks)
│   ├── Multi-Head Self-Attention
│   ├── Feed-Forward Network  
│   ├── Residual Connections
│   └── Layer Normalization
│
├── Decoder (Stack of 6 DecoderBlocks)  
│   ├── Masked Multi-Head Self-Attention
│   ├── Multi-Head Cross-Attention
│   ├── Feed-Forward Network
│   ├── Residual Connections
│   └── Layer Normalization
│
├── Input Embedding + Positional Encoding
└── Output Linear Layer
```

### Key Features

#### 1. **Padding Masks**
Automatically handles variable-length sequences:
```python
# Source mask: prevents attention to padding tokens
src_mask = (source != pad_token_id)  # Shape: (batch, 1, 1, src_len)

# Target mask: combines causal + padding masks  
trg_mask = causal_mask & padding_mask  # Shape: (batch, 1, trg_len, trg_len)
```

#### 2. **Proper Decoder Architecture**
Separate self-attention and cross-attention layers:
```python
# Self-attention: decoder attends to itself (with causal mask)
self_attn_out = self_attention(decoder_input, decoder_input, decoder_input, trg_mask)

# Cross-attention: decoder attends to encoder output
cross_attn_out = cross_attention(encoder_output, decoder_input, encoder_output, src_mask)
```

## 📁 Code Structure

```
src/
├── transformer.py     # Main Transformer class with padding mask support
├── encoder.py        # Encoder and EncoderBlock classes
├── decoder.py        # Decoder and DecoderBlock classes  
├── attention.py      # Multi-Head Attention implementation
├── embed.py          # Embedding and Positional Encoding
├── utils.py          # Utility functions (replicate)
└── test.py           # Comprehensive test suite

docs/
├── images/           # Architecture diagrams
└── Attention.pdf     # Reference materials
```

## 🎓 Educational Value

This implementation is designed for learning and includes:

### **Detailed Documentation**
Every component is thoroughly documented with:
- Clear docstrings explaining parameters and shapes
- Inline comments describing the mathematical operations
- Shape annotations throughout the code

### **Progressive Complexity**
- Start with basic components (embedding, attention)
- Build up to complete encoder/decoder blocks
- Combine into full transformer architecture

### **Real-World Considerations**
- Padding mask implementation for variable sequences
- Proper device handling (CPU/GPU)
- Training compatibility with standard PyTorch workflows
- Memory-efficient operations

## 🚀 Advanced Usage

### Custom Configurations

```python
# Smaller model for experimentation
small_model = Transformer(
    embed_dim=256,
    src_vocab_size=1000,
    target_vocab_size=1000,
    seq_len=32,
    num_blocks=3,
    heads=4
)

# Larger model for serious applications
large_model = Transformer(
    embed_dim=768,
    src_vocab_size=50000,
    target_vocab_size=50000,
    seq_len=512,
    num_blocks=12,
    heads=12
)
```

### Inference Example

```python
# Greedy decoding example
def generate_translation(model, source_tokens, max_length=50):
    model.eval()
    
    # Start with start-of-sequence token
    target_tokens = [1]  # Assuming 1 is SOS token
    
    for _ in range(max_length):
        target_tensor = torch.tensor([target_tokens]).long()
        
        with torch.no_grad():
            logits = model(source_tokens, target_tensor)
            next_token = logits[0, -1].argmax().item()
            
        target_tokens.append(next_token)
        
        if next_token == 2:  # Assuming 2 is EOS token
            break
            
    return target_tokens

# Usage
source = torch.tensor([[3, 15, 42, 8, 1]])  # Some source sequence
translation = generate_translation(model, source)
print(f"Translation: {translation}")
```

## 📚 References

If you are interested in learning more about the Transformer architecture, I recommend the following resources:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar.
- [Pytorch Transformers from Scratch](https://www.youtube.com/watch?v=U0s0f995w14&t=729s) by Aladdin Persson.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) by Sasha Rush.
- [Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) by Sebastian Raschka.
- [TRANSFORMERS FROM SCRATCH](https://peterbloem.nl/blog/transformers) by Peter Bloem.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Beam search decoding
- Different positional encoding schemes  
- Optimization techniques (gradient clipping, learning rate scheduling)
- Additional test cases
- Performance benchmarks

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Vaswani et al. for the original Transformer paper
- The PyTorch team for the excellent framework
- The open-source community for educational resources

---

**Built with ❤️ for learning and understanding Transformers**


