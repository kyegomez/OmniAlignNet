# OmniAlignNet API Reference

Complete reference documentation for all modules, classes, and methods in the OmniAlignNet implementation.

---

## Table of Contents
- [CrossAttention](#crossattention)
- [SelfAttention](#selfattention)
- [OmniNet](#omninet)
- [Type Definitions](#type-definitions)

---

## CrossAttention

Custom implementation of Cross-Attention mechanism for multi-modal attention where queries from one modality attend to keys and values from another modality.

### Class Definition

```python
class CrossAttention(nn.Module)
```

### Constructor

#### `__init__(dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True)`

Initialize the CrossAttention module.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | Required | Embedding dimension for all projections |
| `num_heads` | `int` | Required | Number of attention heads (dim must be divisible by num_heads) |
| `dropout` | `float` | `0.0` | Dropout rate applied to attention weights (0.0 to 1.0) |
| `bias` | `bool` | `True` | Whether to use bias in linear projection layers |

**Returns:** None

**Raises:**
- `AssertionError`: If `dim` is not divisible by `num_heads`

**Attributes:**
- `dim` (int): Embedding dimension
- `num_heads` (int): Number of attention heads
- `head_dim` (int): Dimension per attention head (computed as dim // num_heads)
- `scale` (float): Scaling factor for attention scores (computed as head_dim ** -0.5)
- `q_proj` (nn.Linear): Linear projection for queries
- `k_proj` (nn.Linear): Linear projection for keys
- `v_proj` (nn.Linear): Linear projection for values
- `out_proj` (nn.Linear): Output projection layer
- `dropout` (nn.Dropout): Dropout layer for attention weights

---

### Methods

#### `forward(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor`

Forward pass of cross-attention mechanism.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `torch.Tensor` | Required | Query tensor of shape `(batch_size, seq_len_q, dim)` |
| `key` | `torch.Tensor` | Required | Key tensor of shape `(batch_size, seq_len_k, dim)` |
| `value` | `torch.Tensor` | Required | Value tensor of shape `(batch_size, seq_len_v, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask for masking specific positions |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, seq_len_q, dim)` | Output tensor after cross-attention |

**Processing Steps:**
1. Projects query, key, and value tensors through linear layers
2. Reshapes tensors for multi-head attention
3. Computes scaled dot-product attention scores
4. Applies optional mask
5. Applies softmax and dropout
6. Computes attention-weighted values
7. Reshapes and projects output

---

## SelfAttention

Custom implementation of Self-Attention mechanism where each token attends to all other tokens in the same sequence.

### Class Definition

```python
class SelfAttention(nn.Module)
```

### Constructor

#### `__init__(dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True)`

Initialize the SelfAttention module.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | Required | Embedding dimension for all projections |
| `num_heads` | `int` | Required | Number of attention heads (dim must be divisible by num_heads) |
| `dropout` | `float` | `0.0` | Dropout rate applied to attention weights (0.0 to 1.0) |
| `bias` | `bool` | `True` | Whether to use bias in linear projection layers |

**Returns:** None

**Raises:**
- `AssertionError`: If `dim` is not divisible by `num_heads`

**Attributes:**
- `dim` (int): Embedding dimension
- `num_heads` (int): Number of attention heads
- `head_dim` (int): Dimension per attention head (computed as dim // num_heads)
- `scale` (float): Scaling factor for attention scores (computed as head_dim ** -0.5)
- `qkv_proj` (nn.Linear): Combined linear projection for queries, keys, and values
- `out_proj` (nn.Linear): Output projection layer
- `dropout` (nn.Dropout): Dropout layer for attention weights

---

### Methods

#### `forward(x: Tensor, mask: Optional[Tensor] = None) -> Tensor`

Forward pass of self-attention mechanism.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | Required | Input tensor of shape `(batch_size, seq_len, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask for masking specific positions |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, seq_len, dim)` | Output tensor after self-attention |

**Processing Steps:**
1. Projects input through combined QKV projection
2. Splits into query, key, and value tensors
3. Reshapes tensors for multi-head attention
4. Computes scaled dot-product attention scores
5. Applies optional mask
6. Applies softmax and dropout
7. Computes attention-weighted values
8. Reshapes and projects output

---

## OmniNet

Main OmniAlignNet module implementing dual-stream multimodal alignment architecture with cross-attention, self-attention, and CLIP-style contrastive learning.

### Class Definition

```python
class OmniNet(nn.Module)
```

### Constructor

#### `__init__(dim: int, num_heads: int, query_heads: int = 1, dropout: float = 0.0, kv_heads: int = 1, temperature: float = 0.07, parallel: bool = True)`

Initialize the OmniNet module.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | Required | Embedding dimension for all modalities |
| `num_heads` | `int` | Required | Number of attention heads for cross-attention and self-attention layers |
| `query_heads` | `int` | `1` | Number of query heads for grouped query attention (future use) |
| `dropout` | `float` | `0.0` | Dropout rate for attention layers (0.0 to 1.0) |
| `kv_heads` | `int` | `1` | Number of key-value heads for grouped query attention (future use) |
| `temperature` | `float` | `0.07` | Temperature parameter for contrastive loss scaling |
| `parallel` | `bool` | `True` | Whether to use parallelized forward pass implementation |

**Returns:** None

**Attributes:**
- `dim` (int): Embedding dimension
- `num_heads` (int): Number of attention heads
- `query_heads` (int): Number of query heads
- `dropout` (float): Dropout rate
- `kv_heads` (int): Number of key-value heads
- `temperature` (float): Temperature for contrastive loss
- `parallel` (bool): Flag for parallel execution
- `query_embedding` (nn.Parameter): Learnable query embedding of shape `(1, 1, dim)`
- `vision_cross_attn` (CrossAttention): Cross-attention layer for vision stream
- `audio_cross_attn` (CrossAttention): Cross-attention layer for audio stream
- `vision_self_attn` (SelfAttention): Self-attention layer for vision stream
- `audio_self_attn` (SelfAttention): Self-attention layer for audio stream
- `vision_l2_norm` (nn.LayerNorm): Layer normalization for vision embeddings
- `audio_l2_norm` (nn.LayerNorm): Layer normalization for audio embeddings

---

### Methods

#### `forward(vision_input: Tensor, audio_input: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]`

Main forward pass through the OmniAlignNet module. Routes to either parallel or standard implementation based on the `parallel` flag.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_input` | `torch.Tensor` | Required | Vision tokens of shape `(batch_size, seq_len, dim)` |
| `audio_input` | `torch.Tensor` | Required | Audio tokens of shape `(batch_size, seq_len, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, 1, dim)` | Vision embeddings after processing (omni-modal space) |
| `torch.Tensor` | `(batch_size, 1, dim)` | Audio embeddings after processing (omni-modal space) |
| `torch.Tensor` | Scalar | CLIP-style contrastive loss value |

**Return Type:** `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

---

#### `forward_parallel(vision_input: Tensor, audio_input: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]`

Ultra-parallelized forward pass using torch.jit.script optimization for maximum performance on GPU.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_input` | `torch.Tensor` | Required | Vision tokens of shape `(batch_size, seq_len, dim)` |
| `audio_input` | `torch.Tensor` | Required | Audio tokens of shape `(batch_size, seq_len, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, 1, dim)` | Vision embeddings after processing |
| `torch.Tensor` | `(batch_size, 1, dim)` | Audio embeddings after processing |
| `torch.Tensor` | Scalar | CLIP-style contrastive loss value |

**Return Type:** `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

**Optimizations:**
- JIT compilation with `torch.jit.optimized_execution`
- Parallel stream processing
- Vectorized CLIP loss computation

---

#### `_forward(vision_input: Tensor, audio_input: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]`

Standard (non-parallelized) forward pass through the OmniAlignNet module.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_input` | `torch.Tensor` | Required | Vision tokens of shape `(batch_size, seq_len, dim)` |
| `audio_input` | `torch.Tensor` | Required | Audio tokens of shape `(batch_size, seq_len, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, 1, dim)` | Vision embeddings after processing |
| `torch.Tensor` | `(batch_size, 1, dim)` | Audio embeddings after processing |
| `torch.Tensor` | Scalar | CLIP-style contrastive loss value |

**Return Type:** `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

**Processing Steps:**
1. Expands query embedding to match batch size
2. Processes vision and audio streams through respective cross-attention and self-attention layers
3. Computes CLIP-style contrastive loss

---

#### `_process_vision_stream(query: Tensor, vision_input: Tensor, mask: Optional[Tensor] = None) -> Tensor`

Internal method to process vision stream with optimized operations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `torch.Tensor` | Required | Query tensor of shape `(batch_size, 1, dim)` |
| `vision_input` | `torch.Tensor` | Required | Vision input tokens of shape `(batch_size, seq_len, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, 1, dim)` | Processed vision embeddings in omni-modal space |

**Processing Steps:**
1. Cross-attention: query attends to vision tokens
2. Self-attention: processes cross-attention output
3. Layer normalization: normalizes final embeddings

---

#### `_process_audio_stream(query: Tensor, audio_input: Tensor, mask: Optional[Tensor] = None) -> Tensor`

Internal method to process audio stream with optimized operations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `torch.Tensor` | Required | Query tensor of shape `(batch_size, 1, dim)` |
| `audio_input` | `torch.Tensor` | Required | Audio input tokens of shape `(batch_size, seq_len, dim)` |
| `mask` | `Optional[torch.Tensor]` | `None` | Optional attention mask |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, 1, dim)` | Processed audio embeddings in omni-modal space |

**Processing Steps:**
1. Cross-attention: query attends to audio tokens
2. Self-attention: processes cross-attention output
3. Layer normalization: normalizes final embeddings

---

#### `_compute_clip_loss(vision_embeddings: Tensor, audio_embeddings: Tensor) -> Tensor`

Compute CLIP-style contrastive loss between vision and audio embeddings (standard implementation with loops).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_embeddings` | `torch.Tensor` | Required | Vision embeddings of shape `(batch_size, seq_len, dim)` |
| `audio_embeddings` | `torch.Tensor` | Required | Audio embeddings of shape `(batch_size, seq_len, dim)` |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | Scalar | Contrastive loss value (symmetric) |

**Loss Formula:**
```
L_o-align = (1/2)(L_v->a + L_a->v)

where:
- L_v->a: Vision-to-audio contrastive loss
- L_a->v: Audio-to-vision contrastive loss
```

**Processing Steps:**
1. Normalizes embeddings for cosine similarity
2. Computes similarity matrix between vision and audio
3. Computes vision-to-audio loss using cross-entropy with diagonal as positives
4. Computes audio-to-vision loss using cross-entropy with diagonal as positives
5. Returns symmetric loss average

---

#### `_compute_clip_loss_vectorized(vision_embeddings: Tensor, audio_embeddings: Tensor) -> Tensor`

Compute CLIP-style contrastive loss using highly vectorized operations (optimized implementation).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_embeddings` | `torch.Tensor` | Required | Vision embeddings of shape `(batch_size, vision_seq_len, dim)` |
| `audio_embeddings` | `torch.Tensor` | Required | Audio embeddings of shape `(batch_size, audio_seq_len, dim)` |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | Scalar | Contrastive loss value (symmetric) |

**Optimizations:**
- Batch matrix operations (`torch.bmm`) instead of loops
- In-place operations for memory efficiency
- Simultaneous bidirectional loss computation
- Vectorized diagonal extraction

**Loss Formula:**
```
L_o-align = (1/2)(L_v->a + L_a->v)
```

---

#### `get_similarity_matrix(vision_embeddings: Tensor, audio_embeddings: Tensor) -> Tensor`

Compute cosine similarity matrix between vision and audio embeddings.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vision_embeddings` | `torch.Tensor` | Required | Vision embeddings of shape `(batch_size, vision_seq_len, dim)` |
| `audio_embeddings` | `torch.Tensor` | Required | Audio embeddings of shape `(batch_size, audio_seq_len, dim)` |

**Returns:**

| Type | Shape | Description |
|------|-------|-------------|
| `torch.Tensor` | `(batch_size, vision_seq_len, audio_seq_len)` | Cosine similarity matrix |

**Processing Steps:**
1. Normalizes both embeddings using L2 normalization
2. Computes matrix multiplication: `vision_norm @ audio_norm.T`
3. Returns unnormalized (non-temperature-scaled) similarity scores

**Use Cases:**
- Visualizing cross-modal attention patterns
- Analyzing alignment quality
- Debugging model behavior
- Zero-shot retrieval tasks

---

## Type Definitions

### Torch Tensor Types

All tensor parameters and returns use PyTorch tensors (`torch.Tensor`). Below are the common shapes used throughout the API:

#### Input Shapes

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| Vision Input | `(batch_size, vision_seq_len, dim)` | Vision modality token embeddings |
| Audio Input | `(batch_size, audio_seq_len, dim)` | Audio modality token embeddings |
| Query | `(batch_size, query_len, dim)` | Query tensor for cross-attention |
| Key | `(batch_size, key_len, dim)` | Key tensor for cross-attention |
| Value | `(batch_size, value_len, dim)` | Value tensor for cross-attention |
| Self-Attention Input | `(batch_size, seq_len, dim)` | Input for self-attention |
| Mask | `(batch_size, num_heads, seq_len, seq_len)` | Optional attention mask |

#### Output Shapes

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| Vision Embeddings | `(batch_size, 1, dim)` | Aligned vision embeddings in omni-modal space |
| Audio Embeddings | `(batch_size, 1, dim)` | Aligned audio embeddings in omni-modal space |
| CLIP Loss | Scalar | Contrastive loss value (single number) |
| Similarity Matrix | `(batch_size, vision_seq_len, audio_seq_len)` | Cosine similarity between vision and audio |
| Attention Output | `(batch_size, seq_len_q, dim)` | Output from attention layers |

### Optional Types

From Python's `typing` module:

```python
from typing import Optional, Tuple
from torch import Tensor

Optional[Tensor] = Union[Tensor, None]
Tuple[Tensor, Tensor, Tensor] = (Tensor, Tensor, Tensor)
```

---

## Complete Method Summary

### CrossAttention Methods

| Method | Input Types | Output Type | Description |
|--------|-------------|-------------|-------------|
| `__init__` | `dim: int, num_heads: int, dropout: float, bias: bool` | `None` | Initialize cross-attention module |
| `forward` | `query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]` | `Tensor` | Forward pass of cross-attention |

### SelfAttention Methods

| Method | Input Types | Output Type | Description |
|--------|-------------|-------------|-------------|
| `__init__` | `dim: int, num_heads: int, dropout: float, bias: bool` | `None` | Initialize self-attention module |
| `forward` | `x: Tensor, mask: Optional[Tensor]` | `Tensor` | Forward pass of self-attention |

### OmniNet Methods

| Method | Input Types | Output Type | Description |
|--------|-------------|-------------|-------------|
| `__init__` | `dim: int, num_heads: int, query_heads: int, dropout: float, kv_heads: int, temperature: float, parallel: bool` | `None` | Initialize OmniNet module |
| `forward` | `vision_input: Tensor, audio_input: Tensor, mask: Optional[Tensor]` | `Tuple[Tensor, Tensor, Tensor]` | Main forward pass (routes to parallel or standard) |
| `forward_parallel` | `vision_input: Tensor, audio_input: Tensor, mask: Optional[Tensor]` | `Tuple[Tensor, Tensor, Tensor]` | Parallelized forward pass with JIT optimization |
| `_forward` | `vision_input: Tensor, audio_input: Tensor, mask: Optional[Tensor]` | `Tuple[Tensor, Tensor, Tensor]` | Standard forward pass implementation |
| `_process_vision_stream` | `query: Tensor, vision_input: Tensor, mask: Optional[Tensor]` | `Tensor` | Process vision modality stream |
| `_process_audio_stream` | `query: Tensor, audio_input: Tensor, mask: Optional[Tensor]` | `Tensor` | Process audio modality stream |
| `_compute_clip_loss` | `vision_embeddings: Tensor, audio_embeddings: Tensor` | `Tensor` | Compute contrastive loss (standard) |
| `_compute_clip_loss_vectorized` | `vision_embeddings: Tensor, audio_embeddings: Tensor` | `Tensor` | Compute contrastive loss (vectorized) |
| `get_similarity_matrix` | `vision_embeddings: Tensor, audio_embeddings: Tensor` | `Tensor` | Compute similarity matrix |

---

## Usage Examples

### Basic Usage

```python
import torch
from main import OmniNet

# Initialize model
model = OmniNet(
    dim=512,
    num_heads=8,
    query_heads=4,
    dropout=0.1,
    kv_heads=2,
    temperature=0.07,
    parallel=True
)

# Create input tensors
batch_size = 4
vision_tokens = torch.randn(batch_size, 10, 512)  # 10 vision tokens
audio_tokens = torch.randn(batch_size, 8, 512)    # 8 audio tokens

# Forward pass
vision_emb, audio_emb, loss = model(vision_tokens, audio_tokens)

print(f"Vision embeddings shape: {vision_emb.shape}")  # (4, 1, 512)
print(f"Audio embeddings shape: {audio_emb.shape}")    # (4, 1, 512)
print(f"Contrastive loss: {loss.item():.4f}")          # scalar value
```

### Using Individual Components

```python
from main import CrossAttention, SelfAttention

# Cross-attention between modalities
cross_attn = CrossAttention(dim=512, num_heads=8, dropout=0.1)

query = torch.randn(2, 5, 512)
key = torch.randn(2, 10, 512)
value = torch.randn(2, 10, 512)

output = cross_attn(query, key, value)  # Shape: (2, 5, 512)

# Self-attention within modality
self_attn = SelfAttention(dim=512, num_heads=8, dropout=0.1)
x = torch.randn(2, 10, 512)
output = self_attn(x)  # Shape: (2, 10, 512)
```

### Computing Similarity Matrix

```python
# After forward pass
vision_emb, audio_emb, loss = model(vision_tokens, audio_tokens)

# Get similarity matrix for analysis
similarity = model.get_similarity_matrix(vision_emb, audio_emb)
print(f"Similarity matrix shape: {similarity.shape}")  # (batch_size, 1, 1)

# For visualization or retrieval tasks
import matplotlib.pyplot as plt
plt.imshow(similarity[0].detach().cpu().numpy())
plt.colorbar()
plt.title("Vision-Audio Similarity")
plt.show()
```

### Training Loop Example

```python
import torch.optim as optim

# Initialize model and optimizer
model = OmniNet(dim=512, num_heads=8, temperature=0.07)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for vision_batch, audio_batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        vision_emb, audio_emb, clip_loss = model(vision_batch, audio_batch)
        
        # Backward pass
        clip_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {clip_loss.item():.4f}")
```

---

## Performance Considerations

### Memory Usage

- **Attention Complexity**: O(n²) for sequence length n
- **Memory Scales With**: batch_size × seq_len² × dim
- **Optimization**: Use gradient checkpointing for long sequences

### Speed Optimizations

1. **Parallel Mode** (`parallel=True`): Uses JIT compilation
2. **Vectorized Loss**: Eliminates Python loops
3. **Batch Operations**: Uses `torch.bmm` for batch matrix multiplication
4. **In-place Operations**: Reduces memory allocations

### Recommended Settings

| Use Case | dim | num_heads | dropout | parallel |
|----------|-----|-----------|---------|----------|
| Research/Prototyping | 512 | 8 | 0.1 | True |
| Production/Fast Inference | 512 | 8 | 0.0 | True |
| Small Models | 256 | 4 | 0.1 | False |
| Large Models | 1024 | 16 | 0.1 | True |

---

## Version Information

- **Implementation Version**: 1.0
- **PyTorch Version**: 2.0+
- **Python Version**: 3.7+

---

## Notes

- All methods expect tensors on the same device (CPU or GPU)
- Input embeddings should be pre-normalized or scaled appropriately
- The `query_heads` and `kv_heads` parameters are reserved for future grouped query attention implementations
- For production use, consider using mixed precision training with `torch.cuda.amp`

---

## Support and Issues

For questions, issues, or contributions, please visit the GitHub repository or contact the maintainer at kye@swarms.world.

