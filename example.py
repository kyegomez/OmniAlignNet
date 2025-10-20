import torch
from main import OmniAlignNet

# Initialize the OmniAlignNet module
model = OmniAlignNet(
    dim=512,
    num_heads=8,
    query_heads=4,
    dropout=0.1,
    kv_heads=2,
    temperature=0.07,
)

# Create sample input data
batch_size = 2
vision_seq_len = 10
audio_seq_len = 8
dim = 512

vision_tokens = torch.randn(batch_size, vision_seq_len, dim)
audio_tokens = torch.randn(batch_size, audio_seq_len, dim)

with torch.no_grad():
    vision_embeddings, audio_embeddings, clip_loss = model(
        vision_tokens, audio_tokens
    )
    
    print(f"Vision embeddings: {vision_embeddings.shape}")
    print(f"Audio embeddings: {audio_embeddings.shape}")
    