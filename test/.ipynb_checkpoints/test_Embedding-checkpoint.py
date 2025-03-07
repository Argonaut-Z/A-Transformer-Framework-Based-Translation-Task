import sys
import os

sys.path.append('../')
from model.Embedding import TokenEmbedding, PositionalEncoding
import torch

if __name__ == '__main__':
    
    x = torch.tensor([[1, 3, 5, 7, 9],
                      [2, 4, 6, 8, 10]], dtype=torch.long)
    x = x.transpose(0, 1)   # [src_len, batch_size]
    print("input shape [src_len, batch_size]:", x.shape)
    token_embedding = TokenEmbedding(vocab_size=11, emb_size=512)
    x = token_embedding(tokens=x)
    print("token embedding shape [src_len, batch_size, embed_dim]:", x.shape)
    pos_embedding = PositionalEncoding(d_model=512)
    x = pos_embedding(x)
    print("pos embedding shape [src_len, batch_size, d_model]:", x.shape)
    