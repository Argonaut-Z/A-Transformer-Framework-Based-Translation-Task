import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        """
        初始化词嵌入模块
        :param vocab_size: 词汇表大小
        :param emb_size: 嵌入维度
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    
    def forward(self, tokens):
        """
        计算词嵌入
        :param tokens: 输入序列 [seq_len, batch_size]
        :return: 嵌入序列 [seq_len, batch_size, embed_dim]
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码模块
        :param d_model: 嵌入维度
        :param max_len: 最大序列长度
        :param dropout: 丢弃概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)    # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)    # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        为输入张量添加位置编码
        :param x: 输入序列 [seq_len, batch_size, embed_dim]
        :return: 添加位置编码后的序列 [seq_len, batch_size, embed_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)