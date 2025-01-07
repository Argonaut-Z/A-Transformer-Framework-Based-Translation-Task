import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_

is_print_shape = False


def multi_head_attention_forward(query,     # [tgt_len, batch_size, embed_dim]
                                 key,       # [src_len, batch_size, embed_dim]
                                 value,     # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj,  # [embed_dim = v_dim * num_heads, embed_dim = v_dim * num_heads]
                                 training=True,
                                 key_padding_mask=None, # [batch_size, src_len/tgt_len]
                                 q_proj=None,   # [embed_dim, k_dim * num_heads]
                                 k_proj=None,   # [embed_dim, k_dim * num_heads]
                                 v_proj=None,   # [embed_dim, v_dim * num_heads]
                                 attn_mask=None,    # [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]
                                 ):
    q = q_proj(query)
    # [tgt_len, batch_size, embed_dim] × [embed_dim, num_heads * k_dim] = [tgt_len, batch_size, num_heads * k_dim]
    
    k = k_proj(key)
    # [src_len, batch_size, embed_dim] × [embed_dim, num_heads * k_dim] = [src_len, batch_size, num_heads * k_dim]
    
    v = v_proj(value)
    # [src_len, batch_size, embed_dim] × [embed_dim, num_heads * k_dim] = [src_len, batch_size, num_heads * k_dim]
    
    tgt_len, bsz, embed_dim = query.size()  # [tgt_len, batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads   # num_heads * head_dim = embed_dim
    scaling = float(embed_dim) ** -0.5
    q = q * scaling     # [tgt_len, batch_size, k_dim * num_heads]
    
    if attn_mask is not None:   # [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of 3D attn_mask is not correct.')
    
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [tgt_len, batch_size, kdim * num_heads] ==>  [batch_size * num_heads, tgt_len, kdim]
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads, src_len, kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads, src_len, kdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads, tgt_len, k_dim] × [batch_size * num_heads, k_dim, src_len]
    # = [batch_size * num_heads, tgt_len, src_len]
    
    if attn_mask is not None:
        attn_output_weights += attn_mask    # [batch_size * num_heads, tgt_len, src_len]
    
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 变成 [batch_size, num_heads, tgt_len, src_len] 的形状
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf')
        )
        # 扩展维度，将 key_padding_mask 从 [batch_size, src_len] 变成 [batch_size, 1, 1, src_len]
        # 然后再对 attn_output_weights 进行填充
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
        # [batch_size * num_heads, tgt_len, src_len]
    
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)    # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    
    attn_output = torch.bmm(attn_output_weights, v)  # num_heads 个 Attention(Q,K,V) 的结果
    # Z = [batch_size * num_heads, tgt_len, src_len] × [batch_size * num_heads, src_len, v_dim]
    # = [batch_size * num_heads, tgt_len, v_dim]
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # [batch_size * num_heads, tgt_len, v_dim] => [tgt_len, batch_size, num_heads * v_dim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    
    Z = out_proj(attn_output)
    # [tgt_len, batch_size, num_heads * v_dim] => [tgt_len, batch_size, embed_dim]
    return Z, attn_output_weights.sum(dim=1) / num_heads    


class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.k_dim = self.head_dim
        self.v_dim = self.head_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == self.embed_dim
        # 上面的限制条件就是论文中的 d_k = d_v = d_model / n_head 条件
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # W_q,  embed_dim = q_dim * num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # W_k,  embed_dim = k_dim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # W_v,  embed_dim = v_dim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中，编码时 query, key, value 都是同一个输入，解码时 输入的部分也都是同一个输入
        解码和编码交互时 key, value 指的是 memory, query 指的是 tgt
        param query:    [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列长度
        param key:      [src_len, batch_size, embed_dim], src_len 表示源序列长度
        param value:    [src_len, batch_size, embed_dim], src_len 表示源序列长度
        param attn_mask: [tgt_len, src_len] or [batch_size * num_heads, tgt_len, src_len]
        param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        return:
            attn_output: [tgt_len, batch_size, embed_dim]
            attn_output_weights: [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MyMultiheadAttention(d_model, n_head, dropout=dropout)
        
        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu
        
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # src2.shape = [src_len, batch_size, embed_dim]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)   # 层归一化
        
        src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)   # 层归一化
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output  # [src_len, batch_size, num_heads * k_dim] <=> [src_len, batch_size, embed_dim]


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # 掩码自注意力机制
        tgt2 = self.self_attn(tgt, tgt, tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 多头交叉注意力机制
        tgt2 = self.multihead_attn(tgt, memory, memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈全连接层
        tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        
        return output

        
class MyTransformer(nn.Module):
    def __init__(self,
                 d_model=512,
                 n_head=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super(MyTransformer, self).__init__()
        # ================= 编码部分 ===================        
        encoder_layer = MyTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        # ================= 解码部分 =================== 
        decoder_layer = MyTransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self._reset_parameters()
        
        self.d_model = d_model
        self.n_head= n_head
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask