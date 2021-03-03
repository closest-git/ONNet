import torch.nn as nn
import torch
import math
import torch.nn.functional as F
# from .sparse_max import sparsemax, entmax15

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class QK_Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        #mini batch多句话得长度并不一致,需要按照最大得长度对短句子进行补全，也就是padding零，mask起来，填充一个负无穷（-1e9这样得数值），这样计算就可以为0了，等于把计算遮挡住。
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # p_attn = entmax15(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_project = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = QK_Attention()
        self.dropout = nn.Dropout(p=dropout) if dropout>0 else None

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        if self.attention is None:
            x = self.dropout(x)         #Very interesting, why self-attention is so useful?
        else:
            if self.h == 1:
                # query, key, value = [l(x) for l, x in zip(self.linear_project, (x, x, x))]
                query, key, value = x,x,x
            else:
            # 1) Do all the linear projections in batch from d_model => h x d_k
                query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                    for l, x in zip(self.linear_project, (x, x, x))]
            # query, key, value = (x,x,x)

            # 2) Apply attention on all the projected vectors in batch.
            x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

            # 3) "Concat" using a view and apply a final linear.
            if self.h > 1:
                x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

#keep structure simple ,no norm,no dropout!!!
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()        
        self.norm = nn.LayerNorm(dim)       #why this is so good
        # self.norm = nn.BatchNorm1d(64)          #nearly same as layernorm
        # self.norm = nn.Identity()
        # self.norm = nn.BatchNorm1d(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.fn is None:
            x = self.norm(x)
        else:
            x = self.fn(self.norm(x), **kwargs)
        return x

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = GELU()
        # self.activation = nn.ReLU()     # maybe use ReLU

    def forward(self, x):
        if self.dropout is None:
            return self.w_2(self.activation(self.w_1(x)))
        else:
            return self.w_2(self.dropout(self.activation(self.w_1(x))))

class BTransformer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout,clip_grad=""):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        
        super().__init__()
        print(f"attn_heads={attn_heads}")
        self.clip_grad = clip_grad
        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        # self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)        
        # self.attn = SublayerConnection(size=hidden, dropout=dropout)
        # self.ff = SublayerConnection(size=hidden, dropout=dropout)
        if self.clip_grad == "agc":
            self.attn = Residual( MultiHeadedAttention(h = attn_heads, d_model=hidden, dropout=dropout) )        
            self.ff = Residual( PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout) )
        else:
            self.attn = Residual(PreNorm(hidden, MultiHeadedAttention(h = attn_heads, d_model=hidden, dropout=dropout)))        
            self.ff = Residual(PreNorm(hidden, PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, mask):
        # x = self.attn(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # x = self.ff(x, self.feed_forward)
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        if self.dropout is not None:
            return self.dropout(x)
        else:
            return x

        
class AttentionQKV(nn.Module):
    def __init__(self, hidden, attn_heads, dropout):
        super(AttentionQKV, self).__init__()
        self.attn = Residual(PreNorm(hidden, MultiHeadedAttention(h = attn_heads, d_model=hidden, dropout=dropout))) 
    
    def forward(self, x, mask=None):
        shape = list(x.shape)
        if len(shape)==2:
            x = x.unsqueeze(1)
        x = self.attn(x, mask=mask)
        if len(shape)==2:
            x = x.squeeze(1)
        return x
