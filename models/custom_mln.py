import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import numpy as np
from einops.layers.torch import Rearrange, Reduce

class MultiHeadAttention(nn.Module):
  def __init__(self, emb_size: int = 768, num_heads=8, dropout= 0,cross=False):
    super().__init__()
    self.emb_size = emb_size
    self.num_heads = num_heads
    self.cross=cross
   
    self.qkv = nn.Linear(emb_size, emb_size * 3)
    if self.cross:
        self.q_proj=nn.Linear(emb_size,emb_size)
        self.k_proj= nn.Linear(emb_size,emb_size)
        self.v_proj=nn.Linear(emb_size,emb_size)


    self.att_drop = nn.Dropout(dropout)
    self.projection = nn.Linear(emb_size, emb_size)
    self.count=0
    self.arr=[]


  def forward(self, x, k=None,mask=None):
   
    if self.cross:
        queries=rearrange(self.q_proj(x),"b n (h d) -> b h n d",h=self.num_heads)
        keys=rearrange(self.k_proj(k),"b n (h d) -> b h n d",h=self.num_heads)
        values=rearrange(self.v_proj(k),"b n (h d) -> b h n d",h=self.num_heads)
    else:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

    energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
    if mask is not None:
      fill_value = torch.finfo(torch.float32).min
      energy.mask_fill(~mask, fill_value)

    scaling = self.emb_size ** (1/2)
    att = F.softmax(energy, dim = -1) / scaling
    att_inter=att
    att = self.att_drop(att)
    
    # sum up over the third axis
    out = torch.einsum('bhal, bhlv -> bhav', att, values)
    out = rearrange(out, "b h n d -> b n (h d)")
    out = self.projection(out)
    return out,att_inter,energy
  
class Pair_attention(nn.Module):
    def __init__(self, embed_size, heads,dropout):
        super(Pair_attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.att_drop = nn.Dropout(dropout)
        self.count=0
        self.arr=[]
    def forward(self, pairs):
        # Get number of training examples
        N = pairs.shape[0]


        # Split the embedding into self.heads different pieces
        pair_head = pairs.reshape(N, self.heads, self.head_dim)

        # Differentiating between value, key, query
        values = self.values(pair_head)
        keys = self.keys(pair_head)
        queries = self.queries(pair_head)

        # Calculating similarity between query and keys for each head
        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        # Normalize energy values similarly
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        att_inter=attention
        
        attention = self.att_drop(attention)
        # Output is a attention weighted combination of each value
        out = torch.einsum("hqk,khd->qhd", [attention, values]).reshape(
            N, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out, energy, att_inter
