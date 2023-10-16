from typing import Optional

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from scdiff.modules.layers.basic import FeedForward
from scdiff.utils.misc import default, exists, max_neg_value
from scdiff.utils.modules import BatchedOperation, create_norm, zero_module


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, *, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mnv = max_neg_value(sim)-torch.finfo(sim.dtype).max
            if sim.shape[1:] == sim.shape[1:]:
                mask = repeat(mask, 'b ... -> (b h) ...', h=h)
            else:
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, mnv)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int = 64,
        self_attn: bool = True,
        cross_attn: bool = False,
        ts_cross_attn: bool = False,
        final_act: Optional[nn.Module] = None,
        dropout: float = 0.,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = False,
        qkv_bias: bool = False,
        linear_attn: bool = False,
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
            # assert not (self_attn or linear_attn)
            # attn_cls = TokenSpecificCrossAttention
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention
        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )  # is self-attn if context is none
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )  # is a self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act
        self.checkpoint = checkpoint
        assert not self.checkpoint, 'Checkpointing not available yet'  # FIX: remove checkpiont option

    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:
            x = self.attn1(self.norm1(x), context=context, mask=cross_mask, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), mask=self_mask, **kwargs) + x
        x = self.ff(self.norm3(x)) + x
        if self.act is not None:
            x = self.act(x)
        return x


class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, norm="groupnorm32"):
        super().__init__()
        self.in_dim = in_dim
        inner_dim = n_heads * d_head
        self.norm = create_norm(norm, in_dim)

        self.proj_in = nn.Linear(in_dim, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Linear(inner_dim, out_dim))

    def forward(self, x, context=None):
        # NOTE: if no context is given, cross-attention defaults to self-attention
        x = x.unsqueeze(1)
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = self.proj_out(x)
        return (x + x_in)[:, 0, :]
