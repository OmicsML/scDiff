import torch
import torch.nn as nn

from scdiff.modules.layers.attention import BasicTransformerBlock
from scdiff.modules.layers.basic import FeedForward
from scdiff.utils.diffusion import ConditionEncoderWrapper
from scdiff.utils.modules import create_norm


class Encoder(nn.Module):

    def __init__(
        self,
        depth,
        dim,
        num_heads,
        dim_head,
        *,
        dropout=0.,
        cond_type='crossattn',
        cond_cat_input=False,
    ):
        super().__init__()

        self.cond_cat_input = cond_cat_input

        if cond_type == 'crossattn':
            self.blocks = nn.ModuleList([
                BasicTransformerBlock(dim, num_heads, dim_head, self_attn=False, cross_attn=True, context_dim=dim,
                                      qkv_bias=True, dropout=dropout, final_act=None)
                for _ in range(depth)])
        elif cond_type == 'mlp':
            self.blocks = nn.ModuleList([
                ConditionEncoderWrapper(nn.Sequential(
                    nn.Linear(dim, dim),
                    "gelu",
                    create_norm("layernorm", dim),
                    nn.Dropout(dropout),
                )) for _ in range(depth)])
        elif cond_type == 'stackffn':
            self.blocks = nn.ModuleList([
                ConditionEncoderWrapper(
                    FeedForward(dim, mult=4, glu=False, dropout=dropout)
                ) for _ in range(depth)])
        else:
            raise ValueError(f'Unknown conditioning type {cond_type!r}')

    def forward(self, x, context_list, cond_emb_list):
        # XXX: combine context_list and cond_emb_list in conditioner?..
        x = x.unsqueeze(1)

        stack = zip(self.blocks, reversed(context_list), reversed(cond_emb_list))
        for i, (blk, ctxt, cond_emb) in enumerate(stack):
            full_cond_emb_list = list(filter(lambda x: x is not None, (ctxt, cond_emb)))
            if self.cond_cat_input:
                full_cond_emb_list.append(x)
            full_cond_emb = torch.cat(full_cond_emb_list, dim=1) if full_cond_emb_list else None

            x = blk(x, context=full_cond_emb)

        return x.squeeze(1)
