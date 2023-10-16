from itertools import repeat
from typing import List, Literal, Dict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from scdiff.ext.gears.model import GEARS_Conditioner
from scdiff.modules.layers.basic import FeedForward, MLPLayers
from scdiff.utils.modules import create_activation, create_norm

ATTN_MASK_MODE = Literal["nonzero", "subset_nonzero"]


def select_pe_encoder(*args, **kwargs):
    # FIX: remove pes
    raise NotImplementedError("Deprecated, please remove")


class EmbeddingDict(nn.Module):
    TEXT_EMB_DIR = './data/ontology_resources'

    def __init__(self, num_embed_dict, embedding_dim, depth, embedding_tokens=1,
                 norm_layer=None, freeze=False, mask_ratio=0.0, text_emb=None,
                 text_emb_file=None, freeze_text_emb=True, text_proj_type='linear',
                 stackfnn_glu_flag=False, text_proj_hidden_dim=512, text_proj_act=None,
                 text_proj_num_layers=2, text_proj_norm=None, text_proj_dropout=0.,
                 gears_flag=False, gears_mode="single", num_perts=None, gears_hidden_size=64,
                 gears_mlp_layers=2, gears_norm=None, num_go_gnn_layers=1):
        super().__init__()
        size = embedding_dim * embedding_tokens
        n = embedding_tokens
        d = embedding_dim

        self.keys = sorted(num_embed_dict)  # ensure consistent ordering
        self.mask_ratio = mask_ratio

        self.emb_dict = nn.ModuleDict()
        for key in self.keys:
            self.emb_dict[key] = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding(
                        num_embed_dict[key],
                        size,
                        _freeze=freeze,
                    ),
                    create_norm(norm_layer, size),
                    Rearrange('b (n d) -> b n d', n=n, d=d),
                )
                for _ in range(depth)
            ])

        if text_emb is not None or text_emb_file is not None:
            if text_emb is None:
                text_emb = torch.load(f'{self.TEXT_EMB_DIR}/{text_emb_file}')
            if text_proj_type == 'linear':
                text_proj = nn.Linear(text_emb.shape[1], size)
            elif text_proj_type == 'stackffn':
                text_proj = FeedForward(text_emb.shape[1], dim_out=size, mult=4, glu=stackfnn_glu_flag)
            elif text_proj_type == 'mlp':
                text_proj = MLPLayers(text_emb.shape[1], size, text_proj_hidden_dim, text_proj_num_layers,
                                      text_proj_dropout, text_proj_norm, text_proj_act)
            else:
                raise NotImplementedError(f"Unsupported text_proj_type {text_proj_type}")

            text_act = create_activation(text_proj_act)
            if text_proj_norm is None and norm_layer is not None:
                text_norm = create_norm(norm_layer, size)
            else:
                text_norm = create_norm(text_proj_norm, size)
            self.keys.append("text")
            self.emb_dict['text'] = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding.from_pretrained(text_emb, freeze=freeze_text_emb),
                    text_proj,
                    text_norm,
                    text_act,
                    Rearrange('b (n d) -> b n d', n=n, d=d),
                )
                for _ in range(depth)
            ])

        if num_perts is not None and gears_flag:
            self.keys.append('pert')
            self.gears_mode = gears_mode
            gears_kwargs = dict(num_perts=num_perts, out_dim=size, mode=gears_mode,
                                hidden_size=gears_hidden_size, mlp_layers=gears_mlp_layers)
            if gears_mode == "single":
                self.emb_dict['pert'] = nn.ModuleList([
                    nn.Sequential(
                        GEARS_Conditioner(num_go_gnn_layers=num_go_gnn_layers, **gears_kwargs),
                        create_norm(gears_norm, size),
                        Rearrange('b (n d) -> b n d', n=n, d=d),
                    )
                    for _ in range(depth)
                ])
            else:
                self.emb_dict['pert'] = nn.ModuleList([
                    GEARS_Conditioner(num_go_gnn_layers=depth, **gears_kwargs),
                    nn.ModuleList([create_norm(gears_norm, size) for _ in range(depth)]),
                    Rearrange('b (n d) -> b n d', n=n, d=d),
                ])

    def __iter__(self):
        yield from self.keys

    def __getitem__(self, key):
        return self.emb_dict[key]

    def forward(self, input: Dict[str, torch.Tensor], aug_graph=None) -> List[torch.Tensor]:
        # Outer list: condition types; inner list: layer depth
        out = []
        for key in self.keys:
            if self.training:
                # NOTE: NULL condition token added during dataset init, and is
                # set to be the first token (index zero).
                mask = torch.rand_like(input[key].float()) < self.mask_ratio
                masked_input = input[key].long()
                if key != 'text' and key != "pert":
                    masked_input[mask] = 0
            else:
                masked_input = input[key].long()

            if (
                isinstance(self[key][0], GEARS_Conditioner)  # single
                or isinstance(self[key][0][0], GEARS_Conditioner)  # parallel | sequential
            ):
                emb_list = []
                if self.gears_mode == "single":
                    for emb in self[key]:
                        gears_out = emb[0](masked_input, aug_graph)
                        emb_list.append(emb[1:](gears_out))
                else:
                    gears_out = self[key][0](masked_input, aug_graph)
                    stack = zip(gears_out, self[key][1], repeat(self[key][2]))
                    for emb, norm, rearrange in stack:
                        emb_list.append(rearrange(norm(emb)))
            else:
                emb_list = [emb(masked_input) for emb in self[key]]

            out.append(emb_list)

        # Consolidate by concatenating along the token dimention in each layer
        out = [torch.cat(embs, dim=1) for embs in zip(*out)]

        return out
