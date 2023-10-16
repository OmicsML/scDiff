import torch.nn as nn

from scdiff.utils.modules import create_activation, create_norm
from scdiff.modules.layers.scmodel import EmbeddingDict  # FIX: EmbeddingDict will be refactored to CombinedConditioner


class Decoder(nn.Module):
    def __init__(self, dim, out_dim, dropout=0., norm_type="layernorm", num_layers=1, cond_num_dict=None,
                 cond_emb_dim=None, cond_mask_ratio=0., act="gelu", out_act=None):
        super().__init__()
        if isinstance(act, str) or act is None:
            act = create_activation(act)
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)

        self.cond_num_dict = cond_num_dict
        if self.cond_num_dict is not None:
            cond_emb_dim = cond_emb_dim if cond_emb_dim is not None else dim
            self.cond_embed = EmbeddingDict(cond_num_dict, cond_emb_dim, 1, 1, None, mask_ratio=cond_mask_ratio)
        else:
            self.cond_embed = None

        self.layers = nn.ModuleList()  # FIX: use MLP layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim),
                act,
                create_norm(norm_type, dim),
                nn.Dropout(dropout),
            ))
        self.layers.append(nn.Sequential(nn.Linear(dim, out_dim), out_act))

    def forward(self, x, conditions=None):
        if self.cond_embed is not None:
            cond_emb = self.cond_embed(conditions)[0]
            x = x + cond_emb.squeeze(1)

        for layer in self.layers:
            x = layer(x)

        return x
