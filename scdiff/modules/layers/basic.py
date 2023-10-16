import torch
import torch.nn as nn
import torch.nn.functional as F

from scdiff.utils.misc import default
from scdiff.utils.modules import batch_apply_norm, create_activation, create_norm


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class PreNormResidual(nn.Module):
    def __init__(self, layer, dim, norm="layernorm"):
        super().__init__()
        self.layer = layer
        self.norm = create_norm(norm, dim)

    def float(self, x):
        return self.layer(self.norm(x)) + x


class FFN(PreNormResidual):
    def __init__(self, dim, **kwargs):
        ff = FeedForward(dim, **kwargs)
        super().__init__(ff, dim)


class MLPLayers(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, dropout, norm=None, act="prelu"):
        super().__init__()
        layer_dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                create_activation(act),
                nn.Dropout(dropout),
            ))
            self.norms.append(create_norm(norm, layer_dims[i+1]))

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = batch_apply_norm(norm, x)
        return x


class ResMLPLayers(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, dropout, norm):
        super().__init__()
        assert num_layers > 1, 'At least two layers for MLPs.'
        layer_dims = [in_dim] + [hidden_dim * (num_layers - 1)] + [out_dim]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(layer_dims)-2):
            self.layers.append(nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.PReLU(),
                nn.Dropout(dropout),
            ))
            self.norms.append(create_norm(norm, layer_dims[i+1]))
        self.out_layer = nn.Sequential(
            nn.Linear(sum(layer_dims[:-1]), layer_dims[-1]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.out_norm = create_norm(norm, layer_dims[-1])

    def forward(self, x):
        hist = []
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = batch_apply_norm(norm, x)
            hist.append(x)
        out = self.out_layer(torch.cat(hist, 1))
        out = batch_apply_norm(self.out_norm, out)
        return out
