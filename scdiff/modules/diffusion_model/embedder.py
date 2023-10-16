import torch
import torch.nn as nn
import torch.nn.functional as F

from scdiff.utils.modules import create_activation, create_norm


class Embedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hidden, norm, activation='gelu', dropout=0.,
                 gene_emb=None, fix_embedding=False):
        super().__init__()

        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = {j: i for i, j in enumerate(pretrained_gene_list)}

        if gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
        else:
            num_genes = len(pretrained_gene_list)
            self.emb = nn.Parameter(torch.randn([num_genes, num_hidden], dtype=torch.float32) * 0.005)

        if fix_embedding:
            self.emb.requires_grad = False

        self.post_layer = nn.Sequential(
            create_activation(activation),
            create_norm(norm, num_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x, pe_input=None, input_gene_list=None, input_gene_idx=None):
        assert pe_input is None  # FIX: deprecate pe_input

        if input_gene_idx is not None:
            gene_idx = input_gene_idx
        elif input_gene_list is not None:
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError(
                    'The input gene size is not the same as the pretrained gene list. '
                    'Please provide the input gene list.',
                )
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)

        feat = F.embedding(gene_idx, self.emb)
        out = torch.sparse.mm(x, feat)
        out = self.post_layer(out)

        return out, gene_idx
