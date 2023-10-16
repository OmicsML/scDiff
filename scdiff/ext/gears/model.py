import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
from torch_geometric.utils import scatter


class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)


class GEARS_Model(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene/globel perturbation embedding dictionary lookup
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))


        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')

        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                               hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')

    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            # get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))
                                                 ).repeat(num_graphs, ).to(self.args['device']))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))
                                                    ).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            # get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))

            # augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            # add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                # in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            # apply the first MLP
            base_emb = self.transform(base_emb)
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis=2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)

            # uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)

            return torch.stack(out)


class GEARS_Conditioner(torch.nn.Module):
    def __init__(self, num_perts, out_dim, hidden_size=64, num_go_gnn_layers=1,
                 mlp_layers=2, enable_inference_cache=True, mode="single"):
        super().__init__()

        assert mlp_layers >= 1, f"GEARS MLP layers must be greater than 1, got {mlp_layers}"
        assert mode in ("single", "parallel", "sequential", "mlpparallel", "mlpsequential"), f"Unknown mode {mode!r}"
        assert mode != "mlpsequential" or hidden_size == out_dim, "mlpsequential requires equal hidden and out dim"
        self.mode = mode

        self.num_perts = num_perts
        self.num_layers = num_go_gnn_layers

        # NOTE: we use the first (index 0) embedding as the control embedding
        self.pert_emb = nn.Embedding(num_perts + 1, hidden_size, max_norm=True)

        self.pert_fuse = nn.ModuleList([
            MLP([*([hidden_size] * mlp_layers), out_dim], last_layer_act='ReLU')
            for _ in range(1 if mode == "single" else self.num_layers)])

        self.sim_layers = nn.ModuleList([SGConv(hidden_size, hidden_size, 1)
                                         for _ in range(self.num_layers)])

        self.enable_inference_cache = enable_inference_cache
        self.clear_emb_cache()

    @property
    def use_cache(self) -> bool:
        return self.enable_inference_cache and not self.training

    @property
    def cached_emb(self):
        return self._cached_emb

    def clear_emb_cache(self):
        self._cached_emb = None

    def get_pert_global_emb(self, aug_graph):
        # augment global perturbation embedding with GNN
        G_sim = aug_graph["G_go"]
        G_sim_weight = aug_graph["G_go_weight"]

        pert_global_emb = self.pert_emb.weight
        ctrl_emb = self.pert_emb.weight[0:1]

        pert_global_emb_list = [pert_global_emb]
        for idx, layer in enumerate(self.sim_layers):
            pert_emb = layer(pert_global_emb_list[0 if self.mode == "parallel" else -1][1:], G_sim, G_sim_weight)
            pert_emb = pert_emb if idx == self.num_layers - 1 else pert_emb.relu()
            pert_global_emb_list.append(torch.cat([ctrl_emb, pert_emb], dim=0))

        return pert_global_emb_list[1:]  # skip base embedings

    def forward(self, pert_idx, aug_graph):
        """
        Forward pass of the model
        """
        # NOTE: We use the first embedding as the control embedding and shift
        # everything else by an index of one. We only assign control embedding
        # when all perturbations of the current samlpe are controls.
        pert_index = []
        for idx, i in enumerate(pert_idx.tolist()):
            if all(map(lambda x: x == -1, i)):  # all control -> control
                pert_index.append([idx, 0])
            else:
                pert_index.extend([[idx, j + 1] for j in i if j != -1])
        pert_index = torch.tensor(pert_index, device=pert_idx.device).T

        if self.use_cache:
            # At inference (sampling) time, the global perturbation condition
            # embeddings do not change, so we dont need to recalculate
            if self.cached_emb is None:
                self._cached_emb = [i.detach() for i in self.get_pert_global_emb(aug_graph)]
            pert_global_emb = self.cached_emb

        else:
            self.clear_emb_cache()
            pert_global_emb = self.get_pert_global_emb(aug_graph)

        if self.mode == "single":
            emb = scatter(pert_global_emb[-1][pert_index[1]], pert_index[0], dim=0)
            out = self.pert_fuse[0](emb)
        elif self.mode in ("parallel", "sequential"):
            out = []
            for pert_emb, pert_fuse in zip(pert_global_emb, self.pert_fuse):
                emb = scatter(pert_emb[pert_index[1]], pert_index[0], dim=0)
                out.append(pert_fuse(emb))
        elif self.mode in ("mlpparallel", "mlpsequential"):
            out = [scatter(pert_global_emb[-1][pert_index[1]], pert_index[0], dim=0)]
            for pert_fuse in self.pert_fuse:
                out.append(pert_fuse(out[0 if self.mode == "mlpparallel" else -1]))
            out = out[:-1]
        else:
            raise ValueError(f"Unknown mode {self.mode!r}, should have been caught earlier.")

        return out
