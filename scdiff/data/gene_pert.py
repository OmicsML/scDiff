import os.path as osp
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from scdiff.data.base import TargetDataset
from scdiff.ext.gears import PertData
from scdiff.ext.gears.utils import get_similarity_network, GeneSimNetwork


GO_FILE = 'go_essential_all.csv'
GENE2GO_FILE = 'gene2go_all.pkl'
ESSENTIAL_GENES_FILE = 'essential_all_data_pert_genes.pkl'
DATASETS = {
    'adamson': 'adamson/perturb_processed.h5ad',
    'dixit': 'dixit/perturb_processed.h5ad',
    'norman': 'norman/perturb_processed.h5ad',
}
SPLIT_TYPES = {
    'adamson': ['simulation', 'single'],
    'dixit': ['simulation', 'single'],
    'norman': ['simulation', 'combo_seen0', 'combo_seen1', 'combo_seen2'],
}


def extend_pert_list(x, extend_key):
    if len(x) == 1 and x[0] == extend_key:
        return [extend_key, extend_key]
    else:
        return x


class GenePerturbationBase(ABC):
    def __init__(self, datadir='./data', dataset='adamson', test_cell_types=None, save_processed=False,
                 post_cond_flag=True, ignore_cond_flag=False, pretrained_gene_list=None, split_type='simulation',
                 pretrained_gene_list_path=None, subset_flag=False, seed=1, coexpress_threshold=0.4,
                 num_similar_genes_go_graph=20):
        assert dataset in ['adamson', 'dixit', 'norman']
        assert split_type in SPLIT_TYPES[dataset]
        self.celltype_key = 'cell_type'
        self.batch_key = 'batch'
        self.pert_key = 'condition'
        self.ctrl_key = 'control'
        self.ctrl_value = 'ctrl'
        self.datadir = datadir
        self.dataset = dataset
        self.split_type = split_type
        self.seed = seed
        self.return_raw = False
        self.subset_flag = subset_flag
        self.save_processed = save_processed
        self.post_cond_flag = post_cond_flag
        self.test_cell_types = test_cell_types
        self.ignore_cond_flag = ignore_cond_flag
        self.coexpress_threshold = coexpress_threshold
        self.num_similar_genes_go_graph = num_similar_genes_go_graph
        if pretrained_gene_list is None and pretrained_gene_list_path is not None:
            assert pretrained_gene_list_path.endswith('npy')
            pretrained_gene_list = np.load(pretrained_gene_list_path, allow_pickle=True)
        self.pretrained_gene_list = pretrained_gene_list
        self._read_and_split(datadir=datadir, dataset=dataset, split_type=split_type)
        self._init_condiitons()
        self._prepare()

    def _read_and_split(self, datadir='./data', dataset='adamson', split_type='single'):
        self.pert_data = PertData(datadir)
        self.pert_data.load(data_path=osp.join(datadir, dataset))

        self.pert_data.prepare_split(split=split_type, seed=self.seed)
        self.cell_graphs = self.pert_data.get_cell_graphs()  # only y is needed, x contains control cells

        self.adata = self.pert_data.adata
        self.adata.obs[self.batch_key] = "null"  # NOTE: these datasets do not contain batch info
        self.adata.obs["split"] = "na"
        for split_name, split_conds in self.pert_data.set2conditions.items():
            self.adata.obs.loc[self.adata.obs["condition"].isin(split_conds), "split"] = split_name

        self.pert_list = self.pert_data.pert_names.tolist()
        self.num_perts = len(self.pert_list)
        self.split = self.pert_data.split
        self.train_gene_set_size = self.pert_data.train_gene_set_size
        self.set2conditions = self.pert_data.set2conditions
        self.default_pert_graph = self.pert_data.default_pert_graph
        self.node_map_pert = self.pert_data.node_map_pert

    def _init_condiitons(self):
        # all the datasets only have one cell type and one batch
        self.celltype_enc = LabelEncoder()
        self.celltype_enc.classes_ = np.array(
            sorted(self.adata.obs[self.celltype_key].astype(str).unique())
        )  # NOTE: these datasets only have one cell type, so do not need to add null

        self.batch_enc = LabelEncoder()
        self.batch_enc.classes_ = np.array(
            sorted(self.adata.obs[self.batch_key].astype(str).unique())
        )

        if self.post_cond_flag:
            self.cond_num_dict = {
                'cell_type': len(self.celltype_enc.classes_),
            }
            self.post_cond_num_dict = {'batch': len(self.batch_enc.classes_)}
        else:
            self.cond_num_dict = {
                'batch': len(self.batch_enc.classes_),
                'cell_type': len(self.celltype_enc.classes_),
            }
            self.post_cond_num_dict = None

    def _load(self):
        self.input_graphs = self.cell_graphs[self.SPLIT]  # train, val, test

        self.target = self.extras = None

        pert_idx_list = [data.pert_idx for data in self.input_graphs]
        max_num_pert = max(map(len, pert_idx_list))
        for i in pert_idx_list:  # pad with ctrl idx (-1) to ensure consistent dimension
            if len(i) < max_num_pert:
                i.extend([-1] * (max_num_pert - len(i)))
        self.pert_idx = torch.tensor(pert_idx_list, dtype=torch.long)

        if self.SPLIT != 'train':
            self.input = torch.cat([data.x for data in self.input_graphs], dim=1).T.contiguous()
            self.target = torch.cat([data.y for data in self.input_graphs], dim=0).contiguous()

            # XXX: convert full condition name to condition name (assumes one-to-one)
            fullcond_to_cond = defaultdict(set)
            for fullcond, cond in self.adata.obs[["condition_name", "condition"]].values:
                fullcond_to_cond[fullcond].add(cond)
            len_dict = {i: len(j) for i, j in fullcond_to_cond.items()}
            assert all(i == 1 for i in len_dict.values()), f"Conditions not one-to-one: {len_dict}"
            fullcond_to_cond = {i: j.pop() for i, j in fullcond_to_cond.items()}

            gene_to_idx = {j: i for i, j in enumerate(self.adata.var.index.tolist())}

            gene_rank_dict, ndde20_dict = {}, {}
            for fullname, name in fullcond_to_cond.items():
                pert_idx = self.pert_data.get_pert_idx(name)
                assert all(isinstance(i, (int, np.int64)) for i in pert_idx), f"{pert_idx=!r}"

                gene_order = self.adata.uns["rank_genes_groups_cov_all"][fullname]
                gene_rank_dict[tuple(pert_idx)] = [gene_to_idx[i] for i in gene_order.tolist()]

                ndde20 = self.adata.uns["top_non_dropout_de_20"][fullname]
                ndde20_dict[tuple(pert_idx)] = [gene_to_idx[i] for i in ndde20.tolist()]

            self.extras = {"rank_genes_groups_cov_all_idx_dict": gene_rank_dict,
                           "top_non_dropout_de_20": ndde20_dict}
        else:
            self.input = torch.cat([data.y for data in self.input_graphs], dim=0).contiguous()

        self.gene_names = self.adata.var.index.tolist()
        self.celltype = self.celltype_enc.transform(self.adata.obs[self.celltype_key].astype(str))
        self.batch = self.batch_enc.transform(self.adata.obs[self.batch_key].astype(str))
        self.cond = {
            'batch': torch.tensor(self.batch).float(),
            'cell_type': torch.tensor(self.celltype).float(),
            'pert': self.pert_idx,
        }

        # calculating gene ontology similarity graph
        edge_list = get_similarity_network(network_type='go',
                                           adata=self.adata,
                                           threshold=self.coexpress_threshold,
                                           k=self.num_similar_genes_go_graph,
                                           pert_list=self.pert_list,
                                           data_path=self.datadir,
                                           data_name=self.dataset,
                                           split=self.split_type, seed=self.seed,
                                           train_gene_set_size=self.train_gene_set_size,
                                           set2conditions=self.set2conditions,
                                           default_pert_graph=self.default_pert_graph)

        sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map=self.node_map_pert)
        self.G_go = sim_network.edge_index
        self.G_go_weight = sim_network.edge_weight

        if self.pretrained_gene_list is not None:
            pretrained_gene_index = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))
            self.input_gene_idx = torch.tensor([
                pretrained_gene_index[o] for o in self.gene_list
                if o in pretrained_gene_index
            ]).long()

    @abstractmethod
    def _prepare(self):
        ...


class GenePerturbationTrain(TargetDataset, GenePerturbationBase):
    SPLIT = "train"
    TARGET_KEY = "gene_pert_target"


class GenePerturbationValidation(TargetDataset, GenePerturbationBase):
    SPLIT = "val"
    TARGET_KEY = "gene_pert_target"


class GenePerturbationTest(TargetDataset, GenePerturbationBase):
    SPLIT = "test"
    TARGET_KEY = "gene_pert_target"


# class GenePerturbationGeneration(TargetDataset, GenePerturbationBase):
#     ...
