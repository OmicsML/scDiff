# Jurkat: https://www.10xgenomics.com/resources/datasets/jurkat-cells-1-standard-1-1-0
# 293T: https://www.10xgenomics.com/resources/datasets/293-t-cells-1-standard-1-1-0
# PBMC_1K_URL = "https://ndownloader.figshare.com/files/36088667"

from abc import ABC, abstractmethod

import numpy as np
import anndata as ad
import scanpy as sc
import os.path as osp
import torch
from sklearn.preprocessing import LabelEncoder

from scdiff.data.base import MaskDataset
from scdiff.utils.data import mask_data_offline


class DenoisingBase(ABC):
    def __init__(self, datadir='./data', seed=10, normalize=True, n_genes=None, dataset='Jurkat', fname='Jurkat_processed.h5ad', 
                 save_processed=True, splits={'train':0.8, 'valid':0.1, 'test': 0.1}, subsample_ratio=None, mask_type='mar',
                 force_split=False, post_cond_flag=False, return_raw=False, mask_strategy='none_zero', mask_offline=True, 
                 pretrained_gene_list=None, pretrained_gene_list_path=None):
        assert dataset in ['PBMC1K', 'Jurkat', '293T']
        self.celltype_key = 'cell_type'
        self.batch_key = 'batch'
        self.datadir = datadir
        self.normalize = normalize
        self.return_raw = return_raw
        self.mask_offline = mask_offline
        self.save_processed = save_processed
        self.post_cond_flag = post_cond_flag
        if pretrained_gene_list is None and pretrained_gene_list_path is not None:
            assert pretrained_gene_list_path.endswith('npy')
            pretrained_gene_list = np.load(pretrained_gene_list_path, allow_pickle=True)
        self.pretrained_gene_list = pretrained_gene_list 
        self._read(datadir=datadir, normalize=normalize, n_genes=n_genes, dataset=dataset, fname=fname)
        self._prepare_split(splits=splits, seed=seed, fname=fname, mask_strategy=mask_strategy, mask_type=mask_type,
                            subsample_ratio=subsample_ratio, force_split=force_split)
        self._init_condiitons()
        self._prepare()

    def _read(self, datadir='./data', normalize=True, n_genes=None, dataset='Jurkat', fname='Jurkat_processed.h5ad'):
        if osp.exists(osp.join(datadir, fname)) and fname.endswith('.h5ad'):
            self.adata = ad.read_h5ad(osp.join(datadir, fname))
        else:
            if osp.exists(osp.join(datadir, f'{dataset}.h5ad')):
                self.adata = ad.read_h5ad(osp.join(datadir, f'{dataset}.h5ad'))
            else:
                pass #TODO: add download instructions

            self.adata.var_names_make_unique()
            self.adata.obs[self.celltype_key] = 'unknown'
            self.adata.obs[self.batch_key] = 0
            sc.pp.filter_genes(self.adata, min_cells=1)
            sc.pp.filter_cells(self.adata, min_genes=1)
            self.adata.layers['counts'] = self.adata.X.copy()
            if normalize:
                sc.pp.normalize_total(self.adata, target_sum=1e4, key_added='library_size')
                sc.pp.log1p(self.adata)
            if n_genes is not None:
                sc.pp.highly_variable_genes(self.adata, n_top_genes=n_genes)

        if self.pretrained_gene_list is not None:
            self.gene_list = self.adata.var.index.to_list()
            self.gene_list = [x for x in self.gene_list if x in self.pretrained_gene_list]
            self.adata = self.adata[:, self.gene_list]

    def _prepare_split(self, splits={'train':0.8, 'valid':0.1, 'test': 0.1}, mask_strategy='random', mask_type='mar',
                       seed=10, fname='Denoising_processed.h5ad', subsample_ratio=None, force_split=False):
        # if 'split' in self.adata.obs.columns and sorted(splits) == sorted(np.unique(self.adata.obs['split'])) and not force_split:
        if (
            force_split
            or not (
                ('split' in self.adata.obs.columns and sorted(splits) == sorted(np.unique(self.adata.obs['split'])))
                or all(i in self.adata.layers for i in ('train_mask', 'valid_mask', 'test_mask'))
            )
        ):
            if subsample_ratio is not None:
                assert 0 < subsample_ratio <= 1
                obs = self.adata.obs
                obs_sub = obs.groupby(self.batch_key, group_keys=False).apply(lambda x: x.sample(int(len(x) * subsample_ratio), random_state=seed))
                self.adata = self.adata[obs_sub.index]
            assert 'train' in splits and 'test' in splits
            assert sum([splits[k] for k in splits.keys()]) == 1
            self.adata.obs['split'] = 'train'

            # assert split_strategy in ['random', 'cell_type', 'batch']
            # if split_strategy == 'random':
            #     rng = np.random.default_rng(seed)
            #     N = len(self.adata)
            #     perm = rng.permutation(range(N))
            #     self.adata.obs['split'][perm[int(N * splits['train']):int(N * (splits['train'] + splits['valid']))]] = 'valid'
            #     if 'test' in splits:
            #         self.adata.obs['split'][perm[int(N * (splits['train'] + splits['valid'])):]] = 'test'
            # else:
            #     group_key = self.celltype_key if split_strategy == 'cell_type' else self.batch_key
            #     obs = self.adata.obs
            #     obs_valid = obs.groupby(group_key, group_keys=False).apply(lambda x: x.sample(int(len(x) * splits['valid']), random_state=seed))
            #     self.adata.obs['split'][obs_valid.index] = 'valid'
            #     if 'test' in splits:
            #         obs = obs[~obs.index.isin(obs_valid.index)]
            #         test_ratio = splits['test'] / (splits['train'] + splits['test'])
            #         obs_test = obs.groupby(group_key, group_keys=False).apply(lambda x: x.sample(int(len(x) * test_ratio), random_state=seed))
            #         self.adata.obs['split'][obs_test.index] = 'test'

            train_mask, valid_mask, test_mask = mask_data_offline(self.adata, mask_strategy, mask_type, valid_mask_rate=splits['valid'], 
                                                                  test_mask_rate=splits['test'], seed=seed)
            self.adata.layers['train_mask'] = train_mask
            self.adata.layers['valid_mask'] = valid_mask
            self.adata.layers['test_mask'] = test_mask

            if self.save_processed and fname is not None:
                print(f"Saving processed file to {osp.join(self.datadir, fname)}")
                self.adata.write_h5ad(osp.join(self.datadir, fname), compression='gzip')

    def _init_condiitons(self):
        self.celltype_enc = LabelEncoder()
        self.celltype_enc.classes_ = np.array(
            ["null"] + sorted(self.adata.obs[self.celltype_key].astype(str).unique())
        )

        self.batch_enc = LabelEncoder()
        self.batch_enc.classes_ = np.array(
            ["null"] + sorted(self.adata.obs[self.batch_key].astype(str).unique())
        )

        if self.post_cond_flag:
            self.cond_num_dict = {'cell_type': len(self.celltype_enc.classes_)}
            self.post_cond_num_dict = {'batch': len(self.batch_enc.classes_)}
        else:
            self.cond_num_dict = {
                'batch': len(self.batch_enc.classes_),
                'cell_type': len(self.celltype_enc.classes_),
            }
            self.post_cond_num_dict = None

    def _load(self):
        self.input = torch.tensor(self.adata.X.A if self.normalize else self.adata.layers['counts'].A).float()
        if self.SPLIT == 'test':
            self.target = self.input.clone()
        self.mask = self.adata.layers[f'{self.SPLIT}_mask']
        if self.mask_offline:
            train_mask = self.adata.layers['train_mask']
            self.input[~train_mask] = 0.
        if self.normalize and self.return_raw:
            self.raw_input = self.adata.layers['counts'].A

        self.celltype = self.celltype_enc.transform(self.adata.obs[self.celltype_key].astype(str))
        self.batch = self.batch_enc.transform(self.adata.obs[self.batch_key].astype(str))
        self.cond = {
            'batch': torch.tensor(self.batch).float(),
            'cell_type': torch.tensor(self.celltype).float(),
        }

        if self.pretrained_gene_list is not None:
            pretrained_gene_index = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))
            self.input_gene_idx = torch.tensor([
                pretrained_gene_index[o] for o in self.gene_list
                if o in pretrained_gene_index
            ]).long()

    @abstractmethod
    def _prepare(self):
        ...


class DenoisingTrain(MaskDataset, DenoisingBase):
    SPLIT = "train"


class DenoisingValidation(MaskDataset, DenoisingBase):
    SPLIT = "valid"


class DenoisingTest(MaskDataset, DenoisingBase):
    SPLIT = "test"
