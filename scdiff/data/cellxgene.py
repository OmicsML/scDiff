import os.path as osp
from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder

from scdiff.data.base import MaskDataset, SplitDataset, GenerationDataset
from scdiff.modules.text import SimpleEmbeddingGenerator, CLEmbeddingGenerator
from scdiff.utils.data import mask_data_offline, dict_to_list_of_tuples
from scdiff.utils.misc import default


DATASETS = {
    'HLCA': {
        'fname': 'HLCA_zstd_processed.h5ad',
        'batch_key': 'batch'
    },
    'HLCA_sub': {
        'fname': 'HLCA_zstd_sub.h5ad',
        'batch_key': 'batch'
    },
    'HLCA_naw': {
        'fname': 'HLCA_zstd_Nawijin_GRO-09.h5ad',
        'batch_key': 'batch'
    },
    'Immune': {
        'fname': 'Immune_processed.h5ad',
        'batch_key': 'donor_id'
    },
    'Immune_sub': {
        'fname': 'Immune_sub.h5ad',
        'batch_key': 'donor_id'
    },
    'Liver': {
        'fname': 'Liver_processed.h5ad',
        'batch_key': 'donor_id',
        'raw_path': '/egr/research-dselab/shared/dance/cellxgene/datasets/human/a43aa46b-bd16-47fe-bc3e-19a052624e79.h5ad',
    },
    'Brain': {
        'fname': 'Brain_processed.h5ad',
        'batch_key': 'donor_id',
        'raw_path': '/egr/research-dselab/shared/dance/cellxgene/datasets/human_manual_download_2023-09-27/c05e6940-729c-47bd-a2a6-6ce3730c4919.h5ad',
    },
}


class CellXGeneBase(ABC):
    N_PRESERVE = 5
    LIB_SIZE_FACTOR = 1e4
    TEXT_COND_KEYS = {
        'simple': ['cell_type', 'sex'],
        'CL': ['cell_type_ontology_term_id']
    }
    GENE_LIST_FNAME='HLCA_gene_list.npy'

    def __init__(self, datadir='./data', seed=10, normalize=True, n_genes=None, dataset='HLCA_sub',
                 save_processed=False, splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}, split_strategy='random',
                 subsample_ratio=None, force_split=False, post_cond_flag=False, return_raw=False, rescale=False,
                 text_cond_flag=False, text_emb_model='michiyasunaga/BioLinkBERT-large', text_emb_type='CL',
                 pretrained_gene_list_fname=None, text_null_flag=False, reduce_type='full',
                 test_cell_types=None, train_cell_types=None, overwrite_test=False, n_preserve=None,
                 disgard_flag=True, disgard_threshold=10):
        self.batch_key = DATASETS[dataset]['batch_key']
        self.default_cond_key_dict = dict(batch=self.batch_key, cell_type="cell_type")
        self.default_post_cond_key_dict = dict(batch=self.batch_key)
        self.seed = seed
        self.datadir = datadir
        self.rescale = rescale
        self.normalize = normalize
        self.return_raw = return_raw
        self.save_processed = save_processed
        self.post_cond_flag = post_cond_flag
        self.text_cond_flag = text_cond_flag
        self.text_emb_model = text_emb_model
        self.text_emb_type = text_emb_type
        self.text_null_flag = text_null_flag
        self.reduce_type = reduce_type
        self.pretrained_gene_list_fname = pretrained_gene_list_fname
        self.n_preserve = default(n_preserve, self.N_PRESERVE)
        self.dataset = dataset
        fname = DATASETS[dataset]['fname']
        self._read(datadir=datadir, normalize=normalize, rescale=rescale, n_genes=n_genes, fname=fname)

        self.disgard_threshold = disgard_threshold
        if disgard_flag:
            cell_type_counts = self.adata.obs['cell_type'].value_counts()
            disgard_ct = cell_type_counts[cell_type_counts <= disgard_threshold].index.tolist()
            self.adata = self.adata[~self.adata.obs['cell_type'].isin(disgard_ct)]

        test_cell_types = test_cell_types.split(' | ') if test_cell_types is not None else None
        train_cell_types = train_cell_types.split(' | ') if train_cell_types is not None else None
        if train_cell_types is not None and overwrite_test:
            self.target_cell_types = list(set(self.adata.obs['cell_type']) - set(train_cell_types))
        else:
            self.target_cell_types = test_cell_types
        self.target_cell_types = default(self.target_cell_types, list(set(self.adata.obs['cell_type'])))
        self._prepare_split(splits=splits, split_strategy=split_strategy, seed=seed, fname=fname,
                            subsample_ratio=subsample_ratio, force_split=force_split)
        self._init_condiitons()
        self._prepare()

    def _read(self, datadir='./data', normalize=True, rescale=False, n_genes=None, fname='HLCA_zstd_sub.h5ad'):
        if osp.exists(osp.join(datadir, fname)) and fname.endswith('.h5ad'):
            self.adata = ad.read_h5ad(osp.join(datadir, fname))
        else:
            self.adata = ad.read_h5ad(DATASETS[self.dataset]['raw_path'])  # currently only for Brain and Liver
            self.adata.var = self.adata.var.reset_index().set_index('feature_name')
            self.adata.var_names_make_unique()
            self.adata.X = self.adata.raw.X.copy()
            sc.pp.filter_genes(self.adata, min_cells=1)
            sc.pp.filter_cells(self.adata, min_genes=1)
            self.adata.layers['counts'] = self.adata.X.copy()
            if self.pretrained_gene_list_fname is not None:
                assert self.pretrained_gene_list_fname.endswith('npy')
                pretrained_gene_list_path = osp.join(datadir, self.pretrained_gene_list_fname)
                pretrained_gene_list = np.load(pretrained_gene_list_path, allow_pickle=True)
                self.gene_list = self.adata.var.index.to_list()
                self.gene_list = [x for x in self.gene_list if x in pretrained_gene_list]
                self.adata = self.adata[:, self.gene_list]
            if normalize:
                sc.pp.normalize_total(self.adata, target_sum=self.LIB_SIZE_FACTOR, key_added='library_size')
                sc.pp.log1p(self.adata)
            if rescale:
                self.adata.X /= np.log(self.LIB_SIZE_FACTOR + 1)
            if n_genes is not None:
                sc.pp.highly_variable_genes(self.adata, n_top_genes=n_genes)
        

    def _prepare_split(self, splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}, split_strategy='random', seed=10,
                       fname='HLCA_zstd_sub.h5ad', subsample_ratio=None, force_split=False):
        if split_strategy == 'reduce':  # No validation
            assert self.reduce_type in ['full', 'partial']
            if (
                (
                    self.reduce_type == 'full'
                    and False
                )
                or (
                    self.reduce_type == 'partial'
                    and f'split_partial_{self.n_preserve}' in self.adata.uns.keys()
                    and 'preserve_idx' in self.adata.uns[f'split_partial_{self.n_preserve}'].keys()
                    and all(
                        x in sorted(self.adata.uns[f'split_partial_{self.n_preserve}']['preserve_idx'])
                        for x in sorted(self.target_cell_types)
                    )
                )
            ):
                pass
            else:
                if self.reduce_type == 'full':
                    target_cell_types_flag = self.adata.obs['cell_type'].isin(self.target_cell_types).values
                    if 'split_full' in self.adata.obs.columns:
                        del self.adata.obs['split_full']
                    self.adata.obs['split_full'] = 'train'
                    self.adata.obs['split_full'][target_cell_types_flag] = 'test'
                elif self.reduce_type == 'partial':  # save a separate file
                    for n_preserve in range(1, self.disgard_threshold + 1):
                        self.adata.uns[f'split_partial_{n_preserve}'] = {
                            'reduce_type': self.reduce_type,
                            'n_preserve': n_preserve
                        }
                        rng = np.random.default_rng(seed)
                        preserve_idx = {}
                        for ct in self.target_cell_types:
                            test_cell_types_idx = np.where(self.adata.obs['cell_type'] == ct)[0]
                            preserve_idx[ct] = rng.choice(test_cell_types_idx, n_preserve, replace=False).tolist()
                            # self.adata.obs['split'][preserve_idx[ct]] = 'train'
                        self.adata.uns[f'split_partial_{n_preserve}'].update({
                            'preserve_idx': preserve_idx,
                        })
                if self.save_processed and fname is not None:
                    print(f"Saving processed file to {osp.join(self.datadir, fname)}")
                    self.adata.write_h5ad(osp.join(self.datadir, fname), compression='gzip')
        else:
            if (
                ('split' in self.adata.obs.columns)
                and sorted(splits) == sorted(np.unique(self.adata.obs['split']))
                and not force_split and split_strategy != 'reduce'
            ):
                pass
            else:
                if subsample_ratio is not None:
                    assert 0 < subsample_ratio <= 1
                    obs = self.adata.obs
                    obs_sub = obs.groupby(self.batch_key, group_keys=False).apply(
                        lambda x: x.sample(int(len(x) * subsample_ratio), random_state=seed))
                    self.adata = self.adata[obs_sub.index]
                assert 'train' in splits and 'valid' in splits
                assert sum([splits[k] for k in splits.keys()]) == 1
                assert split_strategy in ['random', 'cell_type', 'batch']
                self.adata.obs['split'] = 'train'
                if split_strategy == 'random':
                    rng = np.random.default_rng(seed)
                    N = len(self.adata)
                    perm = rng.permutation(range(N))
                    self.adata.obs['split'][
                        perm[int(N * splits['train']):int(N * (splits['train'] + splits['valid']))]] = 'valid'
                    if 'test' in splits:
                        self.adata.obs['split'][perm[int(N * (splits['train'] + splits['valid'])):]] = 'test'
                else:
                    group_key = self.celltype_key if split_strategy == 'cell_type' else self.batch_key
                    obs = self.adata.obs
                    obs_valid = obs.groupby(group_key, group_keys=False).apply(
                        lambda x: x.sample(int(len(x) * splits['valid']), random_state=seed))
                    self.adata.obs['split'][obs_valid.index] = 'valid'
                    if 'test' in splits:
                        obs = obs[~obs.index.isin(obs_valid.index)]
                        test_ratio = splits['test'] / (splits['train'] + splits['test'])
                        obs_test = obs.groupby(group_key, group_keys=False).apply(
                            lambda x: x.sample(int(len(x) * test_ratio), random_state=seed))
                        self.adata.obs['split'][obs_test.index] = 'test'
                if self.save_processed and fname is not None:
                    print(f"Saving processed file to {osp.join(self.datadir, fname)}")
                    self.adata.write_h5ad(osp.join(self.datadir, fname), compression='gzip')

    def _init_condiitons(self):
        self.le_dict = {}
        for key, raw_key in self.default_cond_key_dict.items():
            self.le_dict[key] = LabelEncoder()
            self.le_dict[key].classes_ = np.array(
                ["null"] + sorted(self.adata.obs[raw_key].astype(str).unique())
            )

        if self.post_cond_flag:
            cond_keys = list(set(self.default_cond_key_dict) - set(self.default_post_cond_key_dict))
            self.cond_num_dict = {
                k: len(self.le_dict[k].classes_)
                for k in cond_keys
            }
            self.post_cond_num_dict = {
                k: len(self.le_dict[k].classes_)
                for k in self.default_post_cond_key_dict
            }
        else:
            self.cond_num_dict = {
                k: len(self.le_dict[k].classes_)
                for k in self.default_cond_key_dict
            }
            self.post_cond_num_dict = None

        if self.text_cond_flag:
            text_cond_dict = {
                k: self.adata.obs[k].values.tolist()
                for k in self.TEXT_COND_KEYS[self.text_emb_type]
            }
            self.unique_cond_dict = pd.DataFrame(text_cond_dict).drop_duplicates().to_dict(orient='list')
            self.unique_cond_list = dict_to_list_of_tuples(self.unique_cond_dict)
            if self.text_null_flag:
                self.unique_cond_list = ["null"] + self.unique_cond_list
            if self.text_emb_type == 'simple':
                self.text_emb_generator = SimpleEmbeddingGenerator(
                    unique_cond_list=self.unique_cond_list,
                    model=self.text_emb_model,
                    sep=', ',
                )
            elif self.text_emb_type == 'CL':
                self.text_emb_generator = CLEmbeddingGenerator(
                    unique_cond_list=self.unique_cond_list,
                    model=self.text_emb_model,
                    savedir=f'{self.datadir}/ontology_resources',
                    null_flag=self.text_null_flag,
                    data_emb_fname=f'{self.dataset}-cl-emb.pt',
                )
            else:
                raise NotImplementedError

            self.le_dict['text'] = LabelEncoder()
            self.le_dict['text'].classes_ = np.array(self.unique_cond_list)

    def _load(self):
        self.input = torch.tensor(self.adata.X.A if self.normalize else self.adata.layers['counts'].A).float()
        if self.normalize and self.return_raw:
            self.raw_input = self.adata.layers['counts'].A

        self.cond = {
            key: torch.tensor(self.le_dict[key].transform(self.adata.obs[raw_key].astype(str))).long()
            for key, raw_key in self.default_cond_key_dict.items()
        }

        if self.text_cond_flag:
            self.le_dict.pop('cell_type')
            self.cond_num_dict.pop('cell_type')
            self.cond.pop('cell_type')
            text_cond_dict = {
                k: self.adata.obs[k].values.tolist()
                for k in self.TEXT_COND_KEYS[self.text_emb_type]
            }
            text_cond_list = self.text_emb_generator.dict_to_list_of_tuples(text_cond_dict)
            self.cond['text'] = torch.tensor(self.le_dict['text'].transform(text_cond_list)).long()

    @abstractmethod
    def _prepare(self):
        ...


class CellXGeneTrain(SplitDataset, CellXGeneBase):
    SPLIT = "train"


class CellXGeneValidation(SplitDataset, CellXGeneBase):
    SPLIT = "valid"


class CellXGeneNotTest(SplitDataset, CellXGeneBase):
    SPLIT = "train"

    def _prepare(self):
        if 'test' in np.unique(self.adata.obs['split']):
            self.adata = self.adata[self.adata.obs['split'] != 'test']
        self._load()


class CellXGeneTest(SplitDataset, CellXGeneBase):
    SPLIT = "test"


class CellXGeneMaskedTest(MaskDataset, CellXGeneBase):
    SPLIT = "test"

    def __init__(self, *args, mask_strategy="random", mask_rate=0.2, mask_offline=True, **kwargs):
        self.mask_strategy = mask_strategy
        self.mask_rate = mask_rate
        self.mask_offline = True
        super().__init__(*args, **kwargs)

    def _prepare(self):
        if 'test' in np.unique(self.adata.obs['split']):
            self.adata = self.adata[self.adata.obs['split'] == 'test']
        self._load()

        train_mask, valid_mask, test_mask = mask_data_offline(self.adata, self.mask_strategy,
                                                              test_mask_rate=self.mask_rate, seed=self.seed)
        self.adata.layers['train_mask'] = train_mask
        self.adata.layers['valid_mask'] = valid_mask
        self.adata.layers['test_mask'] = test_mask
        self.mask = self.adata.layers[f'{self.SPLIT}_mask']

        if self.SPLIT == 'test':
            self.target = self.input.clone()
        if self.mask_offline:
            self.input[~train_mask] = 0


class CellXGeneFewShotTrain(SplitDataset, CellXGeneBase):
    SPLIT = "train"

    def _prepare(self):
        preserve_idx = np.concatenate([
            v for k, v in self.adata.uns[f'split_partial_{self.n_preserve}']['preserve_idx'].items()
            if k in self.target_cell_types
        ])
        train_idx = np.where(~self.adata.obs['cell_type'].isin(self.target_cell_types))[0]
        train_idx = sorted(np.concatenate([train_idx, preserve_idx]))
        self.adata = self.adata[train_idx]
        self._load()


class CellXGeneFewShotFinetune(SplitDataset, CellXGeneBase):
    SPLIT = "train"

    def _prepare(self):
        preserve_idx = np.concatenate([
            v for k, v in self.adata.uns[f'split_partial_{self.n_preserve}']['preserve_idx'].items()
            if k in self.target_cell_types
        ])
        self.adata = self.adata[preserve_idx]
        self._load()


class CellXGenePretrainFilterByCounts(SplitDataset, CellXGeneBase):
    SPLIT = "train"

    def __init__(self, *args, threshold=1000, choice='top', **kwargs):
        assert choice in ['top', 'bottom']  # 'random'
        self.threshold = threshold
        self.choice = choice
        super().__init__(*args, **kwargs)

    def _prepare(self):
        cell_type_counts = self.adata.obs['cell_type'].value_counts()
        if self.choice == 'top':
            self.target_cell_types = cell_type_counts[cell_type_counts >= self.threshold].index.tolist()
        elif self.choice == 'bottom':
            self.target_cell_types = cell_type_counts[cell_type_counts < self.threshold].index.tolist()
        self.adata = self.adata[self.adata.obs['cell_type'].isin(self.target_cell_types)]
        self._load()


class CellXGeneTargetCellTypeTest(SplitDataset, CellXGeneBase):
    SPLIT = "test"

    def _prepare(self):
        preserve_idx = np.concatenate([
            v for k, v in self.adata.uns[f'split_partial_{self.n_preserve}']['preserve_idx'].items()
            if k in self.target_cell_types
        ])
        rest_idx = np.setdiff1d(np.arange(len(self.adata)), preserve_idx)
        self.adata = self.adata[rest_idx]
        self.adata = self.adata[self.adata.obs['cell_type'].isin(self.target_cell_types)]
        self._load()


class CellXGeneTopKFewShotTrain(SplitDataset, CellXGeneBase):
    SPLIT = "train"

    def __init__(self, *args, threshold=1000, choice='top', num_cell_types=3, **kwargs):
        assert choice in ['top', 'bottom']
        self.choice = choice
        self.threshold = threshold
        self.num_cell_types = num_cell_types
        super().__init__(*args, **kwargs)

    def _prepare(self):
        cell_type_counts = self.adata.obs['cell_type'].value_counts()
        if self.choice == 'top':  # cts >= thresholds are used for pretrain
            cell_types_candidates = cell_type_counts[cell_type_counts < self.threshold].index.tolist()
            self.target_cell_types = cell_types_candidates[:self.num_cell_types]

        elif self.choice == 'bottom':  # cts < thresholds are used for pretrain
            cell_types_candidates = cell_type_counts[cell_type_counts >= self.threshold].index.tolist()
            self.target_cell_types = cell_types_candidates[-self.num_cell_types:]

        preserve_idx = np.concatenate([
            v for k, v in self.adata.uns[f'split_partial_{self.n_preserve}']['preserve_idx'].items()
            if k in self.target_cell_types
        ])
        self.adata = self.adata[preserve_idx]
        self._load()


class CellXGeneTopKFewShotTest(SplitDataset, CellXGeneBase):
    SPLIT = "train"

    def __init__(self, *args, threshold=1000, choice='top', num_cell_types=3, **kwargs):
        assert choice in ['top', 'bottom']
        self.choice = choice
        self.threshold = threshold
        self.num_cell_types = num_cell_types
        super().__init__(*args, **kwargs)

    def _prepare(self):
        cell_type_counts = self.adata.obs['cell_type'].value_counts()
        if self.choice == 'top':  # cts >= thresholds are used for pretrain
            cell_types_candidates = cell_type_counts[cell_type_counts < self.threshold].index.tolist()
            self.target_cell_types = cell_types_candidates[:self.num_cell_types]

        elif self.choice == 'bottom':  # cts < thresholds are used for pretrain
            cell_types_candidates = cell_type_counts[cell_type_counts >= self.threshold].index.tolist()
            self.target_cell_types = cell_types_candidates[-self.num_cell_types:]

        preserve_idx = np.concatenate([
            v for k, v in self.adata.uns[f'split_partial_{self.n_preserve}']['preserve_idx'].items()
            if k in self.target_cell_types
        ])
        rest_idx = np.setdiff1d(np.arange(len(self.adata)), preserve_idx)
        self.adata = self.adata[rest_idx]
        self.adata = self.adata[self.adata.obs['cell_type'].isin(self.target_cell_types)]
        self._load()


class CellXGeneGeneration(GenerationDataset, CellXGeneBase):
    ...
