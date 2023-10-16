from abc import abstractmethod
from math import ceil
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset, IterableDataset

from scdiff.utils.data import get_candidate_conditions, dict_of_tensors_to_tensor


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class SCIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for single-cell data chainable
    """

    def __init__(self, num_cells=0, valid_ids=None, size=256):
        super().__init__()
        self.num_cells = num_cells
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class SplitDataset(Dataset):
    SPLIT: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, index):
        item_dict = {
            "input": self.input[index],
            "cond": {k: self.cond[k][index] for k in list(self.cond)},
        }
        if getattr(self, "normalize", False) and getattr(self, "return_raw", False):
            item_dict['raw_input'] = self.raw_input[index]
        if all(hasattr(self, i) for i in ('G_go', 'G_go_weight')):
            item_dict["aug_graph"] = dict(G_go=self.G_go, G_go_weight=self.G_go_weight)
        if getattr(self, "extras", None) is not None:
            item_dict["extras"] = self.extras
        return item_dict

    def _prepare(self):
        assert self.SPLIT is not None, "Please specify SPLIT class attr."
        if self.SPLIT in np.unique(self.adata.obs["split"]):
            self.adata = self.adata[self.adata.obs["split"] == self.SPLIT]
        self._load()


class FullDatasetMixin:
    SPLIT = "train"

    def __init__(self, *args, **kwargs):
        kwargs["splits"] = {"train": 1.0, "valid": 0.0, "test": 0.0}
        super().__init__(*args, **kwargs)


class MaskDataset(SplitDataset):
    SPLIT: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item_dict = {
            "input": self.input[index],
            "cond": {k: self.cond[k][index] for k in list(self.cond)},
            "mask": self.mask[index],
        }
        if self.SPLIT == 'test':
            item_dict['masked_target'] = self.target[index]
        if self.normalize and self.return_raw:
            item_dict['raw_input'] = self.raw_input[index]
        return item_dict


class TargetDataset(SplitDataset):
    SPLIT: Optional[str] = None
    TARGET_KEY = "target"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        item_dict = super().__getitem__(index)
        if self.target is not None:
            if len(self.target) == len(self.input):
                item_dict[self.TARGET_KEY] = self.target[index]
            else:
                item_dict[self.TARGET_KEY] = self.target
        if self.SPLIT != 'train' and hasattr(self, 'gene_names'):
            item_dict['gene_names'] = self.gene_names
        return item_dict


class GenerationDataset(IterableDataset):
    """Cell generation task dataset.

    The loading process of this dataset work as follows at a high level:

        1. Prepare context cells whose conditions match the specified
           context conditions.
        2. Iterate over this list of context cells in the specified
           ``context_batch_size``. If ``context_batch_size`` is ``None``, then
           use all context cells in one batch.
        3. For each context cell batch, generate a batch of
           ``generation_batch_size`` cells with query conditions. If
           ``generation_batch_size`` is ``None``, then use the same batch size
           as ``context_batch_size``.
        4. Step 2 and step 3 are repeated until either we meet the number of
           ``n_batches_to_generate``, or all valid context cells have been used
           for generation. The whole process is repeated again for ``n_trials``
           times if it is set to a number greater than one.

    Note:
        Only works with automatic batching turned off, i.e., batch_size=None.

    Args:
        use_split: Which split to use for constructing context cells.
        context_cond_candidates_cfg: Configuration for extracting candidate
            conditions for context cells, see
            :meth:`scdiff.utils.data.get_candidate_conditions` for more info.
        generation_cond_candidates_cfg: Same as the above, but for extracting
            candidate conditions for the generated cells.
        context_batch_size: Batch size for sampling context cells. If set to
            ``None``, then use all candidate context cells (i.e., full batch).
        generation_batch_size: Batch size for generating cells. If set to
            ``None``, then use the same bath size as ``context_batch_size``.
        dropout: Dropout applied to the context cells.
        n_trials: Number of times to generate using batches of context cells.
        n_batches_to_generate: Number of batches to sample per trial. If not
            set, then run until we have cycled through all valid context cells.
            A warning message will be displayed if we ran out of samlpes before
            hitting the specified number of ``n_batches_to_generate``.

    """

    def __init__(
        self,
        use_split: str = "train",
        context_cond_candidates_cfg: Optional[DictConfig] = None,
        generation_cond_candidates_cfg: Optional[DictConfig] = None,
        batch_size: Optional[int] = 4096,
        dropout: float = 0.0,
        n_trials: int = 1,
        n_batches_to_generate: int = 1,
        **kwargs,
    ):
        self.use_split = use_split
        self.context_cond_candidates_cfg = context_cond_candidates_cfg
        self.generation_cond_candidates_cfg = generation_cond_candidates_cfg
        self.batch_size = batch_size
        self.dropout = dropout
        self.n_trials = n_trials
        self.n_batches_to_generate = n_batches_to_generate
        super().__init__(**kwargs)

    def _prepare(self):
        if self.use_split != "all":
            assert self.use_split in np.unique(self.adata.obs["split"])
            self.adata = self.adata[self.adata.obs["split"] == self.use_split]
        self._load()

        self.context_cond_candidates = get_candidate_conditions(
            self.context_cond_candidates_cfg,
            self.le_dict,
        )
        self.generation_cond_candidates = get_candidate_conditions(
            self.generation_cond_candidates_cfg,
            self.le_dict,
        )

    def __iter__(self):
        """Iterator for preparing context and query pairs."""
        n_batches_to_generate = self.n_batches_to_generate
        context_cond_candidates = dict_of_tensors_to_tensor(self.context_cond_candidates)
        generation_cond_candidates = dict_of_tensors_to_tensor(self.generation_cond_candidates)

        # Indicator that a cell falls into any one of the candidate conditions
        cond_tensor = dict_of_tensors_to_tensor(self.cond)
        context_candidate_ind = (cond_tensor.unsqueeze(0)
                                 == context_cond_candidates.unsqueeze(1)).all(-1).any(0)
        context_candidate_idx = torch.where(context_candidate_ind)[0]
        num_context_cells = len(context_candidate_idx)

        # Query conditions for generation used in each minibatch of context cells
        batch_size = self.batch_size or len(context_candidate_idx)
        assert batch_size >= len(generation_cond_candidates)
        cond = generation_cond_candidates.repeat(
            ceil(batch_size / len(generation_cond_candidates)), 1)

        query_cond = cond[:batch_size]
        query_cond = {
            sorted(self.cond)[i]: query_cond[:, i] for i in range(len(self.cond))
        }

        for _ in range(self.n_trials):
            # Shuffle candidate context cells
            # rand_idx = torch.randperm(len(context_candidate_idx))
            # context_candidate_idx = context_candidate_idx[rand_idx].contiguous()
            # curr_idx = 0

            # batch_idx = 0
            for _ in range(n_batches_to_generate):
                # next_idx = min(num_context_cells, curr_idx + context_batch_size)
                # select_idx = context_candidate_idx[curr_idx:next_idx]
                select_idx = torch.randint(len(context_candidate_idx), (batch_size,))

                x = F.dropout(self.input[select_idx], self.dropout)
                cell_ids = self.adata.obs.iloc[select_idx].index.tolist()

                yield {"input": x, "cond": query_cond, "context_cell_ids": cell_ids}

                # curr_idx += context_batch_size
                # batch_idx += 1

                # if n_batches_to_generate and (batch_idx >= n_batches_to_generate):
                #     break

            # if n_batches_to_generate and batch_idx < n_batches_to_generate:
            #     warnings.warn(
            #         f"Insufficient context cells to perform {n_batches_to_generate} "
            #         f"batches of generation. Early exciting at batch #{batch_idx}. "
            #         "Consider lowering the # of batch generation or the context size.",
            #         UserWarning,
            #         stacklevel=2,
            #     )


class PerturbationDataset(Dataset):
    SPLIT: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return {
            "input": self.input[index],
            "target": self.target[index],
            "cond": self.cond[index],
            "cond_names": self.cond_names,
            "cond_mapping_dict": self.cond_mapping_dict,
            "top_de_dict": self.top_de_dict
        }

    def _prepare(self):
        assert self.SPLIT is not None, "Please specify SPLIT class attr."
        assert self.SPLIT in np.unique(self.adata.obs["split"])
        self._load(self.SPLIT)
