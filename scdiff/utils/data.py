import warnings
from functools import partial
from itertools import chain, product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder


def get_candidate_conditions(
    context_cond_candidates_cfg: Optional[DictConfig],
    le_dict: Dict[str, LabelEncoder],
) -> torch.Tensor:
    # NOTE: currently only support celltype and batch conditions
    if context_cond_candidates_cfg is None:
        warnings.warn(
            "context_cond_candidates_cfg not specified, using 'grid' mode by default",
            UserWarning,
            stacklevel=2,
        )
        mode, options = "grid", None
    else:
        mode = context_cond_candidates_cfg.mode
        options = context_cond_candidates_cfg.get("options", None)

    if mode == "grid":
        # Example config option
        # mode: grid  # use all possible conditions
        cond = torch.LongTensor(list(product(*[range(le_dict[k].classes_.size) for k in sorted(le_dict)])))

    elif mode == "select":
        # Example config option
        # mode: select  # select specific combinations of conditions
        # options:
        #   - [batch1, celltype1]
        #   - [batch5, celltype2]
        cond_list = [
            le_dict[k].transform(np.array(options.get(k, le_dict[k].classes_)))
            for k in sorted(le_dict)
        ]
        cond = torch.LongTensor(list(map(list, zip(*cond_list))))

    elif mode == "partialgrid":
        # Example config option
        # mode: partialgrid  # use the specified options and grid the rest
        # options:
        #   cond1:
        #     - celltype1
        #     - celltype2
        cond_list = [
            le_dict[k].transform(np.array(options.get(k, le_dict[k].classes_)))
            for k in sorted(le_dict)
        ]
        cond = torch.LongTensor(list(product(*cond_list)))

    else:
        raise ValueError(f"Unknown mode {mode!r}, supported options are: "
                         "['grid', 'select', partialgrid]")

    cond = {
        sorted(le_dict)[i]: cond[:, i] for i in range(len(le_dict))
    }
    return cond


def extract_from_dictlist(dictlist: List[Dict[str, Any]], key: str) -> List[Any]:
    """Extract specific item from each dictionary in a list of dictionaries."""
    # Example (key="a"): [{"a": 1, "b": 0}, {"a": 3}] -> [1, 3]
    return [i[key] for i in dictlist]


def combine_predictions(
    data: LightningDataModule,
    predictions: List[Any],
    save_path: Optional[str] = None,
) -> AnnData:
    """Combine generation predictions into an AnnData and optionally save."""
    # predictions = [{"x_gen": tensor, "query_conditions": tensor
    #                 "context_cell_ids": list[str]}, ...]
    pred_data = data.datasets["predict"]
    extract_items = partial(extract_from_dictlist, dictlist=predictions)
    x_gen = torch.cat(extract_items(key="x_gen"), dim=0).cpu().numpy().astype(np.float32)
    gen_cond_list = extract_items(key="query_conditions")
    gen_cond = {}
    for k in gen_cond_list[0].keys():
        temp = torch.cat([x[k] for x in gen_cond_list], dim=0).cpu().numpy().astype(int)
        gen_cond[k] = pred_data.le_dict[k].inverse_transform(temp)
    context_cell_ids = list(chain.from_iterable(extract_items(key="context_cell_ids")))

    obs = pd.DataFrame(gen_cond)
    adata = AnnData(X=x_gen, obs=obs, var=pred_data.adata.var,
                    uns={"context_cell_ids": context_cell_ids})

    if save_path:
        print(f"Saving results to {save_path}")
        adata.write_h5ad(save_path)

    return adata


def mask_data_offline(adata: AnnData, mask_strategy: Optional[str] = "random", mask_type: Optional[str] = "mar",
                      valid_mask_rate: Optional[float] = 0., test_mask_rate: Optional[float] = 0.1,
                      seed: Optional[int] = 10):

    def _get_probs(vec, distr='exp'):
        from scipy.stats import expon
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(distr)

    rng = np.random.default_rng(seed)
    feat = adata.layers['counts'].A
    if mask_strategy == 'none_zero':
        train_mask = np.ones(feat.shape, dtype=bool)
        valid_mask = np.zeros(feat.shape, dtype=bool)
        test_mask = np.zeros(feat.shape, dtype=bool)
        row, col = np.nonzero(feat)
        nonzero_counts = np.array(feat[row, col])
        num_nonzeros = len(row)
        n_test = int(np.floor(num_nonzeros * test_mask_rate))
        n_valid = int(np.floor(num_nonzeros * valid_mask_rate))
        idx_mask = np.ones(num_nonzeros, dtype=bool)

        # Randomly mask positive counts according to masking probability.
        if mask_type == "mcar":
            distr = "uniform"
        elif mask_type == "mar":
            distr = "exp"
        else:
            raise NotImplementedError(f"Expect mask_type in ['mar', 'mcar'], but found {mask_type}")
        mask_prob = _get_probs(nonzero_counts, distr)
        mask_prob = mask_prob / sum(mask_prob)
        test_idx = rng.choice(np.arange(num_nonzeros), n_test, p=mask_prob, replace=False)
        train_mask[row[test_idx], col[test_idx]] = False
        test_mask[row[test_idx], col[test_idx]] = True

        idx_mask[test_idx] = False
        masked_mask_prob = mask_prob[idx_mask] / sum(mask_prob[idx_mask])
        valid_idx = rng.choice(np.arange(num_nonzeros)[idx_mask], n_valid, p=masked_mask_prob, replace=False)
        train_mask[row[valid_idx], col[valid_idx]] = False
        valid_mask[row[valid_idx], col[valid_idx]] = True

    elif mask_strategy == 'random':
        test_mask = rng.random(feat.shape) < (test_mask_rate + valid_mask_rate)
        valid_mask = test_mask.copy()

        nonzero_idx = np.where(test_mask)
        test_to_val_ratio = test_mask_rate / (test_mask_rate + valid_mask_rate)
        split_point = int(nonzero_idx[0].size * test_to_val_ratio)
        test_idx, val_idx = np.split(rng.permutation(nonzero_idx[0].size), [split_point])

        test_mask[nonzero_idx[0][val_idx], nonzero_idx[1][val_idx]] = False
        valid_mask[nonzero_idx[0][test_idx], nonzero_idx[1][test_idx]] = False
        train_mask = ~(test_mask | valid_mask)

    else:
        raise NotImplementedError(f'Unsupported mask_strategy {mask_strategy}')

    return train_mask, valid_mask, test_mask


def dict_of_tensors_to_tensor(input_dict):
    tensor_list = []
    for key in sorted(input_dict):
        tensor_list.append(input_dict[key])
    return torch.stack(tensor_list).T


def dict_to_list_of_tuples(input_dict):
    if len(list(input_dict)) > 1:
        input_list = [input_dict[k] for k in input_dict.keys()]
        return list(map(tuple, zip(*input_list)))
    else:
        return input_dict[list(input_dict)[0]]
