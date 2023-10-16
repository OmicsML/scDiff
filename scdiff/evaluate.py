import warnings

import scib
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from matplotlib import pyplot
from scib.metrics.ari import ari
from scib.metrics.nmi import nmi
from typing import Dict, List, Optional
from adjustText import adjust_text

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from scdiff.typing import TensorArray
from scdiff.utils.misc import as_array, as_tensor


def PearsonCorr(y_pred, y_true):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c, 1)
        / torch.sqrt(torch.sum(y_true_c * y_true_c, 1))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1))
    )
    return pearson


def PearsonCorr1d(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c)
        / torch.sqrt(torch.sum(y_true_c * y_true_c))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c))
    )
    return pearson


@torch.inference_mode()
def evaluate_annotation(
    true: TensorArray,
    pred: TensorArray,
    name: Optional[str],
) -> Dict[str, float]:
    true_array = as_array(true, assert_type=True)
    pred_array = as_array(pred, assert_type=True)

    le = LabelEncoder()
    le.classes_ = np.array(sorted(set(np.unique(true_array).tolist() + np.unique(pred_array).tolist())))

    true = torch.LongTensor(le.transform(true_array))
    pred = torch.LongTensor(le.transform(pred_array))

    num_classes = le.classes_.size
    # num_classes = int(max(true.max(), pred.max())) + 1
    # num_unique_classes = max(true.unique().numel(), pred.unique().numel())
    # if (num_classes == num_unique_classes + 1) and (0 not in true):
    #     warnings.warn(
    #         "Implicitly removing null label (index 0)",
    #         UserWarning,
    #         stacklevel=2,
    #     )
    #     true, pred, num_classes = true - 1, pred - 1, num_classes - 1
    # elif num_classes != num_unique_classes:
    #     warnings.warn(
    #         f"Number of unique classes {num_unique_classes} mismatch the "
    #         f"number of classes inferred by max index {num_classes}",
    #         UserWarning,
    #         stacklevel=2,
    #     )

    suffix = "" if name is None else f"_{name}"

    out = {}
    out[f"acc{suffix}"] = multiclass_accuracy(true, pred, num_classes).item()
    out[f"f1{suffix}"] = multiclass_f1_score(true, pred, num_classes).item()
    out[f"precision{suffix}"] = multiclass_precision(true, pred, num_classes).item()
    out[f"recall{suffix}"] = multiclass_recall(true, pred, num_classes).item()

    return out


def masked_rmse(pred, true, mask):
    pred_masked = pred * mask
    true_masked = true * mask
    size = mask.sum()
    return (F.mse_loss(pred_masked, true_masked, reduction='sum') / size).sqrt()


def masked_stdz(x, mask):
    size = mask.sum(1, keepdim=True).clamp(1)
    x = x * mask
    x_ctrd = x - (x.sum(1, keepdim=True) / size) * mask
    # NOTE: multiplied by the factor of sqrt of N
    x_std = x_ctrd.pow(2).sum(1, keepdim=True).sqrt()
    return x_ctrd / x_std


def masked_corr(pred, true, mask):
    pred_masked_stdz = masked_stdz(pred, mask)
    true_masked_stdz = masked_stdz(true, mask)
    corr = (pred_masked_stdz * true_masked_stdz).sum(1).mean()
    return corr


@torch.inference_mode()
def denoising_eval(true: TensorArray, pred: TensorArray, mask: TensorArray):
    true = as_tensor(true, assert_type=True)
    pred = as_tensor(pred, assert_type=True)
    mask = as_tensor(mask, assert_type=True).bool()

    rmse_normed = masked_rmse(pred, true, mask).item()
    corr_normed = masked_corr(pred, true, mask).item()
    global_corr_normed = PearsonCorr1d(pred[mask], true[mask]).item()

    # nonzero_masked = (true > 0) * mask
    # rmse_normed_nonzeros = masked_rmse(pred, true, nonzero_masked).item()
    # corr_normed_nonzeros = masked_corr(pred, true, nonzero_masked).item()

    corr_normed_all = PearsonCorr(pred, true).item()
    rmse_normed_all = F.mse_loss(pred, true).sqrt().item()

    r = scipy.stats.linregress(pred[mask].cpu().numpy(), true[mask].cpu().numpy())[2]
    # r_all = scipy.stats.linregress(pred.ravel().cpu().numpy(), true.ravel().cpu().numpy())[2]

    return {
        'denoise_rmse_normed': rmse_normed,
        'denoise_corr_normed': corr_normed,
        'denoise_global_corr_normed': global_corr_normed,
        'denoise_global_r2_normed': r ** 2,
        # 'denoise_rmse_normed_nonzeros': rmse_normed_nonzeros,
        # 'denoise_corr_normed_nonzeros': corr_normed_nonzeros,
        'denoise_rmse_normed_all': rmse_normed_all,
        'denoise_corr_normed_all': corr_normed_all,
        # 'denoise_global_r2_normed_all': r_all ** 2,
    }



@torch.inference_mode()
def perturbation_eval(
    true,
    pred,
    control,
    true_conds=None,
    gene_names=None,
    path_to_save=None,
    de_gene_idx_dict=None,
    ndde20_idx_dict=None,
    de_gene_idx=None,
    ndde20_idx=None,
):
    if true_conds is not None:  # summarize condition wise evaluation
        assert de_gene_idx_dict is not None, "GEARS eval require DE gene index dict"
        assert ndde20_idx_dict is not None, "GEARS eval require top20 none dropout DE gene index dict"
        if path_to_save:
            warnings.warn(
                f"Cant save with multiple conds, got {path_to_save=}. Ignoring save option",
                UserWarning,
                stacklevel=2,
            )
        unique_true_conds = true_conds.unique(dim=0)
        score_dict_list = []
        for cond in unique_true_conds:
            cond_ind = (true_conds == cond).all(1)
            true_sub, pred_sub = true[cond_ind], pred[cond_ind]
            cond_idx_tuple = tuple(i for i in cond.tolist() if i != -1)  # XXX: specificially designed for GEARS
            score_dict_list.append(perturbation_eval(true_sub, pred_sub, control, gene_names=gene_names,
                                                     de_gene_idx=de_gene_idx_dict[cond_idx_tuple],
                                                     ndde20_idx=ndde20_idx_dict[cond_idx_tuple]))
        scores = reduce_score_dict_list(score_dict_list)
        return scores

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        adata_pred = ad.AnnData(pred.detach().cpu().numpy(),
                                obs={'condition': ["pred"] * len(pred)})
        adata_true = ad.AnnData(true.detach().cpu().numpy(),
                                obs={'condition': ["stim"] * len(true)})
        adata_ctrl = ad.AnnData(control.detach().cpu().numpy(),
                                obs={'condition': ["ctrl"] * len(control)})
        adata = ad.concat([adata_true, adata_ctrl])
        if gene_names is not None:
            adata.var.index = gene_names
            adata_pred.var.index = gene_names
        sc.tl.rank_genes_groups(adata, groupby='condition', method="wilcoxon")
        diff_genes = adata.uns["rank_genes_groups"]["names"]['stim']
        diff_genes_idx = [np.where(np.array(gene_names) == x)[0].item() for x in diff_genes]
        adata = ad.concat([adata, adata_pred])
        adata.obs_names_make_unique()
        scores = reg_mean_plot(
            adata,
            condition_key='condition',
            axis_keys={"x": "pred", "y": 'stim', "x1": "ctrl"},
            gene_list=diff_genes[:10] if gene_names is not None else None,
            top_100_genes=diff_genes[:100],
            labels={"x": "predicted", "y": "ground truth", "x1": "ctrl"},
            path_to_save=path_to_save,
            title='scDiff',
            show=False,
            legend=False,
        )

    true_mean = true.mean(0)
    pred_mean = pred.mean(0)
    control_mean = control.mean(0)
    true_delta_mean = true_mean - control_mean
    pred_delta_mean = pred_mean - control_mean

    scores.update({
        # MAE
        'mae': (pred_mean - true_mean).abs().mean().item(),
        'mae_top_100': (pred_mean[diff_genes_idx[:100]] - true_mean[diff_genes_idx[:100]]).abs().mean().item(),
        'mae_delta': (pred_delta_mean - true_delta_mean).abs().mean().item(),
        # MSE
        'mse': F.mse_loss(pred_mean, true_mean).item(),
        'mse_top_100': F.mse_loss(pred_mean[diff_genes_idx[:100]], true_mean[diff_genes_idx[:100]]).item(),
        'mse_delta': F.mse_loss(pred_delta_mean, true_delta_mean).item(),
        # RMSE
        'rmse': np.sqrt(F.mse_loss(pred_mean, true_mean).item()),
        'rmse_top_100': np.sqrt(F.mse_loss(pred_mean[diff_genes_idx[:100]],
                                           true_mean[diff_genes_idx[:100]]).item()),
        'rmse_delta': np.sqrt(F.mse_loss(pred_delta_mean, true_delta_mean).item()),
        # Correlation
        'corr': PearsonCorr1d(pred_mean, true_mean).item(),
        'corr_top_100': PearsonCorr1d(pred_mean[diff_genes_idx[:100]],
                                      true_mean[diff_genes_idx[:100]]).item(),
        'corr_delta': PearsonCorr1d(pred_delta_mean, true_delta_mean).item(),
        # # Cosine similarity
        # 'cos': F.cosine_similarity(pred_mean.unsqueeze(0), true_mean.unsqueeze(0))[0].item(),
        # 'cos_top_100': F.cosine_similarity(pred_mean[diff_genes_idx[:100]].unsqueeze(0),
        #                                    true_mean[diff_genes_idx[:100]].unsqueeze(0))[0].item(),
        # 'cos_delta': F.cosine_similarity(pred_delta_mean.unsqueeze(0),
        #                                  true_delta_mean.unsqueeze(0))[0].item(),
    })

    if de_gene_idx is not None:
        for num_de in (20, 50, 100, 200):
            if num_de > len(de_gene_idx):
                warnings.warn(
                    f"Skipping {num_de} DE gene num eval since max num DE available is {len(de_gene_idx)}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if num_de > true.shape[1]:
                warnings.warn(
                    f"Skipping {num_de} DE gene num eval since max num genes available is {true.shape[1]}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            idx = de_gene_idx[:num_de]
            scores.update(de_eval(pred_mean[idx], true_mean[idx], control_mean[idx], f"de{num_de}"))

    if ndde20_idx is not None:
        scores.update(de_eval(pred_mean[ndde20_idx], true_mean[ndde20_idx], control_mean[ndde20_idx], "ndde20"))

    return scores


def de_eval(true, pred, ctrl, name):
    true_delta = true - ctrl
    pred_delta = pred - ctrl
    return {
        # MAE
        f'mae_{name}': (pred - true).abs().mean().item(),
        f'mae_delta_{name}': (pred_delta - true_delta).abs().mean().item(),
        # MSE
        f'mse_{name}': F.mse_loss(pred, true).item(),
        f'mse_delta_{name}': F.mse_loss(pred_delta, true_delta).item(),
        # RMSE
        f'rmse_{name}': np.sqrt(F.mse_loss(pred, true).item()),
        f'rmse_delta_{name}': np.sqrt(F.mse_loss(pred_delta, true_delta).item()),
        # Correlation
        f'corr_{name}': PearsonCorr1d(pred, true).item(),
        f'corr_delta_{name}': PearsonCorr1d(pred_delta, true_delta).item(),
    }


def reg_mean_plot(adata, condition_key, axis_keys, labels, path_to_save="./reg_mean.pdf",
                  gene_list=None, top_100_genes=None, show=False, legend=True, title=None,
                  x_coeff=3, y_coeff=0, fontsize=14, **kwargs):
    """
        Adapted from https://github.com/theislab/scgen-reproducibility/blob/master/code/scgen/plotting.py
        Plots mean matching figure for a set of specific genes.

        # Parameters
            adata: `~anndata.AnnData`
                Annotated Data Matrix.
            condition_key: basestring
                Condition state to be used.
            axis_keys: dict
                dictionary of axes labels.
            path_to_save: basestring
                path to save the plot.
            gene_list: list
                list of gene names to be plotted.
            show: bool
                if `True`: will show to the plot after saving it.
    """
    import seaborn as sns
    sns.set()
    sns.set(color_codes=True)
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.A
    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        pred_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = np.average(pred_diff.X, axis=0)
        y_diff = np.average(stim_diff.X, axis=0)
        m, b, r_value_diff, p_value_diff, std_err_diff = scipy.stats.linregress(x_diff, y_diff)
        # print(r_value_diff ** 2)
    x = np.average(pred.X, axis=0)
    y = np.average(stim.X, axis=0)
    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    # print(r_value ** 2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})

    if path_to_save:
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df, scatter_kws={'rasterized': True})
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))

        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)

    if "x1" in axis_keys.keys():
        ctrl = adata[adata.obs[condition_key] == axis_keys["x1"]]
        x1 = np.average(ctrl.X, axis=0)
        x_delta = x - x1
        y_delta = y - x1
        _, _, r_value_delta, _, _ = scipy.stats.linregress(x_delta, y_delta)
        if diff_genes is not None:
            ctrl_diff = ctrl[:, diff_genes]
            x1_diff = np.average(ctrl_diff.X, axis=0)
            x_delta_diff = x_diff - x1_diff
            y_delta_diff = y_diff - x1_diff
            _, _, r_value_delta_diff, _, _ = scipy.stats.linregress(x_delta_diff, y_delta_diff)
        # _p2 = pyplot.scatter(x, y1, marker="*", c="red", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")

    if path_to_save:
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, 'o', color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(texts, x=x, y=y, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5),
                        force_points=(0.0, 0.0))
        if legend:
            pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)

        ax.text(x_coeff, y_coeff, r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' +
                f"{r_value ** 2:.4f}", fontsize=kwargs.get("textsize", fontsize))
        if diff_genes is not None:
            ax.text(x_coeff, y_coeff + 0.6, r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' +
                    f"{r_value_diff ** 2:.4f}", fontsize=kwargs.get("textsize", fontsize))
        if path_to_save is not None:
            pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
        if show:
            pyplot.show()
        pyplot.close()

    scores = {'R^2': r_value ** 2}
    if diff_genes is not None:
        scores['R^2_top_100'] = r_value_diff ** 2
    if "x1" in axis_keys.keys():
        scores['R^2_delta'] = r_value_delta ** 2
        if diff_genes is not None:
            scores['R^2_delta_top_100'] = r_value_delta_diff ** 2
    return scores


def dict_of_arrays_to_tensor(input_dict):
    tensor_list = []
    for key in sorted(input_dict):
        tensor_list.append(torch.tensor(input_dict[key]))
    return torch.stack(tensor_list).T


def calculate_batch_r_squared(pred, true, conditions):
    conditions = dict_of_arrays_to_tensor(conditions)
    unique_cond = conditions.unique(dim=0)
    r_squared_list = []
    for i in range(len(unique_cond)):
        cond_flag = torch.all((conditions == unique_cond[i]), dim=1)
        x = pred[cond_flag].mean(0).numpy()
        y = true[cond_flag].mean(0).numpy()
        _, _, r_value, _, _ = scipy.stats.linregress(x, y)
        r_squared_list.append(r_value ** 2)
    return r_squared_list


def reduce_score_dict_list(score_dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    assert isinstance(score_dict_list, list)

    score_keys = sorted(score_dict_list[0])
    assert all(sorted(i) == score_keys for i in score_dict_list), "All score dicts must contain same score keys"

    scores = {score_key: np.mean([i[score_key] for i in score_dict_list]) for score_key in score_keys}

    return scores
