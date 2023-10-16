"""Gather results from W&B.

Example:
$ python scripts/download_wnb_results.py --prefix v5.5

"""
import os.path as osp
import warnings
from typing import Optional

import click
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from scdiff.config import RESULTSDIR

SUMMARY_SKIP_PREFIXS = ["gradnorm", "train", "_"]
SUMMARY_SKIP_NAMES = ["lr-AdamW"]


def clean_summary_dict(summary_dict):
    out = {}
    for key, val in summary_dict.items():
        if any(key == i for i in SUMMARY_SKIP_NAMES) or any(key.startswith(i) for i in SUMMARY_SKIP_PREFIXS):
            continue
        out[key] = val
    return out


def clean_config_dict(config_dict):
    return {k: v for k, v in config_dict.items() if not k.startswith("_")}


def mean_std_agg(x, k: int = 5) -> str:
    try:
        x = x.astype(float)
        if np.isnan(x).all():
            out = None
        else:
            mean = np.nanmean(x)
            std = np.nanstd(x)
            out = f"{mean:.0{k}f} +/- {std:.0{k}f}"
    except ValueError:
        out = None
    return out


@click.command()
@click.option("--entity", type=str, default="danceteam")
@click.option("--project", type=str, default="scDiff")
@click.option("--prefix", type=str, default="")
@click.option("--nametag", type=str, default="")
@click.option("--nosummary", is_flag=True)
@click.option("--from_csv", type=str, default=None)
@click.option("--timeout", type=int, default=60)
def main(
    entity: str,
    project: str,
    prefix: str,
    nametag: str,
    nosummary: bool,
    from_csv: Optional[str],
    timeout: int,
):
    fname = "wandb_dump"
    if prefix:
        fname += f"_{prefix}"
    if nametag:
        fname += f"_{nametag}"

    if from_csv is None:
        if prefix:
            regex_str = rf"^{prefix}"
            filters = {"display_name": {"$regex": regex_str}}
        else:
            filters = None

        runs = wandb.Api(timeout=timeout).runs(path=f"{entity}/{project}", filters=filters)

        summary_list, config_list, name_list = [], [], []
        for run in tqdm(runs, desc="Gathering results"):
            if not run.name.startswith(prefix):
                warnings.warn(
                    f"regex filter failed to filter for prefix {regex_str!r}, but got {run.name}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            summary_list.append(clean_summary_dict(run.summary._json_dict))
            config_list.append(clean_config_dict(run.config))
            name_list.append(run.name)

        runs_df = pd.DataFrame(summary_list)
        runs_df["config"] = config_list
        runs_df["name"] = name_list

        path = osp.join(RESULTSDIR, f"{fname}.csv")
        runs_df.to_csv(path, index=False)
        print(f"Results saved to {path}")

        if "config" in runs_df.columns:
            path = osp.join(RESULTSDIR, f"{fname}_noconfig.csv")
            runs_noconfig_df = runs_df.drop(columns="config")
            runs_noconfig_df.to_csv(path, index=False)
            print(f"Plain results (no config) saved to {path}")

    else:
        runs_df = pd.read_csv(from_csv)

    if not nosummary:
        runs_summary_df = runs_df.copy()
        runs_summary_df["exp_name"] = runs_summary_df["name"].str.split("_r.", expand=True)[0]
        if 'config' in runs_summary_df.columns:
            runs_summary_df.drop(columns='config', inplace=True)
        runs_summary_df = runs_summary_df.groupby("exp_name").agg(mean_std_agg).reset_index()

        path = osp.join(RESULTSDIR, f"{fname}_summary.csv")
        runs_summary_df.to_csv(path, index=False)
        print(f"Summary results saved to {path}")


if __name__ == "__main__":
    main()
