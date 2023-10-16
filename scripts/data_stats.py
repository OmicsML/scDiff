import os.path as osp
from pathlib import Path

import anndata as ad
import click
import pandas as pd


HOMEDIR = Path(__file__).resolve().parents[1]
DATA_INFO_PATH = HOMEDIR / "data_info.csv"


def get_count(adata, key):
    if key == "none":
        return "--"
    else:
        return f"{adata.obs[key].unique().size:,}"


def get_stats(path, batch_key, cell_type_key, condition_key):
    print(f"Reading data from {path}")
    adata = ad.read_h5ad(path, backed="r")
    num_cells, num_genes = adata.shape

    return {
        "# cells": f"{num_cells:,}",
        "# genes": f"{num_genes:,}",
        "# batches": get_count(adata, batch_key),
        "# cell types": get_count(adata, cell_type_key),
        "# conditions": get_count(adata, condition_key),
    }


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), default="./data")
@click.option("--data_info_path", type=click.Path(exists=True), default=DATA_INFO_PATH)
def main(data_dir, data_info_path):
    data_info_df = pd.read_csv(data_info_path)

    stats_list = []
    for _, row in data_info_df.iterrows():
        path = osp.join(data_dir, row.processed_filename)
        stats = get_stats(path, row.batch_key, row.cell_type_key, row.condition_key)
        stats_list.append({"task": row.task, "name": row.dataset_name, **stats})

    full_data_stats = pd.DataFrame(stats_list)
    print(full_data_stats)


if __name__ == "__main__":
    main()
