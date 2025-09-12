from pathlib import Path
import sys

sys.path.insert(0, "./")

from cancerfoundation.data.dataset import DatasetDir
from argparse import ArgumentParser
from bionemo.scdl.io.single_cell_collection import SingleCellCollection

import scanpy as sc
import pandas as pd
import json
import os

GENE_ID = "_cf_gene_id"
CLS_TOKEN = "<cls>"
PAD_TOKEN = "<pad>"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--h5ad-path", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--vocab-path", type=Path, required=False)
    parser.add_argument("--merge_tech", type=str, default="default")
    return parser.parse_args()


def _generate_vocab_from_h5ads(
    h5ads: Path, cls_token: str, pad_token: str
) -> dict[str, int]:
    genes = set()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        var_names = sc.read_h5ad(path, backed="r").var_names
        genes.update(var_names)
    vocab = {gene: i for i, gene in enumerate([cls_token, pad_token] + list(genes))}
    return vocab


def _save_vocab_to_dir(vocab: dict[str, int], data_dir: DatasetDir) -> None:
    with open(data_dir.vocab_path, "w") as f:
        json.dump(vocab, f)


def _add_gene_id_to_h5ads(h5ads: Path, vocab: dict[str, int], data_path: Path) -> None:
    (data_path / "h5ads").mkdir()
    for path in h5ads.iterdir():
        if not path.name.endswith(".h5ad"):
            continue
        adata = sc.read_h5ad(path)
        adata.var[GENE_ID] = adata.var_names.map(vocab)
        adata = adata[:, ~adata.var[GENE_ID].isna()].copy()
        adata.var[GENE_ID] = adata.var[GENE_ID].astype(int)
        adata.write_h5ad(data_path / "h5ads" / path.name)


def convert_columns_to_categorical_with_mapping(df):
    df_copy = df.copy()

    category_mappings = {}

    df_categorical = pd.DataFrame(index=df.index)

    for column in df.columns:
        df_copy[column] = df_copy[column].astype("category")

        category_mappings[column] = dict(
            zip(
                df_copy[column].cat.categories,
                range(len(df_copy[column].cat.categories)),
            )
        )

        df_categorical[column] = df_copy[column].cat.codes

    return df_categorical, category_mappings


def get_tech_map(level: str = "default") -> dict:
    """
    Return a technology mapping dictionary.

    Args:
        level (str): one of {"default", "medium", "coarse"}.
            - "default": leaves everything as-is (identity mapping).
            - "medium": merges only closely related protocols.
            - "coarse": collapses into broad technology families.

    Returns:
        dict: mapping from original technology name -> merged category
    """
    if level == "coarse":
        return {
            # droplet-based 3′ UMI
            "10X": "10x-like",
            "10x": "10x-like",
            "Drop-seq": "10x-like",
            "inDrop": "10x-like",
            # microwell / nanowell
            "SeqWell": "microwell-like",
            "Seq-Well": "microwell-like",
            "Seq-Well S3": "microwell-like",
            "Microwell-seq": "microwell-like",
            "Microwell array-based platform": "microwell-like",
            # plate-based 3′ UMI
            "CEL-seq2": "plate-like",
            "MARS-seq": "plate-like",
            # keep separate
            "SmartSeq2": "SmartSeq2",
            "iCell8": "iCell8",
            "Nanogrid": "Nanogrid",
            "HiSeq 2000": "HiSeq 2000",
        }

    elif level == "medium":
        return {
            # droplet-based 3′ UMI
            "10X": "10x-like",
            "10x": "10x-like",
            "Drop-seq": "Drop-seq",
            "inDrop": "inDrop",
            # microwell / nanowell
            "SeqWell": "SeqWell",
            "Seq-Well": "SeqWell",
            "Seq-Well S3": "SeqWell",
            "Microwell-seq": "Microwell",
            "Microwell array-based platform": "Microwell",
            # plate-based 3′ UMI
            "CEL-seq2": "CEL-seq2",
            "MARS-seq": "MARS-seq",
            # keep separate
            "SmartSeq2": "SmartSeq2",
            "iCell8": "iCell8",
            "Nanogrid": "Nanogrid",
            "HiSeq 2000": "HiSeq 2000",
        }

    elif level == "default":
        # identity mapping: return an empty dict,
        # so .map() leaves values unchanged
        return {}

    else:
        raise ValueError("level must be one of {'default', 'medium', 'coarse'}")


def main():
    args = get_args()

    h5ad_path = args.h5ad_path
    columns = ["sample", "cancer_type", "technology", "tissue"]

    data_path = DatasetDir(args.data_path)
    data_path.mkdir()
    # Generate and save vocabulary
    if args.vocab_path is None:
        vocab = _generate_vocab_from_h5ads(h5ad_path, CLS_TOKEN, PAD_TOKEN)
    else:
        vocab = json.load(args.vocab_path.open())
    _save_vocab_to_dir(vocab, data_path)
    # Add gene IDs to h5ad files
    _add_gene_id_to_h5ads(h5ad_path, vocab, args.data_path)

    # Create and process memmap files
    memmaps = SingleCellCollection(data_path.data_dir / "tmp")
    memmaps.load_h5ad_multi(args.data_path / "h5ads", max_workers=12)

    # Collect observations
    obs_list = []
    for i, fname in enumerate(memmaps.fname_to_mmap.keys()):
        print(f"Processing {i + 1}/{len(memmaps.fname_to_mmap)}: {fname.name}")
        adata = sc.read_h5ad(h5ad_path / (fname.name + ".h5ad"), backed="r")

        # Extract tissue name from the filename (e.g., '..._kidney' -> 'kidney')
        tissue_name = fname.name.split("_")[1]

        # Create the 'tissue' column and assign the extracted name to all cells
        adata.obs["tissue"] = tissue_name

        obs_list.append(adata.obs[columns])

    obs = pd.concat(obs_list)

    tech_map = get_tech_map(args.merge_tech)  # Change level as needed
    obs["technology"] = obs["technology"].map(tech_map).fillna(obs["technology"])

    print(obs.technology.unique())

    obs, mapping = convert_columns_to_categorical_with_mapping(obs)

    obs.to_parquet(data_path.obs_path)

    with data_path.mapping_path.open("w") as f:
        json.dump(mapping, f, indent=4)

    # Flatten and create memmap dataset
    memmaps.flatten(data_path.memmap_path, destroy_on_copy=True)

    print("Conversion completed successfully.")

    # Remove duplicate metadata file that causes issues
    os.remove(data_path.memmap_path / "features/dataframe_00.parquet")


if __name__ == "__main__":
    main()