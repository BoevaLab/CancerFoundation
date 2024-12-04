import os
from omegaconf import DictConfig
import logging
import time
import hydra
import numpy as np
import pandas as pd
import polars as pl
from math import ceil
import scanpy as sc

logging.basicConfig(level=logging.INFO)

def list_h5ad_files(directory):
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith('.h5ad') and os.path.isfile(os.path.join(directory, f))
    ]

def create_memmap(filename, array, dtype):
    if os.path.exists(filename):
        os.remove(filename)
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=array.shape)
    memmap_array[:] = array[:]
    memmap_array.flush()
    return memmap_array

@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config: DictConfig) -> None: 
    
    os.makedirs(config.memmap_dir, exist_ok=True)

    dataset_lst = list_h5ad_files(config.h5ad_dir)

    for dataset in dataset_lst:
        
        h5ad_path = os.path.join(config.h5ad_dir, f"{dataset}.h5ad")
        now = time.time()
        ad = sc.read_h5ad(h5ad_path)
        logging.info(f"loaded AnnData object of dataset ({dataset}): {time.time() - now}s")
        
        dataset_dir = os.path.join(config.memmap_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        now = time.time()
        create_memmap(
            filename=os.path.join(dataset_dir, f"gene_X.data.bin"), 
            array=ad.X.data, 
            dtype=np.float32)
        create_memmap(
            filename=os.path.join(dataset_dir, f"gene_X.indices.bin"), 
            array=ad.X.indices, 
            dtype=np.int32)
        create_memmap(
            filename=os.path.join(dataset_dir, f"gene_X.indptr.bin"), 
            array=ad.X.indptr, 
            dtype=np.int32)
        logging.info(f"written into memmap's: {time.time() - now}s")
        
        ad.obs.to_csv(os.path.join(dataset_dir, "obs.csv"))
        ad.var.to_csv(os.path.join(dataset_dir, "var.csv"))


if __name__ == "__main__":
    main()