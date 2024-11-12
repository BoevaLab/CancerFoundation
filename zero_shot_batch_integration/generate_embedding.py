import sys
sys.path.insert(0, "../")
from model.embedding import embed
import scanpy as sc
import os

model_dir = "../model/assets"
adata_path = "./gene_pertubation/neftel_ss2.h5ad"

adata = sc.read_h5ad(adata_path)
batch_key = "sample"  # The batch identity is used for highly variable gene selection
bio_key = "subtype"

embed_adata = embed(
    adata_or_file=adata,
    model_dir=model_dir,
    batch_key=batch_key,
    batch_size=64,
)
os.makedirs("./gene_pertubation", exist_ok=True)
embed_adata.write_h5ad("./gene_pertubation/CancerGPT_neftel_ss2.h5ad")
