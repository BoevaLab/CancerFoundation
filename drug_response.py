from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union
from anndata import AnnData
import os
from model.cancergpt import CancerGPT, DrugDoseGPT
from model.data_collator import DataCollator
from model.dataset import Dataset

from model.utils import load_pretrained
from model.vocab import GeneVocab
import scanpy as sc
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, SequentialSampler
PathLike = Union[str, os.PathLike]


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

max_length: int = 1200
batch_size: int = 128


model_dir = "model/assets"
device = "cuda"
if device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
model_dir = Path(model_dir)
vocab_file = model_dir / "vocab.json"
model_config_file = model_dir / "args.json"
model_file = model_dir / "model.pth"
pad_token = "<pad>"
pad_value = -2

# vocabulary
vocab = GeneVocab.from_file(vocab_file)




adata = sc.read_h5ad("data/Lincs_L1000.h5ad")
drugs = adata.obs_names[adata.obs["dose"] != 0.0].to_list()[:2000]
control = adata.obs.loc[drugs]["paired_control_index"].to_list()
adata = adata[drugs + control, :]


adata.var["genes"] = adata.var.index
with open(model_config_file, "r") as f:
    model_configs = json.load(f)

vocab.set_default_index(vocab["<pad>"])
genes = adata.var["genes"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)


from model.cancergpt import PAdaptor


drug_adaptor = PAdaptor([1024, 512], 256, 0.1)



# %%
from data.dataset import DrugDoseAnnDataset
from torch.utils.data.dataloader import DataLoader
from data.collator import DrugCollator

collator = DrugCollator(
    do_padding=True,
    pad_token_id=vocab[pad_token],
    pad_value=pad_value,
    do_mlm=False,
    do_binning=True,
    max_length=max_length,
    sampling=True,
    keep_first_n_tokens=1,
)


dataset = DrugDoseAnnDataset(adata, gene_ids, obs_key="cov_drug_name")

pretrained_model = CancerGPT(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=pad_token,
    )

pretrained_model = load_pretrained(pretrained_model, torch.load(
    model_file, map_location=device), verbose=False)

model = DrugDoseGPT(pretrained_model, drug_adaptor, 256)
loss_fn = torch.nn.MSELoss()
model.to(device)
model.train()
model = model.freeze()

optim = torch.optim.Adam(model.parameters(), lr=1e-4)
for data_dict in DataLoader(dataset, batch_size=64, collate_fn=collator):
    optim.zero_grad()
    data_dict = move_to(data_dict, device)
    input_gene_ids = data_dict["gene"]
    src_key_padding_mask = input_gene_ids.eq(
        vocab[pad_token]
    )
    output = model(data_dict["gene"], data_dict["expr"], data_dict["drug"], data_dict["gene"].eq(vocab[pad_token]))
    loss = loss_fn(output, data_dict["target"])
    print(loss)
    loss.backward()
    optim.step()
# %%





