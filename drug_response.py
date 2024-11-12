import json
import os
from pathlib import Path
from random import shuffle
from typing import Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from scipy import sparse
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from gene_pertubation.collator import DrugCollator
from gene_pertubation.dataset import DrugDoseAnnDataset
from model.cancergpt import CancerGPT, DrugDoseGPT, PAdaptor
from model.utils import load_pretrained
from model.vocab import GeneVocab

PathLike = Union[str, os.PathLike]

N_FEATURES = 1024
N_LATEMT = 256

def shuffle_adata(adata):
    """
    Shuffles the `adata`.
    """
    if sparse.issparse(adata.X):
        #adata.X: sparse matrix to array
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata

def train_valid_test(adata: AnnData, split_key):
    '''
    Get train_valid_test dataset
    '''

    shuffled = shuffle_adata(adata)
    train_index = adata.obs[adata.obs[split_key] == 'train'].index.tolist()
    valid_index = adata.obs[adata.obs[split_key] == 'valid'].index.tolist()
    test_index = adata.obs[adata.obs[split_key] == 'test'].index.tolist()
    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()

    if len(train_index) > 0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index) > 0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata = None
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata = None

    return train_adata, valid_adata, test_adata


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


class DrugDoseTrainer:
    def __init__(
            self,
            split_key: str,
            model_dir: str = "model/assets",
            device: str = "cuda",
            max_length: int = 1200,
            batch_size: int = 128,
            n_features: int = 1024,
            n_latent: int = 256,
            learning_rate: float = 1e-4
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_latent = n_latent
        self.learning_rate = learning_rate
        self.split_key = split_key

        # Constants
        self.pad_token = "<pad>"
        self.pad_value = -2

        # Initialize components
        self.vocab = self._load_vocab()
        self.model_configs = self._load_model_configs()
        self.model = self._initialize_model()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _load_vocab(self):
        vocab = GeneVocab.from_file(self.model_dir / "vocab.json")
        vocab.set_default_index(vocab[self.pad_token])
        return vocab

    def _load_model_configs(self):
        with open(self.model_dir / "args.json", "r") as f:
            return json.load(f)

    def _initialize_model(self):
        pretrained_model = CancerGPT(
            ntoken=len(self.vocab),
            d_model=self.model_configs["embsize"],
            nhead=self.model_configs["nheads"],
            d_hid=self.model_configs["d_hid"],
            nlayers=self.model_configs["nlayers"],
            vocab=self.vocab,
            dropout=self.model_configs["dropout"],
            pad_token=self.pad_token,
        )

        pretrained_model = load_pretrained(
            pretrained_model,
            torch.load(self.model_dir / "model.pth", map_location=self.device),
            verbose=False
        )

        drug_adaptor = PAdaptor(self.n_features, n_genes=978, n_latent=self.n_latent)
        model = DrugDoseGPT(pretrained_model, drug_adaptor, self.n_latent)
        model.to(self.device)
        model = model.freeze()
        return model

    def prepare_data(self, adata: AnnData):
        # Prepare gene IDs
        genes = adata.var["genes"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        # Create dataset and collator
        collator = DrugCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.pad_token],
            pad_value=self.pad_value,
            do_mlm=False,
            do_binning=True,
            max_length=self.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )

        # Split gene_pertubation
        train_adata, valid_adata, test_adata = train_valid_test(adata, split_key=self.split_key)

        # Create datasets
        train_dataset = DrugDoseAnnDataset(train_adata, gene_ids,
                                           obs_key="cov_drug_name") if train_adata is not None else None
        valid_dataset = DrugDoseAnnDataset(valid_adata, gene_ids,
                                           obs_key="cov_drug_name") if valid_adata is not None else None
        test_dataset = DrugDoseAnnDataset(test_adata, gene_ids,
                                          obs_key="cov_drug_name") if test_adata is not None else None

        return train_dataset, valid_dataset, test_dataset, collator

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for data_dict in tqdm(train_loader, total=len(train_loader)):
            self.optimizer.zero_grad()
            data_dict = move_to(data_dict, self.device)

            output = self.model(
                data_dict["gene"],
                data_dict["expr"],
                data_dict["drug"],
                data_dict["gene"].eq(self.vocab[self.pad_token])
            )

            loss = self.loss_fn(output, data_dict["target"])
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, valid_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data_dict in valid_loader:
                data_dict = move_to(data_dict, self.device)

                output = self.model(
                    data_dict["gene"],
                    data_dict["expr"],
                    data_dict["drug"],
                    data_dict["gene"].eq(self.vocab[self.pad_token])
                )

                loss = self.loss_fn(output, data_dict["target"])
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, adata: AnnData, max_epochs: int = 100):
        train_dataset, valid_dataset, test_dataset, collator = self.prepare_data(adata)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=collator, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size,
                                  collate_fn=collator) if valid_dataset else None


        for epoch in range(max_epochs):
            train_loss = self.train_epoch(train_loader)

            if valid_loader:
                valid_loss = self.validate(valid_loader)


# Example usage
if __name__ == "__main__":
    # Load gene_pertubation
    adata = sc.read_h5ad("gene_pertubation/L1000_small.h5ad")
    adata.var["genes"] = adata.var.index

    # Initialize and train
    trainer = DrugDoseTrainer(split_key="random_split_0", batch_size=64)
    trainer.train(adata, max_epochs=100)



