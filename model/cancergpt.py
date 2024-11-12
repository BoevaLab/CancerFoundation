import gc
import math
from typing import Dict, List, Mapping, Optional, Tuple, Any, Union
import warnings

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CancerGPT(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        n_input_bins: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_input_bins = n_input_bins

        self.gene_encoder = GeneEncoder(
            ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.encoder = TransformerEncoder(
            encoder_layers, nlayers)

    def encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.gene_encoder(src)

        values = self.value_encoder(values)

        total_embs = src + values

        output = self.encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        return output


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class PAdaptor(nn.Module):
    """
    Constructs the  Perturb-adaptor. This class implements the adaptor part of PRnet. It will chemical perturbation in to 'comb_num' latent space.

    """

    def __init__(self, n_features: int, n_genes: int, n_latent: int, dropout_rate: float=0.1):
        super().__init__() # to run nn.Module's init method
        self.linear1 = nn.Linear(n_features, n_genes)
        self.linear2 = nn.Linear(1, n_latent)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.act(x)
        self.dropout(x)
        x = x.unsqueeze(-1)
        x = self.linear2(x)
        return x


class DrugDoseGPT(nn.Module):
    def __init__(self, cancer_gpt: CancerGPT, drug_adaptor: PAdaptor, n_latent: int):
        super().__init__()
        self.cancer_gpt = cancer_gpt
        self.drug_adaptor = drug_adaptor
        self.mlp = nn.Sequential(*[nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 978)])

    def freeze(self):
        for param in self.cancer_gpt.parameters():
            param.requires_grad = False
        return self

    def forward(self, src: Tensor,
        values: Tensor,
        drug: Tensor,
        src_key_padding_mask: Tensor):

        drug_embedding = self.drug_adaptor(drug)
        src = self.cancer_gpt.gene_encoder(src)

        values = self.cancer_gpt.value_encoder(values)

        total_embs = src + values + drug_embedding

        output = self.cancer_gpt.encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )[:, 0, :]

        return self.mlp(output)



