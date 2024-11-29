import time
import os
from typing import Dict, List, Optional, Union
from .data_sampler import get_balanced_sampler
import transformers
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
import torch
import json

from .data_collator import AnnDataCollator
from .dataset import scDataset
from .utils import load_pretrained
from .gene_tokenizer import GeneVocab
from .model import TransformerModel
from cancerfoundation.loss import criterion_neg_log_bernoulli, get_loss, masked_relative_error
import numpy as np
from safetensors import safe_open
from .loss import LossType


def with_sdp_kernel(func):
    def wrapped_func(*args, **kwargs):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            return func(*args, **kwargs)
    return wrapped_func


class Trainer:
    def __init__(
        self,
        args: Dict,
        n_bins: int,
        input_emb_style: str,
        max_seq_len: int,
        input_style: str,
        mask_ratio: float,
        TRUNC_BY_SAMPLE: bool,
        training_tasks: str,
        batch_size: int,
        eval_batch_size: int,
        embsize: int,
        nheads: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
        lr: float,
        warmup_ratio_or_step: float,
        scheduler_interval: int,
        scheduler_factor: float,
        save_dir: str,
        vocab: Union[str, np.array],
        train_data_path: Union[str, os.PathLike],
        eval_data_path: Union[str, os.PathLike],
        loss_type: LossType = LossType.MSE,
        resume_from_checkpoint: str = None,
        wandb: str = None,
        conditions: List[str] = None,
        mvc_decoder_style: str = "inner product",
        scale_zero_expression: Optional[float] = None,
        accelerator = None,
        do_dat: Optional[bool] = False,
        explicit_zero_prob: Optional[bool] = False,
        balance_primary: Optional[str] = None,
        balance_secondary: Optional[str] = None,
        zero_percentages: Optional[List[float]] = None,
    ):
        self.args = args
        self.n_bins = n_bins
        self.input_emb_style = input_emb_style
        self.max_seq_len = max_seq_len
        self.input_style = input_style
        self.mask_ratio = (
            [0.25, 0.50, 0.75] if training_tasks in [
                "gen", "both"] else mask_ratio
        )
        self.TRUNC_BY_SAMPLE = TRUNC_BY_SAMPLE
        self.training_tasks = training_tasks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.embsize = embsize
        self.nheads = nheads
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.lr = lr
        self.warmup_ratio_or_step = warmup_ratio_or_step
        self.scheduler_interval = scheduler_interval
        self.scheduler_factor = scheduler_factor
        self.save_dir = save_dir
        self.accelerator = accelerator
        self.loss_type = loss_type
        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        self.special_tokens = [self.pad_token, self.cls_token, "<eoc>"]
        self.MVC = True
        self.USE_GENERATIVE_TRAINING = (
            True if self.training_tasks in ["gen", "both"] else False
        )
        self.domain_nums = None

        self.explicit_zero_prob = explicit_zero_prob

        self.do_dat = do_dat

        self.resume_from_checkpoint = resume_from_checkpoint
        self.wandb = wandb
        self.timer = None

        if self.input_emb_style == "category":
            self.mask_value = self.n_bins + 1
            self.pad_value = self.n_bins  # for padding gene expr values
            self.n_input_bins = self.n_bins + 2
        else:
            self.mask_value = -1
            self.pad_value = -2
            self.n_input_bins = self.n_bins

        self.best_val_loss = {"loss": float("inf"), "epoch": -1}
        self.starting_epoch = 0

        self.vocab = None
        self.model = None
        self.has_setup_trainer = False
        self.conditions = conditions
        self.conditions_nums = None
        
        if balance_primary is None and balance_secondary is not None:
            raise ValueError("balance_secondary is not allowed to be set (not None) if balance_primary is None.")
        self.balance_primary = balance_primary
        self.balance_secondary = balance_secondary

        self.zero_percentages = zero_percentages

        self.gene_expr_loss = get_loss(
            loss_type=loss_type, num_classes=self.n_input_bins if self.n_input_bins else None, scale_zero_expression=scale_zero_expression)

        self.__create_vocab(vocab)
        
        self.train_loader = None
        self.eval_loader = None

        self.train_loader = self.__create_datasets(data_path=train_data_path, train=True)
        self.eval_loader = self.__create_datasets(data_path=eval_data_path, train=False)

        self.__set_model(mvc_decoder_style=mvc_decoder_style, gene_expr_out_dim=self.gene_expr_loss.get_in_dim())

    def __initiate_wandb(self, run_id: str = None):
        assert (self.resume_from_checkpoint != None) == (run_id != None)
        self.accelerator.init_trackers(
            project_name=self.wandb,
            config=self.args,
            init_kwargs={
                "wandb": (
                    {
                        "name": self.save_dir,
                        "resume": "allow",
                    }
                    if self.resume_from_checkpoint == None
                    else {"name": self.save_dir, "resume": "must", "id": run_id}
                )
            },
        )

    def __create_vocab(self, vocab: Union[str, np.array]):
        if isinstance(vocab, str) and vocab.endswith(".json"):
            with open(vocab, 'r') as file:
                # Load the JSON data from the file
                vocab_dict = json.load(file)
            vocab = GeneVocab.from_dict(vocab_dict)
        elif isinstance(vocab, str) and vocab.endswith(".npy"):
            genes = np.load(vocab,
                            allow_pickle=True).tolist()
            vocab = GeneVocab(gene_list_or_vocab=genes,
                            specials=self.special_tokens)
        else:
            raise ValueError("Unsupported vocab parameter.")
            
        self.vocab = vocab
    
    def __create_datasets(
        self,
        data_path: Union[str, os.PathLike],
        train: bool,
    ):
        collator = AnnDataCollator(
            do_padding=True if self.max_seq_len is not None else False,
            pad_token_id=self.vocab[self.pad_token],
            pad_value=self.pad_value,
            do_mlm=True,
            do_binning=True if self.input_style == "binned" else False,
            mlm_probability=self.mask_ratio,
            mask_value=self.mask_value,
            max_length=self.max_seq_len,
            sampling=self.TRUNC_BY_SAMPLE,
            data_style=self.training_tasks,
            cls_token_id=self.vocab([self.cls_token])[0],
            n_bins=self.n_bins if self.input_style == "binned" else None,
            conditions=self.conditions,
            zero_percentages=self.zero_percentages,
        )

        if not (self.conditions or self.balance_primary or self.balance_secondary):
            obs_keys = None
        else:
            obs_keys = []
            if self.conditions:
                obs_keys += self.conditions
            if self.balance_primary:
                obs_keys.append(self.balance_primary)
                if self.balance_secondary:
                    obs_keys.append(self.balance_secondary)
            obs_keys = list(set(obs_keys))

        with self.accelerator.main_process_first():
            dataset = scDataset(
                path = data_path,
                metadata = obs_keys,
                test_metadata_completeness = True,
            )

            if train and self.conditions:
                self.conditions_nums = {}
                for cond in self.conditions:
                    self.conditions_nums[cond] = dataset.get_metadata_cardinality(cond)

            batch_size = self.batch_size if train else self.eval_batch_size

            if self.balance_primary and train:
                sampler = get_balanced_sampler(dataset, primary_condition=self.balance_primary, secondary_condition=self.balance_secondary, oversample=True)
            else:
                sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collator,
                drop_last=False,
                num_workers=min(
                    len(os.sched_getaffinity(0)), batch_size),
                pin_memory=True,
                prefetch_factor=4,
            )

    def __set_model(self, mvc_decoder_style: str, gene_expr_out_dim: int):

        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.embsize,
            out_dim=gene_expr_out_dim,
            mvc_decoder_style=mvc_decoder_style,
            nhead=self.nheads,
            d_hid=self.d_hid,
            nlayers=self.nlayers,
            vocab=self.vocab,
            dropout=self.dropout,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            do_mvc=self.MVC,
            conditions=self.conditions_nums,
            input_emb_style=self.input_emb_style,
            n_input_bins=self.n_input_bins,
            use_generative_training=self.USE_GENERATIVE_TRAINING,
            do_dat=self.do_dat,
            explicit_zero_prob=self.explicit_zero_prob,
        )

    def accelerate(self):
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.eval_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.eval_loader,
        )

    def __setup_training_variables(self, epochs: int) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.warmup_ratio_or_step > 0:
            total_num_batches = len(self.train_loader) * epochs
            warmup_steps = (
                int(total_num_batches * self.warmup_ratio_or_step)
                if self.warmup_ratio_or_step < 1
                else int(self.warmup_ratio_or_step)
            )

            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_num_batches,
                last_epoch=-1,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.scheduler_interval, gamma=self.scheduler_factor
            )

    def checkpoint(self, epoch: int):
        self.accelerator.print("Checkpointing...")
        path = f"{self.save_dir}/epoch_{epoch}"
        if self.accelerator.is_main_process:
            os.makedirs(path)
            os.makedirs(f"{path}/accelerate")
            with open(f"{path}/info.json", "w") as json_file:
                info = {
                    "epoch": epoch,
                    "best_val_loss": self.best_val_loss,
                    "run_id": (
                        self.accelerator.get_tracker("wandb").run.id
                        if self.wandb != None
                        else None
                    ),
                }
                json.dump(info, json_file, indent=4)

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(f"{path}/accelerate")
        self.accelerator.print("Checkpointing done!")

    def __log(self, metrics: Dict):

        for key, val in metrics.items():
            metrics[key] = val.item()
        self.accelerator.print("Logging...")
        if self.wandb != None:
            self.accelerator.log(metrics)
        else:
            if self.accelerator.is_main_process:
                path = f"{self.save_dir}/log.json"
                with open(path, "r") as file:
                    data = json.load(file)
                data.append(metrics)
                with open(path, "w") as file:
                    json.dump(data, file, indent=4)
        self.timer = None

    def load_model(self, pretrained_model_path: str, verbose: bool = True) -> Union[torch.nn.Module, None]:
        if pretrained_model_path.endswith(".safetensors"):
            tensors = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k).to(self.accelerator.device) if self.accelerator else f.get_tensor(k)
        elif pretrained_model_path.endswith(".pth") or pretrained_model_path.endswith(".pt"):
            tensors = torch.load(pretrained_model_path)
        else:
            self.accelerator.print("Unsupported file format.")
            return None
        return load_pretrained(self.model, tensors, verbose=verbose)
    
    def setup_training(self, epochs: int, pretrained_model_path: Optional[str] = None) -> None:
        self.__setup_training_variables(epochs)

        if pretrained_model_path:
            self.model = self.load_model(pretrained_model_path)

        self.accelerate()

        if self.resume_from_checkpoint != None:
            self.accelerator.print(
                f"Resume from checkpoint: {self.resume_from_checkpoint}"
            )
            self.save_dir = self.resume_from_checkpoint.rsplit('/', 1)[0]
            self.accelerator.load_state(
                f"{self.resume_from_checkpoint}/accelerate")
            with open(f"{self.resume_from_checkpoint}/info.json", "r") as file:
                # Load the JSON data
                data = json.load(file)
            self.starting_epoch = data["epoch"] + 1
            self.best_val_loss = data["best_val_loss"]
        else:
            if self.accelerator.is_main_process:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                # Create the directory
                path = f"{self.save_dir}/log.json"
                with open(path, "w") as file:
                    json.dump([], file, indent=4)

        if self.wandb != None:
            run_id = None
            if self.resume_from_checkpoint != None:
                run_id = data["run_id"]
            self.__initiate_wandb(run_id=run_id)

        self.has_setup_trainer = True

    @with_sdp_kernel
    def train(self, epoch: int, log_interval: int) -> None:
        """
        Evaluate the model on the validation dataset.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """

        criterion = self.gene_expr_loss

        if self.conditions:
            criterion_conditions = nn.CrossEntropyLoss()

        self.model.train()
        total_loss, total_expr, total_gen, total_mvc, total_error, total_cond = [
            torch.tensor(0.0, device=self.accelerator.device) for _ in range(6)
        ]

        num_batches = len(self.train_loader)

        active_dataloader = self.train_loader

        for batch, data_dict in enumerate(active_dataloader):
            if self.timer == None:
                self.timer = time.time()
            with self.accelerator.accumulate(self.model):
                global_iter = (epoch + 1) * num_batches + batch

                conditions_batch = data_dict["conditions"] if self.conditions else None

                if self.USE_GENERATIVE_TRAINING:
                    pcpt_gene = data_dict["pcpt_gene"]
                    pcpt_expr = data_dict["pcpt_expr"]
                    pcpt_key_padding_mask = pcpt_gene.eq(
                        self.vocab[self.pad_token])
                    gen_gene = data_dict["gen_gene"]
                    gen_expr_target = target_values = data_dict["gen_expr_target"]
                    gen_key_padding_mask = gen_gene.eq(
                        self.vocab[self.pad_token])
                else:
                    input_gene_ids = data_dict["gene"]
                    input_values = data_dict["masked_expr"]
                    target_values = data_dict["expr"]
                    src_key_padding_mask = input_gene_ids.eq(
                        self.vocab[self.pad_token])

                if self.USE_GENERATIVE_TRAINING:
                    output_dict = self.model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        MVC=self.MVC,
                        generative_training=True,
                        conditions=conditions_batch,
                    )
                    gen_expr_preds = output_values = output_dict["gen_preds"]

                    positions_to_match = ~gen_key_padding_mask

                    loss = loss_expr = criterion(
                        gen_expr_preds, gen_expr_target, positions_to_match
                    )

                    if self.MVC:
                        loss_mvc = criterion(
                            output_dict["mvc_output"][:, pcpt_gene.shape[1]:], gen_expr_target, positions_to_match)
                        loss = loss + loss_mvc
                    
                    if self.explicit_zero_prob:
                        loss_zero_log_prob = criterion_neg_log_bernoulli(
                            output_dict["mlm_zero_probs"], gen_expr_target, positions_to_match
                        )
                        loss = loss + loss_zero_log_prob

                        if self.MVC:
                            loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                                output_dict["mvc_zero_probs"], gen_expr_target, positions_to_match
                            )
                            loss = loss + loss_gepc_zero_log_prob

                else:
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        MVC=self.MVC,
                        generative_training=False,
                        conditions=conditions_batch,
                    )
                    output_values = output_dict["mlm_output"]

                    positions_to_match = input_values.eq(
                        self.mask_value
                    )  # the postions to predict
                    loss = loss_expr = criterion(
                        output_values, target_values, positions_to_match
                    )

                    if self.MVC:
                        loss_mvc = criterion(
                            output_dict["mvc_output"], target_values, positions_to_match
                        )
                        loss = loss + loss_mvc
                    
                    if self.explicit_zero_prob:
                        loss_zero_log_prob = criterion_neg_log_bernoulli(
                            output_dict["mlm_zero_probs"], target_values, positions_to_match
                        )
                        loss = loss + loss_zero_log_prob

                        if self.MVC:
                            loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                                output_dict["mvc_zero_probs"], target_values, positions_to_match
                            )
                            #print("loss gepc_ ", loss_gepc_zero_log_prob)
                            loss = loss + loss_gepc_zero_log_prob
                if self.do_dat:
                    if self.conditions:
                        loss_conditions = torch.zeros(
                            loss.shape).to(total_cond.device)
                        for condition in self.conditions:
                            loss_conditions += criterion_conditions(
                                output_dict["condition_output"][condition], conditions_batch[condition].squeeze())
                        loss_conditions /= len(self.conditions)
                        loss += loss_conditions

                if self.USE_GENERATIVE_TRAINING and global_iter > 1000:
                    previous_cell_embs = output_dict["cell_emb"].detach()
                    preds = self.model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        MVC=False,
                        input_cell_emb=previous_cell_embs,
                        generative_training=True,
                        conditions=conditions_batch
                    )["gen_preds"]

                    loss_gen = criterion(
                        preds, gen_expr_target, positions_to_match)
                    loss = loss + loss_gen

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    if self.loss_type != "mse":
                        mre = torch.zeros(1)
                    else:
                        mre = masked_relative_error(
                            output_values, target_values, positions_to_match
                        )

                total_loss += loss
                total_expr += loss_expr

                total_gen += (
                    loss_gen
                    if "loss_gen" in locals()
                    else torch.tensor(0.0, device=self.accelerator.device)
                )
                total_mvc += (
                    loss_mvc
                    if self.MVC
                    else torch.tensor(0.0, device=self.accelerator.device)
                )

                if self.loss_type == "mse":
                    total_error += mre
                total_cond += (
                    loss_conditions
                    if self.conditions and self.do_dat
                    else torch.tensor(0.0, device=self.accelerator.device)
                )

                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    cur_expr = total_expr / log_interval

                    cur_gen = (
                        total_gen / log_interval
                        if "loss_gen" in locals()
                        else torch.tensor(0.0, device=self.accelerator.device)
                    )
                    cur_mvc = (
                        total_mvc / log_interval
                        if self.MVC
                        else torch.tensor(0.0, device=self.accelerator.device)
                    )
                    cur_error = total_error / log_interval
                    cur_cond = (
                        total_cond / log_interval
                        if self.conditions
                        else torch.tensor(0.0, device=self.accelerator.device)
                    )

                    cur_loss, cur_expr, cur_gen, cur_mvc, cur_error, cur_cond = (
                        self.accelerator.gather(
                            (cur_loss, cur_expr,
                             cur_gen, cur_mvc, cur_error, cur_cond)
                        )
                    )
                    metrics = {
                        "train/total_loss": cur_loss.mean(),
                    }

                    if self.USE_GENERATIVE_TRAINING:
                        metrics["train/gen"] = cur_gen.mean()
                    if self.MVC:
                        metrics["train/mvc"] = cur_mvc.mean()
                    if self.conditions:
                        metrics["train/cond"] = cur_cond.mean()
                    if self.loss_type == "mse":
                        metrics["train/mse"] = cur_expr.mean()
                        metrics["train/mre"] = cur_error.mean()
                    else:
                        metrics[f"train/{self.loss_type.value}"] = cur_expr.mean()

                    self.__log(metrics)

                    total_loss, total_expr, total_gen, total_mvc, total_error, total_cond = [
                        torch.tensor(0.0, device=self.accelerator.device) for _ in range(6)
                    ]

    @ with_sdp_kernel
    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        criterion = self.gene_expr_loss

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        total_error = torch.tensor(0.0, device=self.accelerator.device)

        valid_loader = self.eval_loader
        with torch.no_grad():
            for batch, data_dict in enumerate(valid_loader):
                conditions_batch = data_dict["conditions"] if self.conditions else None

                if self.USE_GENERATIVE_TRAINING:
                    pcpt_gene = data_dict["pcpt_gene"]
                    pcpt_expr = data_dict["pcpt_expr"]
                    pcpt_key_padding_mask = pcpt_gene.eq(
                        self.vocab[self.pad_token])
                    gen_gene = data_dict["gen_gene"]
                    gen_expr_target = target_values = data_dict["gen_expr_target"]
                    gen_key_padding_mask = gen_gene.eq(
                        self.vocab[self.pad_token])
                else:
                    input_gene_ids = data_dict["gene"]
                    input_values = data_dict["masked_expr"]
                    target_values = data_dict["expr"]
                    src_key_padding_mask = input_gene_ids.eq(
                        self.vocab[self.pad_token])

                if self.USE_GENERATIVE_TRAINING:
                    output_dict = self.model(
                        pcpt_gene,
                        pcpt_expr,
                        pcpt_key_padding_mask,
                        gen_gene,
                        gen_key_padding_mask,
                        MVC=False,
                        generative_training=True,
                        conditions=conditions_batch,
                    )
                    gen_expr_preds = output_values = output_dict["gen_preds"]

                    positions_to_match = ~gen_key_padding_mask
                else:
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        CCE=False,
                        MVC=False,
                        generative_training=False,
                        conditions=conditions_batch,
                    )
                    output_values = output_dict["mlm_output"]
                    positions_to_match = input_values.eq(self.mask_value)

                loss = criterion(output_values, target_values,
                                 positions_to_match)

                total_loss += loss

                if self.loss_type == "mse":
                    total_error += masked_relative_error(
                        output_values, target_values, positions_to_match
                    )

            total_loss = total_loss / len(valid_loader)
            total_error = total_error / len(valid_loader)

            total_loss, total_error = (
                self.accelerator.gather(
                    (total_loss, total_error)
                ))  # TODO: This is not ideal. Should use gather_for_metrics instead
            total_loss, total_error = total_loss.mean(), total_error.mean()

            best = total_loss < self.best_val_loss["loss"]
            if best:
                self.best_val_loss = {
                    "loss": total_loss.item(), "epoch": epoch}

            if self.loss_type == "mse":
                metrics = {
                    "eval/mse": total_loss,
                    "eval/mre": total_error,
                }
            else:
                metrics = {f"eval/{self.loss_type.value}": total_loss}

            self.__log(metrics)
