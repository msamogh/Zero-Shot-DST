import os
from pathlib import Path
from typing import *

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    AdamW,
    T5Tokenizer,
    BartTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    WEIGHTS_NAME,
    CONFIG_NAME,
)


class DST_Seq2Seq(pl.LightningModule):
    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.clf = torch.nn.Linear(self.model.config.d_model, 1)
        self.save_hyperparameters(ignore=["model"])

    def loss_fn(self, batch):
        output = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = output["decoder_hidden_states"]
        y_pos_sample = F.sigmoid(self.clf(hidden_states[-1][:, -1]))
        pos_mask = (
            (~torch.tensor(batch["is_negative_sample"])).long().to(batch["encoder_input"])
        )
        neg_mask = (
            torch.tensor(batch["is_negative_sample"]).long().to(batch["encoder_input"])
        )
        neg_sampling_loss = -(torch.sum(pos_mask * torch.log(y_pos_sample)) \
                            + torch.sum(neg_mask * torch.log(1 - y_pos_sample)))
        print(f"lm_loss: {output['loss']}; neg_sampling_loss: {neg_sampling_loss}")

        return output["loss"] + neg_sampling_loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = self.loss_fn(batch)
        del batch
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        output = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
            return_dict=True,
        )
        del batch
        loss = output["loss"]
        del output
        self.log("val/loss", loss, sync_dist=True)

        # Run evaluation

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.args["lr"], correct_bias=True)


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, experiment_id: str, every: int, dirpath, fine_tune_prefix=""):
        super().__init__(dirpath=dirpath)
        self.every = every
        self.experiment_id = experiment_id
        self.fine_tune_prefix = fine_tune_prefix

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        if pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = (
                Path(self.dirpath)
                / f"{self.fine_tune_prefix}{self.experiment_id}-{pl_module.global_step}.ckpt"
            )
            pl_module.model.save_pretrained(
                Path(self.dirpath)
                / f"{self.fine_tune_prefix}{self.experiment_id}-{pl_module.global_step}"
            )
            trainer.save_checkpoint(current)
        # Keep a few around
        if pl_module.global_step % (self.every * 5) != 0:
            prev = (
                Path(self.dirpath)
                / f"{self.fine_tune_prefix}{self.experiment_id}-{pl_module.global_step - self.every}.ckpt"
            )
            prev_t5 = (
                Path(self.dirpath)
                / f"{self.fine_tune_prefix}{self.experiment_id}-{pl_module.global_step - self.every}.ckpt"
            )
            prev_t5.unlink(missing_ok=True)
            prev.unlink(missing_ok=True)


def get_save_path(args):
    save_path = os.path.join(args["saving_dir"], args["wandb_run_id"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def get_model_and_tokenizer(args):
    model = T5ForConditionalGeneration.from_pretrained(args["hf_checkpoint"])
    tokenizer = T5Tokenizer.from_pretrained(
        args["base_model"],
        bos_token="[bos]",
        eos_token="[eos]",
        sep_token="[sep]",
        model_max_length=200,
    )
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    return model, tokenizer
