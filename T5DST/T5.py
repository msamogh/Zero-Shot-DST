# Copyright (c) Facebook, Inc. and its affiliates

import os, random
from pathlib import Path
import torch
torch.cuda.empty_cache()
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME)
from data_loader import prepare_data, predict_dataloader, prepare_predict
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import Counter

# def consistency_cross_entropy(lm_logits1, lm_logits2, threshold=0.4):
#     logsoftmax = torch.nn.LogSoftmax(dim=1)
#     softmax = torch.nn.Softmax(dim=1)

#     lm_logits1 = lm_logits1.squeeze()
#     lm_logits2 = lm_logits2.squeeze()
#     # (batch, vocab_size)
#     # give threshold
#     prob2 = softmax(lm_logits2)
#     # the result tuple of two output tensors (max, max_indices)
#     # print(torch.max(prob2, dim=1))
#     prob2_max, prob2_index = torch.max(prob2, dim=1)
#     valid = []
#     for i in range(prob2_max.shape[0]):
#         if (prob2_index[i]==5839 and prob2_max[i]>0.9) or (prob2_index[i]!=5839 and prob2_max[i]>threshold):
#             valid.append(1)
#         else:
#             valid.append(0)

#     #sharpening
#     soft_targets = softmax(lm_logits2/0.5)

#     loss_temp = torch.sum(- soft_targets * logsoftmax(lm_logits1), 1)
#     for i in range(prob2_max.shape[0]):
#         if valid[i]==0:
#             loss_temp[i]=0

#     return torch.mean(loss_temp)

class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, dirpath):
        super().__init__(dirpath=dirpath)
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            pl_module.model.save_pretrained(Path(self.dirpath) / f"t5-{pl_module.global_step}")
            # torch.save(Path(self.dirpath) / pl_module.model.state_dict(), f"t5-{pl_module.global_step}.ckpt")
            trainer.save_checkpoint(current)
        # Keep a few around
        if pl_module.global_step % (self.every * 5) != 0:
            prev = (
                Path(self.dirpath) / f"latest-{pl_module.global_step - self.every}.ckpt"
            )
            prev_t5 = (
                Path(self.dirpath) / f"t5-{pl_module.global_step - self.every}.ckpt"
            )
            prev_t5.unlink(missing_ok=True)
            prev.unlink(missing_ok=True)


class DST_Seq2Seq(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]
        # self.ddp = torch.nn.DataParallel(model)


    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
            return_dict=True
        )
        loss = output.loss

        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    # def validation_step(self, batch, batch_idx):
    #     self.model.eval()
    #     (loss), *_ = self.model(input_ids=batch["encoder_input"],
    #                         attention_mask=batch["attention_mask"],
    #                         labels=batch["decoder_output"]
    #                         )


    #     return {'val_loss': loss, 'log': {'val_loss': loss}}
    #     # return result

    # def validation_epoch_end(self, outputs):
    #     # print(f"Val losses: {outputs[0]['val_loss']}")
    #     # val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
    #     # # show val_loss in progress bar but only log val_loss
    #     # results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
    #     #            'val_loss': val_loss_mean.item()}
    #     return {'progress_bar': {'val_loss': 0}, 'log': {'val_loss': 0},
    #                'val_loss': 0}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)



def train(args, *more):
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"]+"_except_domain_"+args["except_domain"]+ "_slotlang_" +str(args["slot_lang"]) + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    seed_everything(args["seed"])

    print("Training!")

    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    elif "bart" in args["model_name"]:
        model = BartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = BartTokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    
    task = DST_Seq2Seq(args, tokenizer, model)

    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = PeriodicCheckpoint(every=200, dirpath=args["saving_dir"])

    trainer = Trainer(
        default_root_dir=save_path,
        accumulate_grad_batches=args["gradient_accumulation_steps"],
        gradient_clip_val=args["max_norm"],
        max_epochs=args["n_epochs"],
        callbacks=[checkpoint, pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
        devices=args["GPU"],
        deterministic=True,
        num_nodes=1,
        #precision=16,
        accelerator="cuda",
    )

    trainer.fit(task, train_loader)

    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    _ = evaluate_model(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS)

def evaluate_model(args, tokenizer, model, test_loader, save_path, ALL_SLOTS, prefix="zeroshot"):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    for batch in tqdm(test_loader):
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=200,
                                )

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            with open("debug.txt", "a+") as f:
                f.write(f"Input text: {batch['intput_text'][idx]}\n")
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}

            if value!="none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value))

            # analyze slot acc:
            if str(value)==str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1]+=1 # hit
            slot_logger[str(batch["slot_text"][idx])][0]+=1 # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1]/slot_log[0]

    with open(os.path.join(save_path, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    with open(os.path.join(save_path, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"{prefix} result:",evaluation_metrics)

    with open(os.path.join(save_path, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions


def fine_tune(args, *more):
    args = vars(args)
    seed_everything(args["seed"])
    domains = ["hotel", "train", "restaurant", "attraction", "taxi"]
    for domain in domains:
        if domain in args["model_checkpoint"]:
            args["only_domain"] = domain
    assert args["only_domain"]!="none"
    # args["model_checkpoint"] = os.path.join(args["saving_dir"],args["model_name"])
    print(args)

    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
    elif "bart" in args["model_name"]:
        model = BartForConditionalGeneration.from_pretrained(args["model_checkpoint"])
        tokenizer = BartTokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")

    task = DST_Seq2Seq(args, tokenizer, model)
    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, tokenizer)


    checkpoint = ModelCheckpoint(
        dirpath="save2",
        save_last=True,
        every_n_train_steps=5
    )
    trainer = Trainer(
                    default_root_dir=args["model_checkpoint"],
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=20,
                    callbacks=[checkpoint, pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=8,verbose=False, mode='min')],
                    devices=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    # precision=16,
                    strategy="ddp",
                    accelerator="cuda"
                )

    trainer.fit(task, train_loader, val_loader)

    print("test start...")
    #evaluate model
    ratio = "ratio_" + str(args["fewshot"]) + "_seed_" + str(args["seed"])
    _ = evaluate_model(args, task.tokenizer, task.model, test_loader, args["model_checkpoint"], ALL_SLOTS, prefix=ratio)


def predict(args, *more):
    args = vars(args)
    seed_everything(args["seed"])
    
    model = T5ForConditionalGeneration.from_pretrained(args["ckpt_path"])
    tokenizer = T5Tokenizer.from_pretrained(
        args["model_checkpoint"], bos_token="[bos]",
        eos_token="[eos]", sep_token="[sep]"
    )

    ALL_SLOTS, test_loader = prepare_predict(args, tokenizer, path_test="data/escai_dials_2.json")
    # _, _, test_loader, ALL_SLOTS, _, _ = prepare_data(args, tokenizer, path_test="data/escai_dials.json")
    ratio = "ratio_" + str(args["fewshot"]) + "_seed_" + str(args["seed"])

    predictions = evaluate_model(
        args, tokenizer, model, test_loader,
        "escai-predictions", ALL_SLOTS, prefix=ratio
    )

    from pprint import pprint
    with open("predictions.txt", "w") as f:
        pprint(predictions, f)

    return predictions


if __name__ == "__main__":
    args = get_args()
    if args.mode=="train":
        train(args)
    elif args.mode=="finetune":
        fine_tune(args)
    elif args.mode == "predict":
        predict(args)
