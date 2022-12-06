import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import torch
torch.use_deterministic_algorithms(False, warn_only=True)
import pytorch_lightning as pl

torch.cuda.empty_cache()

from tqdm import tqdm
import wandb

from train_data_loader import prepare_data
from prompt_generator import PromptGenerator
from ontology import Ontology
from evaluate import evaluate_metrics
from train_config import get_args
import train_utils

torch.cuda.empty_cache()


def train(args):
    # Initialize
    args = vars(args)
    seed_everything(args["seed"])
    save_path = train_utils.get_save_path(args)

    # Model and tokenizer
    model, tokenizer = train_utils.get_model_and_tokenizer(args)
    task = train_utils.DST_Seq2Seq(args, tokenizer, model)

    # Ontology
    ontology = Ontology.from_files(
        args["dataset"], args["ontology_path"], args["slot_descriptions_path"]
    )
    # Dialogues
    train_loader = prepare_data(
        args,
        task.tokenizer,
        ontology,
        args["path_train"],
        args["train_batch_size"],
        is_training_split=True,
        do_negative_sampling=args["do_negative_sampling_data"],
        shuffle=True,
    )
    val_loader = prepare_data(
        args,
        task.tokenizer,
        ontology,
        args["path_dev"],
        args["dev_batch_size"],
        is_training_split=False,
        do_negative_sampling=False,
    )
    test_loader = prepare_data(
        args,
        task.tokenizer,
        ontology,
        args["path_test"],
        args["test_batch_size"],
        is_training_split=False,
        do_negative_sampling=False,
    )

    # Trainer
    if args["mode"] == "finetune":
        ckpt_args = {"ckpt_path": args["finetune_from_ckpt"]}
    else:
        ckpt_args = {}

    wandb_logger = WandbLogger(project="collaborative-dst", log_model="all")

    trainer = get_trainer(args, save_path)
    trainer.fit(task, train_loader, val_loader, **ckpt_args)

    # Save
    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    # Evaluate
    _ = evaluate_model(
        args, task.tokenizer, task.model, test_loader, save_path, ontology
    )


def generate_ontology_constraints(batch, ontology):
    from transformers import DisjunctiveConstraint

    constraints = []
    for sample in batch:
        pass
    return constraints


def decode_model_output(args, predictions, tokenizer, model, batch, device):
    prev_dialogue_state = {}
    if batch["ID"][0] in predictions and batch["turn_id"][0] > 0:
        prev_dialogue_state = predictions[batch["ID"][0]]["turns"][batch["turn_id"][0] - 1]["pred_belief"]

    input_ids, attention_mask = PromptGenerator.prepend_dialogue_history(
        tokenizer,
        batch["encoder_input"],
        batch["attention_mask"],
        prev_dialogue_state
    )

    return tokenizer.batch_decode(
        model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
            num_beams=args["num_beams"],
        ),
        skip_special_tokens=True,
    )


def initialize_prediction_dict(predictions, batch, idx):
    dial_id = batch["ID"][idx]
    if dial_id not in predictions:
        predictions[dial_id] = {
            "domains": batch["domains"][idx][0],
            "turns": {},
        }
    if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
        predictions[dial_id]["turns"][batch["turn_id"][idx]] = {
            "input_text": {},
            "turn_belief": batch["turn_belief"][idx],
            "pred_belief": {},
        }


def populate_value(predictions, batch, idx, value):
    dial_id = batch["ID"][idx]
    if value != "none":
        predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"][
            str(batch["slot_text"][idx])
        ] = str(value)
        predictions[dial_id]["turns"][batch["turn_id"][idx]]["input_text"][
            str(batch["slot_text"][idx])
        ] = batch["input_text"][idx]


def evaluate_model_2(
    args, tokenizer, model, test_loader, save_path, ontology, prefix="zeroshot"
):
    create_dir_if_not_exists(save_path)

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    predictions = dict()
    for sample in tqdm([sample for sample in test_loader if sample["for_evaluation"]]):
        batch_prediction = decode_model_output(
            args, predictions, tokenizer, model, sample, device
        )
        for idx, value in enumerate(batch_prediction):
            initialize_prediction_dict(predictions, sample, idx)
            populate_value(predictions, sample, idx, value)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ontology)
    evaluation_metrics = {
        "JointAcc": joint_acc_score,
        "TurnAcc": turn_acc_score,
        "JointF1": F1_score,
    }
    with open(os.path.join(save_path, f"{prefix}_prediction.json"), "w") as f:
        json.dump(predictions, f, indent=4)
    with open(os.path.join(save_path, f"{prefix}_result.json"), "w") as f:
        wandb.log(evaluation_metrics)
        json.dump(evaluation_metrics, f, indent=4)
    return predictions, evaluation_metrics


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def evaluate_model(
    args, tokenizer, model, test_loader, save_path, ontology, prefix="zeroshot"
):
    save_path = os.path.join(save_path, "results")
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except FileExistsError:
        pass

    predictions = {}
    top_k_predictions = {}

    # to gpu
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name: [0, 0, 0] for slot_name in ontology.slots}

    for batch in tqdm(test_loader):
        with open(os.path.join(save_path, f"{prefix}_top_k_prediction.json"), "w") as f:
            json.dump(top_k_predictions, f, indent=4)

        constraints = []
        if args["impose_ontology_constraints"]:
            constraints = generate_ontology_constraints(batch, ontology)

        # top_k_model_output = model.generate(
        #     input_ids=batch["encoder_input"].to(device),
        #     attention_mask=batch["attention_mask"].to(device),
        #     eos_token_id=tokenizer.eos_token_id,
        #     max_length=200,
        #     num_beams=args["num_beams"],
        #     do_sample=args["do_sample"],
        #     top_k=args["top_k"],
        #     temperature=args["temperature"],
        #     num_return_sequences=args["top_k"],
        #     length_penalty=args["length_penalty"],
        #     constraints=constraints
        # )
        # top_k_value_batch = tokenizer.batch_decode(top_k_model_output, skip_special_tokens=True)
        # # print(args["top_k"])
        # if len(top_k_value_batch) == args["top_k"] * test_loader.batch_size:
        #     top_k_generations = np.array(top_k_value_batch).reshape(test_loader.batch_size, -1)

        #     for idx, generations in enumerate(top_k_generations):
        #         dial_id = batch["ID"][idx]
        #         if dial_id not in top_k_predictions:
        #             top_k_predictions[dial_id] = {
        #                 "domains": batch["domains"][idx][0],
        #                 "turns": {},
        #             }

        #         if batch["turn_id"][idx] not in top_k_predictions[dial_id]["turns"]:
        #             top_k_predictions[dial_id]["turns"][batch["turn_id"][idx]] = {
        #                 "turn_belief": batch["turn_belief"][idx],
        #                 "top_k_pred_belief": [],
        #             }

        #         top_k_predictions[dial_id]["turns"][batch["turn_id"][idx]][
        #             "top_k_pred_belief"
        #         ].extend(
        #             [
        #                 str(batch["slot_text"][idx]) + "-" + str(candidate)
        #                 for candidate in generations
        #             ]
        #         )

        model_output = model.generate(
            input_ids=batch["encoder_input"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
            num_beams=args["num_beams"],
            # do_sample=args["do_sample"],
            # top_k=args["top_k"],
            # temperature=args["temperature"],
        )
        value_batch = tokenizer.batch_decode(model_output, skip_special_tokens=True)
        for idx, value in enumerate(value_batch):
            # Skip chain of thought samples
            if not batch["for_evaluation"]:
                continue

            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {
                    "domains": batch["domains"][idx][0],
                    "turns": {},
                }

            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {
                    "input_text": {},
                    "turn_belief": batch["turn_belief"][idx],
                    "pred_belief": {},
                }

            if value != "none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"][
                    str(batch["slot_text"][idx])
                ] = str(value)
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["input_text"][
                    str(batch["slot_text"][idx])
                ] = batch["input_text"][idx]

            # analyze slot acc:
            if str(value) == str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1] += 1  # hit
            slot_logger[str(batch["slot_text"][idx])][0] += 1  # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1] / max(slot_log[0], 0.0001)  # type: ignore

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ontology)

    evaluation_metrics = {
        "JointAcc": joint_acc_score,
        "TurnAcc": turn_acc_score,
        "JointF1": F1_score,
    }

    with open(os.path.join(save_path, f"{prefix}_slot_acc.json"), "w") as f:
        json.dump(slot_logger, f, indent=4)
    with open(os.path.join(save_path, f"{prefix}_prediction.json"), "w") as f:
        json.dump(predictions, f, indent=4)
    with open(os.path.join(save_path, f"{prefix}_top_k_prediction.json"), "w") as f:
        json.dump(top_k_predictions, f, indent=4)
    with open(os.path.join(save_path, f"{prefix}_result.json"), "w") as f:
        wandb.log(evaluation_metrics)
        json.dump(evaluation_metrics, f, indent=4)

    return predictions, evaluation_metrics


def get_trainer(args, save_path):
    if args["mode"] == "finetune":
        ckpt_prefix = "finetune-"
    elif args["mode"] == "train":
        ckpt_prefix = ""
    else:
        raise RuntimeError

    periodic_checkpoint_callback = train_utils.PeriodicCheckpoint(
        experiment_id=args["wandb_run_id"],
        every=100,
        dirpath=args["saving_dir"],
        fine_tune_prefix=ckpt_prefix,
    )

    return Trainer(
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        val_check_interval=0.75,
        default_root_dir=save_path,
        # logger=wandb_logger,
        accumulate_grad_batches=args["gradient_accumulation_steps"],
        gradient_clip_val=args["max_norm"],
        max_epochs=args["n_epochs"],
        callbacks=[
            periodic_checkpoint_callback,
            pl.callbacks.EarlyStopping(
                monitor="val/loss",
                min_delta=0.10,
                patience=3,
                verbose=True,
                strict=False,
                mode="min",
            ),
        ],
        devices=args["GPU"],
        deterministic=False,
        num_nodes=1,
        strategy="ddp",
        precision=int(args["precision"]),
        accelerator="cuda",
    )


def predict(args, sub_version=""):
    # Initialize
    args = vars(args)
    seed_everything(args["seed"])

    model, tokenizer = train_utils.get_model_and_tokenizer(args)
    ontology = Ontology.from_files(
        args["dataset"], args["ontology_path"], args["slot_descriptions_path"]
    )
    test_loader = prepare_data(
        args,
        tokenizer,
        ontology,
        dials_path=args["path_predict"] + sub_version,
        batch_size=1,
        shuffle=False,
        is_training_split=False,
    )
    predictions, _ = evaluate_model_2(
        args,
        tokenizer,
        model,
        test_loader,
        args["predictions_output_path"],
        ontology,
        prefix=args["wandb_run_id"],
    )
    # predictions, _ = evaluate_model_2(
    #     args,
    #     tokenizer,
    #     model,
    #     test_loader,
    #     args["predictions_output_path"],
    #     ontology,
    #     prefix=args["wandb_run_id"],
    # )
    return predictions


if __name__ == "__main__":
    args = get_args()
    if not args.disable_wandb:
        wandb.init(
            id=args.wandb_run_id,
            project="collaborative-dst",
            entity="msamogh",
            config=vars(args),
        )
    print(f"Running {args.wandb_run_id}")
    if args.mode == "train":
        train(args)
    elif args.mode == "finetune":
        train(args)
    elif args.mode == "predict":
        predict(args)
