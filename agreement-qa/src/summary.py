from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)
import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from functools import partial

import wandb


wandb_run = wandb.init(
    # set the wandb project where this run will be logged
    project="agreement-tracking",
    # track hyperparameters and run metadata
    config={
        "none_prob": 0.3,
        "ctx_window_size": 4,
        "max_train_epochs": 20,
        "use_early_stopping": True,
        "decoder_type": "bart",
        "decoder_name": "facebook/bart-large",
        "num_steps_between_evals": 500,
        "seed": 42,
        "train_split_n": 10,
        "train_data": "proposals_A_0_proposals_B_1_partial_rejections_A_3_partial_rejections_B_1_partial_accepts_A_3_partial_accepts_B_2",
        "evaluate_all_dfs": True,
    }
)
ROOT_DIR = Path("data/simulated")
SLOTS = {'Working Hours', 'Salary', 'Leased Car', 'Pension Fund', 'Promotion Possibilities', 'Job Description'}


def init_model():
    model_args = Seq2SeqArgs(
        num_train_epochs=wandb_run.config.max_train_epochs,
        no_save=True,
        logging_steps=100,
        evaluate_generated_text=True,
        overwrite_output_dir=True,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True,
        use_early_stopping=wandb_run.config.use_early_stopping,
        early_stopping_delta=0,
        wandb_project="agreement-tracking",
        early_stopping_metric="eval_delta_acc",
        early_stopping_metric_minimize=False,
        early_stopping_patience=2,
        evaluate_during_training_steps=wandb_run.config.num_steps_between_evals,
    )
    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type=wandb_run.config.decoder_type,
        encoder_decoder_name=wandb_run.config.decoder_name,
        args=model_args,
        use_cuda=True,
    )
    return model


def stringify_state(agreements):
    # Randomize order of slots and stringify
    slots = list(agreements.keys())
    random.shuffle(slots)
    return " ".join([f"{slot}={agreements[slot]}" for slot in slots])


def get_utterance(turn):
    if "A" in turn:
        return turn["A"].strip()
    elif "B" in turn:
        return turn["B"].strip()
    return "Go on."


def dials_to_pandas_df(split, data_name):
    dials = json.load(
        Path(ROOT_DIR / split / f"{data_name}.json"
    ).open("r"))
    samples = []
    for dial in dials:
        for i, turn in enumerate(dial["turns"]):
            unmentioned_slots = SLOTS - set(turn["agreements"].keys())
            if unmentioned_slots and random.random() < wandb_run.config.none_prob:
                rand_unmentioned_slots = random.choice(list(unmentioned_slots))
                prev_value =  dial["turns"][max(0, i - wandb_run.config.ctx_window_size)]["agreements"].get(rand_unmentioned_slots, "none")
                samples.append({
                    "target_text": "none",
                    "input_text": f"""[CONTEXT] {' [ENDOFTURN] '.join([get_utterance(turn) for turn in dial["turns"][i - wandb_run.config.ctx_window_size + 1 : i + 1]])} [ENDOFDIALOGUE] [QUESTION] What are the newly added agreements for {rand_unmentioned_slots} between the two speakers in the dialogue? [PREV] {prev_value} [ANSWER] """
                })
            else:
                for slot_key, slot_value in turn["agreements"].items():
                    prev_value = dial["turns"][max(0, i - wandb_run.config.ctx_window_size)]["agreements"].get(slot_key, "none")
                    samples.append({
                        "target_text": slot_value,
                        "input_text": f"""[CONTEXT] {' [ENDOFTURN] '.join([get_utterance(turn) for turn in dial["turns"][i - wandb_run.config.ctx_window_size + 1 : i + 1]])} [ENDOFDIALOGUE] [QUESTION] What are the newly added agreements for {slot_key} between the two speakers in the dialogue? [PREV] {prev_value} [ANSWER] """
                    })
    random.shuffle(samples)
    return pd.DataFrame(samples)


def load_data():
    train_df = dials_to_pandas_df("train", data_name=wandb_run.config.train_data)
    train_df = train_df.sample(
        n=wandb_run.config.train_split_n,
        random_state=wandb_run.config.seed
    ).reset_index(drop=True)

    # Get all filenames in the val and test splits
    val_dfs = {
        filename.stem: dials_to_pandas_df("val", data_name=filename.stem)
        for filename in Path(ROOT_DIR / "val").glob("*.json")
    }
    test_dfs = {
        filename.stem: dials_to_pandas_df("test", data_name=filename.stem)
        for filename in Path(ROOT_DIR / "test").glob("*.json")
    }

    return train_df, val_dfs, test_dfs


def save_model(model, MODEL_DIR):
    model.config.save_pretrained(MODEL_DIR)
    model.model.save_pretrained(MODEL_DIR)
    model.decoder_tokenizer.save_pretrained(MODEL_DIR)


def main():
    random.seed(wandb_run.config.seed)

    # Load data
    train_df, val_dfs, test_dfs = load_data()
    val_df = val_dfs[wandb_run.config.train_data]
    test_df = test_dfs[wandb_run.config.train_data]

    # Define evaluation metric
    def eval_delta_acc(labels, preds):
        print(f"Predictions: {preds}")
        print(f"Labels: {labels}")
        accuracy = (np.array(preds) == np.array(labels)).mean()
        return accuracy

    # Train and evaluate the model
    model = init_model()
    model.train_model(
        train_df,
        eval_data=val_df,
        eval_delta_acc=eval_delta_acc
    )

    if wandb_run.config.evaluate_all_dfs:
        eval_results_over_all_dfs = {
            f"final_eval_result_{key}": model.eval_model(val_df, eval_delta_acc=eval_delta_acc)["eval_delta_acc"]
            for key, val_df in val_dfs.items()
        }
        test_results_over_all_dfs = {
            f"final_test_result_{key}": model.eval_model(test_df, eval_delta_acc=eval_delta_acc)["eval_delta_acc"]
            for key, test_df in test_dfs.items()
        }
    else:
        eval_results_over_all_dfs = {
            "final_eval_result": model.eval_model(val_df, eval_delta_acc=eval_delta_acc)["eval_delta_acc"]
        }
        test_results_over_all_dfs = {
            "final_test_result": model.eval_model(test_df, eval_delta_acc=eval_delta_acc)["eval_delta_acc"]
        }

    save_model(model, f"models/{wandb_run.config.train_data}")

    # Log results
    wandb.log({
        "num_train_examples": len(train_df),
        "num_val_examples": len(val_df),
        "num_test_examples": len(test_df),
        **eval_results_over_all_dfs,
        **test_results_over_all_dfs,
    })

    # Sync wandb
    wandb.join()


if __name__ == "__main__":
    main()
