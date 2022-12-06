# Copyright (c) Facebook, Inc. and its affiliates

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--verbose", action="store_true", default=False)

    add_training_args(parser)
    add_finetuning_args(parser)
    add_model_checkpoint_args(parser)
    add_dataset_args(parser)
    add_ontology_args(parser)
    add_decoding_args(parser)
    add_predict_args(parser)
    add_t5_args(parser)
    add_experiment_args(parser)
    add_collab_dialogue_args(parser)
    add_negative_sampling_args(parser)
    add_offset_too_many_nones_args(parser)

    return parser.parse_args()


def add_offset_too_many_nones_args(parser):
    parser.add_argument("--keep_none_prob", type=float, default=1.0)


def add_negative_sampling_args(parser):
    parser.add_argument(
        "--do_negative_sampling_data", action="store_true", default=False
    )
    parser.add_argument("--ns_hinge_loss_weight", type=float, default=0)
    parser.add_argument("--ns_hinge_loss_margin", type=float, default=0)
    parser.add_argument("--ns_ratio", type=float, default=1)
    parser.add_argument("--ns_loss_fn_weight", type=float, default=1)


def add_decoding_args(parser):
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1)
    parser.add_argument(
        "--impose_ontology_constraints", action="store_true", default=False
    )


def add_training_args(parser):
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--meta_batch_size", type=int, default=1, help="Batch size for meta training"
    )
    parser.add_argument(
        "--dev_batch_size", type=int, default=16, help="Batch size for validation"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size for test"
    )
    parser.add_argument(
        "--predict_batch_size", type=int, default=16, help="Batch size for predict"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--GPU", type=int, default=8, help="number of gpu to use")
    parser.add_argument("--precision", type=str, default="16")


def add_finetuning_args(parser):
    parser.add_argument("--finetune_n_samples", type=int)


def add_model_checkpoint_args(parser):
    parser.add_argument("--pl_checkpoint", type=str, default=None)
    parser.add_argument(
        "--hf_checkpoint",
        type=str,
        default="t5-small",
        help="Path, url or short name of the model",
    )
    parser.add_argument(
        "--saving_dir", type=str, default="save", help="Path for saving"
    )
    parser.add_argument("--finetune_from_ckpt", type=str, default=None)


def add_t5_args(parser):
    parser.add_argument(
        "--max_history", type=int, default=4, help="max number of turns in the dialogue"
    )
    parser.add_argument(
        "--base_model", type=str, default="t5-small", help="use t5 or bart?"
    )
    parser.add_argument(
        "--slot_lang",
        type=str,
        default="none",
        help="use 'none', 'human', 'naive', 'value', 'question', 'slottype', 'slot description'",
    )


def add_dataset_args(parser):
    parser.add_argument("--fix_label", action="store_true")
    parser.add_argument("--path_train", type=str)
    parser.add_argument("--path_dev", type=str)
    parser.add_argument("--path_test", type=str)
    parser.add_argument("--path_predict", type=str)


def add_ontology_args(parser):
    parser.add_argument(
        "--ontology_path", type=str, default="data/multi-woz/MULTIWOZ2 2/ontology.json"
    )
    parser.add_argument(
        "--slot_descriptions_path",
        type=str,
        default="escai/ontology/slot_description.json",
    )


def add_predict_args(parser):
    parser.add_argument("--predictions_output_path", type=str)


def add_experiment_args(parser):
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, required=True)


def add_collab_dialogue_args(parser):
    parser.add_argument("--speaker_label_strategy", type=str)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--baseline_type", type=str, default="none")
