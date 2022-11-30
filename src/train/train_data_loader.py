# Copyright (c) Facebook, Inc. and its affiliates

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
from pprint import pprint
import os
import random
from typing import *
from dataclasses import dataclass, field
from functools import partial
from collections import OrderedDict

from fix_label import fix_general_label_error
from ontology import Ontology
from prompt_generator import IndividualPreferences, JointPreferences, NaiveQuestion

random.seed(577)


class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.data[index]

    def __len__(self):
        return len(self.data)


def read_data(args, path_name, ontology, tokenizer, is_training_split):
    dials = json.load(open(path_name, "r"))

    train_prompt_generators = [
        NaiveQuestion(
            args,
            ontology=ontology,
            tokenizer=tokenizer,
            is_training_split=is_training_split,
            max_history=args["max_history"],
        )
        # IndividualPreferences(
        #     args,
        #     ontology=ontology,
        #     tokenizer=tokenizer,
        #     max_history=args["max_history"],
        #     is_training_split=is_training_split,
        #     speaker="speaker_1",
        # ),
        # IndividualPreferences(
        #     args,
        #     ontology=ontology,
        #     tokenizer=tokenizer,
        #     is_training_split=is_training_split,
        #     max_history=args["max_history"],
        #     speaker="speaker_2",
        # ),
        # JointPreferences(
        #     args,
        #     ontology=ontology,
        #     tokenizer=tokenizer,
        #     is_training_split=is_training_split,
        #     max_history=args["max_history"],
        # ),
    ]
    evaluate_prompt_generators = [
        NaiveQuestion(
            args,
            ontology=ontology,
            tokenizer=tokenizer,
            is_training_split=is_training_split,
            max_history=args["max_history"],
        )
    ]
    prompt_generators = (
        train_prompt_generators if is_training_split else evaluate_prompt_generators
    )

    def flatten_list(l):
        return [item for sublist in l for item in sublist]

    return flatten_list(
        [
            prompt_generator.generate_samples_for_entire_dialogue(
                dial_id=dialogue["dial_id"],
                domains=dialogue["domains"],
                all_turns=dialogue["turns"],
            )
            for dialogue in dials
            for prompt_generator in prompt_generators
        ]
    )


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(
        batch_data["input_text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
        return_attention_mask=True,
    )
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(
        batch_data["output_text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
    )
    print(f"input_batch attention_mask: {input_batch['attention_mask']}")
    print(f"output_batch attention mask: {output_batch['attention_mask']}")
    # replace the padding id to -100 for cross-entropy
    output_batch["input_ids"].masked_fill_(
        output_batch["input_ids"] == tokenizer.pad_token_id, -100
    )
    batch_data["decoder_output"] = output_batch["input_ids"]

    return batch_data


def prepare_data(
    args,
    tokenizer,
    ontology,
    dials_path,
    batch_size,
    is_training_split,
    do_negative_sampling=False,
    shuffle=False,
):
    data = read_data(args, dials_path, ontology, tokenizer, is_training_split)
    data_loader = DataLoader(
        DSTDataset(data, args),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=3,
    )
    return data_loader
