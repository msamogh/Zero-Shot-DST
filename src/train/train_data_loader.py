# Copyright (c) Facebook, Inc. and its affiliates

import json
import torch
import math
from torch.utils.data import DataLoader, TensorDataset, Dataset, SequentialSampler
import ast
from tqdm import tqdm
from pprint import pprint
import os
import random
import functools
from typing import *
from dataclasses import dataclass, field
from functools import partial
from collections import OrderedDict

from fix_label import fix_general_label_error
from ontology import Ontology
from prompt_generator import NegativeContrastiveQuestion, PositiveContrastiveQuestion, NaiveQuestion

random.seed(577)


class TurnWiseSampler(SequentialSampler):
    def __iter__(self):
        """Ensure that no two turns from the same dialog are in the same batch."""
        sorted_samples = sorted(
            range(len(self.data_source)),
            key=functools.cmp_to_key(
                functools.partial(
                    TurnWiseSampler.sort_key_fn, data_source=self.data_source
                )
            ),
        )
        return iter(sorted_samples)

    @staticmethod
    def sort_key_fn(x, y, data_source):
        x, y = data_source[x], data_source[y]
        if x["ID"] != y["ID"]:
            return random.choice([-1, 1])
        if x["turn_id"] <= y["turn_id"]:
            return -1
        else:
            return 1


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
    ]
    if args["do_negative_sampling_data"]:
        train_prompt_generators.extend([
            PositiveContrastiveQuestion(
                args,
                ontology=ontology,
                tokenizer=tokenizer,
                is_training_split=is_training_split,
                max_history=args["max_history"],
            ),
            NegativeContrastiveQuestion(
                args,
                ontology=ontology,
                tokenizer=tokenizer,
                is_training_split=is_training_split,
                max_history=args["max_history"],
            )
        ])
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
        max_length=200,
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
        max_length=10,
    )
    # print(f"input_batch attention_mask: {input_batch['attention_mask']}")
    # print(f"output_batch attention mask: {output_batch['attention_mask']}")
    # replace the padding id to -100 for cross-entropy
    output_batch["input_ids"].masked_fill_(
        output_batch["input_ids"] == tokenizer.pad_token_id, -100
    )
    batch_data["decoder_output"] = output_batch["input_ids"]

    # In the batch, include a concatenation of the input and output text
    # for the purpose of computing the classification loss.
    # This is not used for the decoder.
    batch_data["input_and_output_text"] = [
        f"{input_text} {output_text}"
        for input_text, output_text in zip(
            batch_data["input_text"], batch_data["output_text"]
        )
    ]
    input_and_output_batch = tokenizer(
        batch_data["input_and_output_text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
        max_length=200,
        truncation=True,
    )
    # input_and_output_batch["input_ids"].masked_fill_(
    #     input_and_output_batch["input_ids"]
    #     == tokenizer.pad_token_id,
    #     -100,
    # )
    batch_data["input_and_output_input_ids"] = input_and_output_batch["input_ids"]
    batch_data["input_and_output_attention_mask"] = input_and_output_batch[
        "attention_mask"
    ]

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
        num_workers=0,
        # sampler=TurnWiseSampler(data),
    )
    return data_loader
