# Copyright (c) Facebook, Inc. and its affiliates

import json
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ast
from tqdm import tqdm
import os
import random
from functools import partial
from utils.fix_label import fix_general_label_error
from collections import OrderedDict

EXPERIMENT_DOMAINS = ["attraction", "hotel", "taxi", "escai", "NegoChat"]

random.seed(577)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024


class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += (
                " is " + " or ".join(item_info["value_list"]) + " or none?"
            )
        return item_info

    def __len__(self):
        return len(self.data)


def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        slot_temp = SLOTS
        if dataset == "train" or dataset == "dev":
            if args["except_domain"] != "none":
                slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
            elif args["only_domain"] != "none":
                slot_temp = [k for k in SLOTS if args["only_domain"] in k]
        else:
            if args["except_domain"] != "none":
                slot_temp = [k for k in SLOTS if args["except_domain"] in k]
            elif args["only_domain"] != "none":
                slot_temp = [k for k in SLOTS if args["only_domain"] in k]

        print(f"slot_temp: {slot_temp}")

        if dataset == "train" and args["fewshot"] > 0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[: int(len(dials) * args["fewshot"])]

        for idx, dial_dict in enumerate(dials):
            dialog_history = ""
            dialog_history_list = []

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if (
                args["only_domain"] != "none"
                and args["only_domain"] not in dial_dict["domains"]
            ):
                continue
            if (
                args["except_domain"] != "none"
                and dataset == "test"
                and args["except_domain"] not in dial_dict["domains"]
            ) or (
                args["except_domain"] != "none"
                and dataset != "test"
                and [args["except_domain"]] == dial_dict["domains"]
            ):
                continue

            # Reading data
            for turn_id, turn in enumerate(dial_dict["turns"]):
                # accumulate dialogue utterances
                if args["speaker_label_strategy"] == "raw":
                    dialog_history += (
                        " Employer: "
                        + turn["Employer"]
                        + " Candidate: "
                        + turn["Candidate"]
                    )
                    dialog_history_list.append(
                        " Employer: "
                        + turn["Employer"]
                        + " Candidate: "
                        + turn["Candidate"]
                    )
                else:
                    dialog_history += (
                        " System: " + turn["system"] + " User: " + turn["user"]
                    )
                    dialog_history_list.append(
                        " System: " + turn["system"] + " User: " + turn["user"]
                    )
                if args["fix_label"]:
                    slot_values = fix_general_label_error(
                        turn["state"]["slot_values"], SLOTS
                    )
                else:
                    slot_values = turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "none":
                        slot_values = OrderedDict(
                            [
                                (k, v)
                                for k, v in slot_values.items()
                                if args["except_domain"] not in k
                            ]
                        )
                    elif args["only_domain"] != "none":
                        slot_values = OrderedDict(
                            [
                                (k, v)
                                for k, v in slot_values.items()
                                if args["only_domain"] in k
                            ]
                        )
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_values = OrderedDict(
                            [
                                (k, v)
                                for k, v in slot_values.items()
                                if args["except_domain"] in k
                            ]
                        )
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict(
                            [
                                (k, v)
                                for k, v in slot_values.items()
                                if args["only_domain"] in k
                            ]
                        )

                turn_belief_list = [
                    str(k) + "-" + str(v) for k, v in slot_values.items()
                ]

                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                if "gpt" in args["model_name"]:
                    turn_slots = []
                    turn_slot_values = []
                    if len(dialog_history.split()) > 800:
                        continue
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset != "test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue
                        input_text = (
                            dialog_history
                            + f" {tokenizer.sep_token} {slot}"
                            + " "
                            + tokenizer.bos_token
                        )
                        output_text = (
                            input_text
                            + " "
                            + turn["state"]["slot_values"].get(slot, "none").strip()
                            + " "
                            + tokenizer.eos_token
                        )
                        slot_text = slot
                        value_text = (
                            turn["state"]["slot_values"].get(slot, "none").strip()
                        )

                        data_detail = {
                            "ID": dial_dict["dial_id"],
                            "domains": dial_dict["domains"],
                            "turn_id": turn_id,
                            "dialog_history": dialog_history,
                            "turn_belief": turn_belief_list,
                            "intput_text": input_text,
                            "output_text": output_text,
                            "slot_text": slot_text,
                            "value_text": value_text,
                        }
                        data.append(data_detail)

                else:
                    for slot in slot_temp:

                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset != "test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue

                        output_text = (
                            slot_values.get(slot, "none").strip()
                            + f" {tokenizer.eos_token}"
                        )
                        slot_text = slot
                        value_text = slot_values.get(slot, "none").strip()

                        dialog_history = "".join(
                            dialog_history_list[-args["max_history"] :]
                        )

                        if args["slot_lang"] == "human":
                            slot_lang = description[slot]["description_human"]
                            input_text = (
                                dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                            )
                        elif args["slot_lang"] == "naive":
                            slot_lang = description[slot]["naive"]
                            input_text = (
                                dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                            )
                        elif args["slot_lang"] == "value":
                            slot_lang = description[slot]["naive"]
                            input_text = (
                                dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                            )
                        elif args["slot_lang"] == "question":
                            slot_lang = description[slot]["question"]
                            input_text = (
                                dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                            )
                        elif args["slot_lang"] == "slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = (
                                dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                            )
                        else:
                            input_text = (
                                dialog_history + f" {tokenizer.sep_token} {slot}"
                            )

                        data_detail = {
                            "ID": dial_dict["dial_id"],
                            "domains": dial_dict["domains"],
                            "turn_id": turn_id,
                            "dialog_history": dialog_history,
                            "turn_belief": turn_belief_list,
                            "intput_text": input_text,
                            "output_text": output_text,
                            "slot_text": slot_text,
                            "value_text": value_text,
                            "value_list": description[slot]["values"],
                        }
                        data.append(data_detail)
    # print(len(data))
    # for idx in range(10):
    #     print(data[idx])
    print("domain_counter", domain_counter)
    return data, slot_temp


def get_slot_information(ontology):
    print(ontology)
    print(EXPERIMENT_DOMAINS)
    return [k for k in ontology if k.split("-")[0] in EXPERIMENT_DOMAINS]


def gpt_collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(
        batch_data["output_text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        max_length=1000,
    )
    batch_data["input_ids"] = output_batch["input_ids"]
    return batch_data


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(
        batch_data["intput_text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False,
    )
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(
        batch_data["output_text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=False,
    )
    # replace the padding id to -100 for cross-entropy
    output_batch["input_ids"].masked_fill_(
        output_batch["input_ids"] == tokenizer.pad_token_id, -100
    )
    batch_data["decoder_output"] = output_batch["input_ids"]

    return batch_data


def prepare_predict(args, tokenizer, path_test="data/test_dials.json"):
    ontology = json.load(open(args["predict_ontology_path"], "r"))
    print(f"ontology: {len(ontology)}")
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open(args["predict_slot_descriptions_path"], "r"))
    print(f"ALL_SLOTS: {ALL_SLOTS}")

    data_test, ALL_SLOTS = read_data(
        args, path_test, ALL_SLOTS, tokenizer, description, "test"
    )
    print(f"data_test: {len(data_test)}")
    test_dataset = DSTDataset(data_test, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["test_batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=4,
    )

    return ALL_SLOTS, test_loader


def prepare_data(args, tokenizer):
    ontology = json.load(open(args["predict_ontology_path"], "r"))
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open(args["predict_slot_descriptions_path"], "r"))

    data_train, _ = read_data(
        args, args["path_train"], ALL_SLOTS, tokenizer, description, "train"
    )
    data_dev, _ = read_data(args, args["path_dev"], ALL_SLOTS, tokenizer, description, "dev")
    data_test, ALL_SLOTS = read_data(
        args, args["path_test"], ALL_SLOTS, tokenizer, description, "test"
    )

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    if "gpt" in args["model_name"]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args["train_batch_size"],
            shuffle=True,
            collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer),
            num_workers=2,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args["test_batch_size"],
            shuffle=False,
            collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer),
            num_workers=2,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args["dev_batch_size"],
            shuffle=False,
            collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer),
            num_workers=2,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args["train_batch_size"],
            shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args["test_batch_size"],
            shuffle=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            num_workers=4,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args["dev_batch_size"],
            shuffle=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            num_workers=4,
        )
    fewshot_loader_dev = None
    fewshot_loader_test = None
    return (
        train_loader,
        dev_loader,
        test_loader,
        ALL_SLOTS,
        fewshot_loader_dev,
        fewshot_loader_test,
    )
