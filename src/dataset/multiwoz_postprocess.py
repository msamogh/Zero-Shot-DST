import json
import argparse
import random
from typing import *

from dialogue import Dialogue, DialogueState, Turn

random.seed(4343)


def remove_system_turns(args, dials):
    for dial in dials:
        usr_turns = [turn["user"] for turn in dial["turns"]]
        states = [turn["state"] for turn in dial["turns"]]
        merged_turns = []
        for i, turn in enumerate(usr_turns):
            if i > 1 and random.random() < args.insert_affirm_prob:
                affirm_verbalized = verbalize_newly_added_slots(
                    args, states[i - 2]["slot_values"], states[i - 1]["slot_values"]
                )
                text = f"{affirm_verbalized} {turn}"
            else:
                text = turn
            merged_turns.append(
                {
                    "text": text,
                    "speaker": "speaker_1" if i % 2 == 0 else "speaker_2",
                    "state": states[i],
                }
            )
        dial["turns"] = merged_turns


def verbalize_newly_added_slots(args, state_1, state_2):
    state_1, state_2 = set(list(state_1.items())), set(list(state_2.items()))
    newly_added = list(state_2 - state_1)

    single_slot_affirms = [
        line.strip()
        for line in open(args.affirm_single_slot_verbalizer_path, "r").readlines()
    ]
    multi_slot_affirms = [
        line.strip()
        for line in open(args.affirm_multiple_slots_verbalizer_path, "r").readlines()
    ]

    if len(newly_added) == 1:
        return (
            random.choice(single_slot_affirms)
            .format(key=newly_added[0][0].split("-")[1], value=newly_added[0][1])
            .lower()
        )
    else:
        return random.choice(multi_slot_affirms).lower()


def insert_affirms(dials):
    for dial in dials:
        augmented_turns = []
        for turn in dial["turns"]:
            augmented_turns.append({})
        dials["turns"] = augmented_turns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--insert_affirms", action="store_true", default=True)
    parser.add_argument("--insert_affirm_prob", type=float, default=0.9)
    parser.add_argument("--affirm_single_slot_verbalizer_path", type=str, required=True)
    parser.add_argument(
        "--affirm_multiple_slots_verbalizer_path", type=str, required=True
    )
    args = parser.parse_args()

    dials = json.load(open(args.input_path, "r"))
    remove_system_turns(args, dials)
    json.dump(dials, open(args.output_path, "w"), indent=4)
