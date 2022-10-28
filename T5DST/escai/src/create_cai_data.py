from typing import *
import json

import wandb

from config import get_args
from utils import read_dialogues_from_xlsx


def replace_escai_with_multiwoz(dials):
    with open("../ontology/escai_to_multiwoz_ontology.json", "r") as f:
        SLOTS_MAPPING = json.load(f)
        for dial in dials:
            for turn in dial["turns"]:
                # Replace sys and user utterances
                for slot in SLOTS_MAPPING.values():
                    for orig, new in slot["values"].items():
                        turn["system"] = turn["system"].replace(orig, new)
                        turn["user"] = turn["user"].replace(orig, new)

                new_slot_values = {}
                # Replace state annotation
                for slot_key, slot_value in turn["state"]["slot_values"].items():
                    if slot_key in SLOTS_MAPPING:
                        new_slot_values[
                            SLOTS_MAPPING[slot_key]["name"]
                        ] = SLOTS_MAPPING[slot_key]["values"][slot_value]
                    else:
                        new_slot_values[slot_key] = slot_value
                turn["state"]["slot_values"] = new_slot_values


def create_dials_from_xlsx(args) -> None:
    jsonified = []

    for idx, dialogue in enumerate(read_dialogues_from_xlsx(args)):
        jsonified_session = {
            "dial_id": f"ESCAI{idx}.json",
            "domains": ["escai"],
            "turns": [],
        }

        unannotated_dialogue = True

        for turn_pair in zip(dialogue.turns, dialogue.turns[1:]):
            turn_a, turn_b = turn_pair
            state = turn_b.state
            if not state.is_empty(args.dst_annotation_type):
                unannotated_dialogue = False
            jsonified_session["turns"].append(
                {
                    "system": str(turn_a.utterance),
                    "user": str(turn_b.utterance),
                    "state": {
                        "active_intent": "none",
                        "slot_values": state.get_annotation(
                            annotation_type=args.dst_annotation_type
                        ),
                    },
                }
            )

        # Don't add it to list of dialogues if it's unannotated and the
        # config parameter keep_only_annotated_dials is set to True.
        if (not args.keep_only_annotated_dials) or (not unannotated_dialogue):
            jsonified.append(jsonified_session)

    if args.replace_escai_with_multiwoz:
        replace_escai_with_multiwoz(jsonified)

    json.dump(jsonified, open(args.output_file, "w"))


if __name__ == "__main__":
    args = get_args()
    config = vars(args)
    config["dataset"] = "cai"
    wandb.init(
        id=args.wandb_run_id,
        project="collaborative-dst",
        entity="msamogh",
        config=config,
    )
    create_dials_from_xlsx(args)
