from typing import *
import json

import wandb

from config import get_args
from dataset.utils import read_es_dialogues_from_xlsx, replace_with_source_ontology


def create_dials_from_xlsx(args) -> None:
    jsonified = []

    for idx, dialogue in enumerate(read_es_dialogues_from_xlsx(args, "escai")):
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
                            domain="escai", annotation_type=args.dst_annotation_type
                        ),
                    },
                }
            )

        # Don't add it to list of dialogues if it's unannotated and the
        # config parameter keep_only_annotated_dials is set to True.
        if (not args.keep_only_annotated_dials) or (not unannotated_dialogue):
            jsonified.append(jsonified_session)

    if args.replace_with_source_ontology:
        replace_with_source_ontology(jsonified)

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
