from curses import meta
import json
from typing import *
from pathlib import Path

import pandas as pd

from dialogue import Dialogue, DialogueState, Turn


def read_dialogues(args, domain):
    ROOT_DIR = args.input_file

    dialogues = []

    for dial_file in Path(ROOT_DIR).iterdir():
        utterances = []
        speakers = []
        metadatas = []

        turns = json.load(open(dial_file, "r"))["turns"]
        for turn in turns:
            utterance = turn["data"] if "data" in turn else turn["input"]
            utterances.append(utterance)
            speakers.append(turn["role"])
            metadatas.append(turn["output"])

        dialogue = Dialogue.from_tuple(
            domain=domain,
            utterances=utterances,
            speakers=speakers,
            metadatas=metadatas,
            double_text_strategy=args.double_text_strategy,
        )
        dialogue.idx = str(dial_file)

        dialogues.append(dialogue)

    return dialogues


def replace_with_source_ontology(dials):
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


def read_es_dialogues_from_xlsx(
    args,
    domain,
    session_id_col="Session ID",
    utterance_col="Utterance",
    user_col="User",
    dst_col="State",
) -> List[Dialogue]:
    dialogues = []

    df = pd.read_excel(args.input_file, sheet_name="Negotiation Only")

    if args.keep_only_session_ids is not None:
        session_ids = args.keep_only_session_ids.split(",")
        df = df[df[session_id_col].astype(str).isin(session_ids)]

    for idx, session in enumerate(
        df.groupby(session_id_col)
        .agg({utterance_col: list, user_col: list, dst_col: list})
        .iterrows()
    ):
        _, session = session
        utterances = session[0]
        speakers = session[1]
        state_annotations = session[2]

        dialogues.append(
            Dialogue.from_tuple(
                domain=domain,
                utterances=utterances,
                speakers=speakers,
                state_annotations=state_annotations,
                double_text_strategy=args.double_text_strategy,
            )
        )

    return dialogues
