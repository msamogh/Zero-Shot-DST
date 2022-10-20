from typing import *
from dataclasses import dataclass, field
import argparse
import json
import pandas as pd


@dataclass
class DialogueState:
    a: Dict[Text, Text] = field(default_factory=dict)
    b: Dict[Text, Text] = field(default_factory=dict)
    c: Dict[Text, Text] = field(default_factory=dict)

    def is_empty(self, annotation_type):
        if annotation_type == "cds":
            return not self.c
        raise NotImplementedError

    def get_annotation(self, annotation_type):
        if annotation_type == "cds":
            return {
                f"escai-{k.replace('-', '')}": v for k, v in self.c.items()
            }
        else:
            raise NotImplementedError


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--double_text_strategy", type=str, default="naive", help="naive | merge")
    parser.add_argument("--input_file_sheet_name", type=str, default="Negotiation Only")
    parser.add_argument("--output_file", type=str, default="data/escai_dials_2.json")
    parser.add_argument("--input_file", type=str, default="data/escai/omnibus.xlsx")
    parser.add_argument("--dst_annotation_type", type=str, default="cds")
    parser.add_argument("--keep_only_annotated_dials", type=bool, default=True)

    args = parser.parse_args()
    return args


def parse_state_annotation(annotation: Text) -> DialogueState:
    state = DialogueState()
    if (
        pd.isna(annotation) or \
        annotation is None or \
        annotation == "" or \
        annotation == "-"
    ):
        return state
    for line in annotation.split("\n"):
        slot_key, slot_value = line.split("=")
        interlocutor, slot_key = slot_key.split("-")[0], slot_key[slot_key.index("-") + 1:]
        # For now, we will only use the first element in the list
        # (if multiple slots are mentioned).
        slot_value = slot_value[1:-1].split(", ")[0]
        if interlocutor == "a":
            state.a[slot_key] = slot_value
        elif interlocutor == "b":
            state.b[slot_key] = slot_value
        elif interlocutor == "c":
            state.c[slot_key] = slot_value
    return state


def create_dials_from_xlsx(
    args,
    xlsx_path,
    sheet_name,
    session_id_col="Session ID",
    utterance_col="Utterance",
    user_col="User",
    dst_col="State"
) -> None:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    jsonified = []
    for idx, session in enumerate(df.groupby(session_id_col)\
                    .agg({utterance_col: list, user_col: list, dst_col: list}).iterrows()):
        jsonified_session = {
            "dial_id": f"ESCAI{idx}.json",
            "domains": [
                "escai"
            ],
            "turns": []
        }

        _, session = session

        utterances = session[0]
        speakers = session[1]
        states = session[2]
        turns = list(zip(utterances, speakers, states))

        unannotated_dialogue = True

        if args.double_text_strategy == "naive":
            for idx, turn_pair in enumerate(zip(turns, turns[1:])):
                (utt_a, speaker_a, state_a), (utt_b, speaker_b, state_b) = turn_pair
                state = parse_state_annotation(state_b)
                if not state.is_empty(args.dst_annotation_type):
                    unannotated_dialogue = False
                jsonified_session["turns"].append({
                    "system": str(utt_a),
                    "user": str(utt_b),
                    "state": {
                        "active_intent": "none",
                        "slot_values": state.get_annotation(
                            annotation_type=args.dst_annotation_type
                        )
                    }
                })
        if args.keep_only_annotated_dials and unannotated_dialogue:
            continue
        else:
            jsonified.append(jsonified_session)

    json.dump(jsonified, open(args.output_file, "w"))

if __name__ == "__main__":
    args = get_args()
    create_dials_from_xlsx(
        args,
        args.input_file,
        args.input_file_sheet_name
    )
