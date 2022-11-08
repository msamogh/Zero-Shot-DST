from pathlib import Path
from typing import *
from pprint import pprint, pformat
import wandb
import json

from dialogue import Dialogue, Turn, DialogueState
from utils import read_dialogues, replace_with_source_ontology
from config import get_args


def get_speaker_labels(orig_speaker):
    speaker_label = "a" if orig_speaker == "Employer" else "b"
    other_speaker_label = "b" if speaker_label == "a" else "a"
    return speaker_label, other_speaker_label


def update_dst(args, dialogue_state, turn):
    speaker_label, other_speaker_label = get_speaker_labels(turn.speaker)
    for turn_idx, act in enumerate(turn.metadata):
        dact = list(act.keys())[0]
        if dact == "Greet":
            continue
        if dact == "Accept":
            if isinstance(act[dact], dict):
                slots_waiting_to_be_confirmed = dialogue_state.get_slots_of_speaker(
                    other_speaker_label
                )
                for slot_key, slot_value in dict(slots_waiting_to_be_confirmed).items():
                    if slot_key in act[dact]:
                        dialogue_state.promote_to_c(other_speaker_label, slot_key)
            else:
                # Promote hanging offers from previous turn to "c".
                dialogue_state.promote_all_to_c(other_speaker_label)
        elif dact == "Reject":
            if args.remove_slot_on_reject:
                if isinstance(act[dact], dict):
                    slots_waiting_to_be_confirmed = dialogue_state.get_slots_of_speaker(
                        other_speaker_label
                    )
                    for slot_key, slot_value in dict(
                        slots_waiting_to_be_confirmed
                    ).items():
                        if slot_key in act[dact]:
                            dialogue_state.remove_offer(other_speaker_label, slot_key)
                else:
                    dialogue_state.remove_all_offers(other_speaker_label)

    # Remove all unconfirmed offers from previous turn
    dialogue_state.remove_unconfirmed_offers()

    # Add new offers in this turn
    for turn_idx, act in enumerate(turn.metadata):
        dact = list(act.keys())[0]
        if dact == "Offer":
            for slot_key, slot_value in dict(act[dact]).items():
                dialogue_state.add_offer(speaker_label, slot_key, slot_value)

    return dialogue_state


def negochat_infer_dialogue_states(args, dialogue: Dialogue) -> List[DialogueState]:
    # print(dialogue.idx)
    dialogue_states = []
    for idx, turn in enumerate(dialogue.turns):
        dialogue_state = update_dst(
            args, DialogueState() if idx == 0 else dialogue_states[-1], turn
        )
        dialogue.turns[idx].state = DialogueState.copy_of(dialogue_state)
        dialogue_states.append(dialogue_state)
    assert len(dialogue.turns) == len(dialogue_states)
    return dialogue_states


def get_turn_annotation(args, turn_idx, turn_a, turn_b) -> Dict[Text, Any]:
    state = turn_b.state
    if args.speaker_label_strategy == "raw":
        utterances = {
            turn_a.speaker: str(turn_a.utterance),
            turn_b.speaker: str(turn_b.utterance),
        }
    elif args.speaker_label_strategy == "first_system":
        utterances = {
            "system": str(turn_a.utterance),
            "user": str(turn_b.utterance),
        }
    elif args.speaker_label_strategy == "first_user":
        utterances = {
            "user": str(turn_a.utterance),
            "system": str(turn_b.utterance),
        }
    elif args.speaker_label_strategy == "emp_sys_cand_usr":
        utterances = {
            "system" if turn_a.speaker == "Employer" else "user": str(turn_a.utterance),
            "system" if turn_b.speaker == "Employer" else "user": str(turn_b.utterance),
        }
    elif args.speaker_label_strategy == "cand_sys_emp_usr":
        utterances = {
            "user" if turn_a.speaker == "Employer" else "system": str(turn_a.utterance),
            "user" if turn_b.speaker == "Employer" else "system": str(turn_b.utterance),
        }
    elif args.speaker_label_strategy == "all_system":
        utterances = {
            "system": str(turn_a.utterance),
            "system": str(turn_b.utterance),
        }
    elif args.speaker_label_strategy == "all_user":
        utterances = {
            "user": str(turn_a.utterance),
            "user": str(turn_b.utterance),
        }
    else:
        raise NotImplementedError
    return {
        **utterances,
        "turn_idx": turn_idx,
        "state": {
            "active_intent": "none",
            "slot_values": state.get_annotation(
                domain="NegoChat", annotation_type=args.dst_annotation_type
            ),
        },
    }


def create_negochat_data(args, sub_version=None):
    jsonified = []

    for idx, dialogue in enumerate(read_dialogues(args, "NegoChat")):
        jsonified_session = {
            "dial_id": f"NegoChat-{idx}.json",
            "domains": ["NegoChat"],
            "turns": [],
        }
        negochat_infer_dialogue_states(args, dialogue)

        for turn_idx, turn_pair in enumerate(
            zip(dialogue.turns[::2], dialogue.turns[1:][::2])
        ):
            turn_a, turn_b = turn_pair
            turn_annotation = get_turn_annotation(args, turn_idx, turn_a, turn_b)
            jsonified_session["turns"].append(turn_annotation)
        jsonified.append(jsonified_session)

    if args.replace_with_source_ontology:
        replace_with_source_ontology(jsonified)

    # Split and dump
    sub_version = "" if sub_version is None else f"_{sub_version}"
    split_and_dump(args, jsonified, sub_version)


def split_and_dump(args, jsonified, sub_version):
    from sklearn.model_selection import train_test_split

    dev_split, test_split = [float(x) for x in args.data_splits.split(",")]

    train_dials, dev_and_test_dials = train_test_split(jsonified, test_size=dev_split)
    dev_dials, test_dials = train_test_split(dev_and_test_dials, test_size=test_split)

    json.dump(
        train_dials, open(f"{args.output_file + sub_version}_train", "w"), indent=4
    )
    json.dump(dev_dials, open(f"{args.output_file + sub_version}_dev", "w"), indent=4)
    json.dump(test_dials, open(f"{args.output_file + sub_version}_test", "w"), indent=4)


if __name__ == "__main__":
    args = get_args()
    config = vars(args)
    config["dataset"] = "NegoChat"
    wandb.init(
        id=args.wandb_run_id,
        project="collaborative-dst",
        entity="msamogh",
        config=vars(args),
    )
    if args.speaker_label_strategy == "union":
        args.speaker_label_strategy = "first_system"
        create_negochat_data(args, "1")
        args.speaker_label_strategy = "first_user"
        create_negochat_data(args, "2")
    elif args.speaker_label_strategy == "intersection":
        args.speaker_label_strategy = "first_system"
        create_negochat_data(args, "1")
        args.speaker_label_strategy = "first_user"
        create_negochat_data(args, "2")
    else:
        create_negochat_data(args)
