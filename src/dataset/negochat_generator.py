from pathlib import Path
import json

import wandb

from dialogue import Dialogue
from dataset_generator import DialogueDatasetGenerator
from generator_config import get_args


class NegochatGenerator(DialogueDatasetGenerator):
    def get_dialogues_iter(self):
        for dial_file in Path(self.args["negochat_root_dir"]).iterdir():
            turns = json.load(open(dial_file, "r"))["turns"]
            yield {
                "utterances": [
                    turn["data"] if "data" in turn else turn["input"] for turn in turns
                ],
                "speakers": [turn["role"] for turn in turns],
                "metadatas": [turn["output"] for turn in turns],
            }

    def get_turns_iter(self, raw_dialogue):
        return zip(
            raw_dialogue["utterances"],
            raw_dialogue["speakers"],
            raw_dialogue["metadatas"],
        )

    def get_utterance_text(self, raw_turn):
        return raw_turn[0]

    def get_speaker_id(self, raw_turn):
        return raw_turn[1]

    def get_turn_metadata(self, raw_turn):
        return raw_turn[2]

    def get_state_annotation(self, raw_turn):
        return None

    def _get_speaker_labels(self, orig_speaker):
        speaker_label = "b" if orig_speaker == "Employer" else "a"
        other_speaker_label = "a" if speaker_label == "b" else "b"
        return speaker_label, other_speaker_label

    def should_parse_state(self):
        return False

    def update_dst(self, prev_dialogue_state, turn):
        speaker_label, other_speaker_label = self._get_speaker_labels(turn.speaker)
        for act in turn.metadata:
            dact = list(act.keys())[0]
            if dact == "Greet":
                continue
            elif dact == "Accept":
                if isinstance(act[dact], dict):
                    slots_waiting_to_be_confirmed = (
                        prev_dialogue_state.get_slots_of_speaker(other_speaker_label)
                    )
                    for slot_key, slot_value in dict(
                        slots_waiting_to_be_confirmed
                    ).items():
                        if slot_key in act[dact]:
                            prev_dialogue_state.promote_to_c(
                                other_speaker_label, slot_key
                            )
                else:
                    # Promote hanging offers from previous turn to "c".
                    prev_dialogue_state.promote_all_to_c(other_speaker_label)
            elif dact == "Reject":
                if self.args["remove_slot_on_reject"]:
                    if isinstance(act[dact], dict):
                        slots_waiting_to_be_confirmed = (
                            prev_dialogue_state.get_slots_of_speaker(
                                other_speaker_label
                            )
                        )
                        for slot_key, slot_value in dict(
                            slots_waiting_to_be_confirmed
                        ).items():
                            if slot_key in act[dact]:
                                prev_dialogue_state.remove_offer(
                                    other_speaker_label, slot_key
                                )
                    else:
                        prev_dialogue_state.remove_all_offers(other_speaker_label)

        # Remove all unconfirmed offers from previous turn
        # prev_dialogue_state.remove_unconfirmed_offers()

        # Add new offers in this turn
        for turn_idx, act in enumerate(turn.metadata):
            dact = list(act.keys())[0]
            if dact == "Offer":
                for slot_key, slot_value in dict(act[dact]).items():
                    prev_dialogue_state.add_offer(speaker_label, slot_key, slot_value)

        return prev_dialogue_state


if __name__ == "__main__":
    args = vars(get_args("negochat"))
    generator = NegochatGenerator(args, "negochat")
    generator.generate_dataset()
