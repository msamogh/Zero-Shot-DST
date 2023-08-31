from dataclasses import dataclass
from typing import *
from pathlib import Path
import json

from sklearn.model_selection import train_test_split

from config import get_args
from dialogue import Dialogue, DialogueState
from utils import read_dialogues


class DialogueDatasetGenerator(object):
    def __init__(self, args, dataset_name):
        self.args = args
        self.dataset_name = dataset_name

    def update_dst(self, dialogue_state, turn):
        """Return an updated dialogue state by observing the current turn."""
        raise NotImplementedError

    def init_dialogue_state(self):
        """Initial dialogue state (for the first turn)."""
        return DialogueState()

    def infer_dialogue_states(self, dialogue: Dialogue) -> List[DialogueState]:
        dialogue_states = []
        for idx, turn in enumerate(dialogue.turns):
            dialogue_state = self.update_dst(
                self.init_dialogue_state() if idx == 0 else dialogue_states[-1], turn
            )
            dialogue.turns[idx].state = DialogueState.copy_of(dialogue_state)
            dialogue_states.append(dialogue_state)
        assert len(dialogue.turns) == len(dialogue_states)
        return dialogue_states

    def get_dialogues_iter(self):
        """Return an iterator of raw dialogues."""
        raise NotImplementedError

    def get_turns_iter(self, raw_dialogue):
        """Return an iterator over turns of a dialogue."""
        raise NotImplementedError

    def get_utterance_text(self, raw_turn):
        """Extract the utterance text from a raw dialogue turn."""
        raise NotImplementedError

    def get_speaker_id(self, raw_turn):
        """Extract the speaker ID (1 or 2) from a raw dialogue turn."""
        raise NotImplementedError

    def get_turn_metadata(self, raw_turn):
        """Extract any metadata from a raw dialogue turn."""
        raise NotImplementedError

    def get_state_annotation(self, raw_turn):
        """Extract state annotation from a raw dialogue turn."""
        raise NotImplementedError

    def should_parse_state(self):
        return True

    def read_dialogues(self):
        dialogues = []
        for dial in self.get_dialogues_iter():
            utterances = []
            speakers = []
            metadatas = []
            state_annotations = []
            for turn in self.get_turns_iter(dial):
                utterances.append(self.get_utterance_text(turn))
                speakers.append(self.get_speaker_id(turn))
                metadatas.append(self.get_turn_metadata(turn))
                state_annotations.append(self.get_state_annotation(turn))
            dialogues.append(
                Dialogue.from_tuple(
                    domain=self.dataset_name,
                    utterances=utterances,
                    speakers=speakers,
                    metadatas=metadatas,
                    state_annotations=state_annotations,
                    double_text_strategy=self.args["double_text_strategy"],
                    parse_state=self.should_parse_state(),
                )
            )
        return dialogues

    def generate_dataset(self):
        jsonified = []
        for idx, dialogue in enumerate(self.read_dialogues()):
            jsonified_session = {
                "dial_id": f"{self.dataset_name}-{idx}.json",
                "domains": [self.dataset_name],
                "turns": [],
            }
            self.infer_dialogue_states(dialogue)

            for turn_idx, turn in enumerate(dialogue.turns):
                turn_annotation = self.get_turn_annotation(turn_idx, turn)
                jsonified_session["turns"].append(turn_annotation)

            jsonified.append(jsonified_session)
        self.split_and_dump(jsonified)

    def get_turn_annotation(self, turn_idx, turn) -> Dict[Text, Any]:
        return {
            "speaker": "speaker_1" if turn_idx % 2 == 0 else "speaker_2",
            "text": str(turn.utterance),
            "turn_idx": turn_idx,
            "state": {
                "slot_values": turn.state.get_annotation(
                    domain=self.dataset_name,
                    annotation_type="cds",
                ),
                "slot_values_a": turn.state.get_annotation(
                    domain=self.dataset_name,
                    annotation_type="a",
                ),
                "slot_values_b": turn.state.get_annotation(
                    domain=self.dataset_name,
                    annotation_type="b",
                ),
            },
        }

    def split_and_dump(self, jsonified):
        dev_split, test_split = [float(x) for x in self.args["data_splits"].split(",")]
        train_dials, dev_and_test_dials = train_test_split(
            jsonified, test_size=dev_split
        )
        dev_dials, test_dials = train_test_split(
            dev_and_test_dials, test_size=test_split
        )

        json.dump(
            train_dials, open(f"{self.args['output_dir']}/train.json", "w"), indent=4
        )
        json.dump(dev_dials, open(f"{self.args['output_dir']}/dev.json", "w"), indent=4)
        json.dump(
            test_dials, open(f"{self.args['output_dir']}/test.json", "w"), indent=4
        )
