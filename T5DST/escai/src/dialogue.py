from typing import *
from dataclasses import dataclass, field

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
            return {f"escai-{k.replace('-', '')}": v for k, v in self.c.items()}
        else:
            raise NotImplementedError


@dataclass
class Turn:

    utterance: Text
    speaker: Text
    state_annotation: Optional[Text] = None
    state: Optional[DialogueState] = None

    only_slots: Optional[Text] = None

    def __post_init__(self):
        assert self.state is not None or self.state_annotation is not None
        if self.state is None:
            self.state = self._parse_state_annotation()

    def _parse_state_annotation(self) -> DialogueState:
        state = DialogueState()
        if (
            pd.isna(self.state_annotation)
            or self.state_annotation is None
            or self.state_annotation == ""
            or self.state_annotation == "-"
        ):
            return state
        for line in self.state_annotation.split("\n"):
            slot_key, slot_value = line.split("=")
            interlocutor, slot_key = (
                slot_key.split("-")[0],
                slot_key[slot_key.index("-") + 1 :],
            )
            if self.only_slots is not None and slot_key not in self.only_slots.split(
                ","
            ):
                continue
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


@dataclass
class Dialogue:
    utterances: List[Text]
    speakers: List[Text]
    states: List[Text]

    double_text_strategy: Text = "naive"
    only_slots: Optional[List[Text]] = None

    turns: Optional[List[Turn]] = None

    def __post_init__(self):
        assert len(self.states) == len(self.utterances) and len(self.states) == len(
            self.speakers
        )
        self.turns = [
            Turn(
                utterance=utt,
                speaker=spk,
                state_annotation=state,
                only_slots=self.only_slots,
            )
            for (utt, spk, state) in zip(self.utterances, self.speakers, self.states)
        ]
        if self.double_text_strategy == "naive":
            pass
        elif self.double_text_strategy == "merge":
            self.turns = self._merge_double_texts()
        else:
            raise NotImplementedError

    def _merge_double_texts(self, join_token=" "):
        merged_turns = []
        turn_idx = 0
        while turn_idx < len(self.turns) - 1:
            current_utterance = self.turns[turn_idx].utterance
            while self.turns[turn_idx].speaker == self.turns[turn_idx + 1].speaker:
                current_utterance += join_token + self.turns[turn_idx + 1].utterance
                turn_idx += 1
                if turn_idx == len(self.turns):
                    break
            merged_turns.append(
                Turn(current_utterance, self.turns[turn_idx - 1], self.turns[state - 1])
            )
        return merged_turns
