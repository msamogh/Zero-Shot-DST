from typing import *
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DialogueState:
    a: Dict[Text, Text] = field(default_factory=dict)
    b: Dict[Text, Text] = field(default_factory=dict)
    c: Dict[Text, Text] = field(default_factory=dict)

    @classmethod
    def copy_of(cls, state=None):
        if state is not None and isinstance(state, DialogueState):
            return cls(a=dict(state.a), b=dict(state.b), c=dict(state.c))

    def is_empty(self, annotation_type):
        if annotation_type == "cds":
            return not self.c
        elif annotation_type == "a":
            return not self.a
        elif annotation_type == "b":
            return not self.b
        raise NotImplementedError

    def get_annotation(self, domain, annotation_type):
        if annotation_type == "cds":
            return {f"{domain}-{k.replace('-', '')}": v for k, v in self.c.items()}
        elif annotation_type == "a":
            return {f"{domain}-{k.replace('-', '')}": v for k, v in self.a.items()}
        elif annotation_type == "b":
            return {f"{domain}-{k.replace('-', '')}": v for k, v in self.b.items()}
        else:
            raise NotImplementedError

    def get_slots_of_speaker(self, speaker):
        if speaker == "a":
            return self.a
        elif speaker == "b":
            return self.b
        raise RuntimeError(f"Speaker '{speaker}' has no slots.")

    def add_offer(self, speaker, slot_key, slot_value):
        original_list = self.get_slots_of_speaker(speaker)
        original_list[slot_key] = slot_value

    def remove_offer(self, speaker, slot_key):
        original_list = self.get_slots_of_speaker(speaker)
        if slot_key in original_list:
            del original_list[slot_key]
        else:
            raise RuntimeError(
                f"Slot '{slot_key}' not found in the list of proposed values by {speaker}"
            )

    def remove_all_offers(self, speaker):
        original_list = self.get_slots_of_speaker(speaker)
        for slot_key in list(original_list.keys()):
            del original_list[slot_key]

    def remove_unconfirmed_offers(self):
        self.a = {}
        self.b = {}

    def promote_to_c(self, speaker, slot_key):
        original_list = self.get_slots_of_speaker(speaker)
        if slot_key in original_list:
            slot_value = original_list.pop(slot_key)
            self.c[slot_key] = slot_value
        else:
            raise RuntimeError(
                f"Slot '{slot_key}' not found in the list of proposed values by {speaker}"
            )

    def promote_all_to_c(self, speaker):
        original_list = self.get_slots_of_speaker(speaker)
        for slot_key in list(original_list.keys()):
            self.promote_to_c(speaker, slot_key)


@dataclass
class Turn:

    utterance: Text
    speaker: Text
    state: Optional[DialogueState] = None

    metadata: Optional[Dict[Text, Any]] = None

    @classmethod
    def from_state_annotation(
        cls, utterance, speaker, state_annotation, metadata=None, only_slots=None
    ):
        return cls(
            utterance=utterance,
            speaker=speaker,
            state=Turn.parse_state_annotation(state_annotation, only_slots),
            metadata=metadata,
        )

    @staticmethod
    def parse_state_annotation(state_annotation, only_slots) -> DialogueState:
        state = DialogueState()
        if (
            pd.isna(state_annotation)
            or state_annotation is None
            or state_annotation == ""
            or state_annotation == "-"
        ):
            return state
        for line in state_annotation.split("\n"):
            slot_key, slot_value = line.split("=")
            interlocutor, slot_key = (
                slot_key.split("-")[0],
                slot_key[slot_key.index("-") + 1 :],
            )
            if only_slots is not None and slot_key not in only_slots.split(","):
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
    domain: str
    turns: List[Turn]
    idx: Text = None

    @classmethod
    def from_tuple(
        cls,
        domain,
        utterances,
        speakers,
        state_annotations=None,
        metadatas=None,
        only_slots=None,
        double_text_strategy="naive",
    ):
        if metadatas is None:
            metadatas = [None] * len(utterances)
        if state_annotations is None:
            state_annotations = [None] * len(utterances)

        assert (
            (len(state_annotations) == len(utterances))
            and (len(state_annotations) == len(speakers))
            and (len(state_annotations) == len(metadatas))
        )
        turns = [
            Turn.from_state_annotation(
                utterance=utt,
                speaker=spk,
                state_annotation=state_annotation,
                metadata=metadata,
                only_slots=only_slots,
            )
            for (utt, spk, state_annotation, metadata) in zip(
                utterances, speakers, state_annotations, metadatas
            )
        ]
        if double_text_strategy == "naive":
            pass
        elif double_text_strategy == "merge":
            turns = Dialogue.merge_double_texts(turns)
        else:
            raise NotImplementedError

        return cls(domain=domain, turns=turns)

    @staticmethod
    def merge_double_texts(turns: List[Turn], join_token=" "):
        merged_turns = []
        turn_idx = 0
        while turn_idx < len(turns) - 1:
            current_utterance = turns[turn_idx].utterance
            while turns[turn_idx].speaker == turns[turn_idx + 1].speaker:
                current_utterance += join_token + turns[turn_idx + 1].utterance
                turn_idx += 1
                if turn_idx == len(turns) - 1:
                    break
            merged_turns.append(
                Turn(
                    utterance=current_utterance,
                    speaker=turns[turn_idx].speaker,
                    state=turns[turn_idx].state,
                    metadata=turns[turn_idx].metadata,
                )
            )
            turn_idx += 1
        return merged_turns
