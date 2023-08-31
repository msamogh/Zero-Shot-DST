from typing import *
from dataclasses import dataclass
import random


@dataclass
class InputOutputPair:
    input_text: Text
    output_text: Text
    is_negative_sample: bool
    id: Optional[Text] = None
    turn_id: Optional[Text] = None
    turn_belief: Optional[Dict[Text, Any]] = None
    for_evaluation: Optional[bool] = None
    slot_text: Optional[Text] = None
    domains: Optional[List] = None
    value_text: Optional[Text] = None
    value_list: Optional[List] = None

    def to_dict(self):
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "is_negative_sample": self.is_negative_sample,
            "ID": self.id,
            "turn_belief": self.turn_belief,
            "turn_id": self.turn_id,
            "for_evaluation": self.for_evaluation,
            "slot_text": self.slot_text,
            "domains": self.domains,
            "value_text": self.value_text,
            "value_list": self.value_list,
        }


@dataclass
class PromptGenerator(object):

    args: Dict[Text, Any]
    ontology: Any
    tokenizer: Any
    is_training_split: bool
    include_previous_state: bool = False
    max_history: Optional[int] = None

    def stringify_dialogue_history(self, turns):
        dialogue_history_str = ""
        history_len = len(turns) if self.max_history is None else self.max_history
        for turn in turns[-history_len:]:
            dialogue_history_str += f' {turn["speaker"]}: {turn["text"]} '
            dialogue_history_str += f" {self.tokenizer.sep_token} "
        return dialogue_history_str

    def generate_samples_for_turn(
        self, turn_idx, dial_id, domains, all_turns, prev_dialogue_state=None
    ):
        context = self.get_context(
            turn_idx, all_turns, prev_dialogue_state=prev_dialogue_state
        )
        slot_names, slots_c, questions, answers, wrong_answers = self.get_qas_over_ontology(
            all_turns, turn_idx
        )

        if self.args["verbose"]:
            for question, answer in zip(questions, answers):
                print(f'positive')
                print(f'input_text: {(context + question).replace("  ", " ").strip()}')
                print(f'output_text: {answer.replace("  ", " ").strip()}')
            for question, wrong_answer in zip(questions, wrong_answers):
                print('negative')
                print(f'input_text: {(context + question).replace("  ", " ").strip()}')
                print(f'output_text: {wrong_answer.replace("  ", " ").strip()}')

        pos_samples = [
            InputOutputPair(
                input_text=(context + question).replace("  ", " ").strip(),
                output_text=answer.replace("  ", " ").strip()
                + self.tokenizer.eos_token,
                is_negative_sample=False,
                id=dial_id,
                turn_id=turn_idx,
                turn_belief=slot_c,
                for_evaluation=self.is_for_evaluation(),
                slot_text=slot_name,
                domains=domains,
                value_text=slot_c,
                value_list=list(self.ontology.slots[slot_name]),
            ).to_dict()
            for (slot_name, slot_c, question, answer) in zip(
                slot_names, slots_c, questions, answers
            )
        ]

        if self.args["do_negative_sampling_data"]:
            neg_samples = [
                InputOutputPair(
                    input_text=(context + question).replace("  ", " ").strip(),
                    output_text=wrong_answer.replace("  ", " ").strip()
                    + self.tokenizer.eos_token,
                    is_negative_sample=True,
                    id=dial_id,
                    turn_id=turn_idx,
                    turn_belief=slot_c,
                    for_evaluation=False,
                    slot_text=slot_name,
                    domains=domains,
                    value_text=slot_c,
                    value_list=list(self.ontology.slots[slot_name]),
                ).to_dict()
                for (slot_name, slot_c, question, wrong_answer) in zip(
                    slot_names, slots_c, questions, wrong_answers
                )
            ]
            num_negative_samples = int(self.args["ns_ratio"] * len(pos_samples))
            neg_samples = random.sample(neg_samples, num_negative_samples)

            return pos_samples + neg_samples
        
        return pos_samples

    def generate_samples_for_entire_dialogue(self, dial_id, domains, all_turns):
        def flatten_list(l):
            return [item for sublist in l for item in sublist]

        return flatten_list(
            [
                self.generate_samples_for_turn(turn_idx, dial_id, domains, all_turns)
                for turn_idx in range(len(all_turns))
            ]
        )

    def get_context(self, turn_idx, all_turns, prev_dialogue_state=None):
        context = ""
        if self.include_previous_state and turn_idx > 0:
            if self.is_training_split:
                assert prev_dialogue_state is None
                prev_dialogue_state = self.dialogue_state_for_turn(
                    all_turns, turn_idx - 1
                )
            else:
                assert prev_dialogue_state is not None
            context += (
                PromptGenerator.dialogue_state_to_str(prev_dialogue_state)
                + self.tokenizer.sep_token
                + " "
            )
        context += self.stringify_dialogue_history(all_turns[: turn_idx + 1])
        return context

    def get_question(self, all_turns, turn_idx, slot_name):
        raise NotImplementedError

    def get_answer(self, all_turns, turn_idx, slot_name):
        raise NotImplementedError

    def get_wrong_answer(self, all_turns, turn_idx, slot_name):
        raise NotImplementedError

    def dialogue_state_for_turn(self, all_turns, turn_idx):
        return {
            slot_name: self.get_answer(all_turns, turn_idx, slot_name)
            for slot_name in self.ontology.slots.keys()
        }

    @staticmethod
    def dialogue_state_to_str(dialogue_state):
        dialogue_state_items = list(dialogue_state.items())
        random.shuffle(dialogue_state_items)
        return " ".join(
            [
                f" {slot_name} = {value} ; "
                for slot_name, value in dialogue_state_items
                if value != "none"
            ]
        )

    def get_qas_over_ontology(self, all_turns, turn_idx):
        turn = all_turns[turn_idx]
        slot_names, slots_c, questions, answers, wrong_answers = (
            [],
            [],
            [],
            [],
            []
        )
        for slot_name in self.ontology.slots.keys():
            slot_a, slot_b, slot_c = (
                turn["state"]["slot_values_a"],
                turn["state"]["slot_values_b"],
                turn["state"]["slot_values"],
            )
            # Undersample none value
            if (
                slot_name not in slot_a
                or slot_name not in slot_b
                or slot_name not in slot_c
            ) and random.random() >= self.args["keep_none_prob"]:
                continue

            question = self.get_question(all_turns, turn_idx, slot_name)
            answer = self.get_answer(all_turns, turn_idx, slot_name)
            wrong_answer = self.get_wrong_answer(all_turns, turn_idx, slot_name)

            if question is None or answer is None:
                continue

            slot_names.append(slot_name)
            slots_c.append(slot_c)
            questions.append(question)
            answers.append(answer)
            wrong_answers.append(wrong_answer)
        return slot_names, slots_c, questions, answers, wrong_answers

    @property
    def is_for_evaluation(self):
        raise NotImplementedError


class IndividualPreferences(PromptGenerator):
    def __init__(self, args, ontology, tokenizer, max_history, speaker="speaker_1"):
        super().__init__(args, ontology, tokenizer, max_history)
        self.speaker = speaker

    def get_question(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        slot_description = self.ontology.descriptions[slot_name]["naive"].lower()
        if self.speaker == "speaker_1":
            return f" What does speaker_1 want for {slot_description} ? "
        elif self.speaker == "speaker_2":
            return (
                f" What does speaker_1 want for {slot_description} ? "
                + turn["state"]["slot_values_a"].get(slot_name, "none")
                + " "
                + self.tokenizer.sep_token
                + f" What does speaker_2 want for {slot_description} ? "
            )

    def get_answer(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        if self.speaker == "speaker_1":
            return turn["state"]["slot_values_a"].get(slot_name, "none")
        elif self.speaker == "speaker_2":
            return turn["state"]["slot_values_b"].get(slot_name, "none")

    def get_wrong_answer(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        if self.speaker == "speaker_1":
            return turn["state"]["slot_values_b"].get(slot_name, "none")
        elif self.speaker == "speaker_2":
            return turn["state"]["slot_values_a"].get(slot_name, "none")

    def is_for_evaluation(self):
        return False


class JointPreferences(PromptGenerator):
    def get_question(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        slot_description = self.ontology.descriptions[slot_name]["naive"].lower()
        return (
            f" What does speaker_1 want for {slot_description} ? "
            + turn["state"]["slot_values_a"].get(slot_name, "none")
            + " "
            + self.tokenizer.sep_token
            + f" What does speaker_2 want for {slot_description} ? "
            + turn["state"]["slot_values_b"].get(slot_name, "none")
            + " "
            + self.tokenizer.sep_token
            + f" What do they agree on for {slot_description} ? "
        )

    def get_answer(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        return turn["state"]["slot_values"].get(slot_name, "none")

    def get_wrong_answer(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        if random.random() < 0.5:
            return turn["state"]["slot_values_a"].get(slot_name, "none")
        else:
            return turn["state"]["slot_values_b"].get(slot_name, "none")

    def is_for_evaluation(self):
        return True


class NaiveQuestion(PromptGenerator):
    def get_question(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        slot_description = self.ontology.descriptions[slot_name]["naive"].lower()
        return f" What do speaker_1 and speaker_2 want for {slot_description} ? "

    def get_answer(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        return turn["state"]["slot_values"].get(slot_name, "none")

    def get_wrong_answer(self, all_turns, turn_idx, slot_name):
        turn = all_turns[turn_idx]
        if random.random() < 0.5:
            return turn["state"]["slot_values_a"].get(slot_name, "none")
        else:
            return turn["state"]["slot_values_b"].get(slot_name, "none")

    def is_for_evaluation(self):
        return True
