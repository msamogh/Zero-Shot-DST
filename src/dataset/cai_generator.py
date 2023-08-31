import pandas as pd

from dialogue import Dialogue
from dataset_generator import DialogueDatasetGenerator
from generator_config import get_args


class CAIGenerator(DialogueDatasetGenerator):
    def get_dialogues_iter(self):
        df = pd.read_excel(self.args["excel_file_path"], sheet_name="Negotiation Only")
        if self.args["keep_only_session_ids"] is not None:
            session_ids = self.args["keep_only_session_ids"].split(",")
            df = df[df["Session ID"].astype(str).isin(session_ids)]
        for session in (
            df.groupby("Session ID")
            .agg({"Utterance": list, "User": list, "State": list})
            .iterrows()
        ):
            _, session = session
            yield {
                "utterances": session[0],
                "speakers": session[1],
                "state_annotations": session[2],
            }

    def get_turns_iter(self, raw_dialogue):
        return zip(
            raw_dialogue["utterances"],
            raw_dialogue["speakers"],
            raw_dialogue["state_annotations"],
        )

    def get_utterance_text(self, raw_turn):
        return raw_turn[0]

    def get_speaker_id(self, raw_turn):
        return raw_turn[1]

    def get_turn_metadata(self, raw_turn):
        return None

    def get_state_annotation(self, raw_turn):
        return raw_turn[2]

    def update_dst(self, prev_dialogue_state, turn):
        return turn.state


if __name__ == "__main__":
    args = vars(get_args("escai"))
    generator = CAIGenerator(args, "escai")
    generator.generate_dataset()
