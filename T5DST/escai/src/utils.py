from typing import *
import pandas as pd

from dialogue import Dialogue, DialogueState, Turn


def read_dialogues_from_xlsx(
    args,
    session_id_col="Session ID",
    utterance_col="Utterance",
    user_col="User",
    dst_col="State",
) -> List[Dialogue]:
    dialogues = []

    df = pd.read_excel(args.input_file, sheet_name=args.input_file_sheet_name)

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
        states = session[2]

        dialogues.append(
            Dialogue(
                utterances,
                speakers,
                states,
                double_text_strategy=args.double_text_strategy,
            )
        )

    return dialogues
