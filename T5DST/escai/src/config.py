# Copyright (c) Facebook, Inc. and its affiliates

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # The basics
    parser.add_argument("--wandb_run_id", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="data/omnibus.xlsx")
    parser.add_argument("--output_file", type=str, default="data/escai_dials_2.json")

    # Sanity presevers (for ES-CAI)
    parser.add_argument(
        "--keep_only_annotated_dials", action="store_true", default=True
    )
    parser.add_argument(
        "--keep_only_session_ids",
        type=str,
        default=None,
        help="Only keep the session IDs listed in this comma separated string parameter",
    )
    parser.add_argument(
        "--only_slots",
        type=str,
        help="Restrict ontology to the comma separated list of slot names",
        default=None,
    )

    # DST type
    parser.add_argument("--dst_annotation_type", type=str, default="cds")
    parser.add_argument(
        "--remove_slot_on_reject",
        action="store_true",
        default=False,
        help="Delete a slot from the list of an interlocutor's proposed values when the other interlocutor explicitly rejects it.",
    )

    # Ontology
    parser.add_argument(
        "--replace_with_source_ontology",
        action="store_true",
        help="Replace music/code related terms with terms that the model might be more familiar with (e.g., related to booking and reservations)",
        default=False,
    )
    parser.add_argument(
        "--replace_with_multiwoz_domain",
        type=str,
        help="Which MultiWOZ domain to replace it with",
    )

    # Turn-taking
    parser.add_argument(
        "--double_text_strategy", type=str, default="naive", help="naive | merge"
    )

    # Speaker labels
    parser.add_argument(
        "--speaker_label_strategy",
        type=str,
        help="raw | first_system | first_user | emp_sys_cand_usr | cand_sys_emp_usr | all_system | all_user | union | intersection",
    )

    # Splits
    parser.add_argument(
        "--data_splits",
        type=str,
        help="A,B where A is the % of the whole dataset for the development set, and B is the % of the non-training dataset for the test set",
        default="0.3,0.5",
    )

    args = parser.parse_args()

    return args
