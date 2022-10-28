# Copyright (c) Facebook, Inc. and its affiliates

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # The basics
    parser.add_argument("--wandb_run_id", type=str, required=True)
    parser.add_argument("--input_file_sheet_name", type=str, default="Negotiation Only")
    parser.add_argument("--output_file", type=str, default="data/escai_dials_2.json")
    parser.add_argument("--input_file", type=str, default="data/omnibus.xlsx")

    # Sanity presevers
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

    # Ontology
    parser.add_argument(
        "--replace_escai_with_multiwoz",
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
        help="One of naive, first-system, first-user, all-system, all-user, union, intersection",
        default="naive",
    )

    args = parser.parse_args()

    return args
