import argparse


def get_args(dataset):
    parser = argparse.ArgumentParser()
    parser_fns = {"escai": add_escai_args, "negochat": add_negochat_args}
    add_global_args(parser)
    parser_fns[dataset](parser)
    return parser.parse_args()


def add_global_args(parser):
    parser.add_argument("--dst_annotation_type", type=str, default="cds")
    parser.add_argument(
        "--double_text_strategy", type=str, default="naive", help="naive | merge"
    )
    parser.add_argument(
        "--remove_slot_on_reject",
        action="store_true",
        default=False,
        help="Delete a slot from the list of an interlocutor's proposed values when the other interlocutor explicitly rejects it.",
    )
    parser.add_argument(
        "--data_splits",
        type=str,
        help="A,B where A is the fraction of the whole dataset for the development set, and B is the fraction of the non-training dataset for the test set",
        default="0.3,0.5",
    )
    parser.add_argument("--output_dir", type=str)


def add_negochat_args(parser):
    parser.add_argument("--negochat_root_dir", type=str)


def add_escai_args(parser):
    parser.add_argument("--excel_file_path", type=str)
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
