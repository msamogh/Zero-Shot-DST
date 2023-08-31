import json
import argparse
import random
from typing import *

from dialogue import Dialogue, DialogueState, Turn

random.seed(4343)


def split_speakers(args, dials):
    for dial in dials:
        split_turns = []
        for turn in dial["turns"]:
            split_turns.append({})
            split_turns.append({})
        dial["turns"] = split_turns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    dials = json.load(open(args.input_path, "r"))
    split_speakers(args, dials)
    json.dump(dials, open(args.output_path, "w"), indent=4)
