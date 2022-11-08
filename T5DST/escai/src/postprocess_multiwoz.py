import json
import argparse
from typing import *

from dialogue import Dialogue, DialogueState, Turn


def remove_system_turns(dials):
    for dial in dials:
        usr_turns = [turn["user"] for turn in dial["turns"]]
        states = [turn["state"] for turn in dial["turns"]]
        merged_turns = []
        for i in range(0, len(usr_turns), 2):
            if i + 1 >= len(usr_turns):
                break
            turn_a = usr_turns[i]
            turn_b = usr_turns[i + 1]
            merged_turns.append({
                "system": turn_a,
                "user": turn_b,
                "state": states[i + 1]
            })
        dial["turns"] = merged_turns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    dials = json.load(open(args.input_path, "r"))
    remove_system_turns(dials)
    json.dump(dials, open(args.output_path, "w"))

