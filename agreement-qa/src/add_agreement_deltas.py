import json
from pathlib import Path
from fire import Fire


splits = ["train", "val", "test"]

def add_agreement_delta(dial):
    """To each turn in a dialogue, add a new key called 'agreement_delta' that contains only
    the newly added agreemens since the last turn."""
    for i, turn in enumerate(dial["turns"]):
        if i == 0:
            turn["agreement_delta"] = turn["agreements"]
            turn["prev_agreements"] = {}
        else:
            prev_turn = dial["turns"][i - 1]
            turn["agreement_delta"] = {
                k: v
                for k, v in turn["agreements"].items()
                if k not in prev_turn["agreements"] or prev_turn["agreements"][k] != v
            }
            turn["prev_agreements"] = prev_turn["agreements"]
    return dial

def main(root_dir = "data/simulated/"):
    for split in splits:
        dials = json.load(Path(f"{root_dir}/{split}.json").open("r"))
        for dial in dials:
            dial = add_agreement_delta(dial)
        json.dump(dials, Path(f"{root_dir}/{split}_added.json").open("w"), indent=2)


if __name__ == "__main__":
    Fire(main)

