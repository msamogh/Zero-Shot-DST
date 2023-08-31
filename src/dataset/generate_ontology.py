import json
from typing import *
from collections import defaultdict


def get_unique_values_for_slot(dials_file, domain, slot_key):
    f = json.load(open(dials_file, "r"))
    unique_values = set(
        [
            [
                t["state"]["slot_values"].get(f"{slot_key}", None)
                for d in f
                for t in d["turns"]
            ]
        ][0]
    )
    unique_values.remove(None)
    return list(unique_values)


def get_unique_slots(dials_file: Text, domain: Text) -> List[Text]:
    f = json.load(open(dials_file, "r"))
    slot_names = set()
    for dial in f:
        for turn in dial["turns"]:
            for slot_name in turn["state"]["slot_values"]:
                slot_names.add(slot_name)
    return list(slot_names)


if __name__ == "__main__":
    DIALS_FILE = "../data/negochat_dials.json"
    DOMAIN = "NegoChat"
    ONTOLOGY_PATH = "../ontology/negochat_ontology.json"

    slots: Dict[Text, Any] = defaultdict(list)
    for slot_key in get_unique_slots(DIALS_FILE, DOMAIN):
        slots[slot_key] = get_unique_values_for_slot(DIALS_FILE, "NegoChat", slot_key)

    # Write ontology.json
    json.dump(
        {f"{slot_key}": slot_values for slot_key, slot_values in slots.items()},
        open(ONTOLOGY_PATH, "w"),
        indent=4,
    )
