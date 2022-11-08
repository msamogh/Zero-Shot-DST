import json
import re
import copy
from typing import *


def src_to_tgt_map(
    src_slots: Dict[Text, List[Text]], tgt_slots: Dict[Text, List[Text]]
):
    assert len(src_slots) <= len(tgt_slots)
    name_mappings = {}
    val_mappings = {}
    for src_slot, tgt_slot in zip(src_slots, tgt_slots):
        assert len(src_slots[src_slot]) <= len(tgt_slots[tgt_slot])
        if src_slot in val_mappings:
            raise RuntimeError
        name_mappings[src_slot] = tgt_slot
        for src_slot_val, tgt_slot_val in zip(src_slots[src_slot], tgt_slots[tgt_slot]):
            val_mappings[src_slot_val] = tgt_slot_val
    return name_mappings, val_mappings


def map_text(utt, map):
    map = dict((re.escape(k), v) for k, v in map.items())
    pattern = re.compile("|".join(map.keys()))
    return pattern.sub(lambda m: map[re.escape(m.group(0))], utt)


def map_state(state, name_map, val_map):
    for k, v in copy.copy(state).items():
        del state[k]
        state[name_map[k].replace(" ", "")] = map_text(v, val_map)


def map_turn(turn, name_map, val_map):
    turn["system"] = map_text(turn["system"], val_map)
    turn["user"] = map_text(turn["user"], val_map)
    map_state(turn["state"]["slot_values"], name_map, val_map)


if __name__ == "__main__":
    DIALS_FILE = "../data/negochat_dials.json"
    SRC_ONTOLOGY_FILE = "../ontology/negochat_ontology.json"
    TGT_ONTOLOGY_FILE = "../ontology/ontology.json"
    TGT_ONTOLOGY_MAP_FILE = "../ontology/multiwoz_ontology_map.json"
    SRC_DOMAIN = "NegoChat"
    TGT_DOMAIN = "hotel"

    # Create mapping
    src_slots = json.load(open(SRC_ONTOLOGY_FILE, "r"))
    tgt_slots = {
        k: v
        for k, v in json.load(open(TGT_ONTOLOGY_FILE, "r")).items()
        if k.startswith(TGT_DOMAIN)
    }
    name_mapping, val_mapping = src_to_tgt_map(src_slots, tgt_slots)
    from pprint import pprint

    pprint(name_mapping)
    pprint(val_mapping)

    # Transform
    dials = json.load(open(DIALS_FILE, "r"))
    for dial in dials:
        dial["domain"] = TGT_DOMAIN
        for turn in dial["turns"]:
            map_turn(turn, name_mapping, val_mapping)
            print(turn)

    # Write to file
    json.dump(dials, open(DIALS_FILE + "_mapped", "w"), indent=4)
