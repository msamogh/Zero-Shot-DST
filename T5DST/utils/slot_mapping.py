def replace_escai_with_multiwoz(dials):
    with open("ontology/escai_to_multiwoz_ontology.json", "r") as f:
        SLOTS_MAPPING = json.load(f)
        for dial in dials:
            for turn in dial["turns"]:
                # Replace sys and user utterances
                for slot in SLOTS_MAPPING.values():
                    for orig, new in slot["values"].items():
                        turn["system"] = turn["system"].replace(orig, new)
                        turn["user"] = turn["user"].replace(orig, new)

                new_slot_values = {}
                # Replace state annotation
                for slot_key, slot_value in turn["state"]["slot_values"].items():
                    if slot_key in SLOTS_MAPPING:
                        new_slot_values[
                            SLOTS_MAPPING[slot_key]["name"]
                        ] = SLOTS_MAPPING[slot_key]["values"][slot_value]
                    else:
                        new_slot_values[slot_key] = slot_value
                turn["state"]["slot_values"] = new_slot_values
