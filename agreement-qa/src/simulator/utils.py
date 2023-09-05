from typing import *
import random
random.seed(42)

from structs import SlotValuePair, Proposal


def sample_n_slot_values(n, ontology, slots_to_avoid=None, slot_values_to_avoid=None, slots_to_sample_from=None, slot_values_to_sample_from=None, all_or_sample=None):
    if all_or_sample == "sample":
        slot_values = set()
        for _ in range(n):
            slot_value = sample_slot_value(
                ontology=ontology,
                slots_to_avoid=slots_to_avoid,
                slot_values_to_avoid=slot_values_to_avoid,
                slots_to_sample_from=slots_to_sample_from,
                slot_values_to_sample_from=slot_values_to_sample_from
            )
            if slot_value is not None:
                slot_values.add(slot_value)
        slot_values = list(slot_values)
    elif all_or_sample == "all":
        slot_values = get_valid_slot_value_pairs(
            ontology=ontology,
            slots_to_avoid=slots_to_avoid,
            slot_values_to_avoid=slot_values_to_avoid,
            slots_to_sample_from=slots_to_sample_from,
            slot_values_to_sample_from=slot_values_to_sample_from
        )
    else:
        raise ValueError(f"Invalid value for all_or_sample: {all_or_sample}")
    return slot_values


def slotsof(slot_value_pairs):
    if slot_value_pairs:
        if isinstance(slot_value_pairs[0], Proposal):
            return [slot_value_pair.slot_value.slot for slot_value_pair in slot_value_pairs]
        elif isinstance(slot_value_pairs[0], SlotValuePair):
            return [slot_value_pair.slot for slot_value_pair in slot_value_pairs]
    return []

def valuesof(slot_value_pairs):
    if slot_value_pairs:
        if isinstance(slot_value_pairs[0], Proposal):
            return [slot_value_pairs.slot_value.value for slot_value_pairs in slot_value_pairs]
        elif isinstance(slot_value_pairs[0], SlotValuePair):
            return [slot_value_pairs.value for slot_value_pairs in slot_value_pairs]
    return []


def get_valid_slot_value_pairs(ontology, slots_to_avoid=None, slot_values_to_avoid=None, slots_to_sample_from=None, slot_values_to_sample_from=None):
    """
    Get a list of all valid slot-value pairs from the ontology, considering optional lists of slots and values to avoid or to sample from.

    Args:
        ontology (Dict[str, List[str]]): The ontology dictionary where keys are slot names and values are lists of possible values for each slot.
        slots_to_avoid (List[str], optional): A list of slots to avoid when sampling. Defaults to None.
        slot_values_to_avoid (List[str], optional): A list of slot values to avoid when sampling. Defaults to None.
        slots_to_sample_from (List[str], optional): A list of slots to specifically sample from. Defaults to None.
        slot_values_to_sample_from (List[str], optional): A list of slot values to specifically sample from. Defaults to None.

    Returns:
        List[SlotValuePair]: A list of valid slot-value pairs considering the restrictions given.
    """
    from structs import SlotValuePair

    slots_to_avoid = slots_to_avoid or []
    slot_values_to_avoid = slot_values_to_avoid or []

    slot_choices = slots_to_sample_from if slots_to_sample_from is not None else list(ontology.keys())
    if not slot_choices:
        return []

    valid_slot_value_pairs = []

    for slot in slot_choices:
        if slot in slots_to_avoid or (slots_to_sample_from and slot not in slots_to_sample_from):
            continue

        value_choices = slot_values_to_sample_from or ontology[slot]

        # If slot_values_to_sample_from is provided, ensure that the selected slot contains at least one value from the list
        if slot_values_to_sample_from and not any(val in value_choices for val in ontology[slot]):
            continue

        for value in value_choices:
            if value in slot_values_to_avoid or (slot_values_to_sample_from and value not in slot_values_to_sample_from) or (value not in ontology[slot]):
                continue

            valid_slot_value_pairs.append(SlotValuePair(slot, value))

    return valid_slot_value_pairs



def sample_slot_value(ontology, slots_to_avoid=None, slot_values_to_avoid=None, slots_to_sample_from=None, slot_values_to_sample_from=None):
    """Sample a slot-value pair from the ontology, considering optional lists of slots and values to avoid or to sample from.

    Args:
        ontology (Dict[str, List[str]]): The ontology dictionary where keys are slot names and values are lists of possible values for each slot.
        slots_to_avoid (List[str], optional): A list of slots to avoid when sampling. Defaults to None.
        slot_values_to_avoid (List[str], optional): A list of slot values to avoid when sampling. Defaults to None.
        slots_to_sample_from (List[str], optional): A list of slots to specifically sample from. Defaults to None.
        slot_values_to_sample_from (List[str], optional): A list of slot values to specifically sample from. Defaults to None.

    Returns:
        SlotValuePair: A sampled slot-value pair considering the restrictions given or None if it reaches the maximum number of attempts.
    """
    from structs import SlotValuePair

    slot_attempt_count = 0
    value_attempt_count = 0
    max_attempts = 100

    slots_to_avoid = slots_to_avoid or []
    slot_values_to_avoid = slot_values_to_avoid or []

    slot_choices = slots_to_sample_from if slots_to_sample_from is not None else list(ontology.keys())
    if not slot_choices:
        return None

    while True:
        if slot_attempt_count >= max_attempts:
            return None

        slot = random.choice(slot_choices)
        if slot in slots_to_avoid or (slots_to_sample_from and slot not in slots_to_sample_from):
            slot_attempt_count += 1
            continue

        value_choices = slot_values_to_sample_from or ontology[slot]

        # If slot_values_to_sample_from is provided, ensure that the selected slot contains at least one value from the list
        if slot_values_to_sample_from and not any(val in value_choices for val in ontology[slot]):
            slot_attempt_count += 1
            continue

        while True:
            if value_attempt_count >= max_attempts:
                return None

            value = random.choice(value_choices)
            if value in slot_values_to_avoid or (slot_values_to_sample_from and value not in slot_values_to_sample_from) or (value not in ontology[slot]):
                value_attempt_count += 1
                continue
            return SlotValuePair(slot, value)


def sample_gauss(mean, std=1):
    result = int(random.gauss(mean, std))
    return result
