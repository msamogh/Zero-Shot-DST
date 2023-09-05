from dataclasses import dataclass
from enum import Enum
import random
random.seed(42)


class VerbalizerType(Enum):
    PROPOSE_SLOT_VALUE = "propose_slot_value"
    PROPOSE_VALUE_ONLY = "propose_value_only"

    ACCEPT_SLOT_VALUE = "accept_slot_value"
    ACCEPT_SLOT_ONLY = "accept_slot_only"
    ACCEPT_VALUE_ONLY = "accept_value_only"

    REJECT_SLOT_VALUE = "reject_slot_value"
    REJECT_SLOT_ONLY = "reject_slot_only"
    REJECT_VALUE_ONLY = "reject_value_only"


VERBALIZERS = {
    VerbalizerType.PROPOSE_SLOT_VALUE.value: [
        "I suggest {slot} to be {value}.",
        "How about we do {value} for {slot}?",
        "What do you think about {value} for {slot}?",
    ],
    VerbalizerType.PROPOSE_VALUE_ONLY.value: [
        "I suggest {value}.",
        "How about {value}?",
        "What do you think about {value}?",
    ],
    VerbalizerType.REJECT_SLOT_VALUE.value: [
        "I don't agree with {slot} being {value}.",
        "I don't think {value} is a good idea for {slot}.",
    ],
    VerbalizerType.REJECT_VALUE_ONLY.value: [
        "I don't agree with {value}.",
        "{value} doesn't sound good to me.",
        "{value} is not a good idea."
    ],
    VerbalizerType.REJECT_SLOT_ONLY.value: [
        "I don't agree with your proposal for {slot}.",
    ],
    VerbalizerType.ACCEPT_SLOT_VALUE.value: [
        "I agree with {slot} being {value}.",
    ],
    VerbalizerType.ACCEPT_VALUE_ONLY.value: [
        "I agree with {value}.",
        "{value} sounds good to me."
    ],
    VerbalizerType.ACCEPT_SLOT_ONLY.value: [
        "I agree with your proposal for {slot}.",
    ]
}

def get_random_fragment(verbalizer_type):
    return random.choice(VERBALIZERS[verbalizer_type.value])

def utterance_from_acts(acts, simulation_params):
    utterance = ""
    for act in acts:
        if "propose" in act.act_type.value:
            verbalizer_type = random.choice([
                VerbalizerType.PROPOSE_SLOT_VALUE,
                VerbalizerType.PROPOSE_VALUE_ONLY
            ])
        elif "accept" in act.act_type.value:
            verbalizer_type = random.choice([
                VerbalizerType.ACCEPT_SLOT_VALUE,
                VerbalizerType.ACCEPT_SLOT_ONLY,
                VerbalizerType.ACCEPT_VALUE_ONLY
            ])
        elif "reject" in act.act_type.value:
            verbalizer_type = random.choice([
                VerbalizerType.REJECT_SLOT_VALUE,
                VerbalizerType.REJECT_SLOT_ONLY,
                VerbalizerType.REJECT_VALUE_ONLY
            ])
        else:
            raise NotImplementedError
        utterance += get_random_fragment(verbalizer_type).format(
            slot=act.slot_value.slot,
            value=act.slot_value.value
        ) + " "
    utterance = utterance.strip()
    return utterance
