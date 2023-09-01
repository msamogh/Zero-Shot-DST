from dataclasses import dataclass
from enum import Enum
import random
random.seed(42)

class VerbalizerType(Enum):
    SINGLE_PROPOSE = "single_propose"
    SINGLE_ACCEPT = "single_accept"
    SINGLE_REJECT = "single_reject"

VERBALIZERS = {
    VerbalizerType.SINGLE_PROPOSE: [
        "I suggest {slot} to be {value}.",
        "How about we do {value} for {slot}?",
        "What do you think about {value} for {slot}?",
    ],
    VerbalizerType.SINGLE_REJECT: [
        "I don't agree with {slot} being {value}.",
        "I don't agree with {value}.",
        "I don't agree with your proposal for {slot}"
    ],
    VerbalizerType.SINGLE_ACCEPT: [
        "I agree with {slot} being {value}.",
        "I agree with {value}.",
        "I agree with your proposal for {slot}",
        "{value} sounds good to me."
    ]
}

def get_random_fragment(verbalizer_type):
    return random.choice(VERBALIZERS[verbalizer_type])

def utterance_from_acts(acts):
    from simulator import ActType
    utterance = ""
    for act in acts:
        if act.act_type.value == ActType.PROPOSE.value:
            verbalizer_type = VerbalizerType.SINGLE_PROPOSE
        elif act.act_type.value == ActType.ACCEPT.value:
            verbalizer_type = VerbalizerType.SINGLE_ACCEPT
        elif act.act_type.value == ActType.REJECT.value:
            verbalizer_type = VerbalizerType.SINGLE_REJECT
        else:
            raise NotImplementedError
        utterance += get_random_fragment(verbalizer_type).format(
            slot=act.slot_value.slot,
            value=act.slot_value.value
        ) + " "
    utterance = utterance.strip()
    return utterance
