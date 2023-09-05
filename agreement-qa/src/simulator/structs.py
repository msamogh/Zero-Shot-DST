from dataclasses import dataclass, field
from enum import Enum
from typing import *


# State-related dataclasses
@dataclass(eq=True, frozen=True)
class SlotValuePair:
    slot: str
    value: str

@dataclass(eq=True, frozen=True)
class Proposal:
    speaker: str
    slot_value: SlotValuePair
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Proposal):
            return NotImplemented
        return self.slot_value == __value.slot_value and self.speaker == __value.speaker

@dataclass
class DialogueState:
    open_proposals: List[Proposal] = field(default_factory=list)
    accepted_proposals: List[Proposal] = field(default_factory=list)
    explicit_rejections: List[Proposal] = field(default_factory=list)


# Turn-related dataclasses
class ActType(Enum):
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"

@dataclass
class Turn:
    speaker: str
    acts: List["Act"]
    state: "DialogueState"
    simulation_params: "SimulationParams"
    utterance: Optional[str] = None

    def __post_init__(self):
        from verbalizer import utterance_from_acts
        self.utterance = utterance_from_acts(self.acts, self.simulation_params)


@dataclass
class Act:
    act_type: ActType
    slot_value: SlotValuePair
    extra_info: Dict[str, Any] = field(default_factory=dict)
