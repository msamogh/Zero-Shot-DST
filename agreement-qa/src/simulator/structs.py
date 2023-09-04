from dataclasses import dataclass, field
from enum import Enum
from typing import *


# Config-related dataclasses
@dataclass
class SimulationParams:
    num_dials: int
    ontology: Dict[str, List[str]]

    num_turns: int
    mean_proposals_a: int
    mean_proposals_b: int
    mean_rejects_a: int
    mean_rejects_b: int
    mean_accepts_a: int
    mean_accepts_b: int
    
    min_dist_between_proposal_and_accept_or_reject: int
    
    # Cultural parameters
    # is_multiple_proposals_for_a_slot_allowed: bool = False # TODO
    is_tangential_proposal_implicit_rejection: bool = False
    is_tangential_proposal_implicit_acceptance: bool = False
    does_new_proposal_immediately_undo_agreement: bool = False

    def __post_init__(self):
        # We don't want to allow both implicit rejection and implicit acceptance
        # for tangential proposals simultaneously.
        assert not (
            self.is_tangential_proposal_implicit_rejection and \
            self.is_tangential_proposal_implicit_acceptance
        ), "Cannot have both implicit rejection and implicit acceptance for tangential proposals."
    

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
    simulation_params: SimulationParams
    utterance: Optional[str] = None

    def __post_init__(self):
        from verbalizer import utterance_from_acts
        self.utterance = utterance_from_acts(self.acts, self.simulation_params)


@dataclass
class Act:
    act_type: ActType
    slot_value: SlotValuePair
