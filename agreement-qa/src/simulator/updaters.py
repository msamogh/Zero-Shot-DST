from dataclasses import dataclass, field
from typing import *
from typing import List
from pprint import pprint
import random

from structs import SlotValuePair, Proposal, DialogueState, Act, ActType, Turn
from utils import slotsof, valuesof
import utils


@dataclass
class UpdateStatus:
    """Records the progress of a turn simulation.
    
    A single turn simulation can have multiple changes. For example, a turn simulation
    can have a proposal, an accept, and a reject. The progress of each of these
    changes is recorded in a DialogueStateUpdateS.
    """
    explicit_agreements_added: bool = False
    explicit_rejections_added: bool = False
    implicit_agreements_added: bool = False
    implicit_rejections_added: bool = False
    proposals_added: bool = False


@dataclass
class DialogueStateUpdate:
    """Records the state update of a turn simulation.
    
    A single turn simulation can have multiple state updates. For example, a turn simulation
    can have a proposal, an accept, and a reject. The state update of each of these
    changes is recorded in a StateUpdate.
    """
    curr_turn_idx: int = 0

    explicit_agreements: List[SlotValuePair] = field(default_factory=list)
    explicit_rejections: List[SlotValuePair] = field(default_factory=list)
    implicit_agreements: List[SlotValuePair] = field(default_factory=list)
    implicit_rejections: List[SlotValuePair] = field(default_factory=list)
    undone_agreements: List[SlotValuePair] = field(default_factory=list)
    new_proposals: List[SlotValuePair] = field(default_factory=list)

    status: UpdateStatus = field(default_factory=UpdateStatus)

    @property
    def agreements(self):
        return self.explicit_agreements + self.implicit_agreements
    
    @property
    def rejections(self):
        return self.explicit_rejections + self.implicit_rejections

    @property
    def curr_speaker(self):
        return "A" if self.curr_turn_idx % 2 == 0 else "B"

    @property
    def other_speaker(self):
        return "B" if self.curr_turn_idx % 2 == 0 else "A"
    
    def prepare_for_new_turn(self):
        self.curr_turn_idx += 1
        self.explicit_agreements = []
        self.explicit_rejections = []
        self.implicit_agreements = []
        self.implicit_rejections = []
        self.new_proposals = []
        self.status = UpdateStatus()


@dataclass
class Updater:

    params: "SimulationParams"
    dialogue_state_update: DialogueStateUpdate
    dialogue_state: "DialogueState"
    history: List["Turn"]

    mean_num_slot_values: int = 0

    slots_to_sample_from: Optional[List[str]] = None
    slot_values_to_sample_from: Optional[List[str]] = None
    slots_to_avoid: Optional[List[SlotValuePair]] = None
    slot_values_to_avoid: Optional[List[SlotValuePair]] = None

    all_or_sample: Optional[str] = None

    def _generate_slot_values(self):
        from utils import sample_n_slot_values
        slot_values = sample_n_slot_values(
            n=utils.sample_gauss(self.mean_num_slot_values),
            slots_to_avoid=self.slots_to_avoid,
            slot_values_to_avoid=self.slot_values_to_avoid,
            slots_to_sample_from=self.slots_to_sample_from,
            slot_values_to_sample_from=self.slot_values_to_sample_from,
            ontology=self.params.ontology,
            all_or_sample=self.all_or_sample
        )
        return slot_values
    
    def apply(self):
        # If the updater cannot be applied, return None
        print("Applying", self.__class__.__name__)
        if not self.can_apply():
            self.post_apply()
            return []
        # Otherwise, apply the updater
        # Generate slot values
        generated_slot_values = self._generate_slot_values()
        print("Generated slot values:", generated_slot_values)
        # Populate the in-progress update
        self.populate_in_progress_update(generated_slot_values)
        # Update the dialogue state update
        self.post_apply()

    @staticmethod
    def update_dialogue_state(dialogue_state_update, dialogue_state):
        curr_speaker = dialogue_state_update.curr_speaker

        for slot_value in dialogue_state_update.agreements:
            dialogue_state.accepted_proposals.append(Proposal(curr_speaker, slot_value))
            dialogue_state.open_proposals = [
                p for p in dialogue_state.open_proposals if p.slot_value != slot_value
            ]

        for slot_value in dialogue_state_update.rejections:
            dialogue_state.explicit_rejections.append(Proposal(curr_speaker, slot_value))
            dialogue_state.open_proposals = [
                p for p in dialogue_state.open_proposals if p.slot_value != slot_value
            ]

        for slot_value in dialogue_state_update.new_proposals:
            dialogue_state.open_proposals.append(Proposal(curr_speaker, slot_value))
        dialogue_state.open_proposals = [
            p for p in dialogue_state.open_proposals if p not in dialogue_state.accepted_proposals
        ]
        
        for slot_value in dialogue_state_update.undone_agreements:
            dialogue_state.accepted_proposals = [
                p for p in dialogue_state.accepted_proposals if p.slot_value != slot_value
            ]


    @staticmethod
    def get_acts(dialogue_state_update):
        acts = [
            Act(ActType.PROPOSE, slot_value) for slot_value in dialogue_state_update.new_proposals
        ] + [
            Act(ActType.ACCEPT, slot_value) for slot_value in dialogue_state_update.explicit_agreements
        ] + [
            Act(ActType.REJECT, slot_value) for slot_value in dialogue_state_update.explicit_rejections
        ] + [
            Act(ActType.ACCEPT, slot_value, extra_info={"implicit": True}) for slot_value in dialogue_state_update.implicit_agreements
        ] + [
            Act(ActType.REJECT, slot_value, extra_info={"implicit": True}) for slot_value in dialogue_state_update.implicit_rejections
        ]
        return acts
        
    
    def can_apply(self):
        return self._can_apply(self.dialogue_state_update.status)
    
    def _can_apply(self, status: UpdateStatus):
        raise NotImplementedError

    def populate_in_progress_update(self, slot_values: List[SlotValuePair]):
        raise NotImplementedError

    def post_apply(self):
        raise NotImplementedError


@dataclass
class GenerateProposals(Updater):
    
    def __post_init__(self):
        self.slot_values_to_avoid = valuesof(
            self.dialogue_state.open_proposals + self.dialogue_state.accepted_proposals
        )
        self.all_or_sample = "sample"

    def _can_apply(self, status):
        return status.explicit_rejections_added and \
            status.explicit_agreements_added and \
            (not self.params.is_tangential_proposal_implicit_acceptance or status.implicit_agreements_added) and \
            status.implicit_rejections_added

    def populate_in_progress_update(self, new_proposals: List[SlotValuePair]):
        self.dialogue_state_update.new_proposals.extend(new_proposals)

    def post_apply(self):
        self.dialogue_state_update.status.proposals_added = True


@dataclass
class GenerateExplicitAgreements(Updater):

    def __post_init__(self):
        self.slots_to_sample_from = slotsof(self.dialogue_state.open_proposals)
        self.slot_values_to_sample_from = valuesof(self.dialogue_state.open_proposals)
        self.all_or_sample = "sample"

    def _can_apply(self, status):
        return not (
            status.explicit_agreements_added or \
            status.implicit_agreements_added or \
            status.implicit_rejections_added
        )

    def populate_in_progress_update(self, explicit_accepts: List[SlotValuePair]):
        self.dialogue_state_update.explicit_agreements.extend(explicit_accepts)

    def post_apply(self):
        self.dialogue_state_update.status.explicit_agreements_added = True


@dataclass
class GenerateExplicitRejections(Updater):

    def __post_init__(self):
        self.slots_to_sample_from = slotsof(self.dialogue_state.open_proposals)
        self.slot_values_to_sample_from = valuesof(self.dialogue_state.open_proposals)
        self.slot_values_to_avoid = valuesof(self.dialogue_state.accepted_proposals) + \
            valuesof(self.dialogue_state_update.explicit_agreements)
        self.all_or_sample = "sample"

    def _can_apply(self, status):
        return not (
            status.explicit_rejections_added or \
            status.implicit_agreements_added or \
            status.implicit_rejections_added
        )
    
    def populate_in_progress_update(self, explicit_rejections: List[SlotValuePair]):
        self.dialogue_state_update.explicit_rejections.extend(explicit_rejections)
    
    def post_apply(self):
        self.dialogue_state_update.status.explicit_rejections_added = True


@dataclass
class GenerateImplicitAgreements(Updater):

    def __post_init__(self):
        if self.dialogue_state_update.curr_turn_idx == 0:
            return
        slots_proposed_in_prev_turn = set(slotsof(
            [a.slot_value for a in self.history[-1].acts if a.act_type == ActType.PROPOSE]
        ))
        slots_explicitly_accepted_or_rejected_in_curr_turn = set(slotsof(
            self.dialogue_state_update.explicit_agreements
        )) | set(slotsof(
            self.dialogue_state_update.explicit_rejections
        ))
        implicitly_accepted_slots = (
            slots_proposed_in_prev_turn - slots_explicitly_accepted_or_rejected_in_curr_turn
        )
        self.slots_to_sample_from = list(implicitly_accepted_slots)
        self.slot_values_to_sample_from = valuesof(self.dialogue_state.open_proposals)
        self.all_or_sample = "all"

    def _can_apply(self, status):
        return (
            self.dialogue_state_update.curr_turn_idx > 0 and \
            self.params.is_tangential_proposal_implicit_acceptance and \
            status.explicit_agreements_added and \
            status.explicit_rejections_added
        )

    def populate_in_progress_update(self, slot_values: List[SlotValuePair]):
        self.dialogue_state_update.implicit_agreements.extend(slot_values)

    def post_apply(self):
        self.dialogue_state_update.status.implicit_agreements_added = True


@dataclass
class GenerateImplicitRejections(Updater):

    def __post_init__(self):
        if self.dialogue_state_update.curr_turn_idx == 0:
            return
        slots_proposed_in_prev_turn = set(slotsof(
            [a.slot_value for a in self.history[-1].acts if a.act_type == ActType.PROPOSE]
        ))
        slots_explicitly_accepted_or_rejected_in_curr_turn = set(slotsof(
            self.dialogue_state_update.explicit_agreements
        )) | set(slotsof(
            self.dialogue_state_update.explicit_rejections
        ))
        implicitly_rejected_slots = (
            slots_proposed_in_prev_turn - slots_explicitly_accepted_or_rejected_in_curr_turn
        )
        self.slots_to_sample_from = list(implicitly_rejected_slots)
        self.slot_values_to_sample_from = valuesof(self.dialogue_state.open_proposals)
        self.all_or_sample = "all"

    def _can_apply(self, status):
        return (
            self.dialogue_state_update.curr_turn_idx > 0 and \
            self.params.is_tangential_proposal_implicit_rejection and \
            status.explicit_agreements_added and \
            status.explicit_rejections_added
        )
    
    def populate_in_progress_update(self, slot_values: List[SlotValuePair]):
        self.dialogue_state_update.implicit_rejections.extend(slot_values)

    def post_apply(self):
        self.dialogue_state_update.status.implicit_rejections_added = True


@dataclass
class UndoAgreementsForNewProposals(Updater):

    def __post_init__(self):
        self.slots_to_sample_from = slotsof(self.dialogue_state.accepted_proposals)
        self.slot_values_to_sample_from = valuesof(self.dialogue_state.accepted_proposals)
        self.all_or_sample = "all"

    def _can_apply(self, status):
        return (
            self.params.does_new_proposal_immediately_undo_agreement and \
            status.proposals_added and \
            not status.explicit_rejections_added and \
            not status.explicit_agreements_added and \
            not status.implicit_agreements_added and \
            not status.implicit_rejections_added
        )

    def populate_in_progress_update(self, slot_values: List[SlotValuePair]):
        self.dialogue_state_update.implicit_rejections.extend(slot_values)
        self.dialogue_state_update.undone_agreements.extend(slot_values)

    def post_apply(self):
        pass
