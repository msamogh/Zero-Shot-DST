from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
import copy
import json
import random
random.seed(42)

from fire import Fire
import randomname

from structs import (
    SlotValuePair, SimulationParams, DialogueState,
    Proposal, ActType, Turn, Act
)
import utils


@dataclass
class TurnSimulationChange:
    """Records the progress of a turn simulation.
    
    A single turn simulation can have multiple changes. For example, a turn simulation
    can have a proposal, an accept, and a reject. The progress of each of these
    changes is recorded in a TurnSimulationChange.
    """
    explicit_agreements_added: bool = False
    explicit_rejections_added: bool = False
    implicit_agreements_added: bool = False
    implicit_rejections_added: bool = False
    proposals_added: bool = False

# Dialogue-related dataclasses
@dataclass
class SimulatedDialogue:
    name: str
    params: SimulationParams

    curr_turn_idx: int = 0
    curr_proposing_slot_values: List[SlotValuePair] = field(default_factory=list)
    curr_accepting_slot_values: List[SlotValuePair] = field(default_factory=list)
    curr_rejecting_slot_values: List[SlotValuePair] = field(default_factory=list)

    curr_turn_simulation_change: TurnSimulationChange = field(default_factory=TurnSimulationChange)

    state: DialogueState = field(default_factory=DialogueState)
    history: List[Turn] = field(default_factory=list)


    def generate_proposals(self):
        """Randomly generate proposals for the current speaker.
        
        Returns newly generated proposals.
        """
        assert (self.curr_turn_simulation_change.explicit_rejections_added and \
            self.curr_turn_simulation_change.explicit_agreements_added and \
            self.curr_turn_simulation_change.implicit_agreements_added and \
            self.curr_turn_simulation_change.implicit_rejections_added), \
            
        new_proposals = self._create_random_proposals()
        self._overwrite_existing_proposals(new_proposals)
        self.state.open_proposals.extend(new_proposals)

        self.curr_turn_simulation_change.proposals_added = True

        return new_proposals


    def generate_accepts(self):
        """
        Randomly generate accepts for the current speaker.
        
        Returns:
            list: A list of newly generated accepts.
        """
        num_accepts = utils.sample_gauss(self.curr_mean_accepts)
        new_accepts = []
        for _ in range(num_accepts):
            if not self.state.open_proposals:
                break
            proposal_to_accept = random.choice(self.state.open_proposals)
            if self.is_atleast_min_dist_away(proposal_to_accept):
                self._accept_proposal(proposal_to_accept.slot_value)
                new_accepts.append(proposal_to_accept)
        self.curr_turn_simulation_change.explicit_agreements_added = True
        return new_accepts


    def generate_rejects(self):
        """Randomly generate rejects for the current speaker.

        Returns newly generated rejects.
        """
        num_rejects = utils.sample_gauss(self.curr_mean_rejects)
        new_rejects = []
        for _ in range(num_rejects):
            if not self.state.open_proposals:
                break
            proposal_to_reject = random.choice(self.state.open_proposals)
            if self.is_atleast_min_dist_away(proposal_to_reject):
                self._reject_proposal(proposal_to_reject.slot_value)
                new_rejects.append(proposal_to_reject)
        self.curr_turn_simulation_change.explicit_rejections_added = True
        return new_rejects


    def generate_implicit_acceptances_if_needed(self):
        """If is_tangential_proposal_implicit_acceptance is True, then we need to
        generate implicit accepts for tangential proposals made in the immediately
        previous turn.
        """
        assert (self.curr_turn_simulation_change.explicit_rejections_added and \
            self.curr_turn_simulation_change.explicit_agreements_added), \
            "Explicit agreements and rejections must be added before implicit rejections."

        if self.curr_turn_idx > 0 and self.params.is_tangential_proposal_implicit_acceptance:
            for act in [a for a in self.history[-1].acts if a.act_type == ActType.PROPOSE]:
                self._accept_proposal(act.slot_value)

        self.curr_turn_simulation_change.implicit_agreements_added = True

    def generate_implicit_rejections_if_needed(self):
        """If is_tangential_proposal_implicit_rejection is True, then we need to
        generate implicit rejects for tangential proposals made in the immediately
        previous turn.
        """
        assert (self.curr_turn_simulation_change.explicit_rejections_added and \
            self.curr_turn_simulation_change.explicit_agreements_added), \
            "Explicit agreements and rejections must be added before implicit rejections."
        
        if self.curr_turn_idx > 0 and self.params.is_tangential_proposal_implicit_rejection:
            for act in [a for a in self.history[-1].acts if a.act_type == ActType.PROPOSE]:
                try:
                    self._reject_proposal(act.slot_value)
                except ProposalNotFound:
                    pass

        self.curr_turn_simulation_change.implicit_rejections_added = True

    def undo_agreements_implicitly_if_needed(self):
        assert not (self.curr_turn_simulation_change.explicit_rejections_added or \
            self.curr_turn_simulation_change.explicit_agreements_added or \
            self.curr_turn_simulation_change.implicit_agreements_added or \
            self.curr_turn_simulation_change.implicit_rejections_added or \
            self.curr_turn_simulation_change.proposals_added), \
            "No changes should be made before undoing agreements implicitly."
        
        if self.curr_turn_idx > 0 and self.params.does_new_proposal_immediately_undo_agreement:
            agreed_upon_slots = [p.slot_value.slot for p in self.state.accepted_proposals]
            current_proposal_slots = [p.slot_value.slot for p in self.state.open_proposals]
            for slot in agreed_upon_slots:
                if slot in current_proposal_slots:
                    self.state.accepted_proposals = [
                        p for p in self.state.accepted_proposals
                        if p.slot_value.slot != slot
                    ]

    @classmethod
    def generate_dialogue(cls, name: str, params: SimulationParams):
        """Generate a dialogue with the given parameters."""
        dialogue = cls(name, params)
        for _ in range(params.num_turns):
            dialogue.undo_agreements_implicitly_if_needed()

            new_rejects = dialogue.generate_rejects()
            new_accepts = dialogue.generate_accepts()

            dialogue.generate_implicit_rejections_if_needed()
            dialogue.generate_implicit_acceptances_if_needed()

            new_proposals = dialogue.generate_proposals()

            dialogue.history.append(
                Turn(
                    speaker=dialogue.curr_speaker,
                    state=copy.deepcopy(dialogue.state),
                    acts=[
                        Act(
                            act_type=ActType.PROPOSE,
                            slot_value=proposal.slot_value
                        ) for proposal in new_proposals
                    ] + [
                        Act(
                            act_type=ActType.ACCEPT,
                            slot_value=proposal.slot_value
                        ) for proposal in new_accepts
                    ] + [
                        Act(
                            act_type=ActType.REJECT,
                            slot_value=proposal.slot_value
                        ) for proposal in new_rejects
                    ],
                    simulation_params=params
                )
            )
            dialogue.curr_turn_idx += 1
            dialogue.curr_proposing_slot_values = []
            dialogue.curr_accepting_slot_values = []
            dialogue.curr_rejecting_slot_values = []
        return dialogue


    def to_json(self):
        return {
            "dialogue_id": self.name,
            "turns": [{
                turn.speaker: turn.utterance,
                "agreements": {
                    p.slot_value.slot: p.slot_value.value for p in turn.state.accepted_proposals
                },
                "open_proposals": {
                    p.slot_value.slot: p.slot_value.value for p in turn.state.open_proposals
                },
                "acts": [{
                    act.act_type.value: {
                        act.slot_value.slot: act.slot_value.value
                    }
                } for act in turn.acts]
            } for turn in self.history]
        }


    def is_atleast_min_dist_away(self, proposal):
        return (
            self.curr_turn_idx - proposal.extra_info["turn_idx"] >=
                self.params.min_dist_between_proposal_and_accept_or_reject
        )

    def _create_random_proposals(self):
        """Generate a list of random proposals based on the defined parameters.
        Returns:
            List[Proposal]: A list of generated proposals.
        """
        num_proposals = utils.sample_gauss(self.curr_mean_proposals)
        new_proposals = []
        for _ in range(100):
            if len(new_proposals) == num_proposals:
                break
            # Choose a random slot
            slot = random.choice(list(self.params.ontology.keys()))
            if self._slot_already_proposed(slot, new_proposals):
                continue
            # Choose a random, unique value for the slot
            value = self._select_unique_value(slot)
            if value is not None:
                new_proposals.append(
                    Proposal(
                        self.curr_speaker,
                        SlotValuePair(slot, value),
                        extra_info={"turn_idx": self.curr_turn_idx}
                    )
                )
        return new_proposals
    
    def _select_unique_value(self, slot):
        """Pick a unique value for a given slot that doesn't exist in existing proposals or agreements."""
        available_values = list(self.params.ontology[slot])
        random.shuffle(available_values)
        proposed_values = [p.slot_value for p in self.state.open_proposals]
        agreed_values = [p.slot_value for p in self.state.accepted_proposals]
        for value in available_values:
            if SlotValuePair(slot, value) not in (proposed_values + agreed_values):
                return value
        return None
    
    def _slot_already_proposed(self, slot, proposals):
        """Check if the slot has already been proposed in the current turn.
        Args:
            slot (str): The slot in question.
            proposals (List[Proposal]): The list of current proposals.
        Returns:
            bool: True if the slot has already been proposed, False otherwise.
        """
        return slot in [p.slot_value.slot for p in proposals]

    def _overwrite_existing_proposals(self, new_proposals):
        """Overwrite any existing proposals from the same speaker for the same slot.
        Args:
            new_proposals (List[Proposal]): The list of new proposals.
        """
        for new_proposal in new_proposals:
            for existing_proposal in self.state.open_proposals:
                if (existing_proposal.slot_value.slot == new_proposal.slot_value.slot 
                    and existing_proposal.speaker == self.curr_speaker):
                    self.state.open_proposals.remove(existing_proposal)

    def _accept_proposal(self, slot_value):
        """
        Helper function to accept a proposal. 
        It will remove the proposal from open_proposals and add it to accepted_proposals.
        """
        if slot_value.slot not in [x.slot for x in self.curr_proposing_slot_values + self.curr_accepting_slot_values]:
            try:
                proposal = next(p for p in self.state.open_proposals if p.slot_value == slot_value)
                self.state.open_proposals.remove(proposal)
                self.state.accepted_proposals.append(proposal)
                self.curr_accepting_slot_values.append(
                    Act(act_type=ActType.ACCEPT, slot_value=slot_value).slot_value
                )
            except (ValueError, StopIteration):
                raise ProposalNotFound(
                    f"Slot value {slot_value} not found in open proposals."
                )

    def _reject_proposal(self, slot_value):
        """
        Helper function to reject a proposal. 
        It will remove the proposal from open_proposals and add it to explicit_rejections.
        """
        if slot_value not in self.curr_proposing_slot_values + self.curr_rejecting_slot_values:
            try:
                proposal = next(p for p in self.state.open_proposals if p.slot_value == slot_value)
                self.state.open_proposals.remove(proposal)
                self.state.explicit_rejections.append(proposal)
                self.curr_rejecting_slot_values.append(
                    Act(act_type=ActType.REJECT, slot_value=slot_value).slot_value
                )
            except (ValueError, StopIteration):
                raise ProposalNotFound(
                    f"Slot value {slot_value} not found in open proposals."
                )

    @property
    def curr_speaker(self):
        return "A" if self.curr_turn_idx % 2 == 0 else "B"
    
    @property
    def curr_mean_proposals(self):
        return self.params.mean_proposals_a if self.curr_speaker == "A" else self.params.mean_proposals_b
    
    @property
    def curr_mean_accepts(self):
        return self.params.mean_accepts_a if self.curr_speaker == "A" else self.params.mean_accepts_b
    
    @property
    def curr_mean_rejects(self):
        return self.params.mean_rejects_a if self.curr_speaker == "A" else self.params.mean_rejects_b


class ProposalNotFound(Exception):
    """Raised when a non-existent proposal is attempted to be accepted or rejected."""
    pass


@dataclass
class Simulator:
    params: SimulationParams

    def generate_dialogues(self):
        dialogues = [
            SimulatedDialogue.generate_dialogue(str(i), self.params).to_json()
            for i in range(self.params.num_dials)
        ]
        json.dump(
            dialogues,
            Path(f"data/simulated2/{randomname.get_name()}.json").open("w"),
            indent=2
        )


def main():
    params = SimulationParams(
        ontology={
            'Job Description': ['Project Manager', 'Team Manager', 'QA', 'Programmer'],
            'Salary': ['90,000 USD', '120,000 USD', '60,000 USD'],
            'Leased Car': ['With leased car', 'Without leased car'],
            'Pension Fund': ['0%', '20%', '10%'],
            'Promotion Possibilities': ['Slow promotion track', 'Fast promotion track'],
            'Working Hours': ['9 hours', '10 hours', '8 hours']
        },
        num_turns=10,
        num_dials=100,
        mean_proposals_a=2,
        mean_proposals_b=2,
        mean_rejects_a=1,
        mean_rejects_b=1,
        mean_accepts_a=2,
        mean_accepts_b=2,
        min_dist_between_proposal_and_accept_or_reject=1,
        is_tangential_proposal_implicit_rejection=True,
        is_tangential_proposal_implicit_acceptance=False,
        does_new_proposal_immediately_undo_agreement=False,
        # is_multiple_proposals_for_a_slot_allowed=False,
    )
    simulator = Simulator(params)
    simulator.generate_dialogues()


if __name__ == "__main__":
    Fire(main)
