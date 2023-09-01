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
    is_multiple_proposals_in_one_turn_allowed: bool = False # TODO
    is_overriding_proposal_implicit_rejection: bool = False # TODO
    is_tangential_proposal_implicit_acceptance: bool = False # TODO
    can_agreement_be_overridden: bool = True # TODO
    

# State-related dataclasses
@dataclass(eq=True, frozen=True)
class SlotValuePair:
    slot: str
    value: str

@dataclass
class Proposal:
    speaker: str
    slot_value: SlotValuePair
    extra_info: Dict[str, Any] = field(default_factory=dict)

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
    acts: List["Act"] = field(default_factory=list)
    state: "DialogueState" = field(default_factory=DialogueState)
    utterance: Optional[str] = None

    def __post_init__(self):
        from verbalizer import utterance_from_acts
        self.utterance = utterance_from_acts(self.acts)


@dataclass
class Act:
    act_type: ActType
    slot_value: SlotValuePair


# Dialogue-related dataclasses
@dataclass
class SimulatedDialogue:
    name: str
    params: SimulationParams

    curr_turn_idx: int = 0
    state: DialogueState = field(default_factory=DialogueState)
    history: List[Turn] = field(default_factory=list)

    @staticmethod
    def sample_gauss(mean, std=1):
        return int(random.gauss(mean, std))

    def _select_unique_value(self, slot):
        """Pick a unique value for a given slot that doesn't exist in existing proposals."""
        available_values = list(self.params.ontology[slot])
        random.shuffle(available_values)
        for value in available_values:
            if SlotValuePair(slot, value) not in [proposal.slot_value for proposal in self.state.open_proposals]:
                return value
        return None

    def generate_proposals(self):
        """Randomly generate proposals for the current speaker.
        
        Returns newly generated proposals.
        """
        num_proposals = SimulatedDialogue.sample_gauss(self.curr_mean_proposals)
        new_proposals = []
        for _ in range(100):
            if len(new_proposals) == num_proposals:
                break
            slot = random.choice(list(self.params.ontology.keys()))
            # Don't propose the same slot twice in the same turn.
            if slot in [p.slot_value.slot for p in new_proposals]:
                continue
            # Overwrite existing proposals from the same speaker for the same slot
            # from earlier turns.
            for p in self.state.open_proposals:
                if p.slot_value.slot == slot and p.speaker == self.curr_speaker:
                    self.state.open_proposals.remove(p)
            value = self._select_unique_value(slot)
            if value is None:
                continue
            new_proposals.append(
                Proposal(
                    self.curr_speaker,
                    SlotValuePair(slot, value),
                    extra_info={"turn_idx": self.curr_turn_idx}
                )
            )
        self.state.open_proposals.extend(new_proposals)
        return new_proposals

    def generate_accepts(self):
        """Randomly generate accepts for the current speaker.
        
        Returns newly generated accepts.
        """
        num_accepts = SimulatedDialogue.sample_gauss(self.curr_mean_accepts)
        new_accepts = []
        for _ in range(num_accepts):
            if not self.state.open_proposals:
                break
            proposal_to_accept = random.choice(self.state.open_proposals)
            if self.curr_turn_idx - proposal_to_accept.extra_info["turn_idx"] < self.params.min_dist_between_proposal_and_accept_or_reject:
                continue
            new_accepts.append(proposal_to_accept)
            self.state.open_proposals.remove(proposal_to_accept)
            self.state.accepted_proposals.append(proposal_to_accept)
        return new_accepts

    def generate_rejects(self):
        """Randomly generate rejects for the current speaker.

        Returns newly generated rejects.
        """
        num_rejects = SimulatedDialogue.sample_gauss(self.curr_mean_rejects)
        new_rejects = []
        for _ in range(num_rejects):
            if not self.state.open_proposals:
                break
            proposal_to_reject = random.choice(self.state.open_proposals)
            if self.curr_turn_idx - proposal_to_reject.extra_info["turn_idx"] < self.params.min_dist_between_proposal_and_accept_or_reject:
                continue
            new_rejects.append(proposal_to_reject)
            self.state.open_proposals.remove(proposal_to_reject)
            self.state.explicit_rejections.append(proposal_to_reject)
        return new_rejects

    @classmethod
    def generate_dialogue(cls, name: str, params: SimulationParams):
        dialogue = cls(name, params)
        for _ in range(params.num_turns):
            new_rejects = dialogue.generate_rejects()
            new_accepts = dialogue.generate_accepts()
            new_proposals = dialogue.generate_proposals()
            
            dialogue.curr_turn_idx += 1
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
                )
            )
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
        mean_accepts_a=1,
        mean_accepts_b=1,
        min_dist_between_proposal_and_accept_or_reject=3,
        is_overriding_proposal_implicit_rejection=True,
        is_tangential_proposal_implicit_acceptance=True,
        can_agreement_be_overridden=True,
        is_multiple_proposals_in_one_turn_allowed=True
    )
    simulator = Simulator(params)
    simulator.generate_dialogues()

if __name__ == "__main__":
    Fire(main)
