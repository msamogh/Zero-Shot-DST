from dataclasses import dataclass, field, asdict
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
    DialogueState, ActType, Turn, Act
)
from updaters import DialogueStateUpdate, Updater


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
    is_tangential_proposal_implicit_rejection: bool = False
    is_tangential_proposal_implicit_acceptance: bool = False
    does_new_proposal_immediately_undo_agreement: bool = False

    def __post_init__(self):
        assert not (
            self.is_tangential_proposal_implicit_rejection and \
            self.is_tangential_proposal_implicit_acceptance
        ), "Cannot have both implicit rejection and implicit acceptance for tangential proposals."
    

# Dialogue-related dataclasses
@dataclass
class SimulatedDialogue:
    name: str
    params: SimulationParams

    dialogue_state_update: DialogueStateUpdate = field(default_factory=DialogueStateUpdate)
    dialogue_state: DialogueState = field(default_factory=DialogueState)

    history: List[Turn] = field(default_factory=list)


    @classmethod
    def generate_dialogue(cls, name: str, params: SimulationParams):
        """Generate a dialogue with the given parameters."""
        from updaters import (
            GenerateProposals, GenerateExplicitAgreements, GenerateExplicitRejections,
            GenerateImplicitAgreements, GenerateImplicitRejections,
            UndoAgreementsForNewProposals
        )
        updaters = [
            UndoAgreementsForNewProposals,
            GenerateExplicitAgreements,
            GenerateExplicitRejections,
            GenerateImplicitAgreements,
            GenerateImplicitRejections,
            GenerateProposals,
        ]
        mean_num_slot_values = {
            GenerateExplicitAgreements: params.mean_accepts_a,
            GenerateExplicitRejections: params.mean_rejects_a,
            GenerateProposals: params.mean_proposals_a,
        }
        # Initialize the dialogue
        dialogue = cls(name, params)
        for _ in range(params.num_turns):
            for updater in updaters:
                updater(
                    params,
                    dialogue.dialogue_state_update,
                    dialogue.dialogue_state,
                    dialogue.history,
                    mean_num_slot_values=mean_num_slot_values.get(updater, 0),
                ).apply()
            dialogue.history.append(
                Turn(
                    speaker=dialogue.dialogue_state_update.curr_speaker,
                    state=copy.deepcopy(dialogue.dialogue_state),
                    acts=Updater.get_acts(dialogue.dialogue_state_update),
                    simulation_params=params
                )
            )
            Updater.update_dialogue_state(dialogue.dialogue_state_update, dialogue.dialogue_state)
            dialogue.dialogue_state_update.prepare_for_new_turn()
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
                        act.slot_value.slot: act.slot_value.value,
                        **act.extra_info
                    }
                } for act in turn.acts]
            } for turn in self.history]
        }


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
        mean_rejects_a=2,
        mean_rejects_b=2,
        mean_accepts_a=2,
        mean_accepts_b=2,
        min_dist_between_proposal_and_accept_or_reject=1,
        is_tangential_proposal_implicit_rejection=True,
        is_tangential_proposal_implicit_acceptance=False,
        does_new_proposal_immediately_undo_agreement=True,
    )
    dialogues = [
        SimulatedDialogue.generate_dialogue(str(i), params).to_json()
        for i in range(params.num_dials)
    ]
    name = randomname.get_name()
    json.dump({
        "dialogues": dialogues,
        "params": asdict(params)
    }, Path(f"data/simulated2/{name}.json").open("w"), indent=2)
    print(f"Written to data/simulated2/{name}.json")


if __name__ == "__main__":
    Fire(main)
