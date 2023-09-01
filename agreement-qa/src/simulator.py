import json
from pathlib import Path
from fire import Fire
from pprint import pprint
import random
random.seed(42)
from typing import List, Tuple, Dict, Union, Set, Optional

from add_agreement_deltas import add_agreement_delta
from grid_search import grid_search


def opp(speaker):
    return 'A' if speaker == 'B' else 'B'


def generate_negotiation_dialogue(
    ontology: Dict[str, Set[str]],
    num_turns: int,
    proposals_A: Tuple[float, float],
    proposals_B: Tuple[float, float],
    partial_rejections_A: Tuple[float, float],
    partial_rejections_B: Tuple[float, float],
    partial_accepts_A: Tuple[float, float],
    partial_accepts_B: Tuple[float, float],
    min_dist_between_proposal_and_accept_or_reject: int = 4
) -> List[Dict[str, Union[str, List[Tuple[str, Dict[str, str]]]]]]:
    
    def generate_proposals(
        turn_id: int,
        speaker: str,
        open_proposals: List[Tuple[str, str]],
        ontology: Dict[str, Set[str]],
        speaker_A_params: Tuple[float, float],
        speaker_B_params: Tuple[float, float]
    ) -> List[Tuple[str, Dict[str, str]]]:
        """
        Generate a list of proposals for a given speaker based on ontology and speaker's proposal distribution.
        
        Args:
            - speaker (str): 'A' or 'B'
            - open_proposals (List[Tuple[str, str]]): Open proposals that can be rejected.
            - ontology (Dict[str, Set[str]]): Available slots and values for proposals.
            - speaker_A_params (Tuple[float, float]): Mean and SD for number of proposals for Speaker A.
            - speaker_B_params (Tuple[float, float]): Mean and SD for number of proposals for Speaker B.
        
        Returns:
            - List[Tuple[str, Dict[str, str]]]: Proposals in format [("propose", {slot: value})]
        """

        def get_speaker_params(speaker: str) -> Tuple[float, float]:
            """Retrieve the appropriate speaker parameters."""
            return speaker_A_params if speaker == 'A' else speaker_B_params

        def get_num_proposals(speaker_params: Tuple[float, float]) -> int:
            """Determine the number of proposals for a speaker."""
            return max(1, round(random.gauss(speaker_params[0], speaker_params[1])))

        def select_unique_value(slot: str, existing_proposals: List[Tuple[str, str]]) -> Optional[str]:
            """Pick a unique value for a given slot that doesn't exist in existing proposals."""
            for _ in range(MAX_ATTEMPTS):
                value = random.choice(list(ontology[slot]))
                if (slot, value) not in existing_proposals:
                    return value
            return None

        MAX_ATTEMPTS = 100
        proposals = []
        speaker_params = get_speaker_params(speaker)
        num_proposals = get_num_proposals(speaker_params)

        for _ in range(num_proposals):
            slot = random.choice(list(ontology.keys()))
            value = select_unique_value(slot, open_proposals + proposals)
            
            if value:
                proposals.append((slot, value))

        # Update the open proposals with the new unique proposals.
        open_proposals.extend(proposals)
        
        # Return proposals in the required format.
        return [
            ("propose", {slot: value})
            for slot, value in proposals
        ]

    def get_rejections_for_speaker(
        turn_id: int,
        speaker: str,
        open_proposals: List[Tuple[str, str]],
        min_dist_between_proposal_and_accept_or_reject: int
    ) -> List[Tuple[str, Dict[str, str]]]:
        """
        Generate a list of rejection actions for a given speaker based on open proposals.

        Args:
        - speaker: Either 'A' or 'B' indicating the speaker.
        - open_proposals: A list of current open proposals available for rejection.

        Returns:
        - A list of rejection actions in the format [("reject", {slot: value})].
        """

        def get_number_of_partial_rejections() -> int:
            """
            Get the number of partial rejections for a speaker based on a Gaussian distribution.
            
            Returns:
            - An integer representing the number of partial rejections.
            """
            if speaker == 'A':
                return max(0, round(random.gauss(partial_rejections_A[0], partial_rejections_A[1])))
            return max(0, round(random.gauss(partial_rejections_B[0], partial_rejections_B[1])))

        def formulate_rejections(num_partial: int) -> List[Tuple[str, Dict[str, str]]]:
            """
            Formulate a list of rejections based on a given number and the available open proposals.
            
            Args:
            - num_partial: The number of partial rejections to be formulated.
            
            Returns:
            - A list of rejection actions in the format [("reject", {slot: value})].
            """
            rejections = []

            for _ in range(num_partial):
                if not open_proposals:
                    break
                slot, value = random.choice(open_proposals)
                # proposal_idx = find_proposal_turn_index(turn_id, opp(speaker), slot, value)
                # if turn_id - proposal_idx < min_dist_between_proposal_and_accept_or_reject:
                #     continue
                open_proposals.remove((slot, value))
                rejections.append(("reject", {slot: value}))

            return rejections

        # Use the helper functions to generate and return the rejections
        num_partial = get_number_of_partial_rejections()
        return formulate_rejections(num_partial)

    def get_accepts_for_speaker(
        turn_id: int,
        speaker: str,
        open_proposals: List[Tuple[str, str]],
        min_dist_between_proposal_and_accept_or_reject: int
    ):
        """
        Generate a list of accept actions for the speaker.
        """
        def number_of_accepts():
            """Determine the number of accepts based on speaker preference."""
            if speaker == 'A':
                return max(0, round(random.gauss(partial_accepts_A[0], partial_accepts_A[1])))
            return max(0, round(random.gauss(partial_accepts_B[0], partial_accepts_B[1])))

        num_accepts = number_of_accepts()
        accepts = []

        for _ in range(num_accepts):
            if not open_proposals:
                break
            slot, value = random.choice(open_proposals)
            # proposal_idx = find_proposal_turn_index(turn_id, opp(speaker), slot, value)
            # if turn_id - proposal_idx < min_dist_between_proposal_and_accept_or_reject:
            #     continue
            open_proposals.remove((slot, value))
            accepts.append(("accept", {slot: value}))

        return accepts

    def update_based_on_actions(actions):
        """Update lists (like agreements and open offers) based on the given actions."""
        for action in actions:
            act_type, details = action
            if act_type == "accept" and details:
                record_agreement(*list(details.items())[0])
            elif act_type == "propose":
                open_offers[speaker].append(
                    list(details.items())[0]
                )
            elif act_type in ["reject", "accept"]:
                for slot, value in details.items():
                    if (slot, value) in open_offers[opponent]:
                        open_offers[opponent].remove((slot, value)) 

    def record_agreement(slot, value):
        """Add an agreement if not already present."""
        if (slot, value) not in agreements:
            agreements.append((slot, value))

    def find_proposal_turn_index(curr_turn_id, speaker, slot, value):
        """Find the turn index at which the proposal that was made."""
        breakpoint()
        for turn_idx in range(curr_turn_id - 1, -1, -1):
            turn = dialogue[turn_idx]
            if turn['turn'] == speaker:
                for act, details in turn['acts']:
                    if act == 'propose' and details:
                        if list(details.items())[0] == (slot, value):
                            return turn_idx
        raise RuntimeError("Proposal not found.")

    # Initial setups
    dialogue = []
    open_proposals = []
    agreements = []
    open_offers = {'A': [], 'B': []}
    
    for i in range(num_turns):
        speaker = 'A' if i % 2 == 0 else 'B'
        opponent = opp(speaker)

        # Gather all actions
        rejections = get_rejections_for_speaker(i, speaker, open_proposals, min_dist_between_proposal_and_accept_or_reject)
        accepts = get_accepts_for_speaker(i, speaker, open_proposals, min_dist_between_proposal_and_accept_or_reject)
        proposals = generate_proposals(i, speaker, open_proposals, ontology, proposals_A, proposals_B)

        all_actions = rejections + accepts + proposals

        # Update states
        update_based_on_actions(all_actions)

        # Update dialogue structure
        dialogue.append({
            "turn": speaker,
            "acts": all_actions,
            "agreements": agreements.copy(),
            "open_offers": {
                'A': open_offers['A'].copy(),
                'B': open_offers['B'].copy()
            }
        })

    return dialogue

def add_natural_language_utterances(
    dialogue: List[Dict[str, Union[str, List[Tuple[str, Dict[str, str]]]]]],
    verbalizer: str
) -> None:
    from verbalizer import VERBALIZERS
    # Maps for turning actions into phrases
    action_to_phrase = VERBALIZERS[verbalizer]
    
    for turn in dialogue:
        acts = turn['acts']
        
        turn_phrases = []
        for act in acts:
            action, details = act
            if details:
                for slot, value in details.items():
                    phrase = action_to_phrase[action].format(slot=slot, value=value)
                    turn_phrases.append(phrase)
            else:
                # Handle general acceptance or rejection without specific details
                if action == 'accept':
                    turn_phrases.append('I agree.')
                else:
                    turn_phrases.append('I disagree.')

        natural_turn = ' '.join(turn_phrases)
        turn['text'] = natural_turn


def params_to_filename(params: Dict[str, float]) -> str:
    """Convert a dictionary of parameters to a filename (cannot have =)"""
    return '_'.join([f"{k}_{v}" for k, v in params.items()])


def filename_to_params(filename: str) -> Dict[str, float]:
    """Convert a filename back to its dictionary representation of parameters."""
    # Split the filename based on underscores
    parts = filename.split('_')
    # Reconstruct the dictionary by pairing every two items in the list
    params = {parts[i]: float(parts[i+1]) for i in range(0, len(parts), 2)}
    return params


def main(
    num_turns: int = 20,
    num_points_grid_search: int = 10,
    num_train: int = 300,
    num_val: int = 100,
    num_test: int = 300,
    min_dist_between_proposal_and_accept_or_reject: int = 4,
    verbalizer: str = "explicit_mention_slot_only"
):
    ontology = {
        'Job Description': {'Project Manager', 'Team Manager', 'QA', 'Programmer'},
        'Salary': {'90,000 USD', '120,000 USD', '60,000 USD'},
        'Leased Car': {'With leased car', 'Without leased car'},
        'Pension Fund': {'0%', '20%', '10%'},
        'Promotion Possibilities': {'Slow promotion track', 'Fast promotion track'},
        'Working Hours': {'9 hours', '10 hours', '8 hours'}
    }
    ontology = {k: list(v) for k, v in ontology.items()}
    
    for params in grid_search(params={
        "proposals_A": [0, 1, 2, 3],
        "proposals_B": [0, 1, 2, 3],
        "partial_rejections_A": [0, 1, 2, 3],
        "partial_rejections_B": [0, 1, 2, 3],
        "partial_accepts_A": [0, 1, 2, 3],
        "partial_accepts_B": [0, 1, 2, 3],
    }, total_points=num_points_grid_search):
        for split, num_dials in [
            ("train", num_train),
            ("val", num_val),
            ("test", num_test),
        ]:
            print(f"Generating dialogues for {params}")
            dials = []
            for i in range(num_dials):
                dial = {
                    "dialogue_id": i,
                    "turns": []
                }
                dialogue = generate_negotiation_dialogue(
                    ontology=ontology,
                    num_turns=num_turns,
                    proposals_A=(params["proposals_A"], 0.5),
                    proposals_B=(params["proposals_B"], 0.5),
                    partial_rejections_A=(params["partial_rejections_A"], 0.5),
                    partial_rejections_B=(params["partial_rejections_B"], 0.5),
                    partial_accepts_A=(params["partial_accepts_A"], 0.5),
                    partial_accepts_B=(params["partial_accepts_B"], 0.5),
                    min_dist_between_proposal_and_accept_or_reject=min_dist_between_proposal_and_accept_or_reject
                )
                add_natural_language_utterances(dialogue, verbalizer)
                for i, turn in enumerate(dialogue):
                    dial["turns"].append({
                        turn["turn"]: turn["text"],
                        "agreements": {
                            x[0]: x[1]
                            for x in turn["agreements"]
                        },
                        "acts": turn['acts'],
                        "open_offers": turn['open_offers']
                    })
                    # print(turn['text'])
                    # print(f"Agreements: {turn['agreements']}")
                    # print(f"Open Offers: {turn['open_offers']}")
                    # print()
                dial = add_agreement_delta(dial)
                dials.append(dial)
            json.dump(dials, open(f"data/simulated/{split}/{params_to_filename(params)}.json", "w"), indent=2)


# for turn in dialogue:
#     print(f"Turn {turn['turn']}:")
#     for act, pair in turn['acts']:
#         if pair:
#             slot, value = list(pair.items())[0]
#             print(f"  {act}({slot} = {value})")
#         else:
#             print(f"  {act}()")
#     print(f"  Agreements: {turn['agreements']}")
#     print(f"  Open Offers: {turn['open_offers']}")
#     print()

if __name__ == "__main__":
    Fire(main)
