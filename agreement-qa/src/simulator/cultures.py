from simulator import SimulationParams
from itertools import product


VALUE_TO_BOOL_MAP = {
    "explorer": True,
    "conservative": False,
    "avoidant": True,
    "direct": False,
    "stoic": True,
    "celebratory": False,
    "flexible": True,
    "rigid": False,
}
CULTURE_NAMES = {
    "is_multiple_proposals_in_one_turn_allowed": ["explorer", "conservative"],
    "is_overriding_proposal_implicit_rejection": ["avoidant", "direct"],
    "is_tangential_proposal_implicit_acceptance": ["stoic", "celebratory"],
    "can_agreement_be_overridden": ["flexible", "rigid"],
}
culture_names = product(*CULTURE_NAMES.values())

def get_cultures():
    cultures = {}
    for culture_name in culture_names:
        culture_params = dict(
            zip(CULTURE_NAMES.keys(), [VALUE_TO_BOOL_MAP[c] for c in culture_name])
        )
        cultures["_".join(culture_name)] = culture_params
    return cultures

if __name__ == "__main__":
    print(get_cultures())
