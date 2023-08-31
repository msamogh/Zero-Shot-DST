import random
from itertools import product

def grid_search(params, total_points):
    """
    Generate a grid search with a cross product across all parameters.

    Args:
    - params (dict): Dictionary with keys being parameter names and values being
                     lists of possible values for that parameter.
    - total_points (int): Number of parameter combinations to generate.

    Returns:
    - Iterable of dicts: Each dictionary represents a unique combination of parameters.
    """
    
    # Calculate the cross product of the parameter values
    all_combinations = list(product(*params.values()))
    
    # Randomly sample combinations if there are more combinations than total_points
    sampled_combinations = random.sample(all_combinations, min(total_points, len(all_combinations)))

    # Map the combinations back to the parameter names
    results = [dict(zip(params.keys(), combination)) for combination in sampled_combinations]

    return results


if __name__ == "__main__":
    # Example Usage:
    params = {
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [32, 64, 128, 256],
    }
    total_points = 5

    print(grid_search(params, total_points))
