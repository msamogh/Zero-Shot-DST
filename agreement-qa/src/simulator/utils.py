import random
random.seed(42)


def sample_gauss(mean, std=1):
    return int(random.gauss(mean, std))
