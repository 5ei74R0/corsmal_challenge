"""
https://github.com/OtsuKotsu/MLExp/blob/main/mlexp/utils/reproducibility/seed.py
"""
import random

import numpy
import torch

SEED = 0


def fix_random_seeds(specified_seed: int = SEED):
    random.seed(specified_seed)
    numpy.random.seed(seed=specified_seed)
    torch.manual_seed(seed=specified_seed)
    torch.cuda.manual_seed_all(seed=specified_seed)


def behave_deterministically():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
