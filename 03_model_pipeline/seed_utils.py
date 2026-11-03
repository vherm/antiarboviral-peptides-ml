import os
import random
import numpy as np

SEED = 42

def set_global_seed(seed: int = SEED):
    """Set global seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
