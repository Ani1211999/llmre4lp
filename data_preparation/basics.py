import torch
import random
import numpy as np
def set_seeds(seed=42):
    '''
    Set seed for Python ramdom, torch and numpy.
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed
