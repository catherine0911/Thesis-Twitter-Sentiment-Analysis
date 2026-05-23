import random
import numpy as np
import torch
import os

# Set seed function for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # This ensures that Python's hash-based operations (like dictionary keys) stay consistent
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Turn CuDNN off for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)