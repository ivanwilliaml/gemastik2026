# title: Seed
# when: Awal notebook.
# tags: seed reprodusibel
import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
