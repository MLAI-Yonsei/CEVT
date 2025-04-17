# Example usage
import sys
sys.path.append("../") # go to parent dir
from utils.data_simulator import DataSimulator
import numpy as np
import pandas as pd
import random
from utils.utils_syn import get_f1, summary
from sklearn.metrics import roc_auc_score
import multiprocessing

seed = 0
np.random.seed(seed)
random.seed(seed)

# Number of valid IVs
p = 10

# Are there dependencies within valid IVs?
cond_indp = False

# Max degree of dependencies
d = 5

# Number of invalid IVs
q = 8

# Number of samples
n = 80000

# Create a DataSimulator instance
mdl = DataSimulator(p, cond_indp=cond_indp, d=d, q=q, seed=seed, causal_effect=True, valid_z=True)

# Sample synthetic data from mdl
z, IVs, x, y = mdl.get_samples(n=n)

# Create DataFrame for analysis
data = pd.DataFrame({
    'z': z,
    'x': x,
    't': x,  # 't' should be sampled from BayesNetDataSimulator, correct this if necessary
    'y': y
})

# Display first few rows
print(data.head())
