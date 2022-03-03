import numpy as np

def step_function(x):
    return (x > 0).astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
