import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.rand(7, 3)
layer = Affine(W, 0)
h = layer.forward(c)
print(h)