import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine

x0 = np.array([[1, 0, 0, 0, 0, 0, 0,]])
x1 = np.array([[0, 0, 1, 0, 0, 0, 0,]])

W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

in_layer0 = Affine(W_in, 0)
in_layer1 = Affine(W_in, 0)
out_layer = Affine(W_out, 0)

h0 = in_layer0.forward(x0)
h1 = in_layer1.forward(x1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s)