import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid, step_function

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    #y = step_function(x)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()