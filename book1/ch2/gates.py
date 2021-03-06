import numpy as np

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

def AND_matrix(x1, x2):
    X = np.array([x1, x2])
    W = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(W*X) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    out1 = NAND(x1, x2)
    out2 = OR(x1, x2)
    out3 = AND(out1, out2)

    return out3

if __name__ == '__main__':

    print("-- AND gate --")
    print(AND_matrix(0, 0))
    print(AND_matrix(0, 1))
    print(AND_matrix(1, 0))
    print(AND_matrix(1, 1))

    print("-- NAND gate --")
    print(NAND(0, 0))
    print(NAND(0, 1))
    print(NAND(1, 0))
    print(NAND(1, 1))

    print("-- OR gate --")
    print(OR(0, 0))
    print(OR(0, 1))
    print(OR(1, 0))
    print(OR(1, 1))

    print("-- XOR gate --")
    print(XOR(0, 0))
    print(XOR(0, 1))
    print(XOR(1, 0))
    print(XOR(1, 1))

