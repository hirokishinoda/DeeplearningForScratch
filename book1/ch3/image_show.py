from PIL import Image
import sys, os
sys.path.append(os.pardir)
from lib.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np

def img_show(img):
    #pil_img = Image.fromarray(np.uint8(img))
    #pil_img.show()
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]
    label = y_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)