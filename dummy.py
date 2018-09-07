# import cv2
#
# img = cv2.imread('dataset/test/dagim_kosher/dagim kosher_11.jpg')
#
# cv2.imshow("namedwindow", img)
# cv2.waitKey(0)

# import matplotlib.pyplot as plt
# import numpy as np
# x = list(np.arange(100))
# y = [x*x for x in x]
# plt.plot(x, y, 'b')
# plt.savefig('quadratic.png')
# plt.show()

from datagenerator import datagenerator
from utils import *
from PIL.Image import Image
from keras.preprocessing.image import load_img

train_generator = datagenerator.train_datagenerator(train_batchsize)


def squarePrinter():
    print("This is square Printer!")
    img = load_img("dataset/train/dagim_kosher/dagim kosher_1.jpg")
    print('typ of loaded image', type(img))
    img.show()

def cubePrinter():
    print("This is cube Printer!")


if __name__ == '__main__':
    cubePrinter()
    squarePrinter()