import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def resize(mnist):
    train_data = []
    for img in mnist:
        resized_img = cv2.resize(img.reshape((28,28)), (9, 9))
        train_data.append(resized_img.reshape(81,))
    return train_data

x_train = np.array(resize(x_train))

print(x_train[0])

def format(value):
    return "%d" % value

with open('10.txt', 'w') as f:
    for item in x_train[9]:
        f.write(format(item) + ",\n")