import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
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
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def resize(mnist):
    train_data = []
    for img in mnist:
        resized_img = cv2.resize(img.reshape((28,28)), (9, 9))
        train_data.append(resized_img.reshape(81,))
    return train_data

x_train = np.array(resize(x_train))
x_test = np.array(resize(x_test))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(81,)))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
model.compile(loss='categorical_crossentropy',
               optimizer=RMSprop(),
               metrics=['accuracy'])

history = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

first_layer_weights = np.array(model.layers[0].get_weights()[0]).reshape(81,20)
first_layer_biases  = np.array(model.layers[0].get_weights()[1]).reshape(1,20)

first_layer = np.append(first_layer_biases, first_layer_weights, axis=0)
fst_layer_list = first_layer.reshape(1,1640).tolist()


second_layer_weights = np.array(model.layers[1].get_weights()[0]).reshape(20,10)
second_layer_biases  = np.array(model.layers[1].get_weights()[1]).reshape(1,10)
second_layer = np.append(second_layer_biases, second_layer_weights, axis=0)
snd_layer_list = second_layer.reshape(1,21*10).tolist()

test_image = x_test[0].reshape(1,81)

def format(value):
    return "%.6f" % value

with open('first_layer.txt', 'w') as f:
    for item in fst_layer_list[0]:
        f.write(format(item) + ",\n")

with open('second_layer.txt', 'w') as f:
    for item in snd_layer_list[0]:
        f.write(format(item) + ",\n")

with open('test_image.txt', 'w') as f:
    for item in test_image[0]:
        f.write(format(item) + ",\n")