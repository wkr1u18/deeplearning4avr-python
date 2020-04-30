import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
test_image = pd.read_csv("test_image.txt", header=None).to_numpy()[:,0].reshape(1,81)
first_layer = pd.read_csv("first_layer.txt", header=None).to_numpy()[:,0].reshape(82,20)
second_layer = pd.read_csv("second_layer.txt", header=None).to_numpy()[:,0].reshape(21,10)

pixels = test_image.reshape((9, 9))
plt.imshow(pixels, cmap='gray')
plt.show()

def relu(x):
  return np.maximum(0,x)

def add_bias_unit(X):
    return np.append(np.ones(X.shape[0]).reshape(-1,1), X, axis=1)

print(np.argmax(np.dot(add_bias_unit(relu(np.dot(add_bias_unit(test_image),first_layer))),second_layer)))

print(np.dot(add_bias_unit(test_image),first_layer))