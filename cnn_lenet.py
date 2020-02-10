import numpy as np
import lib
from optimizers import sgd_online
from optimizers import batch_gd
from optimizers import predict
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Keras Mnist dataset 
x_train = np.pad(x_train, [(0,0),(1,1), (1,1)], 'constant').astype(np.float64)
y_train = lib.to_onehot(y_train).astype(np.float64)
x_test = np.pad(x_test, [(0,0),(1,1), (1,1)], 'constant').astype(np.float64)
y_test = lib.to_onehot(y_test).astype(np.float64)

trained_model = sgd_online(x_train, y_train, 5)

predict(trained_model, x_test, y_test)








