import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def NN_plot(nn):
    nn.compile(loss = 'mse')
    xgrid = np.linspace(-10,10,201)
    y = nn.predict(xgrid)
    plt.plot(xgrid,y)
    plt.grid()
    plt.show()

# Control Neural Network: u(x)
inputs = keras.Input(shape=(1))
l1 = layers.Dense(8, activation = 'elu')
l2 = layers.Dense(8, activation = 'elu')
l3= layers.Dense(1, activation = 'linear')
outputs = l1(inputs)
outputs = l2(outputs)
outputs = l3(outputs)
control_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'control_NN')
# control_NN.summary()
# print('control_NN')
# NN_plot(control_NN)

def show():
    print('Control Neural Network: ')
    NN_plot(control_NN)
    