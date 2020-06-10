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


# Value Neural Network: V(x)
inputs = keras.Input(shape=(1))
inputs0 = tf.zeros_like(inputs)
l1 = layers.Dense(8, activation = 'elu')
l2= layers.Dense(8, activation = 'elu')
l3= layers.Dense(1, activation = 'linear')
outputs = l1(inputs)
outputs = l2(outputs)
outputs = l3(outputs)
outputs0 = l1(inputs0)
outputs0 = l2(outputs0)
outputs0 = l3(outputs0)
outputs = outputs - outputs0
value_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'value_NN')
# value_NN.summary()
# print('value_NN')
# NN_plot(value_NN)


# optimal value: rho
inputs = keras.Input(shape=(1))
l= layers.Dense(1, activation = 'linear')
outputs = l(inputs)
rho_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'rho')
# rho_NN.summary()
# print('rho_NN')
# print(rho_NN.predict(np.zeros(1))[0,0])

def show():
    print('Control Neural Network: ')
    NN_plot(control_NN)
    print('Value Neural Network: ')
    NN_plot(value_NN)
    print('Optimal cost: ')
    print(rho_NN.predict(np.zeros(1))[0,0])
    