import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
# tf.random.set_seed(0)



def bm_u_NN():
    inputs = keras.Input(shape=(1))
    l = layers.Dense(1, activation = 'linear', use_bias = False)
    outputs = l(inputs)
    control_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'control_NN')
    return control_NN


def bm_V_NN():
    inputs = keras.Input(shape=(1))
    inputs0 = tf.zeros_like(inputs)
    l1 = layers.Dense(8, activation = 'elu')
    l2= layers.Dense(8, activation = 'elu')
    l3= layers.Dense(1, activation = 'linear')
    outputs = l3(l2(l1(inputs)))
    outputs0 = l3(l2(l1(inputs0)))
    outputs = outputs - outputs0
    outputs = outputs**2
    value_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'value_NN')
    return value_NN


def bm_dV_NN():
    inputs = keras.Input(shape=(1))
    l = layers.Dense(1, activation = 'linear', use_bias = False)
    outputs = l(inputs)
    dvalue_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'dvalue_NN')
    return dvalue_NN
    
def bm_rho_NN():
    inputs = keras.Input(shape=(1))
    l= layers.Dense(1, activation = 'linear', use_bias = False)
    outputs = l(inputs)
    rho_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'rho_NN')
    return rho_NN