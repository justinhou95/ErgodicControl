import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def initial_control_NN(input_dim):
    # Control Neural Network: u(x)
    inputs = keras.Input(shape=(input_dim))
    l1 = layers.Dense(8, activation = 'elu')
    l2 = layers.Dense(8, activation = 'elu')
    l3= layers.Dense(1, activation = 'linear')
    outputs = l1(inputs)
    outputs = l2(outputs)
    outputs = l3(outputs)
    control_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'control_NN')
#     control_NN.summary()
#     NN_plot(control_NN)
#     plt.title('control NN')
#     plt.show()
    return control_NN


def initial_g_NN(input_dim):
    # Control Neural Network: u(x)
    inputs = keras.Input(shape=(input_dim))
    l1 = layers.Dense(8, activation = 'elu')
    l2 = layers.Dense(8, activation = 'elu')
    l3= layers.Dense(1, activation = 'linear')
    outputs = l1(inputs)
    outputs = l2(outputs)
    outputs = l3(outputs)
    g_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'g_NN')
#     control_NN.summary()
#     NN_plot(control_NN)
#     plt.title('control NN')
#     plt.show()
    return g_NN


def initial_value_NN(input_dim):
    # Value Neural Network: V(x)
    inputs = keras.Input(shape=(input_dim))
    inputs0 = tf.zeros_like(inputs)
    l1 = layers.Dense(8, activation = 'elu')
    l2= layers.Dense(8, activation = 'elu')
    l3= layers.Dense(1, activation = 'elu')
    outputs = l1(inputs)
    outputs = l2(outputs)
    outputs = l3(outputs)
    outputs0 = l1(inputs0)
    outputs0 = l2(outputs0)
    outputs0 = l3(outputs0)
    outputs = outputs - outputs0
    value_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'value_NN')
    # value_NN.summary()
#     NN_plot(value_NN)
#     plt.title('value NN')
#     plt.show()
    return value_NN

def initial_rho_NN():
    # optimal value: rho
    inputs = keras.Input(shape=(1))
    l= layers.Dense(1, activation = 'linear')
    outputs = l(inputs)
    rho_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'rho')
    # rho_NN.summary()
#     print('rho_NN: ',rho_NN.predict(np.zeros(1))[0,0])
    return rho_NN

def square_loss(target_y, predicted_y):
    return tf.reduce_mean(tf.math.square(predicted_y - target_y))

custom_optimizer = keras.optimizers.Adam(learning_rate = 0.01)


def NN_plot(nn):
    nn.compile(loss = 'mse')
    xgrid = np.linspace(-10,10,201)
    y = nn.predict(xgrid)
    plt.plot(xgrid,y)
    plt.grid()
    
class MODEL:
    restart_times = 0
    def __init__(self,steps,dt,T,M,n,H):
        self.steps = steps
        self.dt = dt
        self.sqrtdt = np.sqrt(self.dt)
        self.T = self.steps * self.dt
        self.M = M
        self.n = n
        self.H = H
        self.d = n+1
        self.alpha = H + 0.5 
        t_grid = np.arange(steps+1)*dt
        dh = ((10*(1-2*H)**2)/(n*(5-2*H)**2))**(1/5)/T
        h_grid = np.arange(n+1)*dh
        h1_grid = h_grid**(0.5-H)
        h2_grid = h_grid**(1.5-H)
        self.c_grid = (h1_grid[1:] - h1_grid[:-1]) / (0.5-H) / math.gamma(H+0.5) / math.gamma(0.5-H)
        self.gamma_grid = (h2_grid[1:] - h2_grid[0:-1]) / self.c_grid / (1.5-H) / math.gamma(H+0.5) / math.gamma(0.5-H) 
        self.c = np.sum(self.c_grid)
        self.u_optimal = lambda a : a + 10
        self.V_optimal = lambda a : a^2
        self.g_optimal = lambda a : 0
        self.rho_optimal = lambda a : 1
    def start(self,X0): 
        self.X0 = X0    
    def data(self, seed):
        # initial distribution
        x = [np.zeros(shape = (self.M,1)), np.zeros(shape = (self.M,self.n)), self.X0]
        # generate the increments
        Y = self.X0[:,1:]
        X = np.zeros(shape = (self.M,self.d))
        np.random.seed(seed)
        for it in range(self.steps):
            dw =  np.random.normal(0,np.sqrt(self.dt),size = (self.M,1))
            dY = - self.gamma_grid * Y * self.dt + np.tile(dw,(1,self.n))
            Y = Y + dY
            dfBM = np.sum(self.c_grid * dY, axis = -1)
            X[:,0] = dfBM
            X[:,1:] = dY
            x = x + [X, dw]
        y = np.zeros(shape = (self.M,1))
        return x, y    
    def traindata(self,seed):
        self.x_train, self.y_train = self.data(seed+1)
        self.x_valid, self.y_valid = self.data(seed+2)
    def nn(self):
        self.unn = initial_control_NN(self.d)
        self.Vnn = initial_value_NN(self.d)
        self.gnn = initial_g_NN(self.d)
        self.rhonn = initial_rho_NN()
    def build(self):
  
        # Build the main network
        input_aux1 = keras.Input(shape=(1))
        input_aux2 = keras.Input(shape=(self.n))
        zero_helper = tf.zeros_like(input_aux2)
        input_x = keras.Input(shape=(self.d))
        inputs = [input_aux1, input_aux2, input_x]
        X_start = input_x
        X_now = X_start
        loss = tf.zeros_like(input_aux1)
        bsde = tf.zeros_like(input_aux1)
        loss_output = [loss]
        rho = self.rhonn(input_aux1)
        
        for i in range(self.steps):
            input_dfW = keras.Input(shape=(self.d))
            input_dw = keras.Input(shape=(1))
            inputs = inputs + [input_dfW, input_dw]
            u_now = self.unn(X_now)              # Here we use the optimal control
            X_next  = X_now + input_dfW 
            u_change = tf.concat([u_now*self.dt, zero_helper], axis = 1)
            X_next = X_next + u_change
            loss_tmp = (tf.math.square(X_now[:,0:1]) + tf.math.square(u_now))*self.dt
            loss_output = loss_output + [loss_tmp]
            loss = loss + loss_tmp
            bsde_tmp = tf.multiply(-2*self.c*u_now + self.gnn(X_now),input_dw) 
            bsde = bsde + bsde_tmp
            X_now = X_next
        outputs = loss - bsde - rho * self.T + self.Vnn(X_now) - self.Vnn(X_start)
        control_main = keras.Model(inputs=inputs, outputs = outputs, name = 'control_main')
        control_terminal = keras.Model(inputs=inputs, outputs = X_now, name = 'control_terminal')
        control_loss = keras.Model(inputs=inputs, outputs = loss_output, name = 'control_loss')
        self.mainnn = control_main
        self.endnn = control_terminal
        self.lossnn = control_loss
        
    def plot_compare(self):
        NN_plot(self.Vnn)
        xgrid = np.linspace(-10,10,201)
        plt.plot(xgrid, self.value_optimal(xgrid))
        plt.show()
        NN_plot(self.unn)
        plt.plot(xgrid, self.u_optimal(xgrid))
        plt.show()
        tmp = self.rhonn.predict(np.zeros(1))[0,0]
        print('Optimal ergodic cost is: ',tmp)
    def end(self):
        self.endnn.compile(loss = 'mse')
        self.Xend = self.endnn.predict(self.x_train)
    def train(self, epochs = 20, optimizer = 'Adam', verbose = 1):
        self.mainnn.compile(loss = 'mse' , optimizer = optimizer)
        self.mainnn.fit(self.x_train,self.y_train,epochs =epochs,\
                        validation_data = (self.x_valid,self.y_valid), verbose = verbose)
    def autotrain(self, epochs = 20, opt = 'Adam', verbose = 1):
        self.optimal()
        self.train(epochs = epochs, optimizer = optimizer, verbose = verbose)
        self.plot_compare()
        self.end()
        print('Mean and Var of terminal distribution: ',self.Xend.mean(), self.Xend.var())

