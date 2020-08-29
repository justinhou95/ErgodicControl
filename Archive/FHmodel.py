import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Control Neural Network: u(x)\
def control_NN_initial():   
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
    return control_NN


def trainning_data(seed,X0,samplesM,stepsN,dtt):
    np.random.seed(seed)
    x = [X0]
    for i in range(stepsN):
        x = x + [np.random.normal(0,np.sqrt(dtt),size = (samplesM,1))]
    y = np.zeros(shape = (samplesM,1))
    return x, y

def NN_plot(nn):
    nn.compile(loss = 'mse')
    xgrid = np.linspace(-10,10,201)
    y = nn.predict(xgrid)
    plt.plot(xgrid,y)
    plt.grid()
    
class MODEL:
    restart_times = 0
    def __init__(self,steps,dt,T,M):
        self.steps = steps
        self.dt = dt
        self.sqrtdt = np.sqrt(self.dt)
        self.T = self.steps * self.dt
        self.M = M
    def dyna(self,name = 'bm'):
        if name == 'bm':
            def u_optimal(x):
                return -x
            self.u_optimal = u_optimal
            def dynamic(u,x):
                return u
            self.dynamic = dynamic
        if name == 'mr':
            def u_optimal(x):
                a = 2*(np.sqrt(2) - 1)
                b = 2 - np.sqrt(2)
                return -0.5*(a*x + b)
            self.u_optimal = u_optimal
            def dynamic(u,x):
                return (u + 1 -x)  
            self.dynamic = dynamic
    def nn(self,unn):
        self.unn = unn
    def start(self,X0):
        self.X0 = X0
    def traindata(self,seed):
        self.x_train, self.y_train = trainning_data(seed+1,self.X0,self.M,self.steps,self.dt)
        self.x_valid, self.y_valid = trainning_data(seed+2,self.X0,self.M,self.steps,self.dt)
    def build(self):
        # Build the optimal network
        input_x = keras.Input(shape=(1))
        inputs = [input_x]
        X_start = input_x
        X_now = X_start
        loss = tf.zeros_like(X_now)
        bsde = tf.zeros_like(X_now)
        loss_output = [loss]
        for i in range(self.steps):
            input_dW = keras.Input(shape=(1))
            inputs = inputs + [input_dW]
            u_now = self.u_optimal(X_now)  # Here we use the optimal control
            X_next  = X_now + input_dW + self.dynamic(u_now,X_now) * self.dt
            loss_tmp = (tf.math.square(X_now) + tf.math.square(u_now))*self.dt
            loss_output = loss_output + [loss_tmp]
            loss = loss + loss_tmp
            X_now = X_next
        self.optnn = keras.Model(inputs=inputs, outputs = loss/self.T, name = 'control_optimal')
        self.opt_endnn = keras.Model(inputs=inputs, outputs = X_now, name = 'control_optimal')
        
        # Build the main network
        input_x = keras.Input(shape=(1))
        inputs = [input_x]
        X_start = input_x
        X_now = X_start
        loss = tf.zeros_like(X_now)
        bsde = tf.zeros_like(X_now)
        loss_output = [loss]
        for i in range(self.steps):
            input_dW = keras.Input(shape=(1))
            inputs = inputs + [input_dW]
            u_now = self.unn(X_now)
        #     u_now = -0.5*d_value_NN(X_now)    #  Connect u with dV/dx 
            X_next  = X_now + input_dW + self.dynamic(u_now,X_now) * self.dt
            loss_tmp = (tf.math.square(X_now) + tf.math.square(u_now))*self.dt
            loss_output = loss_output + [loss_tmp]
            loss = loss + loss_tmp
            X_now = X_next
        
        self.mainnn = keras.Model(inputs=inputs, outputs = loss/self.T, name = 'control_main')
        self.endnn = keras.Model(inputs=inputs, outputs = X_now, name = 'control_terminal')
        
    def plot_compare(self):
        xgrid = np.linspace(-10,10,201)
        NN_plot(self.unn)
        plt.plot(xgrid, self.u_optimal(xgrid))
        plt.show()
    def end(self):
        self.endnn.compile(loss = 'mse')
        self.Xend = self.endnn.predict(self.x_train)
    def optimal(self):
        self.optnn.compile(loss = 'mean_absolute_error')
        tmp = self.optnn.evaluate(self.x_train, self.y_train)
        print('Cost under optimal ergodic control: ', tmp)
    def train(self, epo = 20, verb = 1):
        self.mainnn.compile(loss = 'mean_absolute_error')
        self.mainnn.fit(self.x_train,self.y_train,epochs =epo,\
                        validation_data = (self.x_valid,self.y_valid), verbose = verb)
        