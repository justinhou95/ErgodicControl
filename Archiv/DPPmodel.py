import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def initial_control_NN():
    # Control Neural Network: u(x)
    inputs = keras.Input(shape=(1))
    l1 = layers.Dense(8, activation = 'elu')
    l2 = layers.Dense(8, activation = 'elu')
    l3= layers.Dense(1, activation = 'linear')
    outputs = l1(inputs)
    outputs = l2(outputs)
    outputs = l3(outputs)
    control_NN = keras.Model(inputs=inputs, outputs=outputs)
    # control_NN.summary()
#     NN_plot(control_NN)
#     plt.title('control NN')
#     plt.show()
    return control_NN

def initial_value_NN():
    # Value Neural Network: V(x)
    inputs = keras.Input(shape=(1))
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
#     outputs = tf.math.square(inputs)
    value_NN = keras.Model(inputs=inputs, outputs=outputs)
    # value_NN.summary()
#     NN_plot(value_NN)
#     plt.title('value NN')
#     plt.show()
    return value_NN

def square_loss(target_y, predicted_y):
    return tf.reduce_mean(tf.math.square(predicted_y - target_y))
custom_optimizer = keras.optimizers.Adam(learning_rate = 0.01)

def mean_loss(target_y, predicted_y):
    return tf.reduce_mean(predicted_y - target_y)

custom_optimizer = keras.optimizers.Adam(learning_rate = 0.01)

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
            def value_optimal(x):
                return x**2
            self.value_optimal = value_optimal
            self.rho_optimal = 1
        if name == 'mr':
            def u_optimal(x):
                a = 2*(np.sqrt(2) - 1)
                b = 2 - np.sqrt(2)
                return -0.5*(a*x + b)
            self.u_optimal = u_optimal
            def dynamic(u,x):
                return (u + 1 -x)  
            self.dynamic = dynamic
            def value_optimal(x):
                a = 2*(np.sqrt(2) - 1)
                b = 2 - np.sqrt(2)
                return a*x**2/2 + b*x
            self.value_optimal = value_optimal
            self.rho_optimal = (2*np.sqrt(2) - 1)/2
    def nn(self):              # Here rho is not a neural network
        self.unn = initial_control_NN()
        self.Vnn1 = initial_value_NN()
        self.Vnn2 = initial_value_NN()
        self.rhon = 1
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
        loss_output = [loss]
        for i in range(self.steps):
            input_dW = keras.Input(shape=(1))
            inputs = inputs + [input_dW]
            u_now = self.u_optimal(X_now)              # Here we use the optimal control
            X_next  = X_now + input_dW + self.dynamic(u_now,X_now) * self.dt
            loss_tmp = (tf.math.square(X_now) + tf.math.square(u_now))*self.dt
            loss_output = loss_output + [loss_tmp]
            loss = loss + loss_tmp
            X_now = X_next
        outputs1 = (loss + self.value_optimal(X_now) - self.value_optimal(X_start))/self.T
        outputs2 = (loss + self.value_optimal(X_now) - self.value_optimal(X_start) - self.T*self.rho_optimal)**2
        control_optimal1 = keras.Model(inputs=inputs, outputs = outputs1)
        control_optimal2 = keras.Model(inputs=inputs, outputs = outputs2)
        self.optnn1 = control_optimal1
        self.optnn2 = control_optimal2
        
        # Build the main network
        input_x = keras.Input(shape=(1))
        inputs = [input_x]
        X_start = input_x
        X_now = X_start
        loss = tf.zeros_like(X_now)
        loss_output = [loss]
        for i in range(self.steps):
            input_dW = keras.Input(shape=(1))
            inputs = inputs + [input_dW]
            u_now = self.unn(X_now)
            X_next  = X_now + input_dW + self.dynamic(u_now,X_now) * self.dt
            loss_tmp = (tf.math.square(X_now) + tf.math.square(u_now))*self.dt
            loss_output = loss_output + [loss_tmp]
            loss = loss + loss_tmp
            X_now = X_next
        outputs1 = (loss + self.Vnn1(X_now) - self.Vnn1(X_start))/self.T
        outputs2 = (loss + self.Vnn1(X_now) - self.Vnn2(X_start) - self.T*self.rhon)**2
        control_main1 = keras.Model(inputs=inputs, outputs = outputs1)
        control_main2 = keras.Model(inputs=inputs, outputs = outputs2)
        control_terminal = keras.Model(inputs=inputs, outputs = X_now)
        control_loss = keras.Model(inputs=inputs, outputs = loss_output)
        self.mainnn1 = control_main1
        self.mainnn2 = control_main2
        self.endnn = control_terminal
        self.lossnn = control_loss
    def plot_compare(self):
        NN_plot(self.Vnn1)
        xgrid = np.linspace(-10,10,201)
        plt.plot(xgrid, self.value_optimal(xgrid))
        plt.show()
        NN_plot(self.Vnn2)
        plt.plot(xgrid, self.value_optimal(xgrid))
        plt.show()
        NN_plot(self.unn)
        plt.plot(xgrid, self.u_optimal(xgrid))
        plt.show()
        print('Optimal ergodic cost is: ',self.rhon)
    def end(self):
        self.endnn.compile(loss = mean_loss)
        self.Xend = self.endnn.predict(self.x_train)
    def optimal(self):
        self.optnn1.compile(loss = mean_loss , optimizer = 'Adam')
        tmp = self.optnn1.evaluate(self.x_train,self.y_train, verbose = 0)
        print('Loss1 under optimal control: ', tmp)
        self.optnn2.compile(loss = mean_loss , optimizer = 'Adam')
        tmp = self.optnn2.evaluate(self.x_train,self.y_train, verbose = 0)
        print('Loss2 under optimal control: ', tmp)
    def endtostart(self):
        self.end()
        self.restart_times += 1
        self.X0 = np.random.normal(self.Xend.mean(),self.Xend.std(),size = (self.M,1))
        self.x_train, self.y_train = trainning_data(10*self.restart_times+1,self.X0,self.M,self.steps,self.dt)
        self.x_valid, self.y_valid = trainning_data(10*self.restart_times+2,self.X0,self.M,self.steps,self.dt)
    def train(self, epo = 20, opt = 'Adam', verb = 1):
        for l in self.Vnn1.layers:
            l.trainable = False
        self.mainnn1.compile(loss = mean_loss , optimizer = 'Adam')
        self.mainnn1.fit(self.x_train,self.y_train,epochs =20,\
                                validation_data = (self.x_valid,self.y_valid), verbose = 1)
        
        self.rhon  = self.mainnn1.evaluate(self.x_valid, self.y_valid)
        
        self.mainnn2.compile(loss = mean_loss , optimizer = 'Adam')
        self.mainnn2.fit(self.x_train,self.y_train,epochs =20,\
                        validation_data = (self.x_valid,self.y_valid), verbose = 1)
        self.Vnn1.set_weights(self.Vnn2.get_weights())
        
        
    def autotrain(self, epo = 20, opt = 'Adam', verb = 1):
        self.optimal()
        self.train(epo = epo, opt = opt, verb = verb)
        self.plot_compare()
        self.end()
        print('Mean and Var of terminal distribution: ',self.Xend.mean(), self.Xend.var())

