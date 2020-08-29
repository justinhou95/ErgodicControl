import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import nnset
# tf.random.set_seed(0)

def initial_NN():
    inputs0 = keras.Input(shape=(2))
    l1 = layers.Dense(1, activation = 'linear',use_bias = False)
    outputs0 = l1(inputs0)
    outputs0 = outputs0**2
    NN = keras.Model(inputs=inputs0, outputs=outputs0)
    
    
    inputs = keras.Input(shape=(2))
    with tf.GradientTape() as t:
        t.watch(inputs)
        outputs1 = NN(inputs)
    outputs2 = t.gradient(outputs1, inputs)
    u_nn = keras.Model(inputs=inputs, outputs= -0.5*outputs2[:,:1])
    V_nn = keras.Model(inputs=inputs, outputs=outputs1)
    dV_nn = keras.Model(inputs=inputs, outputs= outputs2[:,1:])
    return u_nn, V_nn, dV_nn


def initial_V_NN():
    # Value Neural Network: V(x)
    inputs = keras.Input(shape=(2))
    l1 = layers.Dense(1, activation = 'linear',use_bias = False)
    outputs = l1(inputs)
    outputs = outputs0**2
    value_NN = keras.Model(inputs=inputs, outputs=outputs)
    return value_NN

def initial_u_NN():
    inputs = keras.Input(shape=(2))
#     inputs1 = inputs[:,1:] - inputs[:,:1]
    l = layers.Dense(1, activation = 'linear', use_bias = False)
    outputs = l(inputs)
    control_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'control_NN')
    return control_NN


def initial_dV_NN():
    inputs = keras.Input(shape=(2))
#     inputs1 = inputs[:,1:] - inputs[:,:1]
    l = layers.Dense(1, activation = 'linear', use_bias = False)
    outputs = l(inputs)
    dvalue_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'dvalue_NN')
    return dvalue_NN

def initial_rho_NN():
#     tf.random.set_seed(3)
    # optimal value: rho
    inputs = keras.Input(shape=(1))
    l= layers.Dense(1, activation = 'linear', use_bias = False)
    outputs = l(inputs)
    rho_NN = keras.Model(inputs=inputs, outputs=outputs, name = 'rho_NN')
    # rho_NN.summary()
#     print('rho_NN: ',rho_NN.predict(np.zeros(1))[0,0])
    return rho_NN



def square_loss(target_y, predicted_y):
    return tf.reduce_mean(tf.math.square(predicted_y - target_y))
custom_optimizer = keras.optimizers.Adam(learning_rate = 0.01)
    
    
def training_data(seed,U0,Y0,samplesM,stepsN,dtt):
    np.random.seed(seed)
    x = [U0, Y0]
    for i in range(stepsN):
        x = x + [np.random.normal(0,1,size = (samplesM,1))]
    y = np.zeros(shape = (samplesM,1))
    return x, y
    
class MODEL:
    restart_times = 0
    def __init__(self,steps,dt,T,M,gamma):
        self.steps = steps
        self.dt = dt
        self.sqrtdt = np.sqrt(self.dt)
        self.T = self.steps * self.dt
        self.M = M
        self.gamma = gamma
        
        self.u_nn = initial_u_NN()
        self.V_nn = initial_V_NN()
        self.dV_nn = initial_dV_NN()
        self.rho_nn = initial_rho_NN()
#         self.u_nn, self.V_nn, self.dV_nn= initial_NN()

        self.unn = nnset.bm_u_NN()
        self.Vnn = nnset.bm_V_NN()
        self.dVnn = nnset.bm_dV_NN()
        self.rhonn = nnset.bm_rho_NN()
#         self.unn = self.diff_NN(self.Vnn)

        self.a = 1
        self.b = (2+gamma)/ (2*(1+gamma)**2)
        self.c = -2/(1+gamma)
        self.rho_optimal = self.b
        def V_opt(Z):
            Z1 = Z[:,:1]
            Z2 = Z[:,1:]
            return self.a*Z1**2 + self.b*Z2**2 + self.c*Z1*Z2  
        self.V_optimal = V_opt
        def u_opt(Z):
            Z1 = Z[:,:1]
            Z2 = Z[:,1:]
            return -Z1 + Z2/(1+self.gamma) 
        self.u_optimal = u_opt
        def dV_opt(Z):
            Z1 = Z[:,:1]
            Z2 = Z[:,1:]
            return 2*self.b*Z2 + self.c*Z1 
        self.dV_optimal = dV_opt
        def loss(Z):
            Z1 = Z[:,:1]
            Z2 = Z[:,1:]
            return (Z1-Z2)**2 + self.u_optimal(Z)**2
        self.loss = loss
    def start(self,U0,Y0):
        self.U0 = U0
        self.Y0 = Y0
    def traindata(self,seed):
        self.x_train, self.y_train = training_data(
            seed+1, self.U0, self.Y0, self.M, self.steps, self.dt)
        self.x_val, self.y_val = training_data(
            seed+2, self.U0, self.Y0, self.M, self.steps, self.dt)
        self.x_test, self.y_test = training_data(
            seed+3, self.U0, self.Y0, self.M, self.steps, self.dt)
    def build_base(self,case):
        # Build the optimal network
        U_start = keras.Input(shape=(1))
        Y_start = keras.Input(shape=(1))
        inputs = [U_start, Y_start]
        Z_start = tf.concat([U_start, Y_start], axis = 1)
        X_start = Y_start - U_start
        X_now, U_now, Y_now = X_start, U_start, Y_start
        loss = tf.zeros_like(U_start)
        bsde = tf.zeros_like(U_start)
        if case == 'bm':
            u = self.unn
            V = self.Vnn
            dV = self.dVnn
            rho = self.rhonn(tf.ones_like(U_start))  
        if case == 'optimal':
            u = self.u_optimal
            V = self.V_optimal
            dV = self.dV_optimal
            rho = self.rho_optimal
        if case == 'main':
            u = self.u_nn
            V = self.V_nn
            dV = self.dV_nn 
            rho = self.rho_nn(tf.ones_like(U_start)) 
            
        for i in range(self.steps):
            input_dW = keras.Input(shape=(1))
            inputs = inputs + [input_dW]
            Z_now = tf.concat([U_now, Y_now], axis = 1)
            X_now = Y_now - U_now
            if case == 'bm':
                u_now = u(X_now)
            else:
                u_now = u(Z_now) 
            U_next = U_now + u_now * self.dt
            Y_next  = Y_now - self.gamma*Y_now*self.dt + self.sqrtdt * input_dW         #self.dynamic(Y_now,input_dW)
            loss_tmp = (tf.math.square(Y_now - U_now) + tf.math.square(u_now))*self.dt
            loss = loss + loss_tmp
            if case == 'bm':
                bsde_tmp = tf.multiply(dV(X_now),self.sqrtdt * input_dW)
            else:
                bsde_tmp = tf.multiply(dV(Z_now),self.sqrtdt * input_dW)
    #             bsde_tmp = 2*tf.multiply(u(Z_now),self.sqrtdt * input_dW)
            bsde = bsde + bsde_tmp
            U_now = U_next
            Y_now = Y_next
        Z_end = tf.concat([U_now, Y_now], axis = 1)    
        X_end = Y_now - U_now
        if case == 'bm':
            outputs = loss - rho * self.T + V(X_end) - V(X_start) - bsde
        else:
            outputs = loss - rho * self.T + V(Z_end) - V(Z_start) - bsde
            
        if case == 'optimal':
            self.opt_nn = keras.Model(inputs=inputs, outputs = outputs, name = 'control_optimal')
            self.opt_end_nn = keras.Model(inputs=inputs, outputs = Z_end, name = 'help')
            self.opt_nn.compile(loss = 'mse' , optimizer = 'Adam')
            self.opt_end_nn.compile(loss = 'mse' , optimizer = 'Adam')
        if case == 'main':
            self.main_nn = keras.Model(inputs=inputs, outputs = outputs, name = 'control_main')
            self.end_nn = keras.Model(inputs=inputs, outputs = Z_end, name = 'control_terminal')
            self.main_nn.compile(loss = 'mse', optimizer = 'Adam')
            self.end_nn.compile(loss = 'mse', optimizer = 'Adam')
        if case == 'bm':
            self.bm_nn = keras.Model(inputs=inputs, outputs=outputs, name='bmbm')
            self.bm_nn.compile(loss='mse', optimizer='Adam')


    def build(self):
        self.build_base('bm')
        self.build_base('optimal')
        self.build_base('main')
        
        
    def build_test(self):
        pass
    
    
    
    
#     def diff_NN(self,NN):
#         inputs = keras.Input(shape=(2))
#         with tf.GradientTape() as t:
#             t.watch(inputs)
#             outputs1 = NN(inputs)
#         outputs = t.gradient(outputs1, inputs)
#         return keras.Model(inputs=inputs, outputs=outputs[:,0])
        
        
        
        
        
        
        
        
#     def build0(self):
#         # Build the optimal network
#         U_start = keras.Input(shape=(1))
#         Y_start = keras.Input(shape=(1))
#         inputs = [U_start, Y_start]
#         Z_start = tf.concat([U_start, Y_start], axis = 1)
#         U_now = U_start
#         Y_now = Y_start
#         loss = tf.zeros_like(U_now)
#         bsde = tf.zeros_like(U_now)
#         for i in range(self.steps):
#             input_dW = keras.Input(shape=(1))
#             inputs = inputs + [input_dW]
#             Z_now = tf.concat([U_now, Y_now], axis = 1)
#             u_now = self.u_optimal(Z_now) 
#             U_next = U_now + u_now * self.dt
#             Y_next  = Y_now - self.gamma*Y_now*self.dt + self.sqrtdt * input_dW         #self.dynamic(Y_now,input_dW)
#             loss_tmp = (tf.math.square(Y_now - U_now) + tf.math.square(u_now))*self.dt
#             loss = loss + loss_tmp
#             bsde_tmp = tf.multiply(self.dV_optimal(Z_now),self.sqrtdt * input_dW)
#             bsde = bsde + bsde_tmp
#             U_now = U_next
#             Y_now = Y_next
         
#         Z_end = tf.concat([U_now, Y_now], axis = 1)    
#         outputs = loss - self.rho_optimal * self.T + self.V_optimal(Z_end) - self.V_optimal(Z_start) - bsde
#         control_optimal = keras.Model(inputs=inputs, outputs = outputs, name = 'control_optimal')
        
#         helper = keras.Model(inputs=inputs, outputs = Z_end, name = 'help')
        
#         self.opt_nn = control_optimal
#         self.opt_end_nn = helper
#         self.opt_nn.compile(loss = 'mse' , optimizer = 'Adam')
#         self.opt_end_nn.compile(loss = 'mse' , optimizer = 'Adam')
        
        
#         #################################################
#         # Build the main network
#         U_start = keras.Input(shape=(1))
#         Y_start = keras.Input(shape=(1))
#         inputs = [U_start, Y_start]
#         Z_start = tf.concat([U_start - Y_start, Y_start], axis = 1)
#         U_now = U_start
#         Y_now = Y_start
#         loss = tf.zeros_like(U_now)
#         bsde = tf.zeros_like(U_now)
#         loss_output = [loss]
#         rho = self.rho_nn(tf.zeros_like(U_now))
#         for i in range(self.steps):
#             input_dW = keras.Input(shape=(1))
#             inputs = inputs + [input_dW]
#             Z_now = tf.concat([U_now - Y_now, Y_now], axis = 1)
#             u_now = self.u_nn(Z_now)   
#             U_next = U_now + u_now * self.dt
#             Y_next  = Y_now - self.gamma*Y_now*self.dt + self.sqrtdt * input_dW         #self.dynamic(Y_now,input_dW)
#             loss_tmp = (tf.math.square(Y_now - U_now) + tf.math.square(u_now))*self.dt
#             loss = loss + loss_tmp
#             bsde_tmp = tf.multiply(2*u_now,self.sqrtdt * input_dW)
#             bsde_tmp = tf.multiply(self.dV_nn(Z_now),self.sqrtdt * input_dW)
#             bsde = bsde + bsde_tmp
#             U_now = U_next
#             Y_now = Y_next
#         Z_end = tf.concat([U_now - Y_now, Y_now], axis = 1)    
#         outputs = loss - rho * self.T + self.V_nn(Z_end) - self.V_nn(Z_start) - bsde
        
#         control_main = keras.Model(inputs=inputs, outputs = outputs, name = 'control_main')
#         control_terminal = keras.Model(inputs=inputs, outputs = Z_end, name = 'control_terminal')
#         control_loss = keras.Model(inputs=inputs, outputs = loss, name = 'control_loss')
#         self.main_nn = control_main
#         self.end_nn = control_terminal
#         self.loss_nn = control_loss
#         self.main_nn.compile(loss = 'mse', optimizer = 'Adam')
#         self.end_nn.compile(loss = 'mse', optimizer = 'Adam')
#         self.loss_nn.compile(loss = 'mse', optimizer = 'Adam')
 
#         ##############################################################################
#         # Build the main network
#         U_start = keras.Input(shape=(1))
#         Y_start = keras.Input(shape=(1))
#         inputs = [U_start, Y_start]
#         X_start = Y_start - U_start
#         X_now = X_start
#         loss = tf.zeros_like(X_now)
#         bsde = tf.zeros_like(X_now)
#         loss_output = [loss]
#         rho = self.rhonn(tf.zeros_like(X_now))
#         for i in range(self.steps):
#             input_dW = keras.Input(shape=(1))
#             inputs = inputs + [input_dW]
#             u_now = self.unn(X_now)
#             X_next  = X_now + self.sqrtdt * input_dW + u_now * self.dt
#             loss_tmp = (tf.math.square(X_now) + tf.math.square(u_now))*self.dt
#             loss_output = loss_output + [loss_tmp]
#             loss = loss + loss_tmp
#             bsde_tmp = -2*tf.multiply(u_now,self.sqrtdt * input_dW)
#             bsde = bsde + bsde_tmp
#             X_now = X_next
#         outputs = loss - self.T * rho + self.Vnn(X_now) - self.Vnn(X_start) - bsde
#         control_main = keras.Model(inputs=inputs, outputs = outputs, name = 'co')
        
#         bm = keras.Model(
#             inputs=inputs, outputs=outputs, name='control_main')
#         self.bm_nn = bm
#         self.bm_nn.compile(loss='mse', optimizer='Adam')
        
        
        


      


