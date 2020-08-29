import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from IPython import display
# %matplotlib
from mpl_toolkits import mplot3d 

def tamed_gradient(gg):   # tame gradients such that they are bounded in L_inf
    findbreak = False
    for i,g in enumerate(gg):
        imax = tf.norm(g,np.inf)
        if imax > 1e5:
            findbrak = True
            gg[i] = g * (1e5/imax)
    return gg, findbreak

def get_data(seed, dt, N, M, Yvar, UminusYvar): # generate data 
    np.random.seed(seed)
    y = np.random.normal(0, np.sqrt(Yvar), size=(M, 1))
    u = y + np.random.normal(0, np.sqrt(UminusYvar), size=(M, 1))
    U_train = tf.Variable(u)
    Y_train = tf.Variable(y)
    dW_train = [tf.Variable(np.random.normal(
        0, np.sqrt(dt), size=(M, 1))) for i in range(N)]
    return U_train, Y_train, dW_train

class scheme2:
    def __init__(self, N, dt, M, gamma):
        self.N = N
        self.dt = dt
        self.sqrtdt = np.sqrt(self.dt)
        self.T = self.N * self.dt
        self.M = M
        self.gamma = gamma
        self.Y_var_start = 0
        self.UminusY_var_start = 0
        self.a = 1
        self.b = (2+gamma)/ (2*(1+gamma)**2)
        self.c = -2/(1+gamma)
        self.rho_optimal = self.b
    def V_optimal(self, u, y):
        return self.a*u**2 + self.b*y**2 + self.c*u*y
    def dVdu_optimal(self, u, y):
        return 2*self.a*u + self.c*y
    def dVdy_optimal(self, u, y):
        return 2*self.b*y + self.c*u
    def u_optimal(self,u, y):
        return -u + y/(1+self.gamma)
    def dV_optimal(self, u, y):
        return 2*self.b*y + self.c*u
    def optimal(self, x):
        u = x[:, :1]
        y = x[:, 1:]
        return self.V_optimal(u, y)
    def neural_network(self):
        # Optimal Neural Network
        inputs = keras.Input(shape=(2,), dtype='float64')
        outputs = self.optimal(inputs)
        self.model_optimal = keras.Model(inputs=inputs, outputs=outputs)
        self.rho = tf.Variable([[self.rho_optimal]], dtype='float64', trainable=False)

        # Training Neural Network
        inputs = keras.Input(shape=(2,), dtype='float64')
        inputs0 = tf.zeros_like(inputs)
        inputs_inverse = - inputs
        layer1 = layers.Dense(8, activation='elu', dtype='float64')
        layer2 = layers.Dense(8, activation='elu', dtype='float64')
        layer3 = layers.Dense(1, activation='elu', dtype='float64')
        layer4 = layers.Dense(1, activation='elu', use_bias=False, dtype='float64')
        outputs = layer3(layer2(tf.square(layer1(inputs)))) + \
            layer3(layer2(tf.square(layer1(inputs_inverse))))
        outputs0 = 2*layer3(layer2(tf.square(layer1(inputs0))))
        outputs = outputs - outputs0
        # outputs = tf.square(inputs[:,:1] - inputs[:,1:])
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.rho = tf.Variable([[0.]], dtype = 'float64')
    
    def compile(self):
        self.optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        self.train(0)

    @tf.function
    def train_step(self, U0, Y0, dW):
        U = [0]*(self.N+1)
        Y = [0]*(self.N+1)
        X = [0]*(self.N+1)
        V = [0]*(self.N+1)
        dVdU = [0]*self.N
        dVdY = [0]*self.N
        control = [0]*self.N
        with tf.GradientTape(persistent=True) as tape0:
            U[0] = U0
            Y[0] = Y0
            bsde = 0
            loss = 0
            for steps in range(self.N):
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(U[steps])
                    tape.watch(Y[steps])
                    X[steps] = tf.concat([U[steps],Y[steps]],axis = -1)
                    V[steps] = self.model(X[steps])
                dVdU[steps] = tape.gradient(V[steps], U[steps])
                dVdY[steps] = tape.gradient(V[steps], Y[steps])
                control[steps] = - 0.5*dVdU[steps]
                U[steps+1] = U[steps] + control[steps] * self.dt
                Y[steps+1] = Y[steps] + dW[steps] - self.gamma*Y[steps]*self.dt ##############################################################
                loss = loss + (tf.square(U[steps] - Y[steps]) + tf.square(control[steps]))*self.dt
                bsde = bsde + tf.multiply(dVdY[steps],dW[steps])
            X[self.N] = tf.concat([U[self.N], Y[self.N]], axis=-1)
            V[self.N] = self.model(X[self.N])
            outputs = loss - self.rho*self.T + V[self.N] - V[0] - bsde
            Eloss = tf.reduce_mean(tf.square(outputs))
        grad = tape0.gradient(Eloss, self.model.trainable_variables + [self.rho]) 
    #     grad_V = tape0.gradient(Eloss, model.trainable_variables) 
    #     grad_rho = tape0.gradient(Eloss, [rho])  
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_weights + [self.rho]))
    #     optimizer_V.apply_gradients(zip(grad_V, model.trainable_weights))
    #     optimizer_rho.apply_gradients(zip(grad_rho, [rho])) 
    #     train_acc_metric.update_state(y, logits)
        return Eloss, loss, grad, U[-1], Y[-1]   

    def train(self,epochs):
        self.LOSS = []
        self.LOSS_AVERAGE = []
        print('dt:', self.dt, 'N:', self.N, 'M:', self.M, 'T:', self.T)
        print('Y_var_start:', self.Y_var_start,
              'UminusY_var_start:', self.UminusY_var_start)
        print('epochs:', epochs)

        for epoch in range(epochs):
            if epoch < 1:
                U0, Y0, dW = get_data(
                    epoch, self.dt, self.N, self.M, self.Y_var_start, self.Y_var_start)
                print('Compiling' + '.'*10)
            else:
                U0, Y0, dW = get_data(
                    epoch, self.dt, self.N, self.M, Y_end.numpy().var(), (U_end-Y_end).numpy().var())
                U0 = U_end
                Y0 = Y_end
            Eloss, loss, gg, U_end, Y_end = self.train_step(U0, Y0, dW)
            ggg, findbreak = tamed_gradient(gg)
            if findbreak:
                print('break')
            self.LOSS.append(loss.numpy().mean())
            self.LOSS_AVERAGE.append(np.array(self.LOSS[-100:]).mean())
        #     print('Rho: ', rho.numpy()[0,0])
        #     print('Int_loss: ',loss.numpy().mean())
        #         print(grad)
        #         print(model.trainable_weights + [rho])
            if epoch > 0 and (epoch % 50 == 0 or epoch == epochs-1):
                #         display.clear_output(wait = True)
                print('='*100)
                print('Epoch: ', epoch, 'Evaluating Loss: ', Eloss.numpy())
                self.plot_helper2()
                print('Avarage running loss of latest 100 epochs: ', self.LOSS_AVERAGE[-1])
                print('rho:', self.rho.numpy()[0, 0])
                print('X variance:', (U_end-Y_end).numpy().var())
                print('Y variance:', (Y_end).numpy().var())


    def plot_helper2(self):
        u = y = np.arange(-1.0, 1.0, 0.05)
        U, Y = np.meshgrid(u, y)
        Z = np.concatenate([np.reshape(U, [-1,1]),np.reshape(Y,[-1,1])],axis = 1)
        U_tf = tf.constant(Z[:, :1])
        Y_tf = tf.constant(Z[:, 1:])
        with tf.GradientTape(persistent=True) as g:
            g.watch(U_tf)
            g.watch(Y_tf)
            Z_tf = tf.concat([U_tf, Y_tf], axis=-1)
            V_tf = self.model(Z_tf)

        dVdU_tf = g.gradient(V_tf, U_tf)
        dVdY_tf = g.gradient(V_tf, Y_tf)

        V_opt_tf = self.V_optimal(U_tf, Y_tf)
        dVdU_opt_tf = self.dVdu_optimal(U_tf, Y_tf)
        dVdY_opt_tf = self.dVdy_optimal(U_tf, Y_tf)

        V = np.reshape(V_tf, U.shape)
        dVdU = np.reshape(dVdU_tf, U.shape)
        dVdY = np.reshape(dVdY_tf, U.shape)
        V_opt = np.reshape(V_opt_tf, U.shape)
        dVdU_opt = np.reshape(dVdU_opt_tf, U.shape)
        dVdY_opt = np.reshape(dVdY_opt_tf, U.shape)

        # plt.figure()
        # fig1 = plt.subplot(131)

        fig1 = plt.figure(figsize=(30, 8))
        ax = fig1.add_subplot(131, projection='3d')
        ax.plot_surface(U, Y, V, label='Neural Network')
        ax.plot_surface(U, Y, V_opt, label='V')
        ax.set_xlabel('U-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        ax = fig1.add_subplot(132, projection='3d')
        ax.plot_surface(U, Y, dVdU)
        ax.plot_surface(U, Y, dVdU_opt)
        ax.set_xlabel('U-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        ax = fig1.add_subplot(133, projection='3d')
        ax.plot_surface(U, Y, dVdY)
        ax.plot_surface(U, Y, dVdY_opt)
        ax.set_xlabel('U-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        fig2 = plt.figure(figsize=(12, 4))
        ax1 = plt.subplot(121)
        ax1.plot(np.arange(len(self.LOSS[0:])), self.LOSS[0:], lw=1)
        ax1.plot([0, len(self.LOSS[0:])], [self.rho_optimal, self.rho_optimal], lw=1)
        ax1.grid()

        ax2 = plt.subplot(122)
        ax2.plot(np.arange(len(self.LOSS_AVERAGE[0:])), self.LOSS_AVERAGE[0:], lw=1)
        ax2.plot([0, len(self.LOSS_AVERAGE[0:])], [self.rho_optimal, self.rho_optimal], lw=1)
        ax2.grid()

        plt.show(fig1)
        plt.show(fig2)

        del fig1
        del fig2
