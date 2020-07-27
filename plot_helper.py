%matplotlib notebook
from importlib import reload  
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(1)
import OUmodel
reload(OUmodel)
from mpl_toolkits import mplot3d 




def NN_plot(nn):
    nn.compile(loss = 'mse')
    xgrid = np.linspace(-10,10,201)
    y = nn.predict(xgrid)
    plt.plot(xgrid,y)
    plt.grid()
def plot_compare():
    NN_plot(model.Vnn)
    xgrid = np.linspace(-10,10,201)
    plt.plot(xgrid, xgrid**2)
    plt.show()
    NN_plot(model.unn)
    plt.plot(xgrid, xgrid)
    plt.show()
    NN_plot(model.dVnn)
    plt.plot(xgrid, 2*xgrid)
    plt.show()
    tmp = model.rhonn.predict(np.ones(1))[0,0]
    print('Optimal ergodic cost is: ',tmp)
    
def plotNN2d(NN):
    fig = plt.figure() 
    ax = plt.axes(projection ='3d') 
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    # z = (np.sin(X **2) + np.cos(Y **2) ) 
    XY = np.concatenate([np.reshape(X,[X.shape[0]*X.shape[1],1]),np.reshape(Y,[Y.shape[0]*Y.shape[1],1])],axis = 1)
    Z = np.reshape(NN.predict(XY),X.shape)
    ax.plot_surface(X, Y, Z) 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()