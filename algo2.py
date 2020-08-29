import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
from IPython import display
# %matplotlib
from mpl_toolkits import mplot3d 

tf.random.set_seed(1)
np.random.seed(1)


dt = 0.01
N = 100
M = 100
T = dt*N

# Case Brownian Motion
d = 1
gamma = np.array([0])
w = np.array([1])

# Case OU process
# d = 1
# gamma = np.array([1])
# w = np.array([1])

# Case Multi-OU process
# d = 10
# gamma = np.arange(d)+1
# w = np.ones(d)/d


b = -w/(1+gamma)
c = np.zeros(shape = (d,d))
rho_optimal = 0
for i in range(d):
    for j in range(d):
        if i == j:
            c[i,j] = w[i]*w[j]*(gamma[i]+2)/2/(gamma[i]+1)**2
        else:
            c[i,j] = w[i]*w[j]*(gamma[i]+gamma[j]+gamma[i]*gamma[j])\
            /((gamma[i]+1)*(gamma[j]+1)*(gamma[i]+gamma[j]))
        rho_optimal += c[i,j]
b = tf.constant(b,shape = [d,1])
c = tf.constant(c,shape = [d,d])

mean = np.zeros(d+1)
cov = np.zeros((d+1,d+1))
cov_xx = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        cov_xx[i,j] = (gamma[i]*gamma[j]+gamma[i]+gamma[j]+1)/\
        (2*(gamma[i]+gamma[j])*(gamma[i]+1)**2*(gamma[j]+1)**2)
for i in range(d):
    for j in range(d):
        cov[i+1,j+1] = 1/(gamma[i]+gamma[j])
cov[0,0] = np.matmul(np.reshape(w,[1,-1]),np.matmul(cov_xx,np.reshape(w,[-1,1])))[0,0]        
for i in range(d):
    tmp = 0
    for j in range(d):
        tmp += w[i]/((gamma[i]+gamma[j])*(gamma[i]+1)*(gamma[j]+1))
    cov[i+1,0] = tmp
    cov[0,i+1] = tmp
    
mean0 = np.zeros(d+1)
cov0 = np.zeros((d+1,d+1))

def V_optimal(x):
    u = x[:,:1]
    y = x[:,1:]
    opt1 = tf.square(u)
    opt2 = 2*tf.linalg.matmul(tf.multiply(u,y),b)
    opt3 = tf.reduce_sum(y*tf.linalg.matmul(y,c),axis = -1, keepdims = True)
    return opt1 + opt2 + opt3
def dVdu_optimal(x):
    u = x[:,:1]
    y = x[:,1:]
    opt = 2*u + 2*tf.linalg.matmul(y,b)
    return opt
def dVdy_optimal(x):
    u = x[:,:1]
    y = x[:,1:]
    opt1 = 2*tf.linalg.matmul(u,tf.reduce_sum(b, keepdims = True))
    opt2 = 2 * tf.reduce_sum(tf.linalg.matmul(y,c),axis = -1, keepdims = True)
    return opt1 + opt2
def u_optimal(x):
    u = x[:,:1]
    y = x[:,1:]
    opt = -u - tf.linalg.matmul(y,b)
    return opt
rho_optimal