# Step Functions
def step_function(x):
    y = x>0
    return y.astype(np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>=0)*x + (x<0)*0.01*x

def tanh_func(x):
    return np.tanh(x)



import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5.0, 5.0, 0.1)
step = step_function(x)
sigmoid = sigmoid(x)
ReLU = relu(x)
Leak_ReLU = leakyrelu_func(x)
Hyper_tangent = tanh_func(x)
plt.plot(x,step)
plt.plot(x,sigmoid)
plt.plot(x,ReLU)
#plt.plot(x,Leak_ReLU)
#plt.plot(x,Hyper_tangent)
plt.ylim(-.1,3.1)
plt.show()
