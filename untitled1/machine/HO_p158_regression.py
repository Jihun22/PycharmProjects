import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------
#  Sample Dataset: Linear Regression
#  p158
#---------------------------------------------------------------------------
num_sample = 100

X = 2 * np.random.rand(num_sample, 1)        #[0,1] uniform dist. sample 100X1 array로 ...
y = 4 + 3*X +np.random.randn(num_sample, 1)  #평균=0, 분산=1인 normal dist. smaple
print('\nX.shape: {} \ny.shape: {}\n\n'.format(X.shape, y.shape))

plt.scatter(X, y)
plt.show()

os.system('Pause')

#---------------------------------------------------------------------------
#  Sample Dataset: Linear Regression
#  p158
#---------------------------------------------------------------------------
X_b = np.c_[np.ones((num_sample, 1)), X]  # 모든 샘플에 X0=1을 추가해서 X행렬믈 만듦
print('X.shape = {}, X_b.shape = {}\n\n'.format(X.shape, X_b.shape))

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta_estimators by Normal Equation =\n{}\n\n'.format(theta_best))

# x=0와 x=2에서 y값 예측
X_new = np.array([[0], [2]])   # array 표현에 주의
X_new_b0 = np.c_[np.ones((2, 1)), X_new]

y_predict = X_new_b0.dot(theta_best)
print('predicted y of X_new =\n{}\n\n'.format(y_predict))

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

os.system('Pause')
#---------------------------------------------------------
#  scikit learn code for regression
#  p160
#---------------------------------------------------------
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
intercept = lin_reg.intercept_
theta_estimators = lin_reg.coef_
print('sklearn Linear Regression-----------\nintercept =\n{}\ntheta_estimators =\n{}\n\n'.
       format(intercept, theta_estimators))

y_predict_LinearR = lin_reg.predict(X_new)
print('predicted y of X_new by Linear Regression Model=\n{}\n\n'.format(y_predict_LinearR))

#---------------------------------------------------------
#  Gradient Decent
#  page 166
#---------------------------------------------------------

eta = 0.1  # Learning rate
n_iterations = 1000
m = 100    # Number of parameters
delta = 0.01

theta = np.random.randn(2, 1)
#print('Initial valuese of thetas=\n{}\n\n'.format(theta))

for iteration in range(n_iterations):
    gradients = 2/m*X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta*gradients
    #print('valuese of thetas=\n{}\n\n'.format(theta))

print('Estimated Theta by Gradient Decent=\n{}\n\n'.format(theta))

#---------------------------------------------------------
#  서로 다른 Learning Rate에서의 Gradient Descent
# page 167
#---------------------------------------------------------
theta_path_bgd = []

def plot_gradient_decent(theta, eta,theta_path=None):
    m = len(X_b)
    plt.plot(X, y, 'b.')
    n_iterations = 1000

    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b0.dot(theta)
            style = 'b-' if iteration > 0 else 'r--'  # 경우에 따라 그리는 그래프가 다를때
            plt.plot(X_new, y_predict, style)
        gradients = 2/m*X_b.T.dot(X_b.dot(theta)-y)
        theta = theta -eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('$X_1$', fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r'$\eta = {}$'.format(eta), fontsize=16)

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

plt.figure(figsize=(10, 4))
plt.subplot(131); plot_gradient_decent(theta, eta=0.02)
plt.ylabel('$y$', rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_decent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_decent(theta, eta=0.5)
#save_fig('bgd_plot')
plt.show()

#---------------------------------------------------------
#  Stochastic Gradient Decent
# page 168
#---------------------------------------------------------
theta_path_sgd = []
n_epochs = 50
t0, t1 = 5, 50 # 학습 스케쥴 파라미터

def learning_schedule(t):
    return t0/(t+t1)

m = len(X_b)
theta = np.random.randn(2, 1) #무작위 초기화

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i <20:                   #
            y_predict = X_new_b0.dot(theta)        #
            style='b-' if i > 0 else 'r--'         #
            plt.plot(X_new, y_predict, style)      #
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index+1]
        yi = y[random_index: random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta -eta*gradients
        theta_path_sgd.append(theta)               #

plt.plot(X, y, 'b.')
plt.xlabel('$x_1$',fontsize=18)
plt.ylabel('$y$',rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
#save_fig('sgd_plot')
plt.show()

print('Estimated Theta by Stochastic Gradient Decent=\n{}\n\n'.format(theta))

#---------------------------------------------------------
#  SGDRegressor
# page 171
#---------------------------------------------------------
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

theta0_SGDRegressor, theta_SGDRegressor = sgd_reg.intercept_, sgd_reg.coef_
print('intercept by SGDRegressor=\n{}\n\ncoefficients by SGDRegressor=\n{}\n\n'.
                             format(sgd_reg.intercept_, sgd_reg.coef_))
