#---------------------------------------------------------------------------
# Linear Regression
#---------------------------------------------------------------------------
import numpy as np
import numpy.random as rnd

#---------------------------------------------------------------------------
#  Normal Equation
# page 158
#---------------------------------------------------------------------------

X = 2*rnd.rand(100, 1)
y = 4 + 3*X + rnd.randn(100, 1)

X_b = np.c_[np.ones((100,1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta =\n{}\n\n'.format(theta_best))

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
print('-----  Normal Equation  -----')
print('y_predict =\n{}\n\n'.format(y_predict))

import matplotlib
import matplotlib.pyplot as plt

plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.show()
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()

#---------------------------------------------------------------------------
#  Parameter extimation and prediction
# page 160
#---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
intercept = lin_reg.intercept_
slope = lin_reg.coef_
print('-----  sklern Linear Regression: Estimates Parameters  -----')
print('intercept =\n{}\n\n'.format(intercept))
print('slope =\n{}\n\n'.format(slope))

y_X_new = lin_reg.predict(X_new)
print('-----  sklern Linear Regression  -----')
print('y_predict of X_new =\n{}\n\n'.format(y_X_new))

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

def plot_gradient_descent(theta, eta, theta_path=None):
 m = len(X_b)
 plt.plot(X, y, "b.")
 n_iterations = 1000
 for iteration in range(n_iterations):
     if iteration < 10:
         y_predict = X_new_b.dot(theta)
         style = "b-" if iteration > 0 else "r--"
         plt.plot(X_new, y_predict, style)
     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
     theta = theta - eta * gradients
     if theta_path is not None:
         theta_path.append(theta)
 plt.xlabel("$x_1$", fontsize=18)
 plt.axis([0, 2, 0, 15])
 plt.title(r"$\eta = {}$".format(eta), fontsize=16)

rnd.seed(42)
theta = rnd.randn(2,1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
plt.show()

print('\n\n------theta_path_bgd = \n{}\n\n'.format(theta_path_bgd))


#---------------------------------------------------------------------------
#  Stochastic Gradient Descent
#---------------------------------------------------------------------------

theta_path_sgd = []
n_iterations = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

rnd.seed(42)
theta = rnd.randn(2,1)  # random initialization

def learning_schedule(t):
 return t0 / (t + t1)

m = len(X_b)

for epoch in range(n_iterations):
 for i in range(m):
     if epoch == 0 and i < 20:
         y_predict = X_new_b.dot(theta)
         style = "b-" if i > 0 else "r--"
         plt.plot(X_new, y_predict, style)
     random_index = rnd.randint(m)
     xi = X_b[random_index:random_index+1]
     yi = y[random_index:random_index+1]
     gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
     eta = learning_schedule(epoch * m + i)
     theta = theta - eta * gradients
     theta_path_sgd.append(theta)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
intercept_sgd = sgd_reg.intercept_
slope_sgd = sgd_reg.coef_

print('-----  sklern Linear Regression: Stochastic Gradient Decent  -----')
print('intercept_sgd =\n{}\n\n'.format(intercept_sgd))
print('slope_sgd =\n{}\n\n'.format(slope_sgd))


#---------------------------------------------------------------------------
#   Polynomial Regression
#---------------------------------------------------------------------------


m = 100
X = 6 * np.random.rand(m,1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias = False)
X_poly = poly_features.fit_transform(X)
print('\n첫번째 X={}\n첫번째 X의 Polynomial term ={}\n\n'.format(X[0], X_poly[0]))

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print('\nintercept_={}\ncoef_={}\n\n'.format(lin_reg.intercept_, lin_reg.coef_))

#---------------------------------------------------------------------------
#   Graph in Page 174
#---------------------------------------------------------------------------

X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()


#---------------------------------------------------------------------------
#   Learning Curve
#   Page 175
#---------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
   polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
   std_scaler = StandardScaler()
   lin_reg = LinearRegression()
   polynomial_regression = Pipeline((
         ("poly_features", polybig_features),
         ("std_scaler", std_scaler),
         ("lin_reg", lin_reg),
     ))
   polynomial_regression.fit(X, y)
   y_newbig = polynomial_regression.predict(X_new)
   plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

   plt.plot(X, y, "b.", linewidth=3)
   plt.legend(loc="upper left")
   plt.xlabel("$x_1$", fontsize=18)
   plt.ylabel("$y$", rotation=0, fontsize=18)
   plt.axis([-3, 3, 0, 10])
   plt.show()

#---------------------------------------------------------------------------
#   Learning Curve
#   Page 177
#---------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
 train_errors, val_errors = [], []
 for m in range(1, len(X_train)):
     model.fit(X_train[:m], y_train[:m])
     y_train_predict = model.predict(X_train[:m])
     y_val_predict = model.predict(X_val)
     train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
     val_errors.append(mean_squared_error(y_val_predict, y_val))

 plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
 plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Test set")
 plt.legend(loc="upper right", fontsize=14)
 plt.xlabel("Training set size", fontsize=14)
 plt.ylabel("RMSE", fontsize=14)

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])
plt.show()

#---------------------------------------------------------------------------
#   Polynomial Model의 Learning Curve
#   Page 178
#---------------------------------------------------------------------------
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline((
     ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
     ("sgd_reg", LinearRegression()),
 ))

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])
plt.show()

#---------------------------------------------------------------------------
#   Ridge Regression
#   Page 180
#---------------------------------------------------------------------------


from sklearn.linear_model import Ridge

rnd.seed(42)
m = 20
X = 3 * rnd.rand(m, 1)
y = 1 + 0.5 * X + rnd.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

def plot_model(model_class, polynomial, alphas, **model_kargs):
 for alpha, style in zip(alphas, ("b-", "g--", "r:")):
     model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
     if polynomial:
         model = Pipeline((
                 ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                 ("std_scaler", StandardScaler()),
                 ("regul_reg", model),
             ))
     model.fit(X, y)
     y_new_regul = model.predict(X_new)
     lw = 2 if alpha > 0 else 1
     plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
 plt.plot(X, y, "b.", linewidth=3)
 plt.legend(loc="upper left", fontsize=15)
 plt.xlabel("$x_1$", fontsize=18)
 plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100))
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1))
plt.show()

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

sdg_reg = SGDRegressor(penalty='l2', random_state=42)
sdg_reg.fit(X, y.ravel())
sdg_reg.predict([[1.5]])


#---------------------------------------------------------------------------
#   Lasso Regression
#   Page 183
#---------------------------------------------------------------------------

from sklearn.linear_model import Lasso

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1))
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1)
plt.show()

Lasso_reg = Lasso(alpha=0.1)
Lasso_reg.fit(X, y)
Lasso_reg.predict([[1.5]])

#---------------------------------------------------------------------------
#   ElasticNet
#---------------------------------------------------------------------------


from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

rnd.seed(42)
m = 100
X = 6 * rnd.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + rnd.randn(m, 1)

X_train, X_test, y_train, y_test = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline((
     ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
     ("std_scaler", StandardScaler()),
 ))

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_test_poly_scaled = poly_scaler.transform(X_test)

sgd_reg = SGDRegressor(n_iter=1,
                    penalty=None,
                    eta0=0.0005,
                    warm_start=True,
                    learning_rate="constant",
                    random_state=42)

n_epochs = 500
train_errors, test_errors = [], []
for epoch in range(n_epochs):
   sgd_reg.fit(X_train_poly_scaled, y_train)
   y_train_predict = sgd_reg.predict(X_train_poly_scaled)
   y_test_predict = sgd_reg.predict(X_test_poly_scaled)
   train_errors.append(mean_squared_error(y_train_predict, y_train))
   test_errors.append(mean_squared_error(y_test_predict, y_test))

best_epoch = np.argmin(test_errors)
best_test_rmse = np.sqrt(test_errors[best_epoch])

plt.annotate('Best model',
          xy=(best_epoch, best_test_rmse),
          xytext=(best_epoch, best_test_rmse + 1),
          ha="center",
          arrowprops=dict(facecolor='black', shrink=0.05),
          fontsize=16,
         )

best_test_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_test_rmse, best_test_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="Test set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)

plt.show()


#---------------------------------------------------------------------------
#  Logistic Regression
#---------------------------------------------------------------------------

from sklearn import datasets

iris = datasets.load_iris()
list(iris.keys())

from sklearn.linear_model import LogisticRegression

X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 100).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1]>=.5][0]


 #---------------------------------------------------------------------------
 #   decision_boundary
 #---------------------------------------------------------------------------

log_reg.predict([[1.7],[1.5]])

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()


 #---------------------------------------------------------------------------
 #  Softmax Regression
 #---------------------------------------------------------------------------

X = iris["data"][:, (2,3)]  # petal length, width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

log_reg = LogisticRegression(C=10*10)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
     np.linspace(2.9, 7, 500).reshape(-1, 1),
     np.linspace(0.8, 2.7, 200).reshape(-1, 1),
 )
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()

X = iris["data"][:, (2,3)]  # petal length, width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

x0, x1 = np.meshgrid(
     np.linspace(0, 8, 500).reshape(-1, 1),
     np.linspace(0, 3.5, 200).reshape(-1, 1),
 )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolour")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap, linewidth=5)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

softmax_reg.predict([[5, 2]])
softmax_reg.predict_proba([[5,2]])
