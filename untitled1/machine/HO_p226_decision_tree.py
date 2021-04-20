# -------------------------------------------------------------------------
# Decision Tree: iris DataSet
# page 226
# -------------------------------------------------------------------------
import numpy as np
import numpy.random as rnd
import os

import matplotlib
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

export_graphviz(
         tree_clf,
         # out_file = image_path('iris_tree.dot'),
         out_file="iris_tree.dot",
         feature_names=iris.feature_names[2:],
         class_names=iris.target_names,
         rounded=True,
         filled=True
     )

#dot -Tpng iris_tree.dot -o iris_tree.png
with open("iris_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).view()


from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
     x1s = np.linspace(axes[0], axes[1], 100)
     x2s = np.linspace(axes[2], axes[3], 100)
     x1, x2 = np.meshgrid(x1s, x2s)
     X_new = np.c_[x1.ravel(), x2.ravel()]
     y_pred = clf.predict(X_new).reshape(x1.shape)
     custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
     plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
     if not iris:
         custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
         plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
     if plot_training:
         plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
         plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolour")
         plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
         plt.axis(axes)
     if iris:
         plt.xlabel("Petal length", fontsize=14)
         plt.ylabel("Petal width", fontsize=14)
     else:
         plt.xlabel(r"$x_1$", fontsize=18)
         plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
     if legend:
         plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=11)
plt.text(3.2, 1.80, "Depth=1", fontsize=11)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.show()


# Predict probabilities
# page 230
X_1 = ([[5, 1.5]])

class_predict = tree_clf.predict(X_1)
prob_predict = tree_clf.predict_proba(X_1)

print('\nX_1 = {}\n\n'.format(X_1))
print('\nX_1 Class Prediction ={}\n\n'.format(class_predict))
print('\nX_1 Probability Prediction ={}\n\n'.format(prob_predict))


#-----------------------------------------------------------------
# Regularization for Decision Tree
# page234
# 그림 6-3
#-----------------------------------------------------------------
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=14)
plt.subplot(122)
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.show()


#-----------------------------------------------------------------
#  Decision Tree for Regression
# page235
# 그림 6-5
#-----------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor

rnd.seed(42)
m = 200
X = rnd.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + rnd.randn(m, 1) / 10

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

#-----------------------------------------------------------------
#  Decision Tree for Regression
# page235
#-----------------------------------------------------------------
export_graphviz(
         tree_reg,
         out_file="regression_tree.dot",
         feature_names=["x1"],
         rounded=True,
         filled=True
    )

with open("regression_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).view()
#dot -Tpng regression_tree.dot -o regression_tree.png

#from sklearn.tree import DecisionTreeRegressor

 # Quadratic training set + noise


tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
     x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
     y_pred = tree_reg.predict(x1)
     plt.axis(axes)
     plt.xlabel("$x_1$", fontsize=18)
     if ylabel:
         plt.ylabel(ylabel, fontsize=18, rotation=0)
     plt.plot(X, y, "b.")
     plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
     plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=13)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
     plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
     plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)
plt.show()


#-----------------------------------------------------------------
#  Decision Tree for Regression
#  page236
#  그림 6-6
#-----------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor

#tree_reg = DecisionTreeRegressor(max_depth=2)
#tree_reg.fit(X, y)

#from sklearn.tree import DecisionTreeRegressor

 # Quadratic training set + noise
rnd.seed(42)
m = 200
X = rnd.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + rnd.randn(m, 1) / 10

tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(min_samples_leaf=10, random_state=42)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
     x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
     y_pred = tree_reg.predict(x1)
     plt.axis(axes)
     plt.xlabel("$x_1$", fontsize=14)
     if ylabel:
         plt.ylabel(ylabel, fontsize=14, rotation=0)
     plt.plot(X, y, "b.")
     plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
     plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.title('No Restriction', fontsize=14)
plt.legend(loc="upper center", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
     plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
     plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)

plt.title("min_samples_leaf = {}".format(tree_reg2.min_samples_leaf), fontsize=14)
plt.show()



tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=84)
