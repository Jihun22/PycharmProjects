#------------------------------------------------------------------------------
#  SVC Model사용의 경우
#  page 294
#------------------------------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris['data'][:, (2,3)]   #꽃잎 길이, 꽃잎 너비
y = (iris['target'] == 2).astype(np.float)  # Iris-Virginica

svm_clf = Pipeline([
          ('scaler', StandardScaler()),
          ('kernel_linear_svc', SVC(kernel='linear', C=1)),
          ])

svm_clf.fit(X, y)

print('\nIris Prediction by SVC= {}\n\n'.format(svm_clf.predict([[5.5, 1.7]])))
# SVM은 Logistic Regression과 달리 Clsaa에 대한 확률을 제공하지 않음

#------------------------------------------------------------------------------
#   SGDClassifier Model사용의 경우
#  page 294
#------------------------------------------------------------------------------
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris()
X = iris['data'][:, (2,3)]   #꽃잎 길이, 꽃잎 너비
y = (iris['target'] == 2).astype(np.float)  # Iris-Virginica

m = len(iris['target'])
C = 1
alpha = 1/(m*C)

SGD_clf = Pipeline([
          ('scaler', StandardScaler()),
          ('SGDClassifierc', SGDClassifier(loss='hinge', alpha=alpha)),
          ])

SGD_clf.fit(X, y)

print('\nIris Prediction by SGDClassifier = {}\n\n'.format(SGD_clf.predict([[5.5, 1.7]])))

#------------------------------------------------------------------------------
#   다항식 kernel
#  page 294
#------------------------------------------------------------------------------
