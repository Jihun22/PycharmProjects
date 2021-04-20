import os
#--------------------------------------------
#  dataset 불러오기
#--------------------------------------------

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
print(mnist,'\n\n')

#--------------------------------------------
#  dataset 구조이해
#--------------------------------------------
X, y = mnist['data'], mnist['target']
print('X.shape={} \ny.shape={}\n\n'.format(X.shape, y.shape))

os.system('Pause')
#--------------------------------------------
#  dataset 확인
#--------------------------------------------
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[3000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
#plt.imshow(some_digit_image)
plt.axis('off')
plt.show()

print("X[3000]={}\n\n".format(y[3000]))

os.system('Pause')
#--------------------------------------------
#  dataset 분리: train dataset, test dataset
#--------------------------------------------
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
print('X_train={} \ny_train={}'.format(X_train, y_train))

# Cross Validation을 위한 Data Shuffling
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

os.system('Pause')
#--------------------------------------------
# Training Binary Classifier
# 5 를 판단하는 Classifier
#--------------------------------------------

y_train_5 = (y_train == 5)  # 5는 True이고 다른 숫자는 모두 False
y_test_5 = (y_test == 5)

#print('y_train_5.shape: {}'.format(y_train_5.shape))
#os.system('Pause')

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)

predict_value = sgd_clf.predict([some_digit])
print('\n\npredict_value[some_digit] =\n{} \n\npredict_value[some_digit] = {}\n\n'.format(some_digit, predict_value))

os.system('Pause')
#--------------------------------------------
# Cross Validation
#--------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print('\nn_correct/len(y_pred) = {:0.3f}\n\n'.format(n_correct/len(y_pred)))
os.system('Pause')

from sklearn.model_selection import cross_val_score
cvs5 = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print('\n\ncross_val_score(5)={}\n\n'.format(cvs5))
#--------------------------------------------------------
# Dummy Classifier: Never 5
# p129
#--------------------------------------------------------
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cvs_n_5 = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print('cross_val_score(Never 5)={}\n\n'.format(cvs_n_5))

os.system('Pause')
#--------------------------------------------------------
# 오차행렬
# page=129
#--------------------------------------------------------
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_train_5, y_train_pred)
print('confusion matrix(5)=\n{}\n\n'.format(conf_mat))

#conf_mat_perf = confusion_matrix(y_train_5, y_train_perfect_predictions)
#print('confusion matrix(Perfect)=\n{}\n\n'.format(conf_mat_perf))

os.system('Pause')
#--------------------------------------------------------
# 3.3.3 정밀도와 재현율
# page=132
#--------------------------------------------------------
from sklearn.metrics import precision_score, recall_score

ps = precision_score(y_train_5, y_train_pred)
print('precision_score=\n{:0.3f}\n\n'.format(ps))

rs = recall_score(y_train_5, y_train_pred)
print('recall_score=\n{:0.3f}\n\n'.format(rs))

from sklearn.metrics import f1_score
f1 = f1_score(y_train_5, y_train_pred)
print('f1_score=\n{:0.3f}\n\n'.format(f1))

os.system('Pause')
#----------------------------------------------------------
y_scores = sgd_clf.decision_function([some_digit])
print('y_scores=\n{}\n\n'.format(y_scores))

threshold = 0
print('y_scores=({}) > threshold({})?  {}\n\n'.format(y_scores,threshold,(y_scores>threshold)))
threshold = 200000
print('y_scores=({}) > threshold({})?  {}\n\n'.format(y_scores,threshold,(y_scores>threshold)))

#--------------------------------------------------------
# 적정 임계값의 결정
# page=134
#--------------------------------------------------------
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method='decision_function')

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center left')
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'g-')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

plot_precision_vs_recall(precisions, recalls)
plt.show()

y_train_pred_90 = (y_scores > 70000)
print('\nprecision_score(y_train_5, y_train_pred_90): {}\n\n'.format(
                    precision_score(y_train_5, y_train_pred_90)))
print('\nrecall_score(y_train_5, y_train_pred_90): {}\n\n'.format(
                    recall_score(y_train_5, y_train_pred_90)))

os.system('Pause')
#--------------------------------------------------------
# ROC curve, AUC
# page=137
#--------------------------------------------------------
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

plot_roc_curve(fpr, tpr, label=None)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_train_5, y_scores)
print('roc_auc_score=\n{}\n\n'.format(roc_auc))


#--------------------------------------------------------
# RandomForestClassifier와 SGDClassifier의 성능비교
# ROC curve, AUC
# page=139
#--------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                     method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()

#--------------------------------------------------------
# sklearn의 다중분류 작업에 이진분류 알고리즘을 적용할 때 OvO 작동결과
# page=142
#--------------------------------------------------------
sgd_clf.fit(X_train, y_train)
pre_some_digit = sgd_clf.predict([some_digit])
print('[some_digit]=\n{}\n\n'.format(pre_some_digit))

some_digit_scores = sgd_clf.decision_function([some_digit])
print('decision_function[some_digit]=\n{}\n\n'.format(some_digit_scores))

digit_val = np.argmax(some_digit_scores)
print('[some_digit] =\n{}\n\n'.format(digit_val))
print('Class Values of SGD =\n{}\n\n'.format(sgd_clf.classes_))
print('Class list의 6th(Position of Number 5) Value =\n{}\n\n'.format(sgd_clf.classes_[5]))

#--------------------------------------------------------
# sklearn을 이용한 OvO Classifier
# page=143
#--------------------------------------------------------
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_some_digital_pre = ovo_clf.predict([some_digit])
print('[some_digit predict using OvO] =\n{}\n\n'.format(ovo_some_digital_pre))
print('Length of OvO_clf_estimators =\n{}\n\n'.format(len(ovo_clf.estimators_)))
#print('OvO_clf_estimators =\n{}\n\n'.format(ovo_clf.estimators_))

#--------------------------------------------------------
# sklearn을 이용한 RandonForestClssifier Training
# page=143
#--------------------------------------------------------
forest_clf.fit(X_train, y_train)
forest_some_digit =forest_clf.predict([some_digit])
print('[some_digit predicted value in randim forest] =\n{}\n\n'.format(forest_some_digit))
prob_class = forest_clf.predict_proba([some_digit])
print('[some_digit predicted probabilities of classes in randim forest] =\n{}\n\n'.format(prob_class))

cvs_sgd_clf = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
print('Cross Validation Score_SGD =\n{}\n\n'.format(cvs_sgd_clf))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cvs_clf_scaled = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
print('Cross Validation Score_SGD_Scaled =\n{}\n\n'.format(cvs_clf_scaled))


#--------------------------------------------------------
# sklearn을 이용한 Error분석
# page=144
#--------------------------------------------------------
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print('Confusion Matrix =\n{}\n\n'.format(conf_mx))

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx/row_sums

np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

#print('Type(X_aa)={}\n[3]->[3] =\n{}\n\n'.format(type(X_aa), X_aa))
#print('[3]->[5] =\n{}\n\n'.format(X_ab))
#print('[5]->[3] =\n{}\n\n'.format(X_ba))
#print('[5]->[5] =\n{}\n\n'.format(X_bb))

#--------------------------------------------------------
# github jupyter Notebook의 plot_digits함수
# https://github.com/ageron/handson-ml/issues/257
#--------------------------------------------------------

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(8, 8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

#--------------------------------------------------------
# multilabel classification
# page 149
#--------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)     #  y_train 중 2로 나누어 나머지가 1인 것 즉 홀수 인것
y_multilabel = np.c_[y_train_large, y_train_odd]

#print('type(y_train_large)=\n{}\ntype(y_train_odd)=\n{}\ntype(y_multilabel)=\n{}\n'.format
#                (type(y_train_large), type(y_train_odd), type(y_multilabel)))
#print('shape(y_train_large)=\n{}\nshape(y_train_odd)=\n{}\nshape(y_multilabel)=\n{}\n\n'.format
#                (y_train_large.shape, y_train_odd.shape, y_multilabel.shape))

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

some_digit_knn = knn_clf.predict([some_digit])
print('knn_clf.predict([some_digit]=\n{}\n\n'.format(some_digit_knn))

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
#y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
f1_score_multilabel = f1_score(y_multilabel, y_train_knn_pred, average='macro')
print('F1 Score of Multi Labeled Classifier(KNN)=\n{}\n\n'.format(f1_score_multilabel))

#--------------------------------------------------------
# multioutput-multiclass classification
# page 150
#--------------------------------------------------------
noise = rnd.rand.int(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = rnd.rand.int(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train_mod
y_test_mod = X_test

some_index = 5500
plt.subplot(121); plot_digits(X_test_mod[some_digit],1)
plt.subplot(122); plot_digit(sy_test_mod[some_digit],1)
plt.show()

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digits(clean_digit, 1)
