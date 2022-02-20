import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

from keras.layers.normalization import BatchNormalization


# 데이터 세트 로드 함수
from tensorflow_estimator.python.estimator import early_stopping


def load_dataset():
    # h5py 모듈 사용 및, 파일 경로와 파일 접속 모드(읽기, read)를 설정
    all_train_data = h5py.File('C:/Users/ik533/Desktop/h5py/train_happy.h5', "r")
    all_test_data = h5py.File('C:/Users/ik533/Desktop/h5py/test_happy.h5', "r")

    # 모든 훈련 데이터와 테스트 데이터를 파일에서 읽어 numpy 배열에 저장
    x_train = np.array(all_train_data["train_set_x"][:])
    y_train = np.array(all_train_data["train_set_y"][:])

    x_test = np.array(all_test_data["test_set_x"][:])
    y_test = np.array(all_test_data["test_set_y"][:])

    # 데이터 셰이프 변경
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return x_train, y_train, x_test, y_test
# 데이터 로드
X_train, Y_train, X_test, Y_test = load_dataset()

print('Image dimension: ', X_train.shape[1:])
print('Training tensor dimension: ', X_train.shape)
print('Test tensor dimention: ', X_test.shape)
print()
print('Number of training tensor: ', X_train.shape[0])
print('Number of test tensor: ', X_test.shape[0])
print()
print('Training lable dimension: ', Y_train.shape)
print('Test label dimension 차원: ', Y_test.shape)

# 이미지 플롯
plt.imshow(X_train[0])
plt.show()
# 이미지 라벨 출력(smiling = 1, frowning = 0)
print("y = " + str(np.squeeze(Y_train[:, 0])))

#데이터 표준화
#채널의 최댓값인 255를 사용해 각 픽셀 값을 표준화
X_train = X_train / 255.
X_test = X_test / 255.

#라벨 변환
Y_train = Y_train.T
Y_test = Y_test.T

# 상태 출력
print("Number of training examples:" +str(X_train.shape[0]))
print("Number of test examples:" + str(X_test.shape[0]))
print(("X_train shape:" +str(X_train.shape)))
print("Y_train shape:" + str(Y_train.shape))
print("X_test shape:" + str(X_test.shape))
print("Y_test shape:" + str(Y_test.shape))

# 이미지 플롯
plt.imshow(X_train[0])

# 이미지 라벨 출력(smiling = 1, frowning = 0)
print("y = " + str(np.squeeze(Y_train[:, 0])))

# 채널의 최대값인 255를 사용해 각 픽셀값을 표준화
X_train = X_train / 255.
X_test = X_test / 255.

#라벨 변환
Y_train = Y_train.T
Y_test = Y_test.T

#상태 출력
print("Number of training examples:" + str(X_train.shape[0]))
print("Number of test examples:" +str(X_test.shape[0]))
print("X_train shape:" + str(X_train.shape))
print("Y_train shape:" + str(Y_train.shape))
print("X_test shape:" + str(X_test.shape))
print("Y_test shape:" + str(Y_test.shape))


# float32 ndarray로 변환
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
plt.imshow(X_train[3])
plt.show()

#컨볼루셔널 레이어

#컨볼루셔널 뉴럴 네트워크 생성
model = Sequential()

#첫 번째 컨볼루셔널 레이어
model.add(Conv2D(16,(5,5), padding='same', activation='relu',
                 input_shape=(64,64,3)))
model.add(BatchNormalization())
#첫 번째 풀링 레이어
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# 두 번째 컨볼루셔널 레이어
model.add((Conv2D(32, (5,5), padding='same', activation='relu')))
model.add(BatchNormalization())
#두 번째 풀링 레이어
model.add(MaxPooling2D(pool_size=(2,2)))

#드롭아웃 레이어
model.add(Dropout(0.1))

#평탄화 레이어
model.add(Flatten())

#첫번째 완전 연결 레이어
model.add(Dense(128, activation='relu'))

#마지막 출력 레이어
model.add(Dense(1, activation='sigmoid'))

#모델 레이아웃 출력
model.summary()
# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#조기 중단 콜백을 통해 검증 손실을 모니터하고, 필요한 경우 훈련을 중단한다.
from keras.callbacks import EarlyStopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss')

# 훈련 세션 초기화
model.fit(X_train,Y_train,
          validation_data=(X_test,Y_test),
          epochs=20,
          batch_size=50,
          callbacks=[early_stopping])

# 테스트 세트 결과 예측
Y_pred = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# 테스트 정확성, 정밀도 평가 및 sklearn.metrics를 사용해 점수 확인

print("Test accuracy: %s" % accuracy_score(Y_test, Y_pred))
print("Precision: %s" % precision_score(Y_test, Y_pred))
print("Recall: %s" % recall_score(Y_test, Y_pred))
print("F1 score: %s" % f1_score(Y_test, Y_pred))

import seaborn as sns

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True)

predictions = model.predict([X_test])

predictions[103]

Y_test[8]

plt.imshow(X_test[8])
plt.show()

img_tensor = np.expand_dims(X_test[8], axis=0)

img_tensor /= 255.