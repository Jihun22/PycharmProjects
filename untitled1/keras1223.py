#plt 한글 타이틀
import matplotlib

matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False
#로이터 데이터셋 로드
from keras.datasets import  reuters
(train_data , train_labels), (test_data, test_labels)= reuters.load_data(num_words=10000)
print('train_data:',len(train_data))
print('test_data:',len(test_data))
#로이터 데이터셋을 텍스트로 디코딩
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_newswire = ''.join([reverse_word_index.get(i -3, '?')for i in train_data[0]])
print('train_labels:',train_labels[10])

#데이터 인코딩
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 훈련 데이터 벡터 변환
x_train = vectorize_sequences(train_data)
# 테스트 데이터 벡터 변환
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 훈련 레이블 벡터 변환
one_hot_train_labels = to_one_hot(train_labels)
# 테스트 레이블 벡터 변환
one_hot_test_labels = to_one_hot(test_labels)
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
#모델 정의
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics= ['accuracy'])
# 검증 세트 준비
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

 #모델 훈련
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 훈련과 검증 손실 그리기
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss =history.history['val_loss']

epochs = range (1, len(loss) + 1)
plt.plot(epochs,loss, 'bo', label ='Training loss' )
plt.plot(epochs,val_loss,'b', label = 'Validation loss')
plt.title('훈련과 검증 손실')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#훈련과 검증 정확도 그리기

plt.clf()   #그래프 초기화
#['acc] 오류  -> ['accuracy] 로 해야 돌아감
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs , acc , 'bo', label = 'Training acc')
plt.plot(epochs, val_acc , 'b', label = 'Validation accuracy')
plt.title('트레이닝 훈련과 검증 정확도 ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#모델 처음부터 다시 훈련
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print('results:', results)

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print('float:',float(np.sum(hits_array)) / len(test_labels))

# 새로운 데이터 예측하기
predictions = model.predict(x_test)
#predictions  의 각 항목
print("predictions:",predictions[0].shape)
# 벡터의 원소 합
print("np.sum:",np.sum(predictions[0]))
# 가장 큰 값 예측 클래스
print("np.argmax:",np.argmax(predictions[0]))

#레이블과 손실을 다루는 다른 방법
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

#충분히 큰 중간층을 두어야 하는 이유

# 정보 병목이 있는 모델
model = models.Sequential()
model.add(layers.Dense(64, activation='relu' , input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss ='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val,y_val))