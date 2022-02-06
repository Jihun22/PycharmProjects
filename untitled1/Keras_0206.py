import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

print(keras.layers.Dense(10, activation='sigmoid'))

print(keras.Model())

print(keras.models.Sequential())

from tensorflow.keras.layers import Dense, Input, Flatten,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

print(Dense(10, activation='relu'))

print(Flatten(input_shape=[28,28]))

X_train = np.random.randn(5500,2)

print(Input(shape=X_train.shape[1:]))

dense = Dense(10,activation='relu', name='Dense Layer')

print(dense)

dense2 =Dense(15, activation ='softmax')

print(dense2)
#Activation
#Dense layer 에서 활성화함수를 지정할 수도 있지만 때에 따라서 따로 레이어를 만들어줄 수 있음
dense = Dense(10, kernel_initializer= 'he_normal', name ='Dense Layer')
dense = Activation(dense)
print(dense)

#Flatten  배치크기( 데이터 크기)를 제외하고 데이터를 1차원으로 쭉 펼치는 작업

Flatten(input_shape=(28,28))

#Input 모델의 입력을 정의
#shape, dtype 을 포함
# 하나의 모델은 여러개의 입력을 가질수 있음
#summary() 메소드를 통해서는 보이지 않음

input_1 = Input(shape=(28,28), dtype=tf.float32)
input_2 = Input(shape=(8,), dtype=tf.int32)
print(input_1)
print(input_2)


#모델구성방법
#Sequential()
#서브클래싱
#함수형 API

#Sequential()
"""
모델이 순차적으로 진행할 때 사용
간단한 방법
 Sequential 객체 생성후 ,add 를 통한 방법
Sequential 인자에 한번에 추가
다중입력 및 출력이 존재하는 등의 복잡한 모델을 구성할 수 없음
"""
from  tensorflow.keras.layers import  Dense , Input , Flatten
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import plot_model


model = Sequential()
model.add(Input(shape=(28,28)))
model.add(Dense(300,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

# 모델 구조 확인 summary() 이용
model.summary()
#plot_model(model)
#plot_model(model , to_file='/model1.png')

model = Sequential([Input(shape=(28,28) , name= 'Input'),
                    Dense(300, activation='relu', name='Dense1'),
                    Dense(100, activation='relu', name='Dense2'),
                    Dense(10, activation='softmax', name='Output')])
model.summary()
#plot_model(model)
#plot_model(model , to_file='/model2.png')

#함수형 API
