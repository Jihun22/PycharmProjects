import numpy as np
import tensorflow as tf
from keras.layers import Concatenate
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
"""
가장 권장되는 방법
모델을 복잡하고 유연하게 구성 가능
다중 입출력을 다룰수 있음
"""

from  tensorflow.keras.models import Model
from  tensorflow.keras.layers import Input , Flatten, Dense
from tensorflow.keras.utils import  plot_model

inputs = Input(shape=(28,28,1))

x = Flatten(input_shape=(28,28,1))(inputs)
x = Dense(300, activation ='relu')(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(10 , activation = 'softmax')(x)

model = Model(inputs = inputs, outputs =x)
model.summary()
#plot_model(model)


input_layer = Input(shape=(28,28))
hidden1 = Dense(100, activation='relu')(input_layer)
hidden2 = Dense(30, activation='relu')(hidden1)
concat = Concatenate()([input_layer,hidden2])
output = Dense(1)(concat)

model = Model(inputs=[input_layer], outputs=[output])
model.summary()

input_1 = Input(shape=(10,10), name ='input_1')
input_2 = Input(shape=(10,28), name ='input_2')

hidden1 = Dense(100, activation='relu')(input_2)
hidden2 = Dense(10, activation='relu')(hidden1)
concat = Concatenate()([input_1,hidden2])
output = Dense(1,activation='sigmoid', name='output')(concat)

model = Model(inputs = [input_1, input_2], outputs=[output])

model.summary()

input_ = Input(shape=(10,10) , name='input_')

hidden1 = Dense(100, activation= 'relu')(input_)
hidden2 = Dense(10, activation= 'relu')(hidden1)

output = Dense(1, activation='sigmoid', name = 'main_output')(hidden2)
sub_out = Dense(1, name='sum_output')(hidden2)

model = Model(inputs=[input_], outputs=[output, sub_out])

model.summary()

input_1 = Input(shape= (10,10), name='input_1')
input_2 = Input(shape= (10,28), name='input_2')

hidden1 = Dense(100, activation= 'relu')(input_2)
hidden2 = Dense(10, activation='relu')(hidden1)
concat = Concatenate()([input_1, hidden2])
output = Dense(1, activation='sigmoid', name='main_output')(concat)
sub_out = Dense(1, name='sum_output')(hidden2)

model = Model(inputs=[input_1,input_2], outputs=[output, sub_out])
model.summary()
#20220212
'''
서브클래싱 
커스터마이징에 최적화된 방법
model 클래스를 상속받아 model 이 포함하는 기능을 사용할 수 있음
flt() evaluate() , predict9)
save() load()

주로 call() 메소드 안에서 원하는 계산 가능 
for , if 저수준 연산 가능 
권장되는 방법은 아니지만 어느모델의 구현코드를 참고할 때 해석할수 있어야한다.
'''
from  tensorflow.keras.models import  Model
from tensorflow.keras.layers import  Input , Flatten ,Dense
from tensorflow.keras.utils import plot_model

class MyModel(Model):
    def __init__(self, units=30 , activation='relu', **kwargs):
        super(MyModel,self).__inif__(**kwargs)

        self.dense_layer1 = Dense(300, activation= activation)
        self.dense_layer2 = Dense(100, activation= activation)
        self.dense_layer3 = Dense(units, activation=activation)

        self.output_layer = Dense(10, activation='softmax')

    def call(self,inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x)
        x = self.output_layer(x)
        return x
    '''
    모델 가중치 확인 
    '''
    from tensorflow.keras.models import  Model
    from tensorflow.keras.layers import  Input,Flatten, Dense
    from tensorflow.keras.utils import  plot_model

    inputs = Input(shape=(28,28,1))

    x = Flatten(input_shape=(28,28,1)) (inputs)
    x = Dense(300, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs= inputs, outputs=x)

    model.summary()
    #모델의 레이어들이 리스트로 표한됨
    model.layers()
    hidden2 = model.layers[2]
    hidden2.name

