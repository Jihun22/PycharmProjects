import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/ 255.0 , x_test / 255.0

x_train = x_train[...,tf.newaxis].astype("float32")
x_test = x_test[...,tf.newaxis].astype("float32")

# tf.data를 사용하여 데이터셋을 섞고 배치 만듬
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#케라스의 모델 서브클래싱 api 이용하여 tf.keras 모델 만듬
class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(32,3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self,x):
        x =self.conv1(x)
        x =self.flatten(x)
        x =self.d1(x)
        return self.d2(x)
model =MyModel()