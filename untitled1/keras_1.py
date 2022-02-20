#모둘 읽어오기
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

# 데이터를 가공
csv =pd.read_csv("bmi.csv")
csv["weight"] /=100
csv["height"] /= 200

bmi_class = {
    "thin" : [1,0,0],
    "normal": [0,1,0],
    "fat": [0,0,1]
}
y = np.empty((20000,3))
for i , v in enumerate(csv["label"]):
    y[i] = bmi_class[v]

x = csv[["weight", "height"]].values

x_train , y_train = x[1:15001], y[1:15001]
x_test, y_test = x[15001:20001] , y[15001:20001]

#모델을 만듬
model= Sequential()
model.add(Dense(512 , input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile("rmsprop" , "categorical_crossent" , metrics=['accuracy'])

#학습시킴
model.fit(x_train,y_train)
#예측을한뒤:model.predict()
# 정답률을 구함
score =model.evaluate(x_test, y_test)
print("score:",score)
