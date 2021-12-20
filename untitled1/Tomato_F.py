import pandas as pd
import  numpy as np
from keras.models import Sequential
from keras.layers.core import  Dense,Dropout,Activation
from keras.callbacks import EarlyStopping
#csv 파일 읽어오기

file_path='C:/Users/ik533/Desktop/전라북도_농가데이터셋_토마토수정본1.csv'
#[75,82]
#columns=["생장길이,열매수"]
df1 =pd.read_csv(file_path)
#print(df1)
#해당 열 출력 print(df1[["주차","생장길이","열매수"]])
#데이터 가공
df1["생장길이"] /=200
df1["열매수"] /=100

week_class= {
    "4주차": [1,0,0],
    "9주차": [0,1,0],
    "18주차":[0,0,1]
}

y= np.empty((20000,3))
for i,v in enumerate(df1["주차"]):
    y[i] = week_class[v]
print(y[0:3])

x= df1[["생장길이", "열매수"]].values

x_train, y_train = x[1:15001] , y[1:15001]
x_test, y_test = x[15001:20001] , y[150001,20001]

#모델 생성
model =Sequential()
model.add(Dense(512 , input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile("rmsprop", "categorical_crossentropy", metrics= ['accuracy'])
    #레이어 형성
model.compile("rmsprop","categorical_crossentropy",metrics=['accuracy'])

#학습을 시킴
model.fit(x_train,y_train)

#예측을 한뒤 : model.predict()

#정답률
score= model.evaluate(x_test,y_test)
print("score:",score)