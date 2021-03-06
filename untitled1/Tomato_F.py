import pandas as pd
import  numpy as np
from keras.models import Sequential
from keras.layers.core import  Dense,Dropout,Activation
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False
#csv 파일 읽어오기

file_path='C:/Users/ik533/Desktop/전라북도_농가데이터셋_토마토수정본1.csv'
file_pathR='C:/Users/ik533/Desktop/전라북도_농가데이터셋_토마토수정본.csv'
#[75,82]
#columns=["생장길이,열매수"]
df1 =pd.read_csv(file_path)
ds1 =pd.read_csv(file_pathR)
#print(df1)
#해당 열 출력 print(df1[["주차","생장길이","열매수"]])
#데이터 가공
df2=df1['생장길이']
df3=df1['열매수']
df4=df1['주차']

ds2=ds1['생장길이']
ds3=ds1['열매수']
ds4=ds1['주차']

#print(df2)
#print(df3)

tuples= df2
tupp=df3
tupleso = pd.Series(df3)
tupleson = pd.Series(df2)
print(tupleso)
Tomato_df = pd.DataFrame({'열매수':df3,
                          '생장길이':df2,
                          '주차':df4})
#생장길이와 열매수 관계
Tomato_df['평균값'] = (df2 /df3)
print(Tomato_df)

#선형회귀
# x= 열매수  y= 생장길이 그래프
x=df3
y=df2
plt.plot(x,y,'o')
plt.xlabel('열매수')
plt.ylabel('생장길이')
plt.title('4,9,18주데이터')
plt.show()

# 전체 데이터 x= 열매수  y= 생장길이 그래프
x=ds3
y=ds2
plt.plot(x,y,'o')
plt.xlabel('열매수')
plt.ylabel('생장길이')
plt.title('전체데이터')
plt.show()

# x = 주차  y= 열매수
x=df3
y=df4
fig = plt.figure(figsize=(5,5))
font_size =7
plt.plot(x,y,'o')
plt.xlabel('열매수')
plt.ylabel('주차')
plt.title('4,9,18주데이터')
plt.show()

# 전체 데이터 x = 주차  y= 열매수
x=ds3
y=ds4
fig = plt.figure(figsize=(5,5))
font_size =7
plt.plot(x,y,'o')
plt.xlabel('열매수')
plt.ylabel('주차')
plt.title('전체 데이터')
plt.show()
# x = 생장길이  y= 주차
x=df2
y=df4
fig = plt.figure(figsize=(5,5))
font_size =7
plt.plot(x,y,'o')
plt.xlabel('생장길이')
plt.ylabel('주차')
plt.title('4,9,18주데이터')
plt.show()

#전체데이터 x = 생장길이  y= 주차
x=ds2
y=ds4
fig = plt.figure(figsize=(5,5))
font_size =7
plt.plot(x,y,'o')
plt.xlabel('생장길이')
plt.ylabel('주차')
plt.title('전체데이터')
plt.show()
#예측


'''''
y= np.empty((200,3))
for i,v in enumerate(df1["주차"]):
    y[i] = week_class[v]
print(y[0:3])

x= df1[["생장길이", "열매수"]].values

x_train, y_train = x[1:100] , y[1:100]
x_test, y_test = x[2:200] , y[2,200]

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
'''