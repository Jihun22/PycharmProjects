import pandas as pd
import  numpy as np
from keras.models import Sequential
from keras.layers.core import  Dense,Dropout,Activation
from keras.callbacks import EarlyStopping
#csv 파일 읽어오기

file_path='C:/Users/ik533/Desktop/전라북도_농가데이터셋_토마토수정본1.csv'

df1 =pd.read_csv(file_path)
#print(df1.shape)
#print(df1.head())
print(df1['주차'],df1['생장길이'],df1['열매수'])
#print(df1['주차'])
#print(df1['열매수'])
#정규화

#특정열 출력
#print(df1.sorted['생장길이','열매수'].head())