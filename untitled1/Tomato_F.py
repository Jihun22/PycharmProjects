import pandas as pd
from tensorflow import keras

import  numpy as np
import os
#csv 파일 읽어오기
file_path='C:/Users/ik533/Desktop/전라북도_농가데이터셋_토마토수정본.csv'
#[75,82]
#columns=["생장길이,열매수"]
df1 =pd.read_csv(file_path)
#print(df1)
print(df1[["주차","생장길이","열매수"]])
