import pandas as pd
import numpy as np

#Pandas 객체

#Series 객체

s= pd.Series([0,0.25,0.5,0.75,1.0])

print(s)
print(s.values)
print(s.index)
print(s[1])
print(s[1:4])
#인덱스 변경 (따로 구성)
s= pd.Series([0,0.25,0.5 ,0.75,1.0] , index=['a','b','c','d','e'])
print(s)
print(s['c'])
print(s[['c','d','e']])
print('b' in s)

s= pd.Series([0,0.25, 0.5 , 0.75 , 1.0], index=[2,4,6,8,10])
print(s)
print(s[4])
print(s[2:])
print(s.unique())
print(s.value_counts())
print(s.isin([0.25, 0.75]))
pop_tuple = {'서울특별시':9720846,
             '부산광역시':3404423,
             '인천광역시':2947217,
             '대구광역시':2427954,
             '대전광역시':1471040,
             '광주광역시':1455048}
population = pd.Series(pop_tuple)

print(population)
print(population['서울특별시'])
print(population['서울특별시':'인천광역시'])

