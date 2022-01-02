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

#DataFrame 객체
print(pd.DataFrame([{'A':2, 'B':4, 'D':3}, {'A':4 , 'B':5,'C':7}]))

print(pd.DataFrame(np.random.rand(5,5),
             columns=['A','B','C','D','E'],
             index=[1,2,3,4,5]))
#남자인구
male_tuple = {'서울특별시':4732275,
             '부산광역시':1668618,
             '인천광역시':1476813,
             '대구광역시':1198815,
             '대전광역시':734441,
             '광주광역시':720060}
male = pd.Series(male_tuple)

print(male)

#여자인구
female_tuple = {'서울특별시':4988571,
             '부산광역시':1735805,
             '인천광역시':1470404,
             '대구광역시':1229139,
             '대전광역시':736599,
             '광주광역시':734988}
female = pd.Series(female_tuple)

print(female)