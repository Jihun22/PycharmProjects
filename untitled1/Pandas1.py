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

korea_df =pd.DataFrame({'인구수':population,
                        '남자인구수':male,
                        '여자인구수': female})
print(korea_df)

print(korea_df.index)
print(korea_df.columns)
print(korea_df['여자인구수'])
print(korea_df['서울특별시':'인천광역시'])

#index 객체
idx = pd.Index([2,4,6,8,10])
print(idx)
print(idx[1])
print(idx[1:2:2])
print(idx[-1::])
print(idx[::2])
print(idx.size)
print(idx.shape)
print(idx.ndim)
print(idx.dtype)
#idex 연산
idx1 = pd.Index([1,2,4,6,8])
idx2 = pd.Index([2,4,5,6,7])
#append : 색인 객체를 추가한 새로운 색인 반환
print(idx1.append(idx2))

#difference 색인의 차집합 반환
print(idx.difference(idx2))
print(idx1 - idx2)

#intersection 색인의 교집합 반환
print(idx1.intersection(idx2))
print(idx1 & idx2)

#union 색인의 합집합 반환
print(idx1.union(idx2))
print(idx1 | idx2 )

#delete 지우는 값
#print(idx1.delete(0))

#drop 값이 삭제된 새로운 색인 반환
# print(idx.drop(1))

#여집합  공통된걸 뺀 나머지
print(idx1 ^ idx2)

#인덱싱
s = pd.Series([0,0.25,0.5,0.75,1.0], index=['a','b','c','d','e'])
print(s)
print(s['b'])
print('b'in s)
print(s.keys())
print(list(s.items()))
s['f'] = 1.25
print(s)
print(s['a':'d'])
print(s[0:4])
print(s[(s > 0.4 ) & (s < 0.8)])
print(s[['a','c','e']])

# Series 인덱싱
s = pd.Series(['a','b','c','d','e'],
              index=[1,3,5,7,9])
print(s)
print(s[1])

print(s[2:4])

print(s.iloc[1])
print(s.iloc[2:4])
print(s.reindex(range(10)))
print(s.reindex(range(10), method='bfill'))

#DataFreme 인덱싱
print(korea_df['남자인구수'])
print(korea_df.남자인구수)
print(korea_df.여자인구수)
korea_df['남여비율']= (korea_df['남자인구수'] * 100 / korea_df['여자인구수'] )
print(korea_df.남여비율)
print(korea_df.values)
print(korea_df.T)
print(korea_df.values[0])
print(korea_df['인구수'])
print(korea_df.loc[:'인천광역시', :'남자인구수'])
print(korea_df.loc[(korea_df.여자인구수 >1000000)])
print(korea_df.loc[(korea_df.인구수 < 2000000)])
print(korea_df.loc[(korea_df.인구수 > 2500000)])
print(korea_df.loc[korea_df.남여비율 > 100])
print(korea_df.loc[(korea_df.인구수 >2500000) & (korea_df.남여비율 >100)])
print(korea_df.iloc[:3, :2])

#다중인덱싱