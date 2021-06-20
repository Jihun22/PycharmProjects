'''
문) weatherAUS.csv 파일을 시용하여 NB 모델을 생성하시오
  조건1> NaN 값을 가진 모든 row 삭제 
  조건2> 1,2,8,10,11,22,23 칼럼 제외 
  조건3> 7:3 비율 train/test 데이터셋 구성 
  조건4> formula 구성  = RainTomorrow ~ 나머지 변수(16개)
  조건5> model 평가 : accuracy
'''
import pandas as pd
from sklearn.naive_bayes import GaussianNB # model 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score, confusion_matrix # model accuracy

data = pd.read_csv('../data/weatherAUS.csv')
print(data.head())
print(data.info())

# 조건1> NaN 값을 가진 모든 row 삭제
data=data.dropna()
print(data.head())

# 조건2> 1,2,8,10,11,22,23 칼럼 제외 
col = list(data.columns)
print(col)

for i in [1,2,8,10,11,22,23] :    
    col.remove(list(data.columns)[i-1])
print(col)
print(len(col)) # 24-7 = 17

new_data = data[col]
print(new_data.head())
print(new_data.info()) 

# 조건3> 7:3 비율 train/test 데이터셋 구성
train_data, test_data = train_test_split(new_data, test_size=0.3)
train_data.shape # (12164, 17)
type( train_data) # pandas.core.frame.DataFrame

# 조건4> formula 구성  = RainTomorrow ~ 나머지 변수(16개)
model = GaussianNB()
# x변수 : 1~16, y변수 : 'RainTomorrow'
model.fit(X=train_data.iloc[:,:16], y = train_data['RainTomorrow'])

# 조건5> model 평가 : accuracy
y_pred = model.predict(X = test_data.iloc[:,:16]) # 예측치 
y_true = test_data['RainTomorrow']

# 분류정확도 
acc = accuracy_score(y_true, y_pred)
acc #  0.8070579209819716

# 교차분할표 
con_mat = confusion_matrix(y_true, y_pred)
con_mat
'''
array([[3363,  636],
       [ 370,  845]],
'''
acc = (con_mat[0,0]+ con_mat[1,1]) / len(y_true) 
acc
# 정확도 = 0.8070579209819716

con_mat[0,0]/(con_mat[0,0] + con_mat[1,0]) # 0.9025681758009002
con_mat[1,1]/(con_mat[0,1] + con_mat[1,1]) # 0.5434933890048712


