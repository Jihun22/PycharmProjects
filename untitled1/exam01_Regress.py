'''
문) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성 
  조건1> train/test - 7:3비울
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> 모델 평가 : 평균제곱오차[MSE]
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

# 1. data load & 변수 선택
boston = load_boston()
print(boston)

boston_x = boston.data
boston_y = boston.target

boston_x.shape # (506, 13)
boston_y.shape # (506,)

# 2. train/test set (7:3)
x_train, x_test, y_train, y_test = train_test_split(boston_x, boston_y, test_size = 0.3)

x_train.shape # (354, 13)
y_train.shape # (354,)

x_test.shape # (152, 13)
y_test.shape # (152,)

# 3. 회귀모델 생성, 모델 예측치
model = LinearRegression()
model.fit(X = x_train, y = y_train)

y_pred = model.predict(X=x_test) # 예측치
y_true = y_test # 정답

# 4. 모델 평가 : MSE, corr
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
mse # 25.587989432725887

# 단, y 변수 값이 작은 경우에만 MSE 값이 작을수록 정확하다는 것이 맞다.
# 여기서는 y 변수 값 자체가 크기 때문에 MSE가 커진다. 
# MSE는 제곱해서 구하기 때문에 작을수록 더 작은 값을, 클수록 더 큰 값을 도출한다.

import pandas as pd
df = pd.DataFrame({'y_pred' : y_pred, 'y_true' : y_true})
cor = df['y_pred'].corr(df['y_true'])
cor # 0.8387543939759973

# 예측치와 정답의 상관계수가 약 0.84라는 의미 => 1에 가까울수록 정확하다.













