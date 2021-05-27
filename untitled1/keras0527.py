#케라스 3.4 영화 리뷰 분류 : 이진 분류 예제
#IMDB 데이터셋 로드하기
from  keras.datasets import imdb
(train_data,train_labels), (test_data, test_labels) =imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])
print(max([max(sequence) for sequence in train_data]))

# 정수 시퀀스 이진 행렬로 인코딩하기
import numpy as np

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence] = 1.
        return  results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train[0])
#레이블을 벡터로 바꿈
x_train = np.asarray(train_labels).astype('float32')
x_test = np.asarray(test_labels).astype('float32')