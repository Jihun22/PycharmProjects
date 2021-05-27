#케라스 3.4 영화 리뷰 분류 : 이진 분류 예제
#IMDB 데이터셋 로드하기
from  keras.datasets import imdb
(train_data,train_labels), (test_data, test_labels) =imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])
print(max([max(sequence) for sequence in train_data]))