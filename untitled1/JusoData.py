import pandas as pd

CB = pd.read_csv("C:/Users/ik533/Desktop/머신러닝프로젝트/9장_data/CoffeeBean.csv", encoding='CP949', index_col=0, header=0,
                 engine='python')

print(CB.head)  # 작업 내용 확인용 출력

addr = []
for address in CB.address:
    addr.append(str(address).split())
print(addr) #작업내용 확인용 출력