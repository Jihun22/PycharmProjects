import folium as folium
import pandas as pd

CB = pd.read_csv("C:/Users/ik533/Desktop/머신러닝프로젝트/9장_data/CoffeeBean.csv", encoding='CP949', index_col=0, header=0,
                 engine='python')

print("CB:",CB.head)  # 작업 내용 확인용 출력

addr = []
for address in CB.address:
    addr.append(str(address).split())
print("Addr:",addr) #작업내용 확인용 출력

addr2 = []
for i in range(len(addr)):
    if addr[i][0] == "서울":addr[i][0] = "서울특별시"
    elif addr[i][0] == "서울시":
        addr[i][0] = "서울특별시"
    elif addr[i][0] == "부산시":
        addr[i][0] = "부산광역시"
    elif addr[i][0] == "인천":
        addr[i][0] = "인천광역시"
    elif addr[i][0] == "광주":
        addr[i][0] = "광주광역시"
    elif addr[i][0] == "대전시":
        addr[i][0] = "대전광역시"
    elif addr[i][0] == "울산시":
        addr[i][0] = "울산광역시"
    elif addr[i][0] == "세종시":
        addr[i][0] = "세종특별자치시"
    elif addr[i][0] == "경기":
        addr[i][0] = "경기도"
    elif addr[i][0] == "충북":
        addr[i][0] = "충청북도"
    elif addr[i][0] == "충남":
        addr[i][0] = "충청남도"
    elif addr[i][0] == "전북":
        addr[i][0] = "전라북도"
    elif addr[i][0] == "전남":
        addr[i][0] = "전라남도"
    elif addr[i][0] == "경북":
        addr[i][0] = "경상북도"
    elif addr[i][0] == "경남":
        addr[i][0] = "경상남도"
    elif addr[i][0] == "제주":
        addr[i][0] = "제주특별자치도"
    elif addr[i][0] == "제주도":
        addr[i][0] = "제주특별자치도"
    elif addr[i][0] == "제주시":
        addr[i][0] = "제주특별자치도"
    addr2.append(' '.join(addr[i]))
print("Addr2:",addr2)  # 작업 내용 확인용 출력

addr2 = pd.DataFrame(addr2 , columns=['address2'])
CB2 = pd.concat([CB,addr2], axis =1)
print("CB2:",CB2.head)
CB2.to_csv('C:/Users/ik533/Desktop/머신러닝프로젝트/9장_data/CoffeeBean_3.csv',encoding ='CP949',index = False)

map_osm = folium.Map(location = [37.559978,126.975291], zoom_start = 16)
map_osm.save('C:/Users/ik533/Desktop/머신러닝프로젝트/9장_data/map3.html')

CB_geoData = pd.read_csv('C:/Users/ik533/Desktop/머신러닝프로젝트/9장_data/CB_geo2.shp.csv',encoding='cp949',engine='python')
map_CB = folium.Map(location= [37.560284,126.975334],zoom_start=15)
for i, store in CB_geoData.iterrows():
     folium.Marker(location = [store['위도'], store['경도']], popup =
                        store['store'], icon = folium.Icon(color = 'red',
                        icon = 'star')).add_to(map_CB)

map_CB.save('C:/Users/ik533/Desktop/머신러닝프로젝트/9장_data/map_CB3.html')