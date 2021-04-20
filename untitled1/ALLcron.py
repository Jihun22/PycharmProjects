import urllib.request
import urllib.parse
from bs4 import BeautifulSoup

#세팅
baseUrl ='https://search.naver.com/search.naver?where=post&sm=tab_jum&query='
plusUrl = input('검색어를 입력하세요:')
url =baseUrl+urllib.parse.quote_plus(plusUrl) #한글, 영어 가져오는 세팅
html = urllib.request.urlopen(url).read() #html 가져오기
soup = BeautifulSoup(html,'html.parser')

#변수 지정
title = soup.find_all(class_='sh_blog_title')

for i in title:
    print(i.attrs['title'])
    print(i.attrs['href'])
    print()
