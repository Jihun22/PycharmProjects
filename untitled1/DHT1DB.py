import RPi.GPIO as gpio
import Adafruit_DHT
import time
import urllib.request
import pymysql
import  urllib.request

##########마리아 DB#############
def insertDB(temp,humid):
    conn = pymysql.connect(host='localhost', user='JH', password='0019', db='tsst', charset='utf8')

    with conn.cursor() as cursor:
        sql = 'insert into ss(temp, humid) values(%s, %s)'
        cnt = cursor.execute(sql,(humid,temp))
        r = conn.commit()

        if r == 0:
            print("Failed")
        else:
            print("Save Ok")

    conn.close()

###############IOT클라우드#############
def insertCloud(temp,humid):

    api_key ="VEDC88WQ8YZDN3FN"
    url = 'https://api.thingspeak.com/update'
    url = url + '?api_key=%s' % api_key
    url = url + '&field1=%s' % temp
    url = url + '&field2=%s' % humid


    #print(url)
    urllib.request.urlopen(url)



# ref: https://blog.naver.com/dnjswns2280/221402960390
#########################################################


####################################
DAT = 11

gpio.cleanup()
gpio.setmode(gpio.BCM)

while True:
    humidity, temperature = Adafruit_DHT.read_retry(22,DAT)
    humid = round(humidity, 1)
    temp = round(temperature,1)
    print(temp,humid)
    insertDB(temp, humid)
    insertCloud(temp,humid)
    time.sleep(15)