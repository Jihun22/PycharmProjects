import RPi.GPIO as gpio
import Adafruit_DHT
import time
import warnings
warnings.filterwarnings('ignore')
# ref: https://blog.naver.com/dnjswns2280/221402960390

DAT = 11

gpio.cleanup()
gpio.setmode(gpio.BCM)

while True:
    humidity, temperature = Adafruit_DHT.read_retry(22,DAT)
    humid = round(humidity, 1)
    temp = round(temperature,1)
    print(temp,humid)

    time.sleep(2)