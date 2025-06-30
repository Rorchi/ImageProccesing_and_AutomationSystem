import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)

# 22. pine 1 saniye boyunca HIGH sinyali gönder
GPIO.output(21, GPIO.HIGH)
time.sleep(2)
GPIO.output(21, GPIO.LOW)
time.sleep(1)
 
# 22. pine LOW sinyali gönder



GPIO.cleanup()