import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)

# Kullanılacak pinler
servo_pins = [17, 18, 27]

# GPIO ayarları
GPIO.setmode(GPIO.BCM)
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)

# Her servo için PWM başlat (50Hz)
servos = [GPIO.PWM(pin, 50) for pin in servo_pins]

# PWM başlat
for servo in servos:
    servo.start(0)

def set_servo_angle(servo, angle):
    duty = 2.5 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)  # Servo titremesini önlemek için

try:
    # Her servo için farklı açı belirle
    angles = [90, 180, 0]  # 17, 18, 27 numaralı pinlere bağlı servolar için açı değerleri
    for servo, angle in zip(servos, angles):
        set_servo_angle(servo, angle)
    time.sleep(1)
finally:
    for servo in servos:
        servo.stop()
    GPIO.cleanup()