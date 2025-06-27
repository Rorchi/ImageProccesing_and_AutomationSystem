#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path
import RPi.GPIO as GPIO
import math

# PWM ayarları
GPIO.setmode(GPIO.BCM)
servo_pin_1 = 17
servo_pin_2 = 18
servo_pin_3 = 27
dc_motor_pin = 21

GPIO.setup(servo_pin_1, GPIO.OUT)
GPIO.setup(servo_pin_2, GPIO.OUT)
GPIO.setup(servo_pin_3, GPIO.OUT)
GPIO.setup(dc_motor_pin, GPIO.OUT)

servo1 = GPIO.PWM(servo_pin_1, 50)
servo2 = GPIO.PWM(servo_pin_2, 50)
servo3 = GPIO.PWM(servo_pin_3, 50)
dc_pwm = GPIO.PWM(dc_motor_pin, 100)

servo1.start(0)
servo2.start(0)
servo3.start(0)
dc_pwm.start(0)

def set_angle(servo, angle):
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

def rotate_motor(power_percent):
    dc_pwm.ChangeDutyCycle(power_percent)
    time.sleep(1)
    dc_pwm.ChangeDutyCycle(0)

# Blob modeli
nnPath = str((Path(__file__).parent / "/home/pi/Desktop/KOD/result/best.blob").resolve())
if not Path(nnPath).exists():
    raise FileNotFoundError(f"Model bulunamadı: {nnPath}")

labelMap = ["coal", "stone", "unknown"]
syncNN = True
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

camRgb.setPreviewSize(320, 320)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

detectionNetwork.setConfidenceThreshold(0.6)
detectionNetwork.setNumClasses(3)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            label = labelMap[detection.label]
            conf = int(detection.confidence * 100)

            cv2.putText(frame, f"{label} {conf}%", (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.imshow(name, frame)

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        frame = inRgb.getCvFrame()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        detections = inDet.detections
        counter += 1

        for detection in detections:
            if labelMap[detection.label] == "coal":
                x_center = (detection.xmin + detection.xmax) / 2
                y_center = (detection.ymin + detection.ymax) / 2
                x_pixel = int(x_center * frame.shape[1])
                y_pixel = int(y_center * frame.shape[0])
                print(f"Koordinatlar: ({x_pixel}, {y_pixel})")

                # Ölçekleme faktörü (cm/piksel)
                scale_x = 0.2
                scale_y = 0.2
                X_real = x_pixel * scale_x
                Y_real = y_pixel * scale_y

                # Servo kontrol açıları kinematik denklemlere göre hesaplanıyor
                X_real = min(max(X_real, -8), 8)  # X koordinatını -8 ile 8 arasında sınırla
                Y_real = min(max(Y_real, -8), 8)
                set_angle(servo1, 7*math.cos(math.radians(Y_real))+ 8*math.sin(math.radians(X_real)))
                set_angle(servo2, X_real)
                set_angle(servo3, 60)

                rotate_motor(70)  # DC motoru döndür

        cv2.putText(frame, f"FPS: {counter / (time.monotonic() - startTime):.2f}",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        displayFrame("OAK Kamera - Nesne Tespiti", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
GPIO.cleanup()