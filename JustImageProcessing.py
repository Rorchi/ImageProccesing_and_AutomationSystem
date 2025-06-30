#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from pathlib import Path

# .blob model dosyasının yolu
nnPath = str((Path(__file__).parent / "/home/pi/Desktop/KOD/result/best.blob").resolve())

if not Path(nnPath).exists():
    raise FileNotFoundError(f"Model bulunamadı: {nnPath}")

# Etiketler
labelMap = ["coal", "stone", "unknown"]

# Kamera çözünürlüğü ve ayarlar
syncNN = True
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
# frame= cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Removed because 'frame' is not defined yet

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

# Raspberry Pi'de cihaz ile pipeline başlatılıyor
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

        cv2.putText(frame, f"FPS: {counter / (time.monotonic() - startTime):.2f}",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        displayFrame("OAK Kamera - Nesne Tespiti", frame)

        if cv2.waitKey(1) == ord('q'):
            break