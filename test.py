from ultralytics import YOLO
import cv2


cam = cv2.VideoCapture(0)

cam.set(3, 1280)
cam.set(4, 720)

model = YOLO("../YOLO-WEIGHT/yolov8n.pt")

while True:
    success, img = cam.read()
    result = model(img, stream=True)

    cv2.imshow("camera", img)
    cv2.waitKey(1)

