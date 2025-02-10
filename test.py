# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
#
# cam = cv2.VideoCapture(0)
#
# cam.set(3, 1280)
# cam.set(4, 720)
#
# cap = cv2.VideoCapture("videoplayback.mp4")
#
# model = YOLO("best.pt")
#
# # Object detection
# object_detector = cv2.createBackgroundSubtractorMOG2()
#
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#
#     result = model(img, stream=True)
#
#     # Convert to grayscale
#     # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Convert to obejct_detector cv2
#     mask = object_detector.apply(img)
#
#     # Show normal and grayscale images
#     cv2.imshow("Camera - Normal", img)
#     # cv2.imshow("Camera - Grayscale", gray_img)
#     cv2.imshow("Camera - Mask", mask)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cam.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import cvzone
import math

cam = cv2.VideoCapture(0)

cam.set(3, 1280)
cam.set(4, 720)

cap = cv2.VideoCapture("videoplayback.mp4")

model = YOLO("best.pt")

# Object detection
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            conf = detection.conf[0].item()  # Confidence score

            if conf > 0.5:  # Threshold to filter low-confidence detections
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green bounding box
                cv2.putText(img, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert to object_detector cv2
    mask = object_detector.apply(img)

    # Show normal and mask images
    cv2.imshow("Camera - Normal", img)
    cv2.imshow("Camera - Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
