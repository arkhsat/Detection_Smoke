import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from databases import init_db, log_violation
from telegram import send_warning, start_bot_in_thread

# Counting the gesture
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


# Detection person and takes the ROI (Region of Interest) on each person
def detectedroi_person(frame, model_person):
    results = model_person(frame, conf=0.7)
    rois = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                roi = frame[y1:y2, x1:x2].copy()
                rois.append(roi)

    return frame, rois

# Detection cigarette and gesture on every ROI
def process_roi(region, model_cigarette, pose, mp_pose, mp_drawing, full_frame):
    cig_detected = False
    pose_detected = False
    smoke_detected = False

    h, w, _ = region.shape
    bboxes = []

    # Detection cigarette/e-cigarette
    result = model_cigarette(region, conf=0.5)
    for res in result:
        for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
            if int(cls) in [0, 2]:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(region, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(region, "Cigarette", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("🚬 Cigarette detected!")
                bboxes.append(((x1 + x2) // 2, (y1 + y2) // 2))
                cig_detected = True

            elif int (cls) == 1:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(region, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(region, "Smoke", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
                print("💨 Smoke detected!")
                smoke_detected = True

    # For pose estimation
    if not cig_detected:
        return region

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results_pose = pose.process(image_rgb)

    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw pose dan detection gesture
    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark

        try:
            # Left and Right
            points = lambda side: [
                [lm[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x,
                 lm[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y],
                [lm[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].x,
                 lm[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].y],
                [lm[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].x,
                 lm[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].y]
            ]

            # For mouth
            mouth_left = lm[mp_pose.PoseLandmark.MOUTH_LEFT.value]
            mouth_right = lm[mp_pose.PoseLandmark.MOUTH_RIGHT.value]

            # Conversion to pixel
            mouth_left_x = int(mouth_left.x * w)
            mouth_left_y = int(mouth_left.y * h)
            mouth_right_x = int(mouth_right.x * w)
            mouth_right_y = int(mouth_right.y * h)

            mouth_x = (mouth_left_x + mouth_right_x) // 2
            mouth_y = (mouth_left_y + mouth_right_y) // 2

            # Hitung zona dinamis
            mouth_width = abs(mouth_right_x - mouth_left_x)
            zone_size = int(mouth_width * 1.8)  # bisa adjust skala di sini

            # Zona sekitar mulut
            zone_x1 = mouth_x - zone_size
            zone_y1 = mouth_y - zone_size
            zone_x2 = mouth_x + zone_size
            zone_y2 = mouth_y + zone_size

            for side in ["LEFT", "RIGHT"]:
                shoulder, elbow, wrist = points(side)
                angle = calculate_angle(shoulder, elbow, wrist)
                for center_x, center_y in bboxes:
                    if angle < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
                        print("👌🏻 Rokok terdeteksi DI DEKAT MULUT (KIRI)")
                        pose_detected = True

        except Exception as e:
            print(f"Pose error: {e}")

        # Drawing pose
        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


        if cig_detected and pose_detected and smoke_detected:
            print("🚨 Smoking Violation Confirmed! Logging to DB...")
            log_violation(full_frame)
            start_bot_in_thread()
            send_warning()


        # if cig_detected:
        #     print("🚨 Smoking Violation Confirmed! Logging to DB...")
        #     log_violation(full_frame)
        #     start_bot_in_thread()
        #     send_warning()

    # Update ROI with pose
    # region[:, :] = image
    # return region
    return image


# Model
# YOLO MODEL
model_person = YOLO("YOLO/yolov8n.pt")
# model_cigarette = YOLO("YOLO/best.pt")
# model_cigarette = YOLO("YOLO/cigeratte_and_smoke.pt") # second model
model_cigarette = YOLO("YOLO/NEWbest.pt") # third model

# Media pose MODEL
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Source Videos
# Footage smoking
cap = cv2.VideoCapture("footage/ASMR smoking a cigarette with you (no talking n nature sounds).mp4")
# cap = cv2.VideoCapture("footage/Vape.mp4")

# Footage from CCTV
# cap = cv2.VideoCapture("footage/footage 1.mov")
# cap = cv2.VideoCapture("footage/firsdenco1.MP4")
# cap = cv2.VideoCapture("footage/firstdenco2.MP4")

# for test if there are a lot of person in camera
# cap = cv2.VideoCapture("footage/people.mp4")

# For Webcame
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

init_db() # databases

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, rois = detectedroi_person(frame, model_person)

        updated_rois = []
        for roi in rois:
            if roi.size > 0:
                updated_roi = process_roi(roi, model_cigarette, pose, mp_pose, mp_drawing, frame.copy())
                updated_rois.append(updated_roi)

        # Display in main frame
        cv2.imshow("Tracking", frame)

        # Display all Region of interest (ROI)
        for idx, roi in enumerate(updated_rois):
            cv2.imshow(f"ROI {idx + 1}", roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
