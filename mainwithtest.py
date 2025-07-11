import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from databases import init_db, log_violation
from telegram import send_warning, start_bot_in_thread


# Hitung sudut gesture merokok
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Deteksi orang dan ambil ROI-nya
def detectedroi_person(frame, model_person):
    results = model_person(frame, conf=0.7)
    rois = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:  # hanya orang
                x1, y1, x2, y2 = map(int, box[:4])
                roi = frame[y1:y2, x1:x2].copy()
                rois.append(roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, rois

# Proses deteksi dalam ROI
def process_roi(region, model_cigarette, pose, mp_pose, mp_drawing, full_frame, state):
    h, w, _ = region.shape
    bboxes = []

    # Deteksi rokok & asap
    result = model_cigarette(region, conf=0.7) #conf = 8
    for res in result:
        for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            if int(cls) in [0, 2]:  # Rokok / Vape
                cv2.rectangle(region, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(region, "Cigarette", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                state['cig_detected'] = True
                print("🚬 Cigarette detected!")
                bboxes.append(((x1 + x2) // 2, (y1 + y2) // 2))  # simpan center point

            elif int(cls) == 1:  # Asap
                cv2.rectangle(region, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(region, "Smoke", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                state['smoke_detected'] = True
                print("💨 Smoke detected!")

    if not state['cig_detected']:
        return region, state

    # Resize ROI untuk pose
    # resized = cv2.resize(region, (300, 300))
    image_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results_pose = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark

        try:
            def get_points(side):
                return [
                    [lm[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x,
                     lm[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y],
                    [lm[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].x,
                     lm[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].y],
                    [lm[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].x,
                     lm[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].y]
                ]

            left_eye = lm[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
            left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
            nose = lm[mp_pose.PoseLandmark.NOSE.value]

            # Konversi ke piksel
            points = []
            for landmark in [left_eye, right_eye, left_ear, right_ear, nose]:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))

            # Dapatkan bounding box wajah dari titik-titik tersebut
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            face_x1, face_y1 = min(xs), min(ys)
            face_x2, face_y2 = max(xs), max(ys)

            # Bisa tambahkan margin agar tidak terlalu ketat
            margin = int(0.7 * (face_x2 - face_x1))  # misalnya 30% margin

            zone_x1 = face_x1 - margin
            zone_y1 = face_y1 - margin
            zone_x2 = face_x2 + margin
            zone_y2 = face_y2 + margin

            # Gambar zona wajah
            cv2.rectangle(image, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)

            for side in ["LEFT", "RIGHT"]:
                shoulder, elbow, wrist = get_points(side)
                angle = calculate_angle(shoulder, elbow, wrist)
                for center_x, center_y in bboxes:
                    if angle < 25 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
                        print("👌🏻 Rokok terdeteksi DI DEKAT MULUT")
                        state['pose_detected'] = True

        except Exception as e:
            print(f"Pose error: {e}")

        # Gambar landmark
        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        if state['cig_detected'] and state['pose_detected'] and state['smoke_detected']:
            if not state['already_logged']:
                print("🚨 Smoking Violation Confirmed! Logging to DB...")
                log_violation(full_frame)
                start_bot_in_thread()
                send_warning()
                state['already_logged'] = True

    return image, state  # hasil ROI yang di-pose estimation

# === MAIN === #
model_person = YOLO("YOLO/yolov8n.pt")
model_cigarette = YOLO("YOLO/NEWbest.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("footage/firsdenco1.MP4")
# cap = cv2.VideoCapture("fortest/Video_test/2340435-hd_1920_1080_30fps.mp4")
# cap = cv2.VideoCapture("fortest/Video_test/4786187-uhd_4096_2160_25fps.mp4")
# cap = cv2.VideoCapture("fortest/Video_test/new6.mp4")
# cap = cv2.VideoCapture("fortest/Video_test/1 (1).mp4")

# cap = cv2.VideoCapture("footage/ASMR smoking a cigarette with you (no talking n nature sounds).mp4")

init_db()

roi_states = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (1280, 720))

        frame, rois = detectedroi_person(frame, model_person)

        while len(roi_states) < len(rois):
            roi_states.append({'cig_detected': False, 'pose_detected': False, 'smoke_detected': False, 'already_logged': False})

        updated_rois = []
        for idx, roi in enumerate(rois):
            if roi.size > 0:
                updated_roi_img, _ = process_roi(roi, model_cigarette, pose, mp_pose, mp_drawing, frame.copy(), roi_states[idx])
                updated_rois.append(updated_roi_img)

        cv2.imshow("Tracking", frame)
        for idx, roi in enumerate(updated_rois):
            cv2.imshow(f"ROI {idx + 1}", roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()