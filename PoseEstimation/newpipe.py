import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../footage/ASMR smoking a cigarette with you (no talking n nature sounds).mp4")
# cap = cv2.VideoCapture("../fortest/Video_test/2340435-hd_1920_1080_30fps.mp4")
# cap = cv2.VideoCapture("../fortest/Video_test/old4.mp4")

model_cigarette = YOLO("../YOLO/NEWbest.pt")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # YOLO detection
        bboxes = []
        model_results = model_cigarette(frame, stream=True)
        for models in model_results:
            for detection in models.boxes:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                conf = detection.conf[0].item()
                if conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    bboxes.append(((x1 + x2) // 2, (y1 + y2) // 2))  # simpan center point

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle_left = calculate_angle(shoulderL, elbowL, wristL)
            angle_right = calculate_angle(shoulderR, elbowR, wristR)

            # Visualize angle
            cv2.putText(frame, str(angle_left),
                        tuple(np.multiply(elbowL, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(frame, str(angle_right),
                        tuple(np.multiply(elbowR, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
                        )

            if angle_left < 25:
                print("Gerakan")

            if angle_right < 25:
                print("Gerakan kedua")

            # JIKA MENGGUNAKAN WAJAH SEBAGAI DETEKSI ROKOK
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

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
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)

            for center_x, center_y in bboxes:
                if angle_left < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
                    # cv2.putText(frame, "🚬 CIG IN MOUTH", (nose_x - 50, nose_y - 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("🚬 Rokok terdeteksi DI DEKAT HIDUNG (KIRI)")
                elif angle_right < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
                    # cv2.putText(frame, "🚬 CIG IN MOUTH", (nose_x - 50, nose_y - 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("🚬 Rokok terdeteksi DI DEKAT HIDUNG (KANAN)")

        except:
            pass

        # Render pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # frame_resized = cv2.resize(frame, (1280, 720))

        # cv2.imshow('Cigarette Detection', frame_resized)
        cv2.imshow('Cigarette Detection', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
