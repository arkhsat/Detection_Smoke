# import cv2
# import mediapipe as mp
# import numpy as np
# from ultralytics import YOLO
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# # cap = cv2.VideoCapture(0)
# # cap.set(3, 1280)
# # cap.set(4, 720)
# cap = cv2.VideoCapture("../footage/ASMR smoking a cigarette with you (no talking n nature sounds).mp4")
# # cap = cv2.VideoCapture("../footage/footage 1.mov")
# # cap = cv2.VideoCapture("../footage/firsdenco1.MP4")
# # cap = cv2.VideoCapture("../footage/firstdenco2.MP4")
#
# smoke = None
#
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End
#
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#
#     if angle > 180.0:
#         angle = 360 - angle
#
#     return angle
#
#
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         model_cigarette = YOLO("../YOLO/NEWbest.pt")
#
#         h, w, _ = frame.shape
#
#         model = model_cigarette(frame, stream=True)
#         for models in model:
#             for detection in models.boxes:
#                 x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box
#                 conf = detection.conf[0].item()  # Confidence score
#
#                 if conf > 0.5:  # Ambil objek dengan confidence tinggi
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green bounding box
#                     cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     # boundingbox = frame[y1:y2, x1:x2]
#                     boundingbox_x = int((x1 + x2) / 2)
#                     boundingbox_y = int((y1 + y2) / 2)
#
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#
#         # Make detection
#         results = pose.process(image)
#
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # Extract Landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark
#
#             # Get coordinates
#             shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#
#             shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#             elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#             wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#
#             # New coordinates for cigeratte bounding box
#             mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,
#                       landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
#
#             mouth_x = int((landmarks[9].x + landmarks[10].x) / 2 * w)
#             mouth_y = int((landmarks[9].y + landmarks[10].y) / 2 * h)
#
#             # Calculate angle
#             angle_left = calculate_angle(shoulderL, elbowL, wristL)
#             angle_right = calculate_angle(shoulderR, elbowR, wristR)
#
#
#             # Visualize angle
#             cv2.putText(image, str(angle_left),
#                         tuple(np.multiply(elbowL, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                         )
#
#             cv2.putText(image, str(angle_right),
#                         tuple(np.multiply(elbowR, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
#                         )
#
#             # Logic
#                 # FOR LEFT
#             wristleft = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
#             shoulderleft = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
#             # print(f"wrst l: {wristleft}")
#             # print(f"should l: {shoulderleft}")
#
#             yl = shoulderleft > wristleft
#             # print(yl)
#
#             if angle_left < 160 and yl:
#                 # there are rokok
#                 smoke = "Detected Left"
#                 print(smoke)
#
#                 # FOR RIGHT
#             wristright = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
#             shoulderright = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
#
#             yr = shoulderright > wristright
#             # print(yr)
#
#             if angle_right < 160 and yr:
#                 # there are rokok
#                 smoke = "Detected right"
#                 print(smoke)
#
#             # for mouth
#             zone_size = 40  # setengah lebar zona
#             zone_x1 = mouth_x - zone_size
#             zone_y1 = mouth_y - zone_size
#             zone_x2 = mouth_x + zone_size
#             zone_y2 = mouth_y + zone_size
#
#             if zone_x1 < boundingbox_x < zone_x2 and zone_y1 < boundingbox_y < zone_y2:
#                 print("🚬 Rokok terdeteksi DI DEKAT MULUT")
#
#
#             # cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 255), 2)
#
#         except:
#             pass
#
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                   )
#
#         cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 255), 2)
#         cv2.imshow('Mediapipe Feed', image)
#
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


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

            # JIKA MENGGUNAKAN MULUT SEBAGAI DETEKSI WAJAH
            # Landmark mulut
            # mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
            # mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]

            # Konversi ke pixel
            # mouth_left_x = int(mouth_left.x * w)
            # mouth_left_y = int(mouth_left.y * h)
            # mouth_right_x = int(mouth_right.x * w)
            # mouth_right_y = int(mouth_right.y * h)
            #
            # mouth_x = (mouth_left_x + mouth_right_x) // 2
            # mouth_y = (mouth_left_y + mouth_right_y) // 2
            #
            # # Hitung zona dinamis
            # mouth_width = abs(mouth_right_x - mouth_left_x)
            # zone_size = int(mouth_width * 1.8)  # bisa adjust skala di sini
            #
            # # Zona sekitar mulut
            # zone_x1 = mouth_x - zone_size
            # zone_y1 = mouth_y - zone_size
            # zone_x2 = mouth_x + zone_size
            # zone_y2 = mouth_y + zone_size
            #
            # # Gambar zona
            # cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 255), 2)
            # cv2.circle(frame, (mouth_x, mouth_y), 5, (0, 0, 255), -1)

            # Cek apakah rokok ada dalam zona
            # for center_x, center_y in bboxes:
            #     if angle_left < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
            #         cv2.putText(frame, "🚬 CIG IN MOUTH", (mouth_x - 50, mouth_y - 30),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         print("🚬 Rokok terdeteksi DI DEKAT MULUT (KIRI)")
            #     elif angle_right < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
            #         cv2.putText(frame, "🚬 CIG IN MOUTH", (mouth_x - 50, mouth_y - 30),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         print("🚬 Rokok terdeteksi DI DEKAT MULUT (KANAN)")


            # JIKA MENGGUNAKAN HIDUNG UNTUK DI DETEKSI WAJAH
            # nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            # # left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            # # right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            # nose_x = int(nose.x * w)
            # nose_y = int(nose.y * h)
            # # left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            # # right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
            #
            # # Hitung jarak antar mata
            # # eye_distance = np.linalg.norm([left_eye_x - right_eye_x, left_eye_y - right_eye_y])
            #
            # # Gunakan skala proporsional untuk zone_size
            # # zone_size = int(eye_distance * 1.2)  # 1.2 bisa diatur sesuai sensitivitas
            #
            # nose_width = abs(nose_x - nose_y)
            # zone_size = int(nose_width * 0.3)
            #
            # # Zona deteksi sekitar hidung
            # # zone_size = 40  # Bisa disesuaikan jika perlu
            # zone_x1 = nose_x - zone_size
            # zone_y1 = nose_y - zone_size
            # zone_x2 = nose_x + zone_size
            # zone_y2 = nose_y + zone_size
            #
            # # Gambar zona dan titik pusat hidung
            # cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 255), 2)
            # cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
            #
            # for center_x, center_y in bboxes:
            #     if angle_left < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
            #         cv2.putText(frame, "🚬 CIG IN MOUTH", (nose_x - 50, nose_y - 30),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         print("🚬 Rokok terdeteksi DI DEKAT HIDUNG (KIRI)")
            #     elif angle_right < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
            #         cv2.putText(frame, "🚬 CIG IN MOUTH", (nose_x - 50, nose_y - 30),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #         print("🚬 Rokok terdeteksi DI DEKAT HIDUNG (KANAN)")


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



# UNTUK CEK GAMBAR
# import cv2
# import mediapipe as mp
# import numpy as np
# from ultralytics import YOLO
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# # Load gambar
# # frame = cv2.imread("../image test/128_jpg.rf.62ea621c020216921a273a48956f7830.jpg")
# frame = cv2.imread("../image test/203_jpg.rf.4d102a07191ffad4c26fc9a0ac867c63.jpg")
# if frame is None:
#     raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path sudah benar.")
#
# model_cigarette = YOLO("../YOLO/NEWbest.pt")
#
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     return 360 - angle if angle > 180.0 else angle
#
# # Pose estimation
# with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
#     h, w, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)
#
#     # YOLO detection
#     bboxes = []
#     model_results = model_cigarette(frame)
#     for models in model_results:
#         for detection in models.boxes:
#             x1, y1, x2, y2 = map(int, detection.xyxy[0])
#             conf = detection.conf[0].item()
#             if conf > 0.5:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 bboxes.append(((x1 + x2) // 2, (y1 + y2) // 2))
#
#     try:
#         landmarks = results.pose_landmarks.landmark
#
#         shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#         elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#         wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#
#         shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#         elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#         wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#
#         angle_left = calculate_angle(shoulderL, elbowL, wristL)
#         angle_right = calculate_angle(shoulderR, elbowR, wristR)
#
#         cv2.putText(frame, str(angle_left),
#                     tuple(np.multiply(elbowL, [w, h]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#
#         cv2.putText(frame, str(angle_right),
#                     tuple(np.multiply(elbowR, [w, h]).astype(int)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
#
#         mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
#         mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]
#         mouth_x = int((mouth_left.x + mouth_right.x) / 2 * w)
#         mouth_y = int((mouth_left.y + mouth_right.y) / 2 * h)
#         mouth_width = abs(int((mouth_right.x - mouth_left.x) * w))
#         zone_size = int(mouth_width * 1.8)
#
#         zone_x1 = mouth_x - zone_size
#         zone_y1 = mouth_y - zone_size
#         zone_x2 = mouth_x + zone_size
#         zone_y2 = mouth_y + zone_size
#
#         cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 255), 2)
#         cv2.circle(frame, (mouth_x, mouth_y), 5, (0, 0, 255), -1)
#
#         for center_x, center_y in bboxes:
#             if angle_left < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
#                 cv2.putText(frame, "🚬 CIG IN MOUTH (L)", (mouth_x - 50, mouth_y - 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             elif angle_right < 160 and zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:
#                 cv2.putText(frame, "🚬 CIG IN MOUTH (R)", (mouth_x - 50, mouth_y - 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     except:
#         pass
#
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#         )
#
#     cv2.imshow('Cigarette Detection (Image)', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
