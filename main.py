# from ultralytics import YOLO
# import cv2
#
# # Camera setup
# # cam = cv2.VideoCapture("ASMR smoking a cigarette with you (no talking n nature sounds).mp4")
# cam = cv2.VideoCapture(0)
# cam.set(3, 1280)
# cam.set(4, 720)
#
# # YOLO declare model
# model = YOLO("best.pt")
#
# # Initialize CSRT Tracker
# tracker = cv2.TrackerCSRT.create()
# tracking = False  # Status apakah sedang tracking
#
# while True:
#     success, img = cam.read()
#     if not success:
#         break
#
#     if not tracking:
#         # Jika tidak sedang tracking, lakukan deteksi dengan YOLO
#         results = model(img, stream=True, conf=0.5)
#
#         for result in results:
#             for detection in result.boxes:
#                 x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
#                 conf = detection.conf[0].item()  # Confidence score
#
#                 if conf > 0.5:
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green bounding box
#                     cv2.putText(img, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     bbox = (x1, y1, x2 - x1, y2 - y1)  # Format untuk tracker (x, y, w, h)
#                     tracker.init(img, bbox)  # Inisialisasi tracker
#                     tracking = True  # Aktifkan tracking
#                     break  # Keluar setelah deteksi pertama untuk menghindari multi-tracking
#
#     else:
#         # Jika sudah tracking, update objek
#         success, bbox = tracker.update(img)
#         if success:
#             x, y, w, h = map(int, bbox)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Blue bounding box
#             cv2.putText(img, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         else:
#             tracking = False  # Jika tracking gagal, kembali ke deteksi YOLO
#
#     # Display the cam
#     cv2.imshow("Camera - CSRT Tracking", img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cam.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2

# Load Model YOLO
# model = YOLO("best.pt")
model = YOLO("cigeratte_and_smoke.pt")

# Buka Kamera atau Video
# cap = cv2.VideoCapture(0)  # Ganti dengan "video.mp4" jika dari file
# cape = cv2.VideoCapture("ASMR smoking a cigarette with you (no talking n nature sounds).mp4")
cape = cv2.VideoCapture("Vape.mp4")

# Inisialisasi daftar tracker CSRT
trackers = []
frame_count = 0
detect_interval = 10  # Interval deteksi ulang (misalnya setiap 10 frame)

while True:
    # success, img = cap.read()
    success, img = cape.read()
    if not success:
        break

    frame_count += 1

    if frame_count % detect_interval == 0 or len(trackers) == 0:
        # Reset trackers setiap interval deteksi ulang
        trackers = []

        results = model(img, stream=True)
        for result in results:
            for detection in result.boxes:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box
                conf = detection.conf[0].item()  # Confidence score

                if conf > 0.5:  # Ambil objek dengan confidence tinggi
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green bounding box
                    cv2.putText(img, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    tracker = cv2.TrackerCSRT.create()  # Buat CSRT tracker baru
                    bbox = (x1, y1, x2 - x1, y2 - y1)  # Ubah ke format (x, y, w, h)
                    tracker.init(img, bbox)  # Inisialisasi tracker dengan bounding box
                    trackers.append(tracker)

    # Update semua tracker
    for tracker in trackers:
        success, bbox = tracker.update(img)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(img, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Tampilkan hasil tracking
    cv2.imshow("Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cape.release()
cv2.destroyAllWindows()