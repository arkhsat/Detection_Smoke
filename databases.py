import sqlite3
import os
from datetime import datetime
import cv2

# Folder for capture
if not os.path.exists("captures"):
    os.makedirs("captures")

# Database
def init_db():
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()

    # # Hapus tabel jika sudah ada
    # c.execute('DROP TABLE IF EXISTS violations')

    c.execute('''CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    timestamp TEXT,
                    image_path TEXT
                )''')
    conn.commit()
    conn.close()


# Simpan data pelanggaran
def log_violation(image):
    date = datetime.now().strftime("%Y - %m - %d")
    timestamp = datetime.now().strftime("%H : %M : %S")
    fordb = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"captures/violation - {fordb}.jpg"
    cv2.imwrite(filename, image)

    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("INSERT INTO violations (date, timestamp, image_path) VALUES (?, ?, ?)",
              (date, timestamp, filename))
    conn.commit()
    conn.close()