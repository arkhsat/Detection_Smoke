# 🚭 Smoking Violation Detection System

This repository hosts a **real-time system** designed to detect smoking violations in designated no-smoking areas. It combines **smoke detection** using YOLOv8 and **pose estimation** (for smoking gestures via MediaPipe). Detected violations are logged into an **SQLite database** and trigger **instant Telegram notifications**.

---

## ✨ Features

- **Comprehensive Violation Detection**  
  Integrates both smoke detection and smoking gesture recognition for accurate identification.

- **Real-time Smoke Detection**  
  Utilizes YOLOv8 for smoke detection from video streams or image inputs.

- **Smoking Gesture Recognition via Pose Estimation**  
  Uses MediaPipe to detect human gestures commonly associated with smoking (e.g., hand-to-mouth movement).

- **SQLite Logging**  
  Automatically logs detected smoking events and gestures into a local `violations.db` file.

- **Telegram Notifications**  
  Sends real-time alerts with detection details to a configured Telegram chat.

- **Modular Code**  
  Organized and commented Python scripts for easy maintenance and extension.

---

## ⚙️ Installation

### ✅ Prerequisites

- Python 3.10+
- pip (Python package installer)

### 📦 Clone the Repository

```bash
git clone https://github.com/arkhsat/Detection_Smoke.git
cd Detection_Smoke
