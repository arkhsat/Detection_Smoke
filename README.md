# 🚭 Smoking Violation Detection System
This repository hosts a real-time system designed to detect smoking violations in designated no-smoking areas. It achieves this by combining smoke detection with pose estimation to identify smoking gestures. The system leverages YOLOv8 for object detection, logs detected events and potential violations into an SQLite database, and sends instant notifications via Telegram.

## ✨ Features
- **Comprehensive Violation Detection: Integrates both smoke detection and smoking gesture recognition to accurately identify smoking violations.
- **Real-time Smoke Detection: Utilizes advanced deep learning models (YOLOv8) to detect smoke in video streams or image inputs.
- **Smoking Gesture Detection via Pose Estimation: Employs pose estimation techniques to identify specific human gestures associated with smoking, enhancing detection accuracy and reducing false positives.
- **YOLOv8 Integration: Seamlessly integrates with the powerful YOLOv8 object detection framework for efficient and accurate detection of smoke and potentially other relevant objects.
- **SQLite Database Logging: Automatically logs detected smoking events, gestures, and potential violations into a local SQLite database (violations.db).
- **Telegram Notifications: Sends instant alerts and relevant information to a configured Telegram chat upon detecting a smoking violation. Modular Python Code: Organized and commented Python scripts for easy understanding and extension.

## Installation

### Prerequisites
- **Python 3.10+
- **pip (Python package installer)

####1. Clone the repository:
```bash
git clone https://github.com/arkhsat/Detection_Smoke.git
cd Detection_Smoke


####2. Install dependencies:
```bash
pip install ultralytics opencv-python python-telegram-bot
