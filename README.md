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
```

### 📦 Install dependencies:
It's highly recommended to create a requirements.txt file containing all necessary libraries. If you don't have one, you'll need to install the following common libraries manually:

```bash
pip install ultralytics opencv-python python-telegram-bot
```
(Note: ultralytics provides YOLOv8, opencv-python for video processing, and python-telegram-bot for notifications. You might need additional packages depending on your specific model and data handling.)


## Configuration
Before running the application, you'll need to configure your Telegram bot token and chat ID.
1. **Telegram Bot Setup:**
   - **Create a new bot via BotFather on Telegram and obtain your Bot Token.**
   - **Start a chat with your bot and then find your Chat ID (you can use a bot like @userinfobot to get your chat ID).**
   - **Open telegram.py (or the relevant script where Telegram API is used) and replace placeholders with your actual Bot Token and Chat ID.**
3. **Database:**
   The violations.db SQLite database will be created automatically if it doesn't exist when databases.py is initialized.


### 📁 Project Structure
- **mainwithtest.py: The primary script for running the smoking violation detection system, potentially including testing functionalities.**
- **newmain.py: An alternative or updated main script.**
- **telegram.py: Handles sending notifications to Telegram.**
- **databases.py: Manages interactions with the SQLite database (violations.db).**
- **violations.db: The SQLite database file where detection logs are stored.**
- **YOLO/: Contains configurations, weights, or scripts related to the YOLOv8 model.**
- **PoseEstimation/: Contains code or models specifically for pose estimation, used for smoking gesture detection.**
- **footage/: A directory for storing video files to be used for testing the detection system.**
- **.idea/: (Usually ignored) Contains configuration files for PyCharm or IntelliJ IDEA.**

### 📞 Contact
For any questions or inquiries, please open an issue on this repository or contact [arkhananta37@gmail.com].
