🚭 Smoking Violation Detection System
This repository hosts a real-time system designed to detect smoking violations in designated no-smoking areas. It achieves this by combining smoke detection with pose estimation to identify smoking gestures. The system leverages YOLOv8 for object detection, logs detected events and potential violations into an SQLite database, and sends instant notifications via Telegram.

✨ Features
Comprehensive Violation Detection: Integrates both smoke detection and smoking gesture recognition to accurately identify smoking violations.

Real-time Smoke Detection: Utilizes advanced deep learning models (YOLOv8) to detect smoke in video streams or image inputs.

Smoking Gesture Detection via Pose Estimation: Employs pose estimation techniques to identify specific human gestures associated with smoking, enhancing detection accuracy and reducing false positives.

YOLOv8 Integration: Seamlessly integrates with the powerful YOLOv8 object detection framework for efficient and accurate detection of smoke and potentially other relevant objects.

SQLite Database Logging: Automatically logs detected smoking events, gestures, and potential violations into a local SQLite database (violations.db).

Telegram Notifications: Sends instant alerts and relevant information to a configured Telegram chat upon detecting a smoking violation.

Modular Python Code: Organized and commented Python scripts for easy understanding and extension.

🚀 Getting Started
Follow these steps to get your smoking violation detection system up and running.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/arkhsat/Detection_Smoke.git
cd Detection_Smoke

Create a virtual environment (recommended):

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install dependencies:
It's highly recommended to create a requirements.txt file containing all necessary libraries. If you don't have one, you'll need to install the following common libraries manually:

pip install ultralytics opencv-python python-telegram-bot

(Note: ultralytics provides YOLOv8, opencv-python for video processing, and python-telegram-bot for notifications. You might need additional packages depending on your specific model and data handling.)

Configuration
Before running the application, you'll need to configure your Telegram bot token and chat ID.

Telegram Bot Setup:

Create a new bot via BotFather on Telegram and obtain your Bot Token.

Start a chat with your bot and then find your Chat ID (you can use a bot like @userinfobot to get your chat ID).

Open telegram.py (or the relevant script where Telegram API is used) and replace placeholders with your actual Bot Token and Chat ID.

# Example in telegram.py (adjust as per your actual code)
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

Database:
The violations.db SQLite database will be created automatically if it doesn't exist when databases.py is initialized.

Usage
To run the smoking violation detection system, execute the main script.

python mainwithtest.py
# Or, if newmain.py is your primary entry point:
# python newmain.py

The system will start processing video input (likely from a webcam or specified video files in the footage folder).

Detected smoking violations (based on smoke and/or gesture detection) will be logged and notifications sent to Telegram.

📁 Project Structure
mainwithtest.py: The primary script for running the smoking violation detection system, potentially including testing functionalities.

newmain.py: An alternative or updated main script.

telegram.py: Handles sending notifications to Telegram.

databases.py: Manages interactions with the SQLite database (violations.db).

violations.db: The SQLite database file where detection logs are stored.

YOLO/: Contains configurations, weights, or scripts related to the YOLOv8 model.

PoseEstimation/: Contains code or models specifically for pose estimation, used for smoking gesture detection.

footage/: A directory for storing video files to be used for testing the detection system.

.idea/: (Usually ignored) Contains configuration files for PyCharm or IntelliJ IDEA.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Make your changes.

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeature).

Open a Pull Request.

📄 License
This project is open-source and available under the MIT License. (You may want to add a LICENSE file to your repository if you haven't already).

📞 Contact
For any questions or inquiries, please open an issue on this repository or contact [Your Name/Email/GitHub Profile Link].
