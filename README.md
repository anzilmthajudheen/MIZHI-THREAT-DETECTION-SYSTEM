## MIZHI-THREAT-DETECTION-SYSTEM

Mizhi: Enhanced Weapon Threat Detection System
Mizhi is an advanced real-time security surveillance system designed to detect weapons and suspicious behavior using computer vision and deep learning techniques. It features an intuitive web-based interface for easy monitoring and control.

✨ Features
Real-time Weapon Detection: Utilizes YOLOv8 for accurate and fast detection of various weapon types (knives, guns, pistols, rifles, swords, firearms).

Weapon Reality Analysis: Distinguishes between real weapons and images/screens of weapons using a custom CNN classifier and advanced image analysis (edge, texture, screen pattern detection).

Pose-based Suspicious Behavior Detection: Integrates MediaPipe for human pose estimation to identify potentially suspicious postures.

LSTM-based Behavior Analysis: Employs an LSTM model to analyze sequences of frames for complex suspicious behavior patterns.

Web-based User Interface: A modern and responsive React frontend for live video feed, alert monitoring, and system controls.

Flask Backend: A robust Python Flask server to manage detection logic, stream video, handle API requests, and store configurations and alerts.

Configurable Settings: Adjust detection thresholds, model training parameters, and video sources directly from the web UI.

Alert Logging & Categorization: Automatically logs and categorizes alerts (Real Threat, Screen/Photo, Suspicious Behavior) with timestamps and details, saving corresponding frames.

Dynamic Camera Selection: Automatically detects and allows selection of available camera devices.

Video File Input: Supports using local video files as surveillance sources.

Background Model Training: Ability to train the behavior analysis model in the background without interrupting surveillance.

🚀 Technologies Used
Backend (Python):

Flask: Web framework for building the API.

OpenCV (cv2): For video capture, frame processing, and drawing.

TensorFlow / Keras: For building and loading deep learning models (CNN for reality classification, LSTM for behavior analysis).

ultralytics (YOLOv8): State-of-the-art object detection.

MediaPipe: For human pose estimation.

NumPy: For numerical operations.

Scikit-learn: For data splitting during model training.

threading: For concurrent operations (e.g., running surveillance loop in background).

Frontend (Web):

React: JavaScript library for building the user interface.

Tailwind CSS: Utility-first CSS framework for styling.

Lucide Icons: Open-source icon library for UI elements.

HTML5 / CSS3 / JavaScript

📦 Project Structure
mizhi-weapon-detection/
├── .__pycache__/
├── alert_system.cpython-38.pyc
├── mizhi_detector.cpython-38.pyc
├── custom_dataset/
├── runs/
│   └── detect/
├── app.js
├── app.py
├── backend_surveillance.log
├── config.json
├── FullLogo_Transparent.png
├── index.html
├── setup.log
├── setup.py
├── test.py
├── train.py
├── training.log
├── weapon_detection_backend.log
├── yolo_utils.py
└── yolov8s.pt

⚙️ Setup and Installation
Prerequisites
Python 3.8 or higher

pip (Python package installer)

1. Clone the Repository
git clone https://github.com/your-username/mizhi-weapon-detection.git
cd mizhi-weapon-detection

2. Set up Python Environment
It's highly recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

3. Install Python Dependencies
pip install Flask opencv-python ultralytics tensorflow mediapipe scikit-learn

4. Create Necessary Directories
The system will automatically create config.json, models/, and alerts/ directories on first run. However, you can create them manually if you prefer:

mkdir models
mkdir alerts
mkdir alerts/real_threats
mkdir alerts/false_positives
mkdir alerts/suspicious_behavior

5. (Optional) Prepare Dataset for Behavior Model Training
If you plan to train the behavior analysis model, create a dataset directory with normal and suspicious subdirectories containing relevant image data. For example:

dataset/
├── normal/
│   ├── normal_pose_01.jpg
│   └── ...
└── suspicious/
    ├── suspicious_pose_01.jpg
    └── ...

▶️ How to Run
1. Start the Backend Server
Open your terminal or command prompt, navigate to the mizhi-weapon-detection directory, activate your virtual environment, and run the Flask application:

# Ensure your virtual environment is active
python app.py

The server will start, typically on http://0.0.0.0:5000 or http://127.0.0.1:5000. Keep this terminal window open.

2. Open the Frontend
Open the index.html file in your web browser. You can usually do this by double-clicking the file.

Important Note on CORS:
If you encounter issues like SyntaxError: Unexpected token '<' or CORS errors (Cross-Origin Resource Sharing) when opening index.html directly (i.e., file:///path/to/index.html), your browser might be blocking the connection to the Flask backend.

Recommended Solution: Use a Local HTTP Server
The most reliable way to run the frontend is to serve index.html via a simple local HTTP server.

# In a NEW terminal window (keep the Flask backend running in the first one)
# Navigate to the mizhi-weapon-detection directory
cd mizhi-weapon-detection

# Start a simple Python HTTP server
python -m http.server 8000

Then, open your browser and navigate to http://localhost:8000/index.html.

🖥️ Usage
Once the frontend is loaded:

Select Video Source:

Choose an available camera from the "Select Video Source" dropdown.

Alternatively, select "Custom Video File" and enter the full path to a video file on your system (e.g., C:/videos/test.mp4 on Windows, or /home/user/videos/test.mp4 on Linux).

Start Surveillance: Click the "Start" button to begin real-time detection. The live video feed will appear.

Monitor Alerts: The "Recent Alerts" section will display detected threats and suspicious behaviors in real-time.

Red: Real Weapon Threat

Orange: Weapon on Screen/Photo (Possible False Positive)

Yellow: Suspicious Behavior

Corresponding frames for critical alerts will be saved in the alerts/ directory.

Stop Surveillance: Click the "Stop" button to pause the detection.

Clear Alerts: Use the "Clear Alerts" button to empty the alert log displayed on the UI.

System Settings: Click the floating "Settings" button (bottom-right) to open a panel where you can:

Adjust Detection Confidence Threshold, Real Weapon Reality Threshold, and Behavior Alert Threshold.

Configure LSTM Training Epochs and LSTM Training Batch Size.

Provide a Behavior Dataset Path and initiate Train Behavior Model. Ensure surveillance is stopped before training!

🤝 Contributing
Contributions are welcome! If you have suggestions, bug reports, or want to contribute code, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details. (You'll need to create a LICENSE file in your repository if you choose the MIT License).

🙏 Acknowledgements
Ultralytics YOLOv8

TensorFlow

OpenCV

MediaPipe

Flask

React

Tailwind CSS

Lucide Icons
