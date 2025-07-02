from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import json
import yaml
from datetime import datetime
import threading
import time
from ultralytics import YOLO
import logging
from pathlib import Path
from werkzeug.utils import secure檔名

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mizhi_security_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load configuration
config_path = 'config.json'
if not os.path.exists(config_path):
    logger.error(f"Config file not found: {config_path}")
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, 'r') as f:
    config = json.load(f)

class WeaponDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.threat_classes = config.get('weapon_classes', ['Knife', 'Handgun'])
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.data_yaml = Path(config.get('dataset_path', 'data.yaml'))
        self.class_names = self._load_class_names()
        self.load_model()
    
    def _load_class_names(self):
        """Load class names from data.yaml"""
        if not self.data_yaml.exists():
            logger.error(f"Data YAML file not found: {self.data_yaml}")
            raise FileNotFoundError(f"Data YAML file not found: {self.data_yaml}")
        
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config.get('names', {})
    
    def load_model(self):
        """Load the YOLO model for weapon detection"""
        model_path = config.get('model_path', 'runs/detect/mizhi_weapon_detection_small/weights/best.pt')
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom trained model: {model_path}")
            else:
                logger.warning(f"Custom model not found at {model_path}, loading pretrained YOLOv8 model")
                self.model = YOLO('yolov8n.pt')
            
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
    
    def detect_weapons(self, image):
        """Detect weapons in the given image"""
        if not self.model_loaded:
            logger.error("Model not loaded")
            return None, []
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold)
            
            # Process results
            threats_detected = []
            annotated_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'Unknown').lower()
                        confidence = float(box.conf[0])
                        
                        is_threat = class_name in [cls.lower() for cls in self.threat_classes]
                        
                        if is_threat and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw bounding box
                            color = (0, 0, 255)  # Red for threats
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label
                            label = f"{self.class_names.get(class_id, 'Unknown')}: {confidence:.2f}"
                            cv2.putText(annotated_image, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            threats_detected.append({
                                'class': self.class_names.get(class_id, 'Unknown'),
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2]
                            })
            
            return annotated_image, threats_detected
            
        except Exception as e:
            logger.error(f"Error in weapon detection: {e}")
            return None, []

# Initialize weapon detector
weapon_detector = WeaponDetector()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("index.html not found")
        return """
        <h1>MIZHI Security System</h1>
        <p>Backend is running. Please ensure index.html is in the same directory.</p>
        """

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': weapon_detector.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file uploads"""
    try:
        if 'video' not in request.files:
            logger.error("No video file provided in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video in background
        threading.Thread(target=process_uploaded_video, args=(filepath, filename)).start()
        
        logger.info(f"Video uploaded: {filename}")
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename,
            'status': 'processing'
        })
    
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return jsonify({'error': str(e)}), 500

def process_uploaded_video(filepath, filename):
    """Process uploaded video for weapon detection"""
    try:
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        threat_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame to balance speed and accuracy
            if frame_count % 10 == 0:
                annotated_frame, threats = weapon_detector.detect_weapons(frame)
                
                if threats:
                    threat_detections.extend(threats)
                    
                    # Emit alert to frontend
                    socketio.emit('video_analysis_alert', {
                        'filename': filename,
                        'frame': frame_count,
                        'threats': threats,
                        'timestamp': datetime.now().isoformat()
                    })
        
        cap.release()
        
        # Send final analysis result
        socketio.emit('video_analysis_complete', {
            'filename': filename,
            'total_frames': frame_count,
            'threats_detected': len(threat_detections),
            'analysis_complete': True
        })
        logger.info(f"Video processing completed: {filename}, {len(threat_detections)} threats detected")
        
    except Exception as e:
        logger.error(f"Error processing video {filename}: {e}")
        socketio.emit('video_analysis_error', {
            'filename': filename,
            'error': str(e)
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('server_status', {
        'status': 'Connected to MIZHI backend',
        'model_loaded': weapon_detector.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Handle incoming video frames from webcam"""
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Detect weapons
        annotated_image, threats = weapon_detector.detect_weapons(opencv_image)
        
        if annotated_image is not None:
            # Convert back to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
            processed_image_data = f"data:image/jpeg;base64,{processed_image_b64}"
            
            # Prepare response
            response_data = {
                'image': processed_image_data,
                'threat_detected': len(threats) > 0,
                'threats_count': len(threats),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add alerts if threats detected
            if threats:
                alerts = []
                for threat in threats:
                    alerts.append(f"{threat['class']} detected (confidence: {threat['confidence']:.2f})")
                response_data['alerts'] = alerts
            
            emit('processed_frame', response_data)
            logger.info(f"Frame processed, {len(threats)} threats detected")
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        emit('processing_error', {'message': str(e)})

@socketio.on('system_command')
def handle_system_command(data):
    """Handle system commands from frontend"""
    command = data.get('command')
    
    if command == 'reload_model':
        weapon_detector.load_model()
        emit('server_status', {
            'status': 'Model reloaded',
            'model_loaded': weapon_detector.model_loaded
        })
        logger.info("Model reloaded via system command")
    
    elif command == 'get_stats':
        emit('system_stats', {
            'model_loaded': weapon_detector.model_loaded,
            'confidence_threshold': weapon_detector.confidence_threshold,
            'threat_classes': weapon_detector.threat_classes
        })
        logger.info("System stats requested")

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Handle settings configuration"""
    if request.method == 'GET':
        return jsonify({
            'confidence_threshold': weapon_detector.confidence_threshold,
            'threat_classes': weapon_detector.threat_classes
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'confidence_threshold' in data:
            weapon_detector.confidence_threshold = max(0.1, min(1.0, data['confidence_threshold']))
            logger.info(f"Updated confidence threshold to {weapon_detector.confidence_threshold}")
        
        if 'threat_classes' in data:
            weapon_detector.threat_classes = data['threat_classes']
            logger.info(f"Updated threat classes to {weapon_detector.threat_classes}")
        
        return jsonify({'message': 'Settings updated successfully'})

if __name__ == '__main__':
    logger.info("Starting MIZHI Security Surveillance System")
    logger.info(f"Model loaded: {weapon_detector.model_loaded}")
    logger.info(f"Threat classes: {weapon_detector.threat_classes}")
    logger.info(f"Confidence threshold: {weapon_detector.confidence_threshold}")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)