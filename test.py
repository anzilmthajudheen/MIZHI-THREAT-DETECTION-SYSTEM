import os
import cv2
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeaponDetector:
    """Weapon Detection System for Inference"""
    
    def __init__(self, model_path: str, data_yaml: str = 'data.yaml', confidence: float = 0.5):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.confidence = confidence
        self.model = None
        self.class_names = self._load_class_names()
        self.output_dir = Path('runs/detect/inference')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from data.yaml"""
        if not self.data_yaml.exists():
            logger.error(f"Data YAML file not found: {self.data_yaml}")
            raise FileNotFoundError(f"Data YAML file not found: {self.data_yaml}")
        
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('names', {})
    
    def load_model(self):
        """Load the trained YOLO model"""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✓ Model loaded successfully: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, input_path: str, save_results: bool = True, show_results: bool = False) -> List[Dict]:
        """Perform weapon detection on an image or video"""
        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if self.model is None:
            self.load_model()
        
        # Perform inference
        results = self.model.predict(
            source=str(input_path),
            conf=self.confidence,
            save=save_results,
            save_txt=save_results,
            save_conf=save_results,
            project=str(self.output_dir),
            name=input_path.stem,
            exist_ok=True
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            detection_info = []
            
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy.cpu().numpy()[0]
                
                detection_info.append({
                    'class': self.class_names.get(cls_id, 'Unknown'),
                    'confidence': conf,
                    'bbox': {
                        'x1': float(xyxy[0]),
                        'y1': float(xyxy[1]),
                        'x2': float(xyxy[2]),
                        'y2': float(xyxy[3])
                    }
                })
            
            detections.append({
                'image': str(result.path),
                'detections': detection_info
            })
            
            # Visualize results if requested
            if show_results and result.orig_img is not None:
                img = result.orig_img.copy()
                for box in boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.class_names.get(cls_id, 'Unknown')} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"Detections for {Path(result.path).name}")
                plt.show()
        
        # Log detection results
        for detection in detections:
            logger.info(f"Image: {detection['image']}")
            for det in detection['detections']:
                logger.info(f"  Detected {det['class']} with confidence {det['confidence']:.2f} at {det['bbox']}")
        
        return detections

def main():
    """Main function for weapon detection testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MIZHI Weapon Detection Testing System')
    parser.add_argument('--model', default='runs/detect/mizhi_weapon_detection_small/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--input', required=True,
                       help='Path to input image or video')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for detection')
    parser.add_argument('--data', default='data.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--show', action='store_true',
                       help='Show detection results visually')
    
    args = parser.parse_args()
    
    try:
        detector = WeaponDetector(
            model_path=args.model,
            data_yaml=args.data,
            confidence=args.conf
        )
        
        detections = detector.detect(
            input_path=args.input,
            save_results=True,
            show_results=args.show
        )
        
        logger.info("✓ Detection completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())