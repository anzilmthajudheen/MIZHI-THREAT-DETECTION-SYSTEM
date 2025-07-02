import os
import yaml
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required Python packages"""
    try:
        required_packages = [
            'torch>=1.8.0',
            'ultralytics>=8.0.0',
            'numpy>=1.18.5',
            'matplotlib>=3.2.2',
            'seaborn>=0.11.0',
            'pandas>=1.1.3',
            'opencv-python>=4.5.0',
            'pyyaml>=5.3.1',
            'wandb>=0.12.0'
        ]
        logger.info("Installing required packages...")
        for package in required_packages:
            subprocess.check_call(['pip', 'install', package])
        logger.info("✓ All dependencies installed successfully")
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        raise

def verify_dataset(data_yaml_path: str):
    """Verify dataset structure based on data.yaml"""
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        logger.error(f"Dataset config file not found: {data_yaml_path}")
        raise FileNotFoundError(f"Dataset config file not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config.get('path', 'C:/Users/Ajmal/Downloads/Documents/program/mizhi/custom_dataset/dataset'))
    train_path = dataset_path / config.get('train', 'C:/Users/Ajmal/Downloads/Documents/program/mizhi/custom_dataset/dataset/images/train')
    val_path = dataset_path / config.get('val', 'C:/Users/Ajmal/Downloads/Documents/program/mizhi/custom_dataset/dataset/images/val')
    test_path = dataset_path / config.get('test', '') if config.get('test') else None
    
    # Verify directories
    for path_name, path in [('train', train_path), ('val', val_path)]:
        if not path.exists():
            logger.error(f"{path_name.capitalize()} directory not found: {path}")
            raise FileNotFoundError(f"{path_name.capitalize()} directory not found: {path}")
    
    # Verify label directories
    label_train_path = dataset_path / 'labels' / 'train'
    label_val_path = dataset_path / 'labels' / 'val'
    for path_name, path in [('train labels', label_train_path), ('val labels', label_val_path)]:
        if not path.exists():
            logger.error(f"{path_name.capitalize()} directory not found: {path}")
            raise FileNotFoundError(f"{path_name.capitalize()} directory not found: {path}")
    
    # Count images and labels
    stats = {}
    for split, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if path and path.exists():
            images = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png'))
            labels_path = dataset_path / 'labels' / split if split != 'test' else None
            labels = list(labels_path.glob('*.txt')) if labels_path and labels_path.exists() else []
            
            stats[split] = {
                'images': len(images),
                'labels': len(labels),
                'matched': len([img for img in images if labels_path and (labels_path / f"{img.stem}.txt").exists()])
            }
    
    # Log dataset statistics
    logger.info("Dataset Verification Results:")
    for split, data in stats.items():
        if split != 'test' or data['images'] > 0:
            logger.info(f"  {split.capitalize()}: {data['images']} images, {data['labels']} labels, {data['matched']} matched")
            if data['images'] != data['matched'] and split != 'test':
                logger.warning(f"  {split.capitalize()} mismatch: {data['images'] - data['matched']} images without labels")
    
    if stats.get('train', {}).get('images', 0) == 0:
        logger.error("No training images found")
        raise ValueError("No training images found")
    if stats.get('val', {}).get('images', 0) == 0:
        logger.error("No validation images found")
        raise ValueError("No validation images found")
    
    logger.info("✓ Dataset verification passed")
    return stats

def main():
    """Main function to set up the weapon detection environment"""
    try:
        # Install dependencies
        install_dependencies()
        
        # Verify dataset
        data_yaml_path = 'C:/Users/Ajmal/Downloads/Documents/program/mizhi/custom_dataset/dataset.yaml'
        verify_dataset(data_yaml_path)
        
        logger.info("Setup completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1

if __name__ == '__main__':
     exit(main())