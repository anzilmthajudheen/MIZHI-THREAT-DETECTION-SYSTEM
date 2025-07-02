import os
import yaml
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import wandb
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure logging with UTF-8 encoding to avoid UnicodeEncodeError
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MizhiTrainer:
    """Advanced MIZHI Weapon Detection Training System"""
    
    def __init__(self, config_path: str = 'C:/Users/Ajmal/Downloads/Documents/program/mizhi/custom_dataset/dataset.yaml', use_wandb: bool = False):
        self.config_path = Path(config_path).resolve()  # Ensure absolute path
        self.use_wandb = use_wandb
        self.model = None
        self.results_dir = Path('runs/detect')
        self.project_name = 'mizhi_weapon_detection'
        self.device = self._get_best_device()
        
        # Verify config file
        self._verify_config_file()
        
        # Enhanced model configurations
        self.base_models = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt', 
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt',
            'xlarge': 'yolov8x.pt',
            'nano_p6': 'yolov8n6.pt',
            'small_p6': 'yolov8s6.pt',
            'medium_p6': 'yolov8m6.pt'
        }
        
        # Optimized training parameters with epochs set to 10
        self.training_params = {
            'epochs': 10,
            'imgsz': 640,
            'batch': self._auto_detect_batch_size(),
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'patience': 5,
            'save_period': 2,
            'workers': 8,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 2,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'device': self.device
        }
        
        # Data augmentation parameters
        self.augmentation_params = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        # Initialize W&B if requested
        if self.use_wandb:
            self._init_wandb()
    
    def _verify_config_file(self):
        """Verify that the config file exists and is accessible"""
        logger.info(f"Checking for config file at: {self.config_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        if not self.config_path.exists():
            logger.error(f"Dataset config file not found at: {self.config_path}")
            logger.error(f"Available files in directory {self.config_path.parent}: {[f.name for f in self.config_path.parent.glob('*')]}")
            logger.error(f"Parent directory contents {self.config_path.parent.parent}: {[f.name for f in self.config_path.parent.parent.glob('*')]}")
            logger.error(f"Read permissions: {os.access(str(self.config_path), os.R_OK)}")
            # Try relative path as fallback
            relative_path = Path('custom_dataset/dataset.yaml').resolve()
            logger.info(f"Trying relative path: {relative_path}")
            if relative_path.exists():
                logger.info(f"Found config file at relative path: {relative_path}")
                self.config_path = relative_path
                return
            raise FileNotFoundError(f"Dataset config file not found: {self.config_path}")
        logger.info(f"Dataset config file found: {self.config_path}")
        # Log file contents for verification
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                logger.info(f"Config file contents:\n{f.read()}")
        except Exception as e:
            logger.warning(f"Could not read config file contents: {e}")
    
    def _get_best_device(self) -> str:
        """Automatically detect the best available device"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                return ','.join([str(i) for i in range(device_count)])
            return '0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _auto_detect_batch_size(self) -> int:
        """Automatically detect optimal batch size based on available hardware"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 24:
                return 64
            elif gpu_memory >= 16:
                return 48
            elif gpu_memory >= 12:
                return 32
            elif gpu_memory >= 8:
                return 24
            elif gpu_memory >= 6:
                return 16
            else:
                return 8
        else:
            cpu_count = os.cpu_count()
            return min(8, max(2, cpu_count // 2))
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project=self.project_name,
                config={
                    'training_params': self.training_params,
                    'augmentation_params': self.augmentation_params,
                    'device': self.device
                }
            )
            logger.info("Weights & Biases logging initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def check_dataset(self) -> Dict[str, any]:
        """Enhanced dataset validation with detailed statistics"""
        logger.info("Performing comprehensive dataset validation...")
        logger.info(f"Using config file: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read config file {self.config_path}: {e}")
            raise
        
        # Extract paths and resolve to absolute
        dataset_path = Path(config.get('path', '')).resolve()
        train_path = (dataset_path / config.get('train', 'images/train')).resolve()
        val_path = (dataset_path / config.get('val', 'images/val')).resolve()
        test_path = (dataset_path / config.get('test', 'images/test')).resolve() if config.get('test') else None
        
        logger.info(f"Dataset base path: {dataset_path}")
        logger.info(f"Train path: {train_path}")
        logger.info(f"Val path: {val_path}")
        if test_path:
            logger.info(f"Test path: {test_path}")
        
        # Check directory existence
        for path_name, path in [('train', train_path), ('val', val_path)]:
            if not path.exists():
                raise FileNotFoundError(f"{path_name.capitalize()} directory not found: {path}")
        
        # Count and analyze images
        stats = {}
        for split, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
            if path and path.exists():
                images = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png'))
                labels_path = dataset_path / 'labels' / split
                labels = list(labels_path.glob('*.txt')) if labels_path.exists() else []
                
                stats[split] = {
                    'images': len(images),
                    'labels': len(labels),
                    'matched': len([img for img in images if (labels_path / f"{img.stem}.txt").exists()]),
                    'path': str(path)
                }
                
                # Analyze image dimensions if we have images
                if images:
                    sample_images = images[:min(100, len(images))]
                    dimensions = []
                    for img_path in sample_images:
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                dimensions.append(img.shape[:2])
                        except Exception:
                            continue
                    
                    if dimensions:
                        heights, widths = zip(*dimensions)
                        stats[split].update({
                            'avg_height': np.mean(heights),
                            'avg_width': np.mean(widths),
                            'min_height': np.min(heights),
                            'max_height': np.max(heights),
                            'min_width': np.min(widths),
                            'max_width': np.max(widths)
                        })
        
        # Analyze class distribution
        class_names = config.get('names', {})
        class_counts = {name: 0 for name in class_names.values()}
        
        for split in ['train', 'val', 'test']:
            if split in stats:
                labels_path = dataset_path / 'labels' / split
                if labels_path.exists():
                    for label_file in labels_path.glob('*.txt'):
                        try:
                            with open(label_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        class_id = int(parts[0])
                                        if class_id in class_names:
                                            class_counts[class_names[class_id]] += 1
                        except Exception:
                            continue
        
        stats['classes'] = class_counts
        stats['total_annotations'] = sum(class_counts.values())
        
        # Validation checks
        issues = []
        if stats['train']['images'] == 0:
            issues.append("No training images found")
        if stats['val']['images'] == 0:
            issues.append("No validation images found")
        if stats['train']['matched'] < stats['train']['images']:
            issues.append(f"Missing labels: {stats['train']['images'] - stats['train']['matched']} training images without labels")
        if stats['val']['matched'] < stats['val']['images']:
            issues.append(f"Missing labels: {stats['val']['images'] - stats['val']['matched']} validation images without labels")
        
        # Log statistics
        logger.info("Dataset Statistics:")
        for split, data in stats.items():
            if split != 'classes' and split != 'total_annotations':
                logger.info(f"  {split.capitalize()}: {data['images']} images, {data['matched']} with labels")
                if 'avg_height' in data:
                    logger.info(f"    Avg dimensions: {data['avg_width']:.0f}x{data['avg_height']:.0f}")
        
        logger.info(f"  Total annotations: {stats['total_annotations']}")
        logger.info("  Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"    {class_name}: {count}")
        
        if issues:
            logger.warning("Dataset issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Dataset validation passed")
        
        return stats
    
    def prepare_training_config(self, model_size: str, custom_params: Optional[Dict] = None) -> Dict:
        """Prepare optimized training configuration"""
        if model_size not in self.base_models:
            logger.warning(f"Unknown model size '{model_size}'. Using 'small' instead.")
            model_size = 'small'
        
        config = self.training_params.copy()
        config.update(self.augmentation_params)
        
        if custom_params:
            config.update(custom_params)
        
        if model_size in ['nano', 'nano_p6']:
            config.update({
                'lr0': 0.01,
                'warmup_epochs': 3,
                'close_mosaic': 2
            })
        elif model_size in ['xlarge', 'medium_p6']:
            config.update({
                'lr0': 0.005,
                'warmup_epochs': 3,
                'close_mosaic': 2
            })
        
        return config
    
    def train_model(self, model_size: str = 'small', custom_params: Optional[Dict] = None,
                   resume: bool = False, experiment_name: Optional[str] = None) -> Optional[object]:
        """Enhanced model training with advanced features"""
        dataset_stats = self.check_dataset()
        config = self.prepare_training_config(model_size, custom_params)
        base_model = self.base_models[model_size]
        
        logger.info(f"Initializing {model_size} model: {base_model}")
        
        if resume and self.find_latest_checkpoint():
            checkpoint_path = self.find_latest_checkpoint()
            self.model = YOLO(checkpoint_path)
            logger.info(f"Resuming training from: {checkpoint_path}")
        else:
            self.model = YOLO(base_model)
        
        if not experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"{self.project_name}_{model_size}_{timestamp}"
        
        logger.info("Training Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        
        try:
            logger.info(f"Passing data path to YOLO: {self.config_path}")
            results = self.model.train(
                data=str(self.config_path),
                name=experiment_name,
                project=str(self.results_dir),
                **config
            )
            
            logger.info("Training completed successfully!")
            self._post_training_analysis(experiment_name, results, dataset_stats)
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def _post_training_analysis(self, experiment_name: str, results: object, dataset_stats: Dict):
        """Perform comprehensive post-training analysis"""
        results_path = self.results_dir / experiment_name
        
        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'best_model_path': str(results_path / 'weights' / 'best.pt'),
                'last_model_path': str(results_path / 'weights' / 'last.pt'),
            },
            'training_config': self.training_params,
            'dataset_stats': dataset_stats,
            'device_info': {
                'device': self.device,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
        }
        
        summary_path = results_path / 'training_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved: {summary_path}")
        logger.info(f"Best model saved: {summary['model_info']['best_model_path']}")
        
        self._generate_analysis_plots(results_path)
        
        if self.use_wandb:
            wandb.log({
                'training_complete': True,
                'best_model_path': summary['model_info']['best_model_path'],
                'dataset_stats': dataset_stats
            })
    
    def _generate_analysis_plots(self, results_path: Path):
        """Generate additional analysis plots"""
        try:
            results_csv = results_path / 'results.csv'
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('MIZHI Training Analysis', fontsize=16)
                
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', alpha=0.8)
                axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', alpha=0.8)
                axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', alpha=0.8)
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', alpha=0.8)
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', alpha=0.8)
                axes[0, 1].set_title('Validation mAP')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', alpha=0.8)
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', alpha=0.8)
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                if 'lr/pg0' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['lr/pg0'], alpha=0.8)
                    axes[1, 1].set_title('Learning Rate Schedule')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Learning Rate')
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(results_path / 'training_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("Analysis plots generated")
                
        except Exception as e:
            logger.warning(f"Failed to generate analysis plots: {e}")
    
    def validate_model(self, model_path: Optional[str] = None, save_results: bool = True) -> Dict:
        """Enhanced model validation with detailed metrics"""
        if model_path is None:
            model_path = self.find_latest_model()
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Validating model: {model_path}")
        
        model = YOLO(model_path)
        metrics = model.val(
            data=str(self.config_path),
            device=self.device,
            save_json=save_results,
            save_hybrid=save_results
        )
        
        validation_results = {
            'model_path': model_path,
            'validation_time': datetime.now().isoformat(),
            'metrics': {
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'f1_score': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)) if (metrics.box.mp + metrics.box.mr) > 0 else 0.0
            },
            'per_class_metrics': {}
        }
        
        if hasattr(metrics.box, 'ap_class_index') and len(metrics.box.ap_class_index) > 0:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            class_names = config.get('names', {})
            
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                if class_idx in class_names:
                    validation_results['per_class_metrics'][class_names[class_idx]] = {
                        'ap50': float(metrics.box.ap50[i]),
                        'ap50_95': float(metrics.box.ap[i])
                    }
        
        logger.info("Validation Results:")
        logger.info(f"  mAP@0.5: {validation_results['metrics']['mAP50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {validation_results['metrics']['mAP50_95']:.4f}")
        logger.info(f"  Precision: {validation_results['metrics']['precision']:.4f}")
        logger.info(f"  Recall: {validation_results['metrics']['recall']:.4f}")
        logger.info(f"  F1-Score: {validation_results['metrics']['f1_score']:.4f}")
        
        if validation_results['per_class_metrics']:
            logger.info("  Per-class AP@0.5:")
            for class_name, metrics_dict in validation_results['per_class_metrics'].items():
                logger.info(f"    {class_name}: {metrics_dict['ap50']:.4f}")
        
        if save_results:
            results_path = Path(model_path).parent.parent / 'validation_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"Validation results saved: {results_path}")
        
        return validation_results
    
    def find_latest_model(self) -> Optional[str]:
        """Find the most recently trained model"""
        if not self.results_dir.exists():
            return None
        
        mizhi_dirs = [d for d in self.results_dir.iterdir() 
                     if d.is_dir() and self.project_name in d.name]
        
        if not mizhi_dirs:
            return None
        
        latest_dir = max(mizhi_dirs, key=lambda d: d.stat().st_mtime)
        model_path = latest_dir / 'weights' / 'best.pt'
        
        return str(model_path) if model_path.exists() else None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for resuming training"""
        if not self.results_dir.exists():
            return None
        
        mizhi_dirs = [d for d in self.results_dir.iterdir() 
                     if d.is_dir() and self.project_name in d.name]
        
        if not mizhi_dirs:
            return None
        
        latest_dir = max(mizhi_dirs, key=lambda d: d.stat().st_mtime)
        checkpoint_path = latest_dir / 'weights' / 'last.pt'
        
        return str(checkpoint_path) if checkpoint_path.exists() else None
    
    def export_model(self, model_path: Optional[str] = None, 
                    formats: List[str] = ['onnx', 'torchscript', 'tflite']) -> Dict[str, str]:
        """Enhanced model export with multiple formats"""
        if model_path is None:
            model_path = self.find_latest_model()
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found for export")
        
        logger.info(f"Exporting model: {model_path}")
        
        model = YOLO(model_path)
        export_results = {}
        
        for format_type in formats:
            try:
                exported_path = model.export(
                    format=format_type,
                    imgsz=640,
                    optimize=True,
                    half=format_type in ['onnx', 'tflite'],
                    int8=format_type == 'tflite'
                )
                export_results[format_type] = str(exported_path)
                logger.info(f"Successfully exported to {format_type}: {exported_path}")
            except Exception as e:
                logger.error(f"Failed to export to {format_type}: {e}")
                export_results[format_type] = None
        
        return export_results
    
    def cleanup_old_runs(self, keep_last: int = 5):
        """Clean up old training runs to save disk space"""
        if not self.results_dir.exists():
            return
        
        mizhi_dirs = [d for d in self.results_dir.iterdir() 
                     if d.is_dir() and self.project_name in d.name]
        
        if len(mizhi_dirs) <= keep_last:
            return
        
        mizhi_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        dirs_to_remove = mizhi_dirs[keep_last:]
        
        for dir_path in dirs_to_remove:
            try:
                import shutil
                shutil.rmtree(dir_path)
                logger.info(f"Removed old run: {dir_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {dir_path}: {e}")

def main():
    """Enhanced main function with comprehensive CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='MIZHI Enhanced Weapon Detection Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model small --config standard
  python train.py --model medium --epochs 10 --batch 16 --wandb
  python train.py --validate --export --cleanup
        """
    )
    
    parser.add_argument('--model', default='small', 
                       choices=['nano', 'small', 'medium', 'large', 'xlarge', 'nano_p6', 'small_p6', 'medium_p6'],
                       help='Model size to use')
    parser.add_argument('--config', default='standard',
                       help='Training configuration preset')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--imgsz', type=int, help='Image size')
    parser.add_argument('--lr0', type=float, help='Initial learning rate')
    parser.add_argument('--device', help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the model after training')
    parser.add_argument('--export', action='store_true',
                       help='Export the model after training')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old training runs')
    parser.add_argument('--name', help='Experiment name')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--data', default='C:/Users/Ajmal/Downloads/Documents/program/mizhi/custom_dataset/dataset.yaml',
                       help='Path to dataset configuration file')
    
    args = parser.parse_args()
    
    trainer = MizhiTrainer(config_path=args.data, use_wandb=args.wandb)
    
    if args.cleanup:
        trainer.cleanup_old_runs()
    
    custom_params = {}
    if args.epochs:
        custom_params['epochs'] = args.epochs
    if args.batch:
        custom_params['batch'] = args.batch
    if args.imgsz:
        custom_params['imgsz'] = args.imgsz
    if args.lr0:
        custom_params['lr0'] = args.lr0
    if args.device:
        custom_params['device'] = args.device
    
    try:
        logger.info(f"Starting training: model={args.model}, config={args.config}")
        
        results = trainer.train_model(
            model_size=args.model,
            custom_params=custom_params if custom_params else None,
            resume=args.resume,
            experiment_name=args.name
        )
        
        if results is None:
            logger.error("Training failed!")
            return 1
        
        if args.validate:
            trainer.validate_model()
        
        if args.export:
            trainer.export_model()
        
        logger.info("Training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())