import os
import yaml
import mlflow
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import gdown
import zipfile

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from models.yolo_model import YOLODetector

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_mlflow(config):
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri(config["mlops"]["experiment_tracking"]["mlflow_tracking_uri"])
    mlflow.set_experiment("object-detection-flow")

def download_bdd100k():
    """Download BDD100K dataset."""
    data_dir = Path("datasets/bdd100k")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not (data_dir / "images").exists():
        print("Downloading BDD100K dataset...")
        # BDD100K requires registration. You'll need to download it manually from:
        # https://bdd-data.berkeley.edu/
        print("Please download BDD100K from https://bdd-data.berkeley.edu/")
        print("Extract it to datasets/bdd100k/")
        return None
    
    return str(data_dir / "data.yaml")

def download_citypersons():
    """Download CityPersons dataset."""
    data_dir = Path("datasets/citypersons")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not (data_dir / "images").exists():
        print("Downloading CityPersons dataset...")
        # CityPersons requires registration. You'll need to download it manually from:
        # https://www.cityscapes-dataset.com/
        print("Please download CityPersons from https://www.cityscapes-dataset.com/")
        print("Extract it to datasets/citypersons/")
        return None
    
    return str(data_dir / "data.yaml")

def train_model(config, model_name):
    """Train/fine-tune the specified model on both datasets."""
    training_config = config["training"]
    
    # Initialize model
    if model_name == "yolo":
        model = YOLODetector(config)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Get dataset paths
    bdd100k_path = download_bdd100k()
    citypersons_path = download_citypersons()
    
    if not bdd100k_path or not citypersons_path:
        print("Please download the datasets and try again.")
        return
    
    # Training on BDD100K
    print("\nTraining on BDD100K for vehicle detection...")
    with mlflow.start_run(run_name=f"{model_name}-bdd100k-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({
            "model_type": model_name,
            "dataset": "bdd100k",
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "epochs": training_config["epochs"]
        })
        
        try:
            # Train on BDD100K
            results = model.train_on_bdd100k(bdd100k_path, training_config["epochs"])
            
            # Evaluate on BDD100K
            val_metrics = model.evaluate(bdd100k_path, 'bdd100k')
            
            # Log metrics
            metrics = {
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_mAP50": val_metrics["mAP50"],
                "val_mAP50-95": val_metrics["mAP50-95"]
            }
            
            mlflow.log_metrics(metrics)
            
            print("\nBDD100K Validation Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Save the vehicle detection model
            model_path = f"models/{model_name}_vehicles.pt"
            model.save_model(model_path)
            mlflow.log_artifact(model_path)
            
        except Exception as e:
            print(f"Error during BDD100K training: {str(e)}")
            mlflow.log_param("error", str(e))
            raise
    
    # Training on CityPersons
    print("\nTraining on CityPersons for pedestrian detection...")
    with mlflow.start_run(run_name=f"{model_name}-citypersons-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({
            "model_type": model_name,
            "dataset": "citypersons",
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "epochs": training_config["epochs"]
        })
        
        try:
            # Train on CityPersons
            results = model.train_on_citypersons(citypersons_path, training_config["epochs"])
            
            # Evaluate on CityPersons
            val_metrics = model.evaluate(citypersons_path, 'citypersons')
            
            # Log metrics
            metrics = {
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_mAP50": val_metrics["mAP50"],
                "val_mAP50-95": val_metrics["mAP50-95"]
            }
            
            mlflow.log_metrics(metrics)
            
            print("\nCityPersons Validation Results:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Save the person detection model
            model_path = f"models/{model_name}_persons.pt"
            model.save_model(model_path)
            mlflow.log_artifact(model_path)
            
        except Exception as e:
            print(f"Error during CityPersons training: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

def main():
    """Main training pipeline."""
    config = load_config()
    setup_mlflow(config)
    
    # Train YOLO model on both datasets
    print("\nTraining YOLO model...")
    train_model(config, "yolo")

if __name__ == "__main__":
    main() 