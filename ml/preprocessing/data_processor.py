import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import albumentations as A

class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, config, transform=None, is_training=True):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotation_dir (str): Directory containing annotations
            config (dict): Configuration dictionary
            transform (callable, optional): Optional transform to be applied on images
            is_training (bool): Whether this is training or validation dataset
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        # Get all image files
        self.image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        # Set up augmentation pipeline
        if is_training:
            self.aug_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5 if config["preprocessing"]["augmentation"]["horizontal_flip"] else 0),
                A.VerticalFlip(p=0.5 if config["preprocessing"]["augmentation"]["vertical_flip"] else 0),
                A.Rotate(limit=config["preprocessing"]["augmentation"]["rotation_range"], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config["preprocessing"]["augmentation"]["brightness_range"],
                    contrast_limit=config["preprocessing"]["augmentation"]["contrast_range"],
                    p=0.5
                ),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation_path = self.annotation_dir / f"{image_path.stem}.txt"
        boxes = []
        labels = []
        
        if annotation_path.exists():
            with open(annotation_path) as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])
                    labels.append(class_id)
        
        # Convert to numpy arrays
        boxes = np.array(boxes)
        labels = np.array(labels)
        
        # Apply augmentations if in training mode
        if self.is_training and self.aug_pipeline:
            transformed = self.aug_pipeline(
                image=image,
                bboxes=boxes if len(boxes) > 0 else np.zeros((0, 4)),
                class_labels=labels if len(labels) > 0 else []
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['class_labels'])
        
        # Resize image
        image = cv2.resize(image, tuple(self.config["preprocessing"]["image_size"]))
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Prepare target dict for Faster R-CNN format
        target = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels)
        }
        
        return image, target

def create_data_loaders(config, image_dir, annotation_dir):
    """Create training and validation data loaders."""
    # Split data into train and validation
    dataset = ObjectDetectionDataset(image_dir, annotation_dir, config, is_training=True)
    val_size = int(len(dataset) * config["training"]["validation_split"])
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """Custom collate function for the data loader."""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets 