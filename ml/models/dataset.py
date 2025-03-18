import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, split: str = 'test'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        # Load annotations
        self.images_dir = self.root_dir / 'images' / split
        self.annotations_dir = self.root_dir / 'annotations' / split
        
        # Get list of image files
        self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        
        # Load class mapping
        with open(self.root_dir / 'classes.json', 'r') as f:
            self.class_to_idx = json.load(f)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def _get_default_transform(self):
        """Get default transform for test data."""
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _load_annotation(self, image_path: Path) -> Dict:
        """Load annotation for an image."""
        annotation_path = self.annotations_dir / f"{image_path.stem}.json"
        with open(annotation_path, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation = self._load_annotation(image_path)
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        
        for obj in annotation['objects']:
            boxes.append([
                obj['bbox'][0],  # x1
                obj['bbox'][1],  # y1
                obj['bbox'][2],  # x2
                obj['bbox'][3]   # y2
            ])
            labels.append(self.class_to_idx[obj['class']])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
        
        # Prepare target dictionary
        target = {
            'boxes': torch.as_tensor(transformed['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(transformed['class_labels'], dtype=torch.int64)
        }
        
        return transformed['image'], target

def get_test_loader(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = False
) -> DataLoader:
    """Create a DataLoader for the test dataset."""
    dataset = ObjectDetectionDataset(
        root_dir=dataset_path,
        split='test'
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets 