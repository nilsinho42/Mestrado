import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class FasterRCNNDetector:
    def __init__(self, config):
        """Initialize Faster R-CNN model with configuration."""
        self.config = config["models"]["faster_rcnn"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(self.config["classes"]) + 1  # +1 for background
        
        # Load pre-trained backbone
        backbone = torchvision.models.resnet50(pretrained=True)
        backbone_output = 2048  # ResNet50 output channels
        
        # Remove the last two layers (avg pool and fc)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Configure anchor generator
        anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in self.config["anchor_sizes"]),
            aspect_ratios=tuple((0.5, 1.0, 2.0) for _ in range(len(self.config["anchor_sizes"])))
        )
        
        # Create Faster R-CNN model
        self.model = FasterRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_train=self.config["rpn_pre_nms_top_n"],
            rpn_pre_nms_top_n_test=self.config["rpn_pre_nms_top_n"],
            rpn_post_nms_top_n_train=self.config["rpn_post_nms_top_n"],
            rpn_post_nms_top_n_test=self.config["rpn_post_nms_top_n"],
            rpn_nms_thresh=self.config["rpn_nms_threshold"],
            box_detections_per_img=100
        )
        
        self.model.to(self.device)
        
    def train_one_epoch(self, data_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for images, targets in data_loader:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        """Evaluate the model."""
        self.model.eval()
        metrics = {
            "precision": [],
            "recall": [],
            "mAP": []
        }
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                predictions = self.model(images)
                
                # Calculate metrics (simplified version)
                for pred, target in zip(predictions, targets):
                    # TODO: Implement proper mAP calculation
                    pass
                
        return metrics
    
    def predict(self, image):
        """Run inference on a single image."""
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([image.to(self.device)])
        return prediction[0]
    
    def save_model(self, path):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        
    def load_model(self, path):
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config'] 