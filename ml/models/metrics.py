import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """Calculate detection metrics including precision, recall, and mAP."""
    metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'mAP50': 0.0,
        'mAP50_95': 0.0,
        'inference_time': 0.0
    }
    
    # Initialize counters
    total_predictions = 0
    total_ground_truth = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Process each image
    for pred, gt in zip(predictions, ground_truth):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        
        gt_boxes = gt['boxes'].cpu().numpy()
        gt_labels = gt['labels'].cpu().numpy()
        
        # Sort predictions by confidence
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        
        for i in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1
            
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and pred_labels[i] == gt_labels[best_gt_idx]:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                false_positives += 1
        
        false_negatives += len(gt_boxes) - len(matched_gt)
        total_predictions += len(pred_boxes)
        total_ground_truth += len(gt_boxes)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['mAP50'] = precision * recall  # Simplified mAP calculation
    metrics['mAP50_95'] = metrics['mAP50']  # In a real implementation, this would be calculated with multiple IoU thresholds
    
    return metrics 