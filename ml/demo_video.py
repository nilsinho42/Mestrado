import argparse
from pathlib import Path
import sys
import yaml

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent))
from models.yolo_model import YOLODetector

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Real-time object detection demo')
    parser.add_argument('--source', type=str, default='0',
                      help='Source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='models/yolo_final.pt',
                      help='Path to trained model')
    parser.add_argument('--save', type=str, default=None,
                      help='Path to save processed video')
    parser.add_argument('--no-display', action='store_true',
                      help='Disable display window')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize model
    model = YOLODetector(config)
    
    # Load trained weights if specified
    if args.model and Path(args.model).exists():
        model.load_model(args.model)
        print(f"Loaded model from {args.model}")
    
    # Process video stream
    try:
        source = 0 if args.source == '0' else args.source
        stats = model.process_video_stream(
            source=source,
            display=not args.no_display,
            save_path=args.save
        )
        
        # Print statistics
        print("\nProcessing Statistics:")
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Total Vehicles Detected: {stats['total_vehicles']}")
        print(f"Total Persons Detected: {stats['total_persons']}")
        print(f"Average Vehicles per Frame: {stats['avg_vehicles_per_frame']:.2f}")
        print(f"Average Persons per Frame: {stats['avg_persons_per_frame']:.2f}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")

if __name__ == "__main__":
    main() 